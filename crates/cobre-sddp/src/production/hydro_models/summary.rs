//! Display-summary builder for the hydro model preprocessing pipeline.
//!
//! Aggregates the resolved production and evaporation models plus their
//! provenance into a `HydroModelSummary` consumed by `cobre-cli` for display.
//! All counts are derived from the already-validated pipeline result; summary
//! construction is infallible.

use cobre_core::System;

use super::types::{
    EvaporationReferenceSource, EvaporationSource, FphaHydroDetail, HydroModelSummary,
    PrepareHydroModelsResult, ProductionModelSource, ResolvedProductionModel,
};
// ── Summary builder ───────────────────────────────────────────────────────────

/// Build a [`HydroModelSummary`] from the pipeline result and the system;
/// infallible (all fields derive from the already-validated inputs).
///
/// `total_planes` sums plane counts from the first study stage only — taken as
/// representative for display.
#[must_use]
pub fn build_hydro_model_summary(
    result: &PrepareHydroModelsResult,
    system: &System,
) -> HydroModelSummary {
    let mut n_constant = 0usize;
    let mut n_fpha = 0usize;
    let mut total_planes = 0usize;
    let mut fpha_details: Vec<FphaHydroDetail> = Vec::new();

    let has_study_stage = system.stages().iter().any(|s| s.id >= 0);
    let representative_stage = 0usize;

    for (hydro_pos, (entity_id, source)) in result.provenance.production_sources.iter().enumerate()
    {
        match source {
            ProductionModelSource::DefaultConstant => {
                n_constant += 1;
            }
            ProductionModelSource::PrecomputedHyperplanes
            | ProductionModelSource::ComputedFromGeometry => {
                n_fpha += 1;

                let n_planes = if has_study_stage {
                    match result.production.model(hydro_pos, representative_stage) {
                        ResolvedProductionModel::Fpha { planes, .. } => planes.len(),
                        ResolvedProductionModel::ConstantProductivity { .. } => 0,
                    }
                } else {
                    0
                };

                total_planes += n_planes;

                let name = system
                    .hydros()
                    .iter()
                    .find(|h| h.id == *entity_id)
                    .map_or_else(|| entity_id.0.to_string(), |h| h.name.clone());

                fpha_details.push(FphaHydroDetail {
                    hydro_id: *entity_id,
                    name,
                    source: *source,
                    n_planes,
                });
            }
        }
    }

    let mut n_evaporation = 0usize;
    let mut n_no_evaporation = 0usize;

    for (_, source) in &result.provenance.evaporation_sources {
        match source {
            EvaporationSource::LinearizedFromGeometry => n_evaporation += 1,
            EvaporationSource::NotModeled | EvaporationSource::DisabledNoArea => {
                n_no_evaporation += 1;
            }
        }
    }

    let mut n_user_supplied_ref = 0usize;
    let mut n_default_midpoint_ref = 0usize;

    for ((_, evap_src), (_, ref_src)) in result
        .provenance
        .evaporation_sources
        .iter()
        .zip(result.provenance.evaporation_reference_sources.iter())
    {
        if *evap_src == EvaporationSource::LinearizedFromGeometry {
            match ref_src {
                EvaporationReferenceSource::UserSupplied => n_user_supplied_ref += 1,
                EvaporationReferenceSource::DefaultMidpoint => n_default_midpoint_ref += 1,
            }
        }
    }

    HydroModelSummary {
        n_constant,
        n_fpha,
        total_planes,
        fpha_details,
        n_evaporation,
        n_no_evaporation,
        n_user_supplied_ref,
        n_default_midpoint_ref,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        Bus, DeficitSegment, EntityId, Hydro, SystemBuilder,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        scenario::CorrelationModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    // The summary tests construct production and evaporation result types owned by
    // sibling submodules; import the full re-exported surface from the directory
    // module rather than only `summary`'s own narrow `use` block.
    use crate::HydroEnergyProductivityOverride;
    use crate::production::hydro_models::*;
    use crate::test_support::minimal_hydros;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn zero_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32, model: HydroGenerationModel) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 100.0,
            max_storage_hm3: 2000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: model,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId::from(10));
        hydro
    }

    /// Build a minimal single-bus `System` with the given hydros and one study stage.
    ///
    /// Uses bus `EntityId(10)` to match the `make_hydro` helper's `bus_id`.
    fn make_system_for_summary(hydros: Vec<Hydro>) -> cobre_core::System {
        let bus = Bus {
            id: EntityId(10),
            name: "B10".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        };
        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(vec![stage])
            .correlation(CorrelationModel::default())
            .build()
            .unwrap()
    }

    /// Build a `PrepareHydroModelsResult` for a set of hydro IDs where every
    /// hydro uses constant productivity and has no evaporation.
    fn make_result_all_constant(hydro_ids: &[i32]) -> PrepareHydroModelsResult {
        let n_hydros = hydro_ids.len();
        let n_stages = 1;
        let models: Vec<Vec<ResolvedProductionModel>> = hydro_ids
            .iter()
            .map(|_| {
                (0..n_stages)
                    .map(|_| ResolvedProductionModel::ConstantProductivity { productivity: 0.95 })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, &minimal_hydros(n_hydros), n_stages);
        let production_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), ProductionModelSource::DefaultConstant))
            .collect();
        let evaporation_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), EvaporationSource::NotModeled))
            .collect();
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        let evap_models: Vec<EvaporationModel> =
            hydro_ids.iter().map(|_| EvaporationModel::None).collect();
        PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        }
    }

    /// Build a `PrepareHydroModelsResult` with mixed constant and FPHA hydros.
    ///
    /// The hydro list is merged and sorted by id ascending (canonical order).
    /// Each FPHA hydro gets `n_planes` hyperplanes at the single study stage.
    fn make_result_mixed(
        constant_ids: &[i32],
        fpha_ids: &[i32],
        n_planes: usize,
    ) -> PrepareHydroModelsResult {
        let n_stages = 1;
        let fpha_plane = FphaPlane {
            intercept: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
        };
        let mut all_ids: Vec<(i32, bool)> = constant_ids
            .iter()
            .map(|&id| (id, false))
            .chain(fpha_ids.iter().map(|&id| (id, true)))
            .collect();
        all_ids.sort_by_key(|(id, _)| *id);

        let n_hydros = all_ids.len();
        let models: Vec<Vec<ResolvedProductionModel>> = all_ids
            .iter()
            .map(|(_, is_fpha)| {
                (0..n_stages)
                    .map(|_| {
                        if *is_fpha {
                            ResolvedProductionModel::Fpha {
                                planes: vec![fpha_plane; n_planes],
                            }
                        } else {
                            ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }
                        }
                    })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, &minimal_hydros(n_hydros), n_stages);
        let production_sources: Vec<(EntityId, ProductionModelSource)> = all_ids
            .iter()
            .map(|(id, is_fpha)| {
                let src = if *is_fpha {
                    ProductionModelSource::PrecomputedHyperplanes
                } else {
                    ProductionModelSource::DefaultConstant
                };
                (EntityId(*id), src)
            })
            .collect();
        let evaporation_sources: Vec<(EntityId, EvaporationSource)> = all_ids
            .iter()
            .map(|(id, _)| (EntityId(*id), EvaporationSource::NotModeled))
            .collect();
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = all_ids
            .iter()
            .map(|(id, _)| (EntityId(*id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        let evap_models: Vec<EvaporationModel> =
            all_ids.iter().map(|_| EvaporationModel::None).collect();
        PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        }
    }

    /// Build a `PrepareHydroModelsResult` with evaporation for a subset of hydros.
    fn make_result_with_evaporation(
        hydro_ids: &[i32],
        evap_ids: &[i32],
    ) -> PrepareHydroModelsResult {
        let n_hydros = hydro_ids.len();
        let n_stages = 1;
        let models: Vec<Vec<ResolvedProductionModel>> = hydro_ids
            .iter()
            .map(|_| {
                (0..n_stages)
                    .map(|_| ResolvedProductionModel::ConstantProductivity { productivity: 0.95 })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, &minimal_hydros(n_hydros), n_stages);
        let production_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), ProductionModelSource::DefaultConstant))
            .collect();
        let evap_set: std::collections::HashSet<i32> = evap_ids.iter().copied().collect();
        let evaporation_sources: Vec<(EntityId, EvaporationSource)> = hydro_ids
            .iter()
            .map(|&id| {
                let src = if evap_set.contains(&id) {
                    EvaporationSource::LinearizedFromGeometry
                } else {
                    EvaporationSource::NotModeled
                };
                (EntityId(id), src)
            })
            .collect();
        let evap_models: Vec<EvaporationModel> = hydro_ids
            .iter()
            .map(|&id| {
                if evap_set.contains(&id) {
                    EvaporationModel::Linearized {
                        coefficients: vec![LinearizedEvaporation {
                            intercept_m3s: 1.0,
                            volume_slope_m3s_per_hm3: 0.01,
                        }],
                        reference_volumes_hm3: vec![500.0],
                    }
                } else {
                    EvaporationModel::None
                }
            })
            .collect();
        // All test hydros use the default midpoint (no per-season volumes in this fixture).
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        }
    }

    /// build_hydro_model_summary counts n_user_supplied_ref and n_default_midpoint_ref correctly.
    #[test]
    fn build_hydro_model_summary_ref_source_counts() {
        // 3 hydros: IDs 1, 2, 3.
        // IDs 1 and 2 have evaporation (1=UserSupplied, 2=DefaultMidpoint).
        // ID 3 has no evaporation (DefaultMidpoint, irrelevant for count).
        let hydro_ids = [1i32, 2, 3];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);

        let n_hydros = hydro_ids.len();
        let n_stages = 1;
        let models: Vec<Vec<ResolvedProductionModel>> = (0..n_hydros)
            .map(|_| vec![ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }])
            .collect();
        let production = ProductionModelSet::new(models, system.hydros(), n_stages);
        let production_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), ProductionModelSource::DefaultConstant))
            .collect();

        let evaporation_sources = vec![
            (EntityId(1), EvaporationSource::LinearizedFromGeometry),
            (EntityId(2), EvaporationSource::LinearizedFromGeometry),
            (EntityId(3), EvaporationSource::NotModeled),
        ];
        let evaporation_reference_sources = vec![
            (EntityId(1), EvaporationReferenceSource::UserSupplied),
            (EntityId(2), EvaporationReferenceSource::DefaultMidpoint),
            (EntityId(3), EvaporationReferenceSource::DefaultMidpoint),
        ];
        let evap_models = vec![
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.01,
                }],
                reference_volumes_hm3: vec![200.0],
            },
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.01,
                }],
                reference_volumes_hm3: vec![300.0],
            },
            EvaporationModel::None,
        ];

        let result = PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        };

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(summary.n_evaporation, 2, "n_evaporation must be 2");
        assert_eq!(summary.n_no_evaporation, 1, "n_no_evaporation must be 1");
        assert_eq!(
            summary.n_user_supplied_ref, 1,
            "n_user_supplied_ref must be 1 (ID 1)"
        );
        assert_eq!(
            summary.n_default_midpoint_ref, 1,
            "n_default_midpoint_ref must be 1 (ID 2; ID 3 has no evaporation)"
        );
    }

    /// All-constant system: n_constant = 4, n_fpha = 0, total_planes = 0.
    #[test]
    fn build_hydro_model_summary_all_constant() {
        let hydro_ids = [1i32, 2, 3, 4];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);
        let result = make_result_all_constant(&hydro_ids);

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(
            summary.n_constant, 4,
            "all-constant system must have n_constant = 4"
        );
        assert_eq!(
            summary.n_fpha, 0,
            "all-constant system must have n_fpha = 0"
        );
        assert_eq!(
            summary.total_planes, 0,
            "all-constant system must have total_planes = 0"
        );
        assert!(
            summary.fpha_details.is_empty(),
            "all-constant system must have empty fpha_details"
        );
    }

    /// Mixed system (2 constant + 2 FPHA, 5 planes each): correct counts and plane total.

    #[test]
    fn build_hydro_model_summary_mixed_counts_and_plane_total() {
        let constant_ids = [1i32, 2];
        let fpha_ids = [3i32, 4];
        let all_ids = [1i32, 2, 3, 4];
        let hydros = all_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);
        let result = make_result_mixed(&constant_ids, &fpha_ids, 5);

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(summary.n_constant, 2, "n_constant must be 2");
        assert_eq!(summary.n_fpha, 2, "n_fpha must be 2");
        assert_eq!(
            summary.total_planes, 10,
            "total_planes must be 10 (2 × 5 planes)"
        );
        assert_eq!(
            summary.fpha_details.len(),
            2,
            "must have 2 fpha_details entries"
        );
        for detail in &summary.fpha_details {
            assert_eq!(
                detail.n_planes, 5,
                "each FPHA detail must have n_planes = 5"
            );
        }
    }

    /// Acceptance criterion: 2 constant + 2 FPHA (10 planes) + 3 evaporation / 1 without.

    #[test]
    // Rationale: acceptance test exercises constant + FPHA + evaporation counts
    // in a single fixture; splitting into helpers would obscure the coverage
    // contract.
    #[allow(clippy::too_many_lines)]
    fn build_hydro_model_summary_acceptance_criterion() {
        // IDs 1,2 constant; IDs 3,4 FPHA with 5 planes each.
        // IDs 1,2,3 have evaporation; ID 4 does not.
        let constant_ids = [1i32, 2];
        let fpha_ids = [3i32, 4];
        let all_ids = [1i32, 2, 3, 4];
        let hydros = all_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);

        let n_stages = 1;
        let fpha_plane = FphaPlane {
            intercept: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
        };
        let mut sorted: Vec<(i32, bool)> = constant_ids
            .iter()
            .map(|&id| (id, false))
            .chain(fpha_ids.iter().map(|&id| (id, true)))
            .collect();
        sorted.sort_by_key(|(id, _)| *id);

        let models: Vec<Vec<ResolvedProductionModel>> = sorted
            .iter()
            .map(|(_, is_fpha)| {
                (0..n_stages)
                    .map(|_| {
                        if *is_fpha {
                            ResolvedProductionModel::Fpha {
                                planes: vec![fpha_plane; 5],
                            }
                        } else {
                            ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }
                        }
                    })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, system.hydros(), n_stages);
        let production_sources: Vec<(EntityId, ProductionModelSource)> = sorted
            .iter()
            .map(|(id, is_fpha)| {
                (
                    EntityId(*id),
                    if *is_fpha {
                        ProductionModelSource::PrecomputedHyperplanes
                    } else {
                        ProductionModelSource::DefaultConstant
                    },
                )
            })
            .collect();
        let evap_set: std::collections::HashSet<i32> = [1, 2, 3].into_iter().collect();
        let evaporation_sources: Vec<(EntityId, EvaporationSource)> = sorted
            .iter()
            .map(|(id, _)| {
                (
                    EntityId(*id),
                    if evap_set.contains(id) {
                        EvaporationSource::LinearizedFromGeometry
                    } else {
                        EvaporationSource::NotModeled
                    },
                )
            })
            .collect();
        let evap_models: Vec<EvaporationModel> = sorted
            .iter()
            .map(|(id, _)| {
                if evap_set.contains(id) {
                    EvaporationModel::Linearized {
                        coefficients: vec![LinearizedEvaporation {
                            intercept_m3s: 1.0,
                            volume_slope_m3s_per_hm3: 0.01,
                        }],
                        reference_volumes_hm3: vec![500.0],
                    }
                } else {
                    EvaporationModel::None
                }
            })
            .collect();
        // All test hydros use the default midpoint (no per-season volumes in this fixture).
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = sorted
            .iter()
            .map(|(id, _)| (EntityId(*id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        let result = PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        };

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(summary.n_constant, 2, "n_constant must be 2");
        assert_eq!(summary.n_fpha, 2, "n_fpha must be 2");
        assert_eq!(summary.total_planes, 10, "total_planes must be 10");
        assert_eq!(summary.n_evaporation, 3, "n_evaporation must be 3");
        assert_eq!(summary.n_no_evaporation, 1, "n_no_evaporation must be 1");
    }

    /// Evaporation counts are derived from provenance, not model variant.
    #[test]
    fn build_hydro_model_summary_evaporation_counts_from_provenance() {
        let hydro_ids = [1i32, 2, 3, 4];
        let evap_ids = [1i32, 3];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);
        let result = make_result_with_evaporation(&hydro_ids, &evap_ids);

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(
            summary.n_evaporation, 2,
            "n_evaporation must be 2 (IDs 1 and 3)"
        );
        assert_eq!(
            summary.n_no_evaporation, 2,
            "n_no_evaporation must be 2 (IDs 2 and 4)"
        );
    }

    /// Summary with one computed-source hydro: `n_fpha == 1` and
    /// `fpha_details[0].source == ComputedFromGeometry`.
    #[test]
    fn computed_source_in_summary_counts_correctly() {
        // Single hydro with Fpha generation model (needed so build_hydro_model_summary
        // can look up the name from the system entity list).
        let hydro = make_hydro(5, HydroGenerationModel::Fpha);
        let system = make_system_for_summary(vec![hydro]);

        let fpha_plane = FphaPlane {
            intercept: 800.0,
            gamma_v: 0.003,
            gamma_q: 0.90,
            gamma_s: -0.005,
        };
        let n_planes = 4;
        let production = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::Fpha {
                planes: vec![fpha_plane; n_planes],
            }]],
            system.hydros(),
            1,
        );

        let result = PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(vec![EvaporationModel::None]),
            provenance: HydroModelProvenance {
                production_sources: vec![(
                    EntityId::from(5),
                    ProductionModelSource::ComputedFromGeometry,
                )],
                evaporation_sources: vec![(EntityId::from(5), EvaporationSource::NotModeled)],
                evaporation_reference_sources: vec![(
                    EntityId::from(5),
                    EvaporationReferenceSource::DefaultMidpoint,
                )],
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        };

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(
            summary.n_fpha, 1,
            "n_fpha must be 1 for one computed-source hydro"
        );
        assert_eq!(summary.n_constant, 0, "n_constant must be 0");
        assert_eq!(
            summary.total_planes, n_planes,
            "total_planes must equal the plane count from the representative stage"
        );
        assert_eq!(
            summary.fpha_details.len(),
            1,
            "must have one fpha_details entry"
        );
        assert_eq!(
            summary.fpha_details[0].source,
            ProductionModelSource::ComputedFromGeometry,
            "fpha_details[0].source must be ComputedFromGeometry"
        );
        assert_eq!(
            summary.fpha_details[0].n_planes, n_planes,
            "fpha_details[0].n_planes must match the fitted plane count"
        );
    }
}
