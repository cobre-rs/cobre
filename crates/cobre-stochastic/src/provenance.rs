//! Provenance metadata for stochastic pipeline components.
//!
//! [`ComponentProvenance`] records whether a component of a [`StochasticContext`](crate::StochasticContext)
//! was computed internally from system data, provided by the caller as an
//! external override, or is absent because the system has no relevant entities.
//!
//! [`StochasticProvenance`] groups per-component provenance into a single value
//! stored on [`StochasticContext`](crate::StochasticContext) and returned by its `provenance()` accessor.

use cobre_core::scenario::SamplingScheme;

/// Origin of a single stochastic pipeline component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentProvenance {
    /// Computed by the stochastic pipeline from system data.
    Generated,
    /// Provided by the caller as an external override.
    UserSupplied,
    /// Component not present — system has no relevant entities.
    NotApplicable,
}

/// Provenance records for all components of a [`StochasticContext`](crate::StochasticContext).
///
/// Set once during construction; never mutated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticProvenance {
    /// Origin of the opening scenario tree.
    pub opening_tree: ComponentProvenance,
    /// Origin of the spatial correlation decomposition.
    pub correlation: ComponentProvenance,
    /// Origin of inflow PAR models (`Generated` when hydros present; `NotApplicable` otherwise).
    pub inflow_model: ComponentProvenance,
    /// Sampling scheme configured for the inflow entity class.
    /// `None` when per-class config is not yet applied.
    pub inflow_scheme: Option<SamplingScheme>,
    /// Sampling scheme configured for the load entity class.
    /// `None` when per-class config is not yet applied.
    pub load_scheme: Option<SamplingScheme>,
    /// Sampling scheme configured for the NCS entity class.
    /// `None` when per-class config is not yet applied.
    pub ncs_scheme: Option<SamplingScheme>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::{
        Bus, DeficitSegment, EntityId, SystemBuilder,
        entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
            SamplingScheme,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use crate::{
        ComponentProvenance,
        context::{ClassSchemes, OpeningTree, OpeningTreeInputs, build_stochastic_context},
    };

    fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus{id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn make_hydro(id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
            },
        }
    }

    fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        }
    }

    fn identity_correlation(entity_ids: &[i32]) -> CorrelationModel {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: entity_ids
                        .iter()
                        .map(|&id| CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(id),
                        })
                        .collect(),
                    matrix,
                }],
            },
        );
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    #[test]
    fn provenance_with_user_tree() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let inflow_models = vec![make_inflow_model(1, 0), make_inflow_model(1, 1)];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let user_tree = OpeningTree::from_parts(vec![1.0_f64; 2 * 2], vec![2, 2], 1);
        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(user_tree),
                historical_library: None,
                external_scenario_counts: None,
                noise_group_ids: None,
            },
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().opening_tree,
            ComponentProvenance::UserSupplied,
            "user-supplied tree must be recorded as UserSupplied"
        );
    }

    #[test]
    fn provenance_without_user_tree() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let inflow_models = vec![make_inflow_model(1, 0), make_inflow_model(1, 1)];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().opening_tree,
            ComponentProvenance::Generated,
            "generated opening tree must be recorded as Generated"
        );
    }

    #[test]
    fn provenance_no_entities() {
        // A system with no hydros and no stochastic load buses has dim == 0.
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .stages(vec![make_stage(0, 0, 3)])
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().opening_tree,
            ComponentProvenance::NotApplicable,
            "dim == 0 means opening tree is NotApplicable"
        );
    }

    #[test]
    fn provenance_correlation_from_system() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let inflow_models = vec![make_inflow_model(1, 0), make_inflow_model(1, 1)];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().correlation,
            ComponentProvenance::Generated,
            "non-empty correlation profiles must produce Generated"
        );
    }

    #[test]
    fn provenance_correlation_empty() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .stages(vec![make_stage(0, 0, 3)])
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().correlation,
            ComponentProvenance::NotApplicable,
            "empty/absent correlation must produce NotApplicable"
        );
    }

    #[test]
    fn provenance_inflow_with_hydros() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let inflow_models = vec![make_inflow_model(1, 0), make_inflow_model(1, 1)];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().inflow_model,
            ComponentProvenance::Generated,
            "system with hydros must produce inflow_model == Generated"
        );
    }

    #[test]
    fn provenance_inflow_without_hydros() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .stages(vec![make_stage(0, 0, 3)])
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().inflow_model,
            ComponentProvenance::NotApplicable,
            "system with no hydros must produce inflow_model == NotApplicable"
        );
    }

    #[test]
    fn test_provenance_per_class_schemes_populated() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .stages(vec![make_stage(0, 0, 3)])
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert!(
            ctx.provenance().inflow_scheme == Some(SamplingScheme::InSample),
            "inflow_scheme must be Some(InSample) when passed as argument"
        );
        assert!(
            ctx.provenance().load_scheme == Some(SamplingScheme::InSample),
            "load_scheme must be Some(InSample) when passed as argument"
        );
        assert!(
            ctx.provenance().ncs_scheme == Some(SamplingScheme::InSample),
            "ncs_scheme must be Some(InSample) when passed as argument"
        );
    }
}
