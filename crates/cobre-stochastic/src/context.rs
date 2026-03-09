//! Top-level stochastic pipeline initialization.
//!
//! [`StochasticContext`] owns the three independently-built stochastic
//! infrastructure components: the PAR coefficient cache, the pre-decomposed
//! spatial correlation, and the pre-generated opening scenario tree.
//!
//! [`build_stochastic_context`] wires these components together from a
//! [`System`] reference, running the full preprocessing pipeline in a
//! well-defined order:
//!
//! 1. Validate PAR model parameters (fatal errors stop the pipeline).
//! 2. Build the [`PrecomputedParLp`] coefficient cache.
//! 3. Decompose the spatial correlation matrix via Cholesky.
//! 4. Generate the opening scenario tree.
//!
//! The caller is responsible for providing the `base_seed` as an explicit
//! `u64`. Seed extraction from external configuration — including handling
//! the `None` case with OS entropy — is an application-level concern that
//! belongs in the calling crate.

use cobre_core::{EntityId, System};

use crate::{
    StochasticError,
    correlation::resolve::DecomposedCorrelation,
    par::{precompute::PrecomputedParLp, validation::validate_par_parameters},
    tree::{generate::generate_opening_tree, opening_tree::OpeningTreeView},
};

pub use crate::tree::opening_tree::OpeningTree;

/// Fully-initialized stochastic pipeline components, owned in one place.
///
/// Built once during initialization from a [`System`] reference and a base
/// seed. After construction all fields are immutable; the context is consumed
/// read-only by iterative optimization algorithms.
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use cobre_core::{
///     Bus, DeficitSegment, EntityId, SystemBuilder,
///     scenario::{
///         CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
///         InflowModel,
///     },
///     temporal::{
///         Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
///         StageStateConfig,
///     },
///     entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
/// };
/// use cobre_stochastic::context::build_stochastic_context;
/// use chrono::NaiveDate;
///
/// # fn make_bus(id: i32) -> Bus {
/// #     Bus {
/// #         id: EntityId(id),
/// #         name: format!("Bus{id}"),
/// #         deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 1000.0 }],
/// #         excess_cost: 0.0,
/// #     }
/// # }
/// # fn make_stage(index: usize, id: i32, bf: usize) -> Stage {
/// #     Stage {
/// #         index,
/// #         id,
/// #         start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
/// #         end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
/// #         season_id: Some(0),
/// #         blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
/// #         block_mode: BlockMode::Parallel,
/// #         state_config: StageStateConfig { storage: true, inflow_lags: false },
/// #         risk_config: StageRiskConfig::Expectation,
/// #         scenario_config: ScenarioSourceConfig { branching_factor: bf, noise_method: NoiseMethod::Saa },
/// #     }
/// # }
/// # fn make_hydro(id: i32) -> Hydro {
/// #     Hydro {
/// #         id: EntityId(id),
/// #         name: format!("H{id}"),
/// #         bus_id: EntityId(0),
/// #         downstream_id: None,
/// #         entry_stage_id: None,
/// #         exit_stage_id: None,
/// #         min_storage_hm3: 0.0,
/// #         max_storage_hm3: 100.0,
/// #         min_outflow_m3s: 0.0,
/// #         max_outflow_m3s: None,
/// #         generation_model: HydroGenerationModel::ConstantProductivity { productivity_mw_per_m3s: 1.0 },
/// #         min_turbined_m3s: 0.0,
/// #         max_turbined_m3s: 100.0,
/// #         min_generation_mw: 0.0,
/// #         max_generation_mw: 100.0,
/// #         tailrace: None,
/// #         hydraulic_losses: None,
/// #         efficiency: None,
/// #         evaporation_coefficients_mm: None,
/// #         diversion: None,
/// #         filling: None,
/// #         penalties: HydroPenalties {
/// #             spillage_cost: 0.0, diversion_cost: 0.0, fpha_turbined_cost: 0.0,
/// #             storage_violation_below_cost: 0.0, filling_target_violation_cost: 0.0,
/// #             turbined_violation_below_cost: 0.0, outflow_violation_below_cost: 0.0,
/// #             outflow_violation_above_cost: 0.0, generation_violation_below_cost: 0.0,
/// #             evaporation_violation_cost: 0.0, water_withdrawal_violation_cost: 0.0,
/// #         },
/// #     }
/// # }
/// # fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
/// #     InflowModel {
/// #         hydro_id: EntityId(hydro_id),
/// #         stage_id,
/// #         mean_m3s: 100.0,
/// #         std_m3s: 30.0,
/// #         ar_coefficients: vec![],
/// #         residual_std_ratio: 1.0,
/// #     }
/// # }
/// # fn identity_correlation(entity_ids: &[i32]) -> CorrelationModel {
/// #     let n = entity_ids.len();
/// #     let matrix: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
/// #     let mut profiles = BTreeMap::new();
/// #     profiles.insert("default".to_string(), CorrelationProfile {
/// #         groups: vec![CorrelationGroup {
/// #             name: "g1".to_string(),
/// #             entities: entity_ids.iter().map(|&id| CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(id) }).collect(),
/// #             matrix,
/// #         }],
/// #     });
/// #     CorrelationModel { method: "cholesky".to_string(), profiles, schedule: vec![] }
/// # }
/// let hydros = vec![make_hydro(1), make_hydro(2)];
/// let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3), make_stage(2, 2, 3)];
/// let inflow_models = vec![
///     make_inflow_model(1, 0), make_inflow_model(1, 1), make_inflow_model(1, 2),
///     make_inflow_model(2, 0), make_inflow_model(2, 1), make_inflow_model(2, 2),
/// ];
///
/// let system = SystemBuilder::new()
///     .buses(vec![make_bus(0)])
///     .hydros(hydros)
///     .stages(stages)
///     .inflow_models(inflow_models)
///     .correlation(identity_correlation(&[1, 2]))
///     .build()
///     .unwrap();
///
/// let ctx = build_stochastic_context(&system, 42).unwrap();
/// assert_eq!(ctx.dim(), 2);
/// assert_eq!(ctx.n_stages(), 3);
/// assert_eq!(ctx.base_seed(), 42);
/// ```
#[derive(Debug)]
pub struct StochasticContext {
    par_lp: PrecomputedParLp,
    correlation: DecomposedCorrelation,
    opening_tree: OpeningTree,
    base_seed: u64,
    dim: usize,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StochasticContext>();
};

impl StochasticContext {
    /// Returns a reference to the PAR(p) LP coefficient cache.
    #[must_use]
    pub fn par_lp(&self) -> &PrecomputedParLp {
        &self.par_lp
    }

    /// Returns a reference to the pre-decomposed spatial correlation.
    #[must_use]
    pub fn correlation(&self) -> &DecomposedCorrelation {
        &self.correlation
    }

    /// Returns a reference to the opening scenario tree.
    #[must_use]
    pub fn opening_tree(&self) -> &OpeningTree {
        &self.opening_tree
    }

    /// Returns a read-only borrowed view over the opening scenario tree.
    #[must_use]
    pub fn tree_view(&self) -> OpeningTreeView<'_> {
        self.opening_tree.view()
    }

    /// Returns the base seed used to generate the opening tree.
    #[must_use]
    pub fn base_seed(&self) -> u64 {
        self.base_seed
    }

    /// Returns the noise dimension (number of hydro plants).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of study stages in the opening tree.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.opening_tree.n_stages()
    }
}

/// Initialize the full stochastic pipeline from a [`System`] reference.
///
/// Runs these steps in order:
///
/// 1. Validate PAR model parameters — fatal errors propagated immediately.
/// 2. Extract study stages (non-negative `stage.id`).
/// 3. Extract canonical hydro entity IDs.
/// 4. Build [`PrecomputedParLp`] from inflow models and study stages.
/// 5. Build [`DecomposedCorrelation`] from the system correlation model.
/// 6. Generate the opening scenario tree from the correlation and study stages.
///
/// The `base_seed` parameter must be supplied explicitly by the caller.
/// Converting `ScenarioSource.seed: Option<i64>` to a `u64` — including
/// the `None` case with OS entropy — is an application-level concern that
/// belongs in the calling crate, not in this infrastructure crate.
///
/// # Errors
///
/// - [`StochasticError::InvalidParParameters`]: a PAR model has AR order > 0
///   with zero standard deviation.
/// - [`StochasticError::InvalidCorrelation`]: the correlation model is empty,
///   ambiguous, or contains an invalid matrix.
/// - [`StochasticError::CholeskyDecompositionFailed`]: a correlation matrix
///   is not positive-definite.
pub fn build_stochastic_context(
    system: &System,
    base_seed: u64,
) -> Result<StochasticContext, StochasticError> {
    let _report = validate_par_parameters(system.inflow_models())?;

    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();

    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let dim = hydro_ids.len();

    let par_lp = PrecomputedParLp::build(system.inflow_models(), &study_stages, &hydro_ids)?;

    // When there are no hydro plants (dim == 0), the system is thermal-only.
    // Skip correlation decomposition (no correlation model is needed) and
    // build an empty opening tree with zero-length noise vectors. The forward
    // and backward passes simply produce no inflow noise in this case.
    let mut correlation = if dim == 0 {
        DecomposedCorrelation::empty()
    } else {
        DecomposedCorrelation::build(system.correlation())?
    };

    let opening_tree =
        generate_opening_tree(base_seed, &study_stages, dim, &mut correlation, &hydro_ids);

    Ok(StochasticContext {
        par_lp,
        correlation,
        opening_tree,
        base_seed,
        dim,
    })
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
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::build_stochastic_context;
    use crate::StochasticError;

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
            },
        }
    }

    fn make_inflow_model(hydro_id: i32, stage_id: i32, std: f64, coeffs: Vec<f64>) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: std,
            ar_coefficients: coeffs,
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
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    #[test]
    fn stochastic_context_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::StochasticContext>();
    }

    /// AC: build succeeds with a valid system; accessors return expected dimensions.
    #[test]
    fn build_succeeds_with_valid_system() {
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
        ];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
            make_inflow_model(1, 2, 30.0, vec![]),
            make_inflow_model(2, 0, 20.0, vec![]),
            make_inflow_model(2, 1, 20.0, vec![]),
            make_inflow_model(2, 2, 20.0, vec![]),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 42).unwrap();

        assert_eq!(ctx.dim(), 2);
        assert_eq!(ctx.n_stages(), 3);
        assert_eq!(ctx.base_seed(), 42);
    }

    /// AC: `par_lp()` returns a cache with the expected dimensions.
    #[test]
    fn par_lp_has_expected_dimensions() {
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
        ];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
            make_inflow_model(1, 2, 30.0, vec![]),
            make_inflow_model(2, 0, 20.0, vec![]),
            make_inflow_model(2, 1, 20.0, vec![]),
            make_inflow_model(2, 2, 20.0, vec![]),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 42).unwrap();

        assert_eq!(ctx.par_lp().n_hydros(), 2);
        assert_eq!(ctx.par_lp().n_stages(), 3);
    }

    /// AC: `opening_tree()` has the expected stage and dimension counts.
    #[test]
    fn opening_tree_has_expected_dimensions() {
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let stages = vec![
            make_stage(0, 0, 5),
            make_stage(1, 1, 5),
            make_stage(2, 2, 5),
        ];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
            make_inflow_model(1, 2, 30.0, vec![]),
            make_inflow_model(2, 0, 20.0, vec![]),
            make_inflow_model(2, 1, 20.0, vec![]),
            make_inflow_model(2, 2, 20.0, vec![]),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 42).unwrap();

        assert_eq!(ctx.opening_tree().n_stages(), 3);
        assert_eq!(ctx.opening_tree().dim(), 2);
    }

    /// AC: `tree_view()` returns a view with matching dimensions.
    #[test]
    fn tree_view_returns_valid_view() {
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let stages = vec![make_stage(0, 0, 4), make_stage(1, 1, 4)];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
            make_inflow_model(2, 0, 20.0, vec![]),
            make_inflow_model(2, 1, 20.0, vec![]),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 7).unwrap();
        let view = ctx.tree_view();

        assert_eq!(view.n_stages(), ctx.opening_tree().n_stages());
        assert_eq!(view.dim(), ctx.opening_tree().dim());
        // Spot-check: first opening of first stage must match.
        assert_eq!(view.opening(0, 0), ctx.opening_tree().opening(0, 0));
    }

    /// AC: invalid PAR parameters (AR order > 0 with zero std) returns `InvalidParParameters`.
    #[test]
    fn build_fails_on_invalid_par() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 3)];
        // AR(1) with std == 0.0 is the fatal case.
        let inflow_models = vec![make_inflow_model(1, 0, 0.0, vec![0.3])];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let result = build_stochastic_context(&system, 42);

        assert!(
            matches!(result, Err(StochasticError::InvalidParParameters { .. })),
            "expected InvalidParParameters, got: {result:?}"
        );
    }

    /// AC: non-positive-definite correlation matrix returns `CholeskyDecompositionFailed`.
    #[test]
    fn build_fails_on_invalid_correlation() {
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let stages = vec![make_stage(0, 0, 3)];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(2, 0, 20.0, vec![]),
        ];

        // A non-positive-definite matrix: rho > 1 is invalid.
        let n = 2usize;
        let rho = 2.0_f64; // off-diagonal > 1 -- not PD.
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
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
                    matrix,
                }],
            },
        );
        let bad_correlation = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(bad_correlation)
            .build()
            .unwrap();

        let result = build_stochastic_context(&system, 42);

        assert!(
            matches!(
                result,
                Err(StochasticError::CholeskyDecompositionFailed { .. })
            ),
            "expected CholeskyDecompositionFailed, got: {result:?}"
        );
    }

    /// Pre-study stages (negative IDs) are excluded from the opening tree.
    #[test]
    fn pre_study_stages_excluded_from_opening_tree() {
        let hydros = vec![make_hydro(1)];
        // Two study stages (id >= 0) and one pre-study stage (id < 0).
        let stages = vec![
            make_stage(0, -1, 3), // pre-study — must be excluded from tree
            make_stage(1, 0, 3),
            make_stage(2, 1, 3),
        ];
        let inflow_models = vec![
            make_inflow_model(1, -1, 30.0, vec![]),
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 0).unwrap();

        // The opening tree must contain only the 2 study stages.
        assert_eq!(
            ctx.n_stages(),
            2,
            "pre-study stage must not appear in opening tree"
        );
    }
}
