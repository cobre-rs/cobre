//! Top-level stochastic pipeline initialization.
//!
//! [`StochasticContext`] owns the four independently-built stochastic
//! infrastructure components: the PAR coefficient cache, the pre-decomposed
//! spatial correlation, the pre-generated opening scenario tree, and the
//! normal noise LP parameter cache.
//!
//! [`build_stochastic_context`] wires these components together from a
//! [`System`] reference, running the full preprocessing pipeline in a
//! well-defined order:
//!
//! 1. Validate PAR model parameters (fatal errors stop the pipeline).
//! 2. Build the [`PrecomputedParLp`] coefficient cache.
//! 3. Decompose the spatial correlation matrix via Cholesky.
//! 4. Generate the opening scenario tree.
//! 5. Build the [`PrecomputedNormalLp`] cache for stochastic load buses.
//!
//! The caller is responsible for providing the `base_seed` as an explicit
//! `u64`. Seed extraction from external configuration — including handling
//! the `None` case with OS entropy — is an application-level concern that
//! belongs in the calling crate.

use cobre_core::{EntityId, System};

use crate::{
    StochasticError,
    correlation::resolve::DecomposedCorrelation,
    normal::precompute::{EntityFactorEntry, PrecomputedNormalLp},
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
/// let ctx = build_stochastic_context(&system, 42, &[]).unwrap();
/// assert_eq!(ctx.dim(), 2);
/// assert_eq!(ctx.n_stages(), 3);
/// assert_eq!(ctx.base_seed(), 42);
/// assert_eq!(ctx.n_load_buses(), 0);
/// ```
#[derive(Debug)]
pub struct StochasticContext {
    par_lp: PrecomputedParLp,
    correlation: DecomposedCorrelation,
    opening_tree: OpeningTree,
    normal_lp: PrecomputedNormalLp,
    base_seed: u64,
    dim: usize,
    n_load_buses: usize,
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

    /// Returns the noise dimension (`n_hydros + n_load_buses`).
    ///
    /// Opening tree noise vectors have this length. Inflow noise occupies
    /// indices `[0, n_hydros)` and load noise occupies `[n_hydros, dim)`.
    /// Use `par_lp().n_hydros()` to find the boundary.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of stochastic load buses included in the noise dimension.
    ///
    /// Load noise occupies indices `[n_hydros, n_hydros + n_load_buses)` in each
    /// opening tree noise vector. Returns `0` when there are no buses with `std_mw > 0`.
    #[must_use]
    pub fn n_load_buses(&self) -> usize {
        self.n_load_buses
    }

    /// Returns a reference to the precomputed normal noise LP parameter cache.
    ///
    /// Contains stage-major mean, standard deviation, and block factor arrays
    /// for all stochastic load buses (those with `std_mw > 0`). The entity
    /// order matches the load bus IDs appended to `entity_order` during context
    /// construction, sorted by `EntityId`.
    pub fn normal_lp(&self) -> &PrecomputedNormalLp {
        &self.normal_lp
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
/// 4. Collect stochastic load bus IDs (buses with at least one [`LoadModel`]
///    entry where `std_mw > 0`), sorted by [`EntityId`] for declaration-order
///    invariance.
/// 5. Build [`PrecomputedParLp`] from inflow models and study stages.
/// 6. Build [`DecomposedCorrelation`] from the system correlation model.
/// 7. Generate the opening scenario tree from the expanded entity order
///    (`hydro_ids` followed by `load_bus_ids`) with `dim = n_hydros + n_load_buses`.
/// 8. Build [`PrecomputedNormalLp`] for the stochastic load buses.
///
/// The `base_seed` parameter must be supplied explicitly by the caller.
/// Converting `ScenarioSource.seed: Option<i64>` to a `u64` — including
/// the `None` case with OS entropy — is an application-level concern that
/// belongs in the calling crate, not in this infrastructure crate.
///
/// The `load_factors` parameter provides per-`(entity_id, stage_id, block_factors)`
/// scaling entries consumed by [`PrecomputedNormalLp`]. Pass an empty slice when
/// no load factor file was loaded; all block factors then default to `1.0`. The
/// caller is responsible for converting any external load factor representation
/// into [`EntityFactorEntry`] slices before calling this function.
///
/// # Errors
///
/// - [`StochasticError::InvalidParParameters`]: a PAR model has AR order > 0
///   with zero standard deviation.
/// - [`StochasticError::InvalidCorrelation`]: the correlation model is empty,
///   ambiguous, or contains an invalid matrix.
/// - [`StochasticError::CholeskyDecompositionFailed`]: a correlation matrix
///   is not positive-definite.
///
/// [`LoadModel`]: cobre_core::scenario::LoadModel
pub fn build_stochastic_context(
    system: &System,
    base_seed: u64,
    load_factors: &[EntityFactorEntry<'_>],
) -> Result<StochasticContext, StochasticError> {
    let _report = validate_par_parameters(system.inflow_models())?;

    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();

    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

    // Collect distinct stochastic bus IDs (std_mw > 0) sorted for declaration-order invariance.
    let load_bus_ids: Vec<EntityId> = {
        let mut ids: Vec<EntityId> = system
            .load_models()
            .iter()
            .filter(|m| m.std_mw > 0.0)
            .map(|m| m.bus_id)
            .collect();
        ids.sort_unstable_by_key(|id| id.0);
        ids.dedup();
        ids
    };
    let n_load_buses = load_bus_ids.len();

    let dim = hydro_ids.len() + n_load_buses;

    let par_lp = PrecomputedParLp::build(system.inflow_models(), &study_stages, &hydro_ids)?;

    // When there are no noise entities (dim == 0) or no correlation profiles
    // were provided, use uncorrelated noise. The dim == 0 case means the
    // system is thermal-only (no inflow or load noise at all). The
    // empty-profiles case means entities exist but the user did not supply a
    // correlation file — treated as independent (identity correlation).
    let mut correlation = if dim == 0 || system.correlation().profiles.is_empty() {
        DecomposedCorrelation::empty()
    } else {
        DecomposedCorrelation::build(system.correlation())?
    };

    // Build the expanded entity order: hydro IDs first (inflow noise indices),
    // then load bus IDs (load noise indices). Load bus IDs are already sorted.
    let entity_order: Vec<EntityId> = hydro_ids
        .iter()
        .copied()
        .chain(load_bus_ids.iter().copied())
        .collect();

    let opening_tree = generate_opening_tree(
        base_seed,
        &study_stages,
        dim,
        &mut correlation,
        &entity_order,
    );

    let max_blocks = study_stages
        .iter()
        .map(|s| s.blocks.len())
        .max()
        .unwrap_or(0);

    let normal_lp = PrecomputedNormalLp::build(
        system.load_models(),
        load_factors,
        &study_stages,
        &load_bus_ids,
        max_blocks,
    )?;

    Ok(StochasticContext {
        par_lp,
        correlation,
        opening_tree,
        normal_lp,
        base_seed,
        dim,
        n_load_buses,
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
            LoadModel,
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

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

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

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

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

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

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

        let ctx = build_stochastic_context(&system, 7, &[]).unwrap();
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

        let result = build_stochastic_context(&system, 42, &[]);

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

        let result = build_stochastic_context(&system, 42, &[]);

        assert!(
            matches!(
                result,
                Err(StochasticError::CholeskyDecompositionFailed { .. })
            ),
            "expected CholeskyDecompositionFailed, got: {result:?}"
        );
    }

    /// When hydro plants exist but no correlation file was provided (empty
    /// profiles), the context should build successfully with uncorrelated
    /// (independent) inflows rather than failing.
    #[test]
    fn build_succeeds_with_hydros_and_empty_correlation() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
        ];

        // Empty correlation — simulates absent correlation.json.
        let empty_correlation = CorrelationModel::default();
        assert!(empty_correlation.profiles.is_empty());

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(empty_correlation)
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

        assert_eq!(ctx.dim(), 1);
        assert_eq!(ctx.n_stages(), 2);
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

        let ctx = build_stochastic_context(&system, 0, &[]).unwrap();

        // The opening tree must contain only the 2 study stages.
        assert_eq!(
            ctx.n_stages(),
            2,
            "pre-study stage must not appear in opening tree"
        );
    }

    fn make_load_model(bus_id: i32, stage_id: i32, mean_mw: f64, std_mw: f64) -> LoadModel {
        LoadModel {
            bus_id: EntityId(bus_id),
            stage_id,
            mean_mw,
            std_mw,
        }
    }

    /// AC: system with hydros + load buses produces correct dim and `n_load_buses`.
    #[test]
    fn context_with_load_buses_has_expanded_dim() {
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
        let load_models = vec![
            make_load_model(10, 0, 100.0, 10.0),
            make_load_model(10, 1, 105.0, 10.0),
            make_load_model(10, 2, 110.0, 10.0),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(10)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

        assert_eq!(
            ctx.dim(),
            3,
            "dim must equal n_hydros + n_load_buses = 2 + 1"
        );
        assert_eq!(ctx.n_load_buses(), 1, "one load bus with std_mw > 0");
    }

    /// AC: system with hydros only produces `dim` = `n_hydros` and `n_load_buses` = 0.
    #[test]
    fn context_without_load_has_original_dim() {
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

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

        assert_eq!(ctx.dim(), 2, "dim must equal n_hydros when no load buses");
        assert_eq!(
            ctx.n_load_buses(),
            0,
            "n_load_buses must be 0 when no load models present"
        );
    }

    /// AC: buses with `std_mw == 0.0` are excluded from the noise dimension.
    #[test]
    fn context_load_bus_deterministic_excluded() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 3), make_stage(1, 1, 3)];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
        ];
        // Bus 10 has std_mw == 0.0 — deterministic, must not enter noise dim.
        let load_models = vec![
            make_load_model(10, 0, 100.0, 0.0),
            make_load_model(10, 1, 105.0, 0.0),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(10)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

        assert_eq!(
            ctx.n_load_buses(),
            0,
            "deterministic load bus (std_mw == 0.0) must not enter noise dim"
        );
        assert_eq!(
            ctx.dim(),
            1,
            "dim must equal n_hydros when no stochastic load buses"
        );
    }

    /// AC: opening tree noise slices have length = `n_hydros` + `n_load_buses`.
    #[test]
    fn opening_tree_noise_length_matches_expanded_dim() {
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let stages = vec![make_stage(0, 0, 4), make_stage(1, 1, 4)];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
            make_inflow_model(2, 0, 20.0, vec![]),
            make_inflow_model(2, 1, 20.0, vec![]),
        ];
        let load_models = vec![
            make_load_model(10, 0, 100.0, 10.0),
            make_load_model(10, 1, 105.0, 10.0),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(10)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 7, &[]).unwrap();

        assert_eq!(ctx.dim(), 3, "expanded dim must be 2 hydros + 1 load bus");
        // Each opening noise vector must have length = dim.
        let view = ctx.tree_view();
        let noise = view.opening(0, 0);
        assert_eq!(
            noise.len(),
            3,
            "opening noise vector length must equal expanded dim"
        );
    }

    /// AC: `normal_lp()` accessor returns correctly built `PrecomputedNormalLp`.
    #[test]
    fn normal_lp_accessible_from_context() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![
            make_stage(0, 0, 3),
            make_stage(1, 1, 3),
            make_stage(2, 2, 3),
        ];
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
            make_inflow_model(1, 2, 30.0, vec![]),
        ];
        let load_models = vec![
            make_load_model(10, 0, 100.0, 10.0),
            make_load_model(10, 1, 110.0, 11.0),
            make_load_model(10, 2, 120.0, 12.0),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(10)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(&system, 42, &[]).unwrap();

        assert_eq!(ctx.n_load_buses(), 1);
        let nlp = ctx.normal_lp();
        assert_eq!(nlp.n_stages(), 3);
        assert_eq!(nlp.n_entities(), 1, "one stochastic load bus");
        // Stage 0, entity 0 (bus 10) — mean and std from load_models.
        assert!(
            (nlp.mean(0, 0) - 100.0).abs() < f64::EPSILON,
            "mean at stage 0 should be 100.0"
        );
        assert!(
            (nlp.std(0, 0) - 10.0).abs() < f64::EPSILON,
            "std at stage 0 should be 10.0"
        );
        assert!(
            (nlp.mean(1, 0) - 110.0).abs() < f64::EPSILON,
            "mean at stage 1 should be 110.0"
        );
        assert!(
            (nlp.mean(2, 0) - 120.0).abs() < f64::EPSILON,
            "mean at stage 2 should be 120.0"
        );
    }
}
