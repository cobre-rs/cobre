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
//! 2. Build the [`PrecomputedPar`] coefficient cache.
//! 3. Decompose the spatial correlation matrix via spectral factorisation.
//! 4. Generate the opening scenario tree.
//! 5. Build the [`PrecomputedNormal`] cache for stochastic load buses.
//!
//! The caller is responsible for providing the `base_seed` as an explicit
//! `u64`. Seed extraction from external configuration — including handling
//! the `None` case with OS entropy — is an application-level concern that
//! belongs in the calling crate.

use cobre_core::{scenario::SamplingScheme, EntityId, LoadModel, System};

/// Per-class sampling scheme selections passed to [`build_stochastic_context`]
/// for provenance tracking.
#[derive(Debug, Clone, Copy)]
pub struct ClassSchemes {
    /// Inflow class sampling scheme, or `None` if not configured.
    pub inflow: Option<SamplingScheme>,
    /// Load class sampling scheme, or `None` if not configured.
    pub load: Option<SamplingScheme>,
    /// NCS class sampling scheme, or `None` if not configured.
    pub ncs: Option<SamplingScheme>,
}

/// Opening-tree inputs passed to [`build_stochastic_context`].
///
/// Groups the two optional caller-provided overrides for the opening scenario
/// tree: a pre-built `OpeningTree` that bypasses generation entirely, and a
/// `HistoricalScenarioLibrary` used when any stage is configured with
/// [`NoiseMethod::HistoricalResiduals`](cobre_core::temporal::NoiseMethod::HistoricalResiduals).
///
/// When both fields are `None` the opening tree is generated from SAA/LHS/QMC
/// noise depending on each stage's `scenario_config.noise_method`.
#[derive(Debug, Default)]
pub struct OpeningTreeInputs<'a> {
    /// A pre-built opening tree that bypasses generation. When `Some`, the
    /// `historical_library` field is ignored.
    pub user_tree: Option<OpeningTree>,
    /// Historical scenario library used for [`NoiseMethod::HistoricalResiduals`](cobre_core::temporal::NoiseMethod::HistoricalResiduals)
    /// stages. Required when any study stage uses that noise method and
    /// `user_tree` is `None`.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
}

use crate::{
    correlation::resolve::DecomposedCorrelation,
    normal::precompute::{EntityFactorEntry, PrecomputedNormal},
    par::{precompute::PrecomputedPar, validation::validate_par_parameters},
    provenance::{ComponentProvenance, StochasticProvenance},
    sampling::historical::HistoricalScenarioLibrary,
    tree::{
        generate::{generate_opening_tree, ClassDimensions},
        opening_tree::OpeningTreeView,
    },
    StochasticError,
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
/// use cobre_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
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
/// #         evaporation_reference_volumes_hm3: None,
/// #         diversion: None,
/// #         filling: None,
/// #         penalties: HydroPenalties {
/// #             spillage_cost: 0.0, diversion_cost: 0.0, fpha_turbined_cost: 0.0,
/// #             storage_violation_below_cost: 0.0, filling_target_violation_cost: 0.0,
/// #             turbined_violation_below_cost: 0.0, outflow_violation_below_cost: 0.0,
/// #             outflow_violation_above_cost: 0.0, generation_violation_below_cost: 0.0,
/// #             evaporation_violation_cost: 0.0, water_withdrawal_violation_cost: 0.0,
/// #             water_withdrawal_violation_pos_cost: 0.0, water_withdrawal_violation_neg_cost: 0.0,
/// #             evaporation_violation_pos_cost: 0.0, evaporation_violation_neg_cost: 0.0,
/// #             inflow_nonnegativity_cost: 1000.0,
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
/// #     CorrelationModel { method: "spectral".to_string(), profiles, schedule: vec![] }
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
/// let ctx = build_stochastic_context(&system, 42, None, &[], &[], OpeningTreeInputs::default(), ClassSchemes { inflow: None, load: None, ncs: None }).unwrap();
/// assert_eq!(ctx.dim(), 2);
/// assert_eq!(ctx.n_stages(), 3);
/// assert_eq!(ctx.base_seed(), 42);
/// assert_eq!(ctx.n_load_buses(), 0);
/// assert_eq!(ctx.forward_seed(), None);
/// ```
#[derive(Debug)]
pub struct StochasticContext {
    par_lp: PrecomputedPar,
    correlation: DecomposedCorrelation,
    opening_tree: OpeningTree,
    normal_lp: PrecomputedNormal,
    ncs_normal: PrecomputedNormal,
    ncs_entity_ids: Vec<EntityId>,
    entity_order: Box<[EntityId]>,
    base_seed: u64,
    /// Seed for `OutOfSample` forward-pass noise generation, independent from
    /// the tree seed (`base_seed`). `None` means the field was not configured
    /// in `stages.json`; the `ForwardSampler` factory validates presence when
    /// constructing an `OutOfSample` sampler.
    forward_seed: Option<u64>,
    dim: usize,
    n_load_buses: usize,
    n_stochastic_ncs: usize,
    provenance: StochasticProvenance,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StochasticContext>();
};

impl StochasticContext {
    /// Returns a reference to the PAR(p) LP coefficient cache.
    #[must_use]
    pub fn par(&self) -> &PrecomputedPar {
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

    /// Returns a borrowed view over the opening scenario tree.
    #[must_use]
    pub fn tree_view(&self) -> OpeningTreeView<'_> {
        self.opening_tree.view()
    }

    /// Returns the base seed used to generate the opening tree.
    #[must_use]
    pub fn base_seed(&self) -> u64 {
        self.base_seed
    }

    /// Returns the seed for `OutOfSample` forward-pass noise generation, independent
    /// from the tree seed. `None` when not specified in `stages.json`.
    #[must_use]
    pub fn forward_seed(&self) -> Option<u64> {
        self.forward_seed
    }

    /// Returns the noise dimension (`n_hydros + n_load_buses + n_stochastic_ncs`).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of stochastic load buses in the noise dimension.
    #[must_use]
    pub fn n_load_buses(&self) -> usize {
        self.n_load_buses
    }

    /// Returns the precomputed normal noise LP parameters for stochastic load buses.
    pub fn normal(&self) -> &PrecomputedNormal {
        &self.normal_lp
    }

    /// Returns the precomputed normal noise LP parameters for NCS entities.
    pub fn ncs_normal(&self) -> &PrecomputedNormal {
        &self.ncs_normal
    }

    /// Returns the sorted entity IDs of NCS entities in the stochastic pipeline.
    #[must_use]
    pub fn ncs_entity_ids(&self) -> &[EntityId] {
        &self.ncs_entity_ids
    }

    /// Returns the canonical entity ID ordering for the noise dimension
    /// (`hydro_ids ++ load_bus_ids ++ ncs_entity_ids`).
    #[must_use]
    pub fn entity_order(&self) -> &[EntityId] {
        &self.entity_order
    }

    /// Returns the number of stochastic NCS entities in the noise dimension.
    #[must_use]
    pub fn n_stochastic_ncs(&self) -> usize {
        self.n_stochastic_ncs
    }

    /// Returns the number of hydro entities in the stochastic model.
    ///
    /// Equal to `dim() - n_load_buses() - n_stochastic_ncs()`. Exposed here
    /// to avoid repeating the three-term subtraction at every call site.
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.dim - self.n_load_buses - self.n_stochastic_ncs
    }

    /// Returns the number of study stages in the opening tree.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.opening_tree.n_stages()
    }

    /// Returns provenance metadata recording how each component was obtained.
    #[must_use]
    pub fn provenance(&self) -> &StochasticProvenance {
        &self.provenance
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
/// 5. Build [`PrecomputedPar`] from PAR model parameters and study stages.
/// 6. Build [`DecomposedCorrelation`] from the system correlation model.
/// 7. Generate the opening scenario tree from the expanded entity order
///    (`hydro_ids` followed by `load_bus_ids` followed by `ncs_ids`) with
///    `dim = n_hydros + n_load_buses + n_stochastic_ncs`.
/// 8. Build [`PrecomputedNormal`] for the stochastic load buses.
///
/// The `base_seed` parameter must be supplied explicitly by the caller.
/// Converting `ScenarioSource.seed: Option<i64>` to a `u64` — including
/// the `None` case with OS entropy — is an application-level concern that
/// belongs in the calling crate, not in this infrastructure crate.
///
/// The `load_factors` parameter provides per-`(entity_id, stage_id, block_factors)`
/// scaling entries consumed by [`PrecomputedNormal`]. Pass an empty slice when
/// no load factor file was loaded; all block factors then default to `1.0`. The
/// caller is responsible for converting any external load factor representation
/// into [`EntityFactorEntry`] slices before calling this function.
///
/// The `ncs_factors` parameter provides per-`(entity_id, stage_id, block_factors)`
/// scaling entries for NCS entities consumed by the NCS [`PrecomputedNormal`].
/// Pass an empty slice when no NCS factor file was loaded.
///
/// # Errors
///
/// - [`StochasticError::InvalidParParameters`]: a PAR model has AR order > 0
///   with zero standard deviation.
/// - [`StochasticError::InvalidCorrelation`]: the correlation model is empty,
///   ambiguous, or contains an invalid matrix.
/// - [`StochasticError::SpectralDecompositionFailed`]: a correlation matrix
///   is not positive-definite.
///
/// [`LoadModel`]: cobre_core::scenario::LoadModel
#[allow(clippy::too_many_lines)]
pub fn build_stochastic_context(
    system: &System,
    base_seed: u64,
    forward_seed: Option<u64>,
    load_factors: &[EntityFactorEntry<'_>],
    ncs_factors: &[EntityFactorEntry<'_>],
    opening_tree_inputs: OpeningTreeInputs<'_>,
    schemes: ClassSchemes,
) -> Result<StochasticContext, StochasticError> {
    let OpeningTreeInputs {
        user_tree: user_opening_tree,
        historical_library,
    } = opening_tree_inputs;
    let _report = validate_par_parameters(system.inflow_models())?;

    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();

    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

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

    // Collect NCS entity IDs that have model entries. Entities with `std = 0`
    // produce deterministic availability at `mean * max_gen`; the noise dimension
    // still exists but contributes zero noise after the transform.
    let ncs_entity_ids: Vec<EntityId> = {
        let mut ids: Vec<EntityId> = system.ncs_models().iter().map(|m| m.ncs_id).collect();
        ids.sort_unstable_by_key(|id| id.0);
        ids.dedup();
        ids
    };
    let n_stochastic_ncs = ncs_entity_ids.len();

    let dim = hydro_ids.len() + n_load_buses + n_stochastic_ncs;

    // Compute provenance BEFORE consuming `user_opening_tree` by pattern match.
    let provenance = {
        let opening_tree_prov = if user_opening_tree.is_some() {
            ComponentProvenance::UserSupplied
        } else if dim > 0 {
            ComponentProvenance::Generated
        } else {
            ComponentProvenance::NotApplicable
        };

        let correlation_prov = if !system.correlation().profiles.is_empty() && dim > 0 {
            ComponentProvenance::Generated
        } else {
            ComponentProvenance::NotApplicable
        };

        let inflow_prov = if hydro_ids.is_empty() {
            ComponentProvenance::NotApplicable
        } else {
            ComponentProvenance::Generated
        };

        StochasticProvenance {
            opening_tree: opening_tree_prov,
            correlation: correlation_prov,
            inflow_model: inflow_prov,
            inflow_scheme: schemes.inflow,
            load_scheme: schemes.load,
            ncs_scheme: schemes.ncs,
        }
    };

    let par_lp = PrecomputedPar::build(system.inflow_models(), &study_stages, &hydro_ids)?;

    let correlation = if dim == 0 || system.correlation().profiles.is_empty() {
        DecomposedCorrelation::empty()
    } else {
        DecomposedCorrelation::build(system.correlation())?
    };

    let entity_order: Vec<EntityId> = hydro_ids
        .iter()
        .copied()
        .chain(load_bus_ids.iter().copied())
        .chain(ncs_entity_ids.iter().copied())
        .collect();

    let opening_tree = if let Some(tree) = user_opening_tree {
        tree
    } else {
        generate_opening_tree(
            base_seed,
            &study_stages,
            dim,
            &correlation,
            &entity_order,
            ClassDimensions {
                n_hydros: hydro_ids.len(),
                n_load_buses,
                n_ncs: n_stochastic_ncs,
            },
            historical_library,
        )?
    };

    let max_blocks = study_stages
        .iter()
        .map(|s| s.blocks.len())
        .max()
        .unwrap_or(0);

    let normal_lp = PrecomputedNormal::build(
        system.load_models(),
        load_factors,
        &study_stages,
        &load_bus_ids,
        max_blocks,
    )?;

    // Build NCS PrecomputedNormal by mapping NcsModel -> LoadModel.
    // The dimensionless availability factors (mean, std) are stored in the
    // LoadModel's mean_mw/std_mw fields; the noise transform in noise.rs
    // applies the max_gen scaling: A_r = max_gen * clamp(mean + std * eta, 0, 1).
    let ncs_normal = if ncs_entity_ids.is_empty() {
        PrecomputedNormal::default()
    } else {
        let ncs_as_load: Vec<LoadModel> = system
            .ncs_models()
            .iter()
            .map(|m| LoadModel {
                bus_id: m.ncs_id,
                stage_id: m.stage_id,
                mean_mw: m.mean,
                std_mw: m.std,
            })
            .collect();
        PrecomputedNormal::build(
            &ncs_as_load,
            ncs_factors,
            &study_stages,
            &ncs_entity_ids,
            max_blocks,
        )?
    };

    Ok(StochasticContext {
        par_lp,
        correlation,
        opening_tree,
        normal_lp,
        ncs_normal,
        ncs_entity_ids,
        entity_order: entity_order.into_boxed_slice(),
        base_seed,
        forward_seed,
        dim,
        n_load_buses,
        n_stochastic_ncs,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::{
        entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
            LoadModel, SamplingScheme,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
        Bus, DeficitSegment, EntityId, SystemBuilder,
    };

    use super::{build_stochastic_context, ClassSchemes, OpeningTreeInputs};
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
            method: "spectral".to_string(),
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

        assert_eq!(ctx.par().n_hydros(), 2);
        assert_eq!(ctx.par().n_stages(), 3);
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

        let ctx = build_stochastic_context(
            &system,
            7,
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

        let result = build_stochastic_context(
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
        );

        assert!(
            matches!(result, Err(StochasticError::InvalidParParameters { .. })),
            "expected InvalidParParameters, got: {result:?}"
        );
    }

    /// AC: non-positive-definite correlation matrix succeeds with spectral decomposition.
    #[test]
    fn build_succeeds_on_non_pd_correlation() {
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
            method: "spectral".to_string(),
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

        let result = build_stochastic_context(
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
        );

        // With spectral decomposition, negative eigenvalues are clipped to
        // zero instead of failing. The build should succeed.
        assert!(
            result.is_ok(),
            "spectral decomposition should handle non-PD matrix, got: {result:?}"
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

        let ctx = build_stochastic_context(
            &system,
            0,
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
            ctx.dim(),
            3,
            "dim must equal n_hydros + n_load_buses + n_ncs = 2 + 1 + 0"
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

        let ctx = build_stochastic_context(
            &system,
            7,
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

    /// AC: `normal_lp()` accessor returns correctly built `PrecomputedNormal`.
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

        assert_eq!(ctx.n_load_buses(), 1);
        let nlp = ctx.normal();
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

    /// AC: `None` produces an identical result to the original 3-argument call.
    ///
    /// Verifies that passing `None` as `user_opening_tree` leaves behaviour
    /// unchanged relative to the pre-refactor signature.
    #[test]
    fn build_with_none_matches_original() {
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

        // Dimensions must match expectations for a 2-hydro, 3-stage, BF=3 system.
        assert_eq!(ctx.dim(), 2, "dim should be 2 (2 hydros, no load buses)");
        assert_eq!(ctx.n_stages(), 3, "n_stages should be 3");
        assert_eq!(ctx.base_seed(), 42, "base_seed should be 42");
        assert_eq!(ctx.opening_tree().n_stages(), 3, "tree must have 3 stages");
        assert_eq!(ctx.opening_tree().dim(), 2, "tree dim must be 2");
        assert_eq!(
            ctx.opening_tree().n_openings(0),
            3,
            "stage 0 must have 3 openings (BF=3)"
        );
    }

    /// AC: a pre-constructed `OpeningTree` passed as `Some(tree)` is used as-is.
    ///
    /// Verifies that `par_lp`, `correlation`, and `normal_lp` are still built
    /// from the system while tree generation is bypassed.
    #[test]
    fn build_with_user_supplied_tree_uses_provided_tree() {
        use crate::context::OpeningTree;

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

        // Construct a known tree: 2 stages, 4 openings per stage, dim=2.
        // Data values are all 99.0 so we can verify the exact values came from
        // the user-supplied tree rather than the generated one.
        let n_stages = 2usize;
        let n_openings = 4usize;
        let dim = 2usize;
        let total = n_stages * n_openings * dim;
        let data = vec![99.0_f64; total];
        let openings_per_stage = vec![n_openings; n_stages];
        let user_tree = OpeningTree::from_parts(data, openings_per_stage, dim);

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(user_tree),
                historical_library: None,
            },
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        // Tree dimensions must match the user-supplied tree, not the system's
        // branching factors.
        assert_eq!(
            ctx.opening_tree().n_stages(),
            2,
            "tree must have 2 stages (user-supplied)"
        );
        assert_eq!(
            ctx.opening_tree().n_openings(0),
            4,
            "stage 0 must have 4 openings (user-supplied)"
        );
        // Verify values come from the user-supplied tree.
        assert_eq!(
            ctx.opening_tree().opening(0, 0),
            &[99.0_f64, 99.0],
            "opening values must match user-supplied data"
        );

        // par_lp, correlation, and normal_lp must still be built from the system.
        assert_eq!(
            ctx.par().n_hydros(),
            2,
            "par_lp must still reflect system hydros"
        );
        assert_eq!(
            ctx.par().n_stages(),
            3,
            "par_lp must still reflect system study stages"
        );
        assert_eq!(ctx.n_load_buses(), 0, "no load buses in this system");
    }

    /// AC: `entity_order()` returns the canonical `hydro_ids` ++ `load_bus_ids` order.
    #[test]
    fn test_entity_order_accessor() {
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
            make_load_model(5, 0, 100.0, 10.0),
            make_load_model(5, 1, 105.0, 10.0),
            make_load_model(5, 2, 110.0, 10.0),
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(5)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(identity_correlation(&[1, 2]))
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
            ctx.entity_order(),
            &[EntityId(1), EntityId(2), EntityId(5)],
            "entity_order must be hydro_ids ++ load_bus_ids"
        );
    }

    /// AC: `entity_order()` is populated even when a user-supplied tree is given.
    #[test]
    fn test_entity_order_with_user_tree() {
        use crate::context::OpeningTree;

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

        let n_stages = 2usize;
        let n_openings = 4usize;
        let dim = 2usize;
        let data = vec![99.0_f64; n_stages * n_openings * dim];
        let openings_per_stage = vec![n_openings; n_stages];
        let user_tree = OpeningTree::from_parts(data, openings_per_stage, dim);

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(user_tree),
                historical_library: None,
            },
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.entity_order(),
            &[EntityId(1), EntityId(2)],
            "entity_order must be populated even with user-supplied tree"
        );
        assert!(
            !ctx.entity_order().is_empty(),
            "entity_order must not be empty"
        );
    }

    /// AC: `forward_seed()` returns Some(seed) when `scenario_source.seed` is supplied.
    #[test]
    fn test_forward_seed_from_config() {
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

        let ctx = build_stochastic_context(
            &system,
            42,
            Some(123),
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
            ctx.forward_seed(),
            Some(123),
            "forward_seed() must return Some(123) when supplied as Some(123)"
        );
    }

    /// AC: `forward_seed()` returns None when `scenario_source.seed` is absent.
    #[test]
    fn test_forward_seed_none_when_absent() {
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
            ctx.forward_seed(),
            None,
            "forward_seed() must return None when not supplied"
        );
    }
}
