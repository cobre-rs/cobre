//! Top-level stochastic pipeline initialization.
//!
//! [`StochasticContext`] owns the precomputed stochastic infrastructure;
//! [`build_stochastic_context`] assembles it from a [`System`] reference.

use std::collections::HashMap;

use cobre_core::{
    EntityId, LoadModel, System,
    scenario::{ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, InflowModel, SamplingScheme},
    temporal::Stage,
};

/// Per-class sampling scheme selections for provenance tracking; each field is
/// `None` when its class is not configured.
#[derive(Debug, Clone, Copy)]
pub struct ClassSchemes {
    /// Inflow class.
    pub inflow: Option<SamplingScheme>,
    /// Load class.
    pub load: Option<SamplingScheme>,
    /// NCS class.
    pub ncs: Option<SamplingScheme>,
}

/// Optional caller overrides for the opening scenario tree. When every field is
/// `None` the tree is generated from SAA/LHS/QMC noise per each stage's
/// `scenario_config.noise_method`.
#[derive(Debug, Default)]
pub struct OpeningTreeInputs<'a> {
    /// A pre-built opening tree that bypasses generation. When `Some`, the
    /// `historical_library` and `external_scenario_counts` fields are ignored.
    pub user_tree: Option<OpeningTree>,
    /// Historical library, required when any study stage uses
    /// [`NoiseMethod::HistoricalResiduals`](cobre_core::temporal::NoiseMethod::HistoricalResiduals)
    /// and `user_tree` is `None`.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-padding external scenario count per stage, clamping openings for
    /// stages with fewer external scenarios than the branching factor. `None`
    /// when no class uses External sampling; when `Some`, length must equal the
    /// number of study stages.
    pub external_scenario_counts: Option<Vec<usize>>,
    /// Per-stage noise-group IDs (indexed by stage array index): stages sharing
    /// a group ID share one noise draw, so weekly stages in a monthly bucket
    /// receive identical noise. `None` draws independent noise per stage; when
    /// `Some`, length must equal the number of study stages.
    pub noise_group_ids: Option<Vec<u32>>,
}

use crate::{
    StochasticError,
    correlation::resolve::DecomposedCorrelation,
    derive_external_sample_moments,
    normal::precompute::{EntityFactorEntry, PrecomputedNormal},
    par::{precompute::PrecomputedPar, validation::validate_par_parameters},
    provenance::{ComponentProvenance, StochasticProvenance},
    sampling::historical::HistoricalScenarioLibrary,
    tree::{
        generate::{ClassDimensions, OpeningTreeGenerationInputs, generate_opening_tree},
        opening_tree::OpeningTreeView,
    },
};

pub use crate::tree::opening_tree::OpeningTree;

/// The entity IDs occupying a [`System`]'s noise vector, one block per class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseEntityOrder {
    /// Hydro IDs, in `System::hydros` canonical order.
    pub hydro_ids: Vec<EntityId>,
    /// Buses carrying load noise (`std_mw > 0`), ID-sorted and deduplicated.
    pub load_bus_ids: Vec<EntityId>,
    /// NCS IDs, ID-sorted and deduplicated.
    pub ncs_entity_ids: Vec<EntityId>,
}

impl NoiseEntityOrder {
    /// The per-class entity counts of this order's three blocks.
    #[must_use]
    #[inline]
    pub fn class_dimensions(&self) -> ClassDimensions {
        ClassDimensions {
            n_hydros: self.hydro_ids.len(),
            n_load_buses: self.load_bus_ids.len(),
            n_ncs: self.ncs_entity_ids.len(),
        }
    }

    /// The noise dimension: the three blocks' combined length.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.class_dimensions().total()
    }

    /// The three blocks concatenated as `hydro_ids ++ load_bus_ids ++ ncs_entity_ids`.
    #[must_use]
    pub fn entity_order(&self) -> Vec<EntityId> {
        self.hydro_ids
            .iter()
            .copied()
            .chain(self.load_bus_ids.iter().copied())
            .chain(self.ncs_entity_ids.iter().copied())
            .collect()
    }
}

/// Derive `system`'s canonical noise-entity layout under `schemes`.
///
/// Single owner: every site sizing or slicing the noise vector calls this rather
/// than re-deriving a class block — a second copy that omits the NCS block sizes
/// the noise vector short, and a caller that re-derives the NCS offset instead
/// of calling [`ClassDimensions::ncs_range`] indexes past the end of a row.
/// An NCS with `std = 0` is included unconditionally: it contributes zero noise
/// after the transform, and dropping it would shift the canonical entity order.
///
/// Load and NCS membership both route through `System`'s own authority
/// ([`System::load_noise_member_bus_ids`], [`System::ncs_noise_member_ids`]) —
/// the single owner every LP/noise-vector site (this function, the external
/// library builders, the LP template builder) calls rather than re-deriving
/// membership. Every caller must pass the SAME resolved `schemes` (the
/// training scenario source) so the external-library width check, the
/// opening-tree layout, and the backward assembly all agree on which
/// entities occupy the vector.
#[must_use]
pub fn noise_entity_order(system: &System, schemes: &ClassSchemes) -> NoiseEntityOrder {
    let load_scheme = schemes.load.unwrap_or(SamplingScheme::InSample);
    let ncs_scheme = schemes.ncs.unwrap_or(SamplingScheme::InSample);

    NoiseEntityOrder {
        hydro_ids: system.hydros().iter().map(|h| h.id).collect(),
        load_bus_ids: system.load_noise_member_bus_ids(load_scheme),
        ncs_entity_ids: system.ncs_noise_member_ids(ncs_scheme),
    }
}

/// Fully-initialized stochastic pipeline components, owned in one place.
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
/// #         operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
/// #     let mut hydro = Hydro {
/// #         id: EntityId(id),
/// #         name: format!("H{id}"),
/// #         operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
/// #         downstream_id: None,
/// #         travel_time_hours: None,
/// #         entry_stage_id: None,
/// #         exit_stage_id: None,
/// #         min_storage_hm3: 0.0,
/// #         max_storage_hm3: 100.0,
/// #         min_outflow_m3s: 0.0,
/// #         max_outflow_m3s: None,
/// #         generation_model: HydroGenerationModel::ConstantProductivity,
/// #         specific_productivity_mw_per_m3s_per_m: None,
/// #         min_turbined_m3s: 0.0,
/// #         max_turbined_m3s: 100.0,
/// #         min_generation_mw: 0.0,
/// #         max_generation_mw: 100.0,
/// #         unit_groups: Vec::new(),
/// #         tailrace: None,
/// #         hydraulic_losses: None,
/// #         efficiency: None,
/// #         evaporation_coefficients_mm: None,
/// #         evaporation_reference_volumes_hm3: None,
/// #         diversion: None,
/// #         filling: None,
/// #         penalties: HydroPenalties {
/// #             spillage_cost: 0.0, diversion_cost: 0.0, turbined_cost: 0.0,
/// #             storage_violation_below_cost: 0.0, filling_target_violation_cost: 0.0,
/// #             turbined_violation_below_cost: 0.0, outflow_violation_below_cost: 0.0,
/// #             outflow_violation_above_cost: 0.0, generation_violation_below_cost: 0.0,
/// #             evaporation_violation_cost: 0.0, water_withdrawal_violation_cost: 0.0,
/// #             water_withdrawal_violation_pos_cost: 0.0, water_withdrawal_violation_neg_cost: 0.0,
/// #             evaporation_violation_pos_cost: 0.0, evaporation_violation_neg_cost: 0.0,
/// #             inflow_nonnegativity_cost: 1000.0,
/// #         },
/// #     };
/// #     hydro.declare_mirror_unit_group(EntityId(0));
/// #     hydro
/// # }
/// # fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
/// #     InflowModel {
/// #         hydro_id: EntityId(hydro_id),
/// #         stage_id,
/// #         mean_m3s: 100.0,
/// #         std_m3s: 30.0,
/// #         ar_coefficients: vec![],
/// #         residual_std_ratio: 1.0,
/// #         annual: None,
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
    entity_order: Box<[EntityId]>,
    base_seed: u64,
    /// Seed for `OutOfSample` forward-pass noise, independent of `base_seed`.
    forward_seed: Option<u64>,
    class_dimensions: ClassDimensions,
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

    /// Install a per-stage opening solve order, sorting each stage's openings by
    /// descending `keys[s]` (see [`OpeningTree::set_solve_order`]). Keys are
    /// caller-computed from setup-constant data, so the order is run-constant and
    /// identical across processes handed the same keys.
    ///
    /// # Errors
    ///
    /// Propagates [`StochasticError::InsufficientData`]
    /// when the key table's stage count or any stage's key count does not match
    /// the tree.
    pub fn set_solve_order(&mut self, keys: &[Vec<f64>]) -> Result<(), StochasticError> {
        self.opening_tree.set_solve_order(keys)
    }

    /// Returns the base seed used to generate the opening tree.
    #[must_use]
    pub fn base_seed(&self) -> u64 {
        self.base_seed
    }

    /// Returns the `OutOfSample` forward-pass noise seed supplied at build time.
    ///
    /// The context never reads it; callers pass it to
    /// [`ForwardSamplerConfig::forward_seed`](crate::ForwardSamplerConfig::forward_seed).
    #[must_use]
    pub fn forward_seed(&self) -> Option<u64> {
        self.forward_seed
    }

    /// Returns the per-class entity counts of the noise dimension.
    #[must_use]
    #[inline]
    pub fn class_dimensions(&self) -> ClassDimensions {
        self.class_dimensions
    }

    /// Returns the noise dimension (the noise-vector layout's total width).
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.class_dimensions.total()
    }

    /// Returns the number of stochastic load buses in the noise dimension.
    #[must_use]
    #[inline]
    pub fn n_load_buses(&self) -> usize {
        self.class_dimensions.n_load_buses
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
        &self.entity_order[self.class_dimensions.ncs_range()]
    }

    /// Returns the canonical entity ID ordering for the noise dimension
    /// (`hydro_ids ++ load_bus_ids ++ ncs_entity_ids`).
    #[must_use]
    pub fn entity_order(&self) -> &[EntityId] {
        &self.entity_order
    }

    /// Returns the number of stochastic NCS entities in the noise dimension.
    #[must_use]
    #[inline]
    pub fn n_stochastic_ncs(&self) -> usize {
        self.class_dimensions.n_ncs
    }

    /// Returns the number of hydro entities in the noise dimension.
    #[must_use]
    #[inline]
    pub fn n_hydros(&self) -> usize {
        self.class_dimensions.n_hydros
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

/// Map each study stage's declared `Stage.id` to its 0-based position in
/// `study_stages` — the declared-domain-id -> canonical-study-index
/// resolution `cobre-io`'s `StageIdResolver` performs for the σ=0 validator
/// (rule 47), replicated here because `cobre-stochastic` cannot depend on
/// `cobre-io` (`cobre-io` depends on `cobre-stochastic`).
/// `ExternalScenarioRow`/`ExternalLoadRow`/`ExternalNcsRow::stage_id` is
/// documented as a declared domain id, "not a 0-based index" — every
/// External-row consumer below resolves through this map, never casts
/// `stage_id as usize` directly.
fn stage_id_to_index(study_stages: &[Stage]) -> HashMap<i32, usize> {
    study_stages
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i))
        .collect()
}

/// Resolve each row's declared `stage_id` to its 0-based study-stage index
/// via `stage_index`, dropping rows whose declared id names no study stage
/// (mirroring an unresolvable id being rejected upstream, never silently
/// mis-indexed) — [`derive_external_sample_moments`]'s own contract requires
/// an already-resolved index, never the raw declared id.
fn resolve_row_stages<R, FR>(
    rows: &[R],
    stage_index: &HashMap<i32, usize>,
    row_fields: FR,
) -> Vec<(EntityId, i32, i32, f64)>
where
    FR: Fn(&R) -> (EntityId, i32, i32, f64),
{
    rows.iter()
        .filter_map(|row| {
            let (entity_id, stage_id, scenario_id, value) = row_fields(row);
            let idx = *stage_index.get(&stage_id)?;
            let idx_i32 = i32::try_from(idx).ok()?;
            Some((entity_id, idx_i32, scenario_id, value))
        })
        .collect()
}

/// Under `External`, build a derived-moment [`LoadModel`] slice — one entry
/// per `(entity, study stage)` with `mean_mw`/`std_mw` set to the sample
/// moments [`derive_external_sample_moments`] computes over `external_rows`.
/// Feeds [`PrecomputedNormal::build`] the SAME `(μ, σ)` pair the
/// standardization side derives from the same rows, so the
/// standardize/reconstruct round trip holds — both sides must keep deriving
/// from the same rows.
fn external_derived_load_models<R, FR>(
    external_rows: &[R],
    entity_ids: &[EntityId],
    study_stages: &[Stage],
    stage_index: &HashMap<i32, usize>,
    row_fields: FR,
) -> Vec<LoadModel>
where
    FR: Fn(&R) -> (EntityId, i32, i32, f64),
{
    let n_entities = entity_ids.len();
    let resolved_rows = resolve_row_stages(external_rows, stage_index, row_fields);
    let moments = derive_external_sample_moments(
        &resolved_rows,
        entity_ids,
        study_stages.len(),
        |&(entity_id, stage_idx, scenario_id, value)| (entity_id, stage_idx, scenario_id, value),
    );
    let mut models = Vec::with_capacity(study_stages.len() * n_entities);
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        for (entity_idx, &bus_id) in entity_ids.iter().enumerate() {
            let (mean, std) = moments[stage_idx * n_entities + entity_idx];
            models.push(LoadModel {
                bus_id,
                stage_id: stage.id,
                mean_mw: mean,
                std_mw: std,
            });
        }
    }
    models
}

/// Under `External`, override each AR(0) hydro's study-stage [`InflowModel`]
/// mean/std with sample moments derived from `system.external_scenarios()`;
/// an AR(p > 0) hydro's models pass through unchanged. AR(0)-ness mirrors
/// `par_probe`'s own per-hydro classification (`PrecomputedPar::order`/
/// `has_annual`, built from the unmodified models), so the override can
/// never disagree with what a later `PrecomputedPar::build` itself would
/// treat as AR(0).
///
/// A hydro/stage with no `inflow_seasonal_stats` row at all — the file is
/// entirely absent — still classifies AR(0) under `PrecomputedPar::build`'s
/// default (`order`/`has_annual` start at their zero value for a hydro with no
/// models), so it is synthesized here from `moments` rather than left out:
/// otherwise the override never fires for that hydro and `PrecomputedPar`
/// defaults its mean/std to `0.0`, which V3.7 then rejects as a
/// deterministic-anchor mismatch against the real external value.
fn external_ar0_inflow_models(
    system: &System,
    hydro_ids: &[EntityId],
    study_stages: &[Stage],
    stage_index: &HashMap<i32, usize>,
    par_probe: &PrecomputedPar,
) -> Vec<InflowModel> {
    let n_stages = study_stages.len();
    let n_hydros = hydro_ids.len();
    let resolved_rows = resolve_row_stages(
        system.external_scenarios(),
        stage_index,
        |row: &ExternalScenarioRow| (row.hydro_id, row.stage_id, row.scenario_id, row.value_m3s),
    );
    let moments = derive_external_sample_moments(
        &resolved_rows,
        hydro_ids,
        n_stages,
        |&(entity_id, stage_idx, scenario_id, value)| (entity_id, stage_idx, scenario_id, value),
    );
    let hydro_index: HashMap<EntityId, usize> = hydro_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut has_row = vec![false; n_stages * n_hydros];
    let mut models: Vec<InflowModel> = system
        .inflow_models()
        .iter()
        .cloned()
        .map(|model| {
            let Some(&stage_idx) = stage_index.get(&model.stage_id) else {
                return model;
            };
            let Some(&h_idx) = hydro_index.get(&model.hydro_id) else {
                return model;
            };
            has_row[stage_idx * n_hydros + h_idx] = true;
            if par_probe.order(h_idx) != 0 || par_probe.has_annual(h_idx) {
                return model;
            }
            let (mean_m3s, std_m3s) = moments[stage_idx * n_hydros + h_idx];
            InflowModel {
                mean_m3s,
                std_m3s,
                ..model
            }
        })
        .collect();

    for (h_idx, &hydro_id) in hydro_ids.iter().enumerate() {
        if par_probe.order(h_idx) != 0 || par_probe.has_annual(h_idx) {
            continue;
        }
        for (stage_idx, stage) in study_stages.iter().enumerate() {
            if has_row[stage_idx * n_hydros + h_idx] {
                continue;
            }
            let (mean_m3s, std_m3s) = moments[stage_idx * n_hydros + h_idx];
            models.push(InflowModel {
                hydro_id,
                stage_id: stage.id,
                mean_m3s,
                std_m3s,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            });
        }
    }

    models
}

/// The PAR model the LP applies to inflow noise: the fitted build over
/// `system`'s study stages and hydros, then the [`SamplingScheme::External`]
/// override when `inflow_scheme` names it.
///
/// # Errors
///
/// Returns [`StochasticError::InvalidParParameters`] when a PAR model has AR
/// order > 0 with zero standard deviation.
pub fn build_inflow_par(
    system: &System,
    inflow_scheme: Option<SamplingScheme>,
) -> Result<PrecomputedPar, StochasticError> {
    validate_par_parameters(system.inflow_models())?;
    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let stage_index = stage_id_to_index(&study_stages);
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let season_map = system.policy_graph().season_map.as_ref();

    let par_lp = PrecomputedPar::build(
        system.inflow_models(),
        system.stages(),
        &hydro_ids,
        season_map,
    )?;
    if inflow_scheme == Some(SamplingScheme::External) {
        let external_models =
            external_ar0_inflow_models(system, &hydro_ids, &study_stages, &stage_index, &par_lp);
        PrecomputedPar::build(&external_models, system.stages(), &hydro_ids, season_map)
    } else {
        Ok(par_lp)
    }
}

/// Initialize the full stochastic pipeline from a [`System`] reference.
///
/// Stage filtering keeps only study stages (non-negative `stage.id`). Load-bus
/// and NCS IDs are sorted by [`EntityId`] for declaration-order invariance, and
/// the noise dimension is laid out as `hydro_ids ++ load_bus_ids ++ ncs_ids`.
///
/// `base_seed` is supplied explicitly: converting `ScenarioSource.seed:
/// Option<i64>` to `u64` (including the `None`-means-OS-entropy case) is an
/// application-level concern, not this infrastructure crate's.
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
///
/// [`LoadModel`]: cobre_core::scenario::LoadModel
// Rationale: extracting sub-steps would thread the same partially-built context
// state through every helper, obscuring the build dependency order without
// reducing real complexity.
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
        external_scenario_counts,
        noise_group_ids,
    } = opening_tree_inputs;

    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let stage_index = stage_id_to_index(&study_stages);

    let noise_order = noise_entity_order(system, &schemes);
    let class_dimensions = noise_order.class_dimensions();
    let dim = class_dimensions.total();
    let entity_order = noise_order.entity_order();
    let NoiseEntityOrder {
        hydro_ids,
        load_bus_ids,
        ncs_entity_ids,
    } = noise_order;

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

    let par_lp = build_inflow_par(system, schemes.inflow)?;

    let correlation = if dim == 0 || system.correlation().profiles.is_empty() {
        DecomposedCorrelation::empty()
    } else {
        DecomposedCorrelation::build(system.correlation(), &entity_order, class_dimensions)?
    };

    let opening_tree = if let Some(tree) = user_opening_tree {
        tree
    } else {
        generate_opening_tree(
            base_seed,
            &study_stages,
            &correlation,
            &entity_order,
            class_dimensions,
            &OpeningTreeGenerationInputs {
                historical_library,
                external_scenario_counts: external_scenario_counts.as_deref(),
                noise_group_ids: noise_group_ids.as_deref(),
            },
        )?
    };

    let max_blocks = study_stages
        .iter()
        .map(|s| s.blocks.len())
        .max()
        .unwrap_or(0);

    let normal_lp = if schemes.load == Some(SamplingScheme::External) {
        let external_models = external_derived_load_models(
            system.external_load_scenarios(),
            &load_bus_ids,
            &study_stages,
            &stage_index,
            |row: &ExternalLoadRow| (row.bus_id, row.stage_id, row.scenario_id, row.value_mw),
        );
        PrecomputedNormal::build(
            &external_models,
            load_factors,
            &study_stages,
            &load_bus_ids,
            max_blocks,
        )?
    } else {
        PrecomputedNormal::build(
            system.load_models(),
            load_factors,
            &study_stages,
            &load_bus_ids,
            max_blocks,
        )?
    };

    // The `mean_mw`/`std_mw` carried here are dimensionless availability factors,
    // not MW; the NCS noise transform applies the `max_gen` scaling.
    let ncs_normal = if ncs_entity_ids.is_empty() {
        PrecomputedNormal::default()
    } else {
        let ncs_as_load: Vec<LoadModel> = if schemes.ncs == Some(SamplingScheme::External) {
            external_derived_load_models(
                system.external_ncs_scenarios(),
                &ncs_entity_ids,
                &study_stages,
                &stage_index,
                |row: &ExternalNcsRow| (row.ncs_id, row.stage_id, row.scenario_id, row.value),
            )
        } else {
            system
                .ncs_models()
                .iter()
                .map(|m| LoadModel {
                    bus_id: m.ncs_id,
                    stage_id: m.stage_id,
                    mean_mw: m.mean,
                    std_mw: m.std,
                })
                .collect()
        };
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
        entity_order: entity_order.into_boxed_slice(),
        base_seed,
        forward_seed,
        class_dimensions,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use cobre_core::{
        Bus, DeficitSegment, EntityId, Hydro, SystemBuilder,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            ExternalLoadRow, ExternalScenarioRow, InflowModel, LoadModel, NcsModel, SamplingScheme,
        },
        temporal::{NoiseMethod, ScenarioSourceConfig, Stage},
        test_support::{BusSpec, HydroSpec, StageSpec, single_block},
    };

    use super::{
        ClassSchemes, OpeningTreeInputs, build_inflow_par, build_stochastic_context,
        noise_entity_order,
    };
    use crate::StochasticError;

    fn make_stage(index: usize, id: i32, branching_factor: usize) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            season_id: Some(0),
            blocks: single_block("SINGLE", 744.0),
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method: NoiseMethod::Saa,
            },
            ..Default::default()
        })
    }

    fn make_bus(id: i32) -> Bus {
        cobre_core::test_support::make_bus(BusSpec {
            id,
            name: format!("Bus{id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            ..Default::default()
        })
    }

    fn make_hydro(id: i32) -> Hydro {
        cobre_core::test_support::make_hydro(HydroSpec {
            id,
            name: format!("H{id}"),
            max_storage_hm3: 100.0,
            max_turbined_m3s: 100.0,
            max_generation_mw: 100.0,
            ..Default::default()
        })
    }

    fn make_inflow_model(hydro_id: i32, stage_id: i32, std: f64, coeffs: Vec<f64>) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: std,
            ar_coefficients: coeffs,
            residual_std_ratio: 1.0,
            annual: None,
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

    /// AC: `build_inflow_par` validates PAR parameters itself — a caller that
    /// reaches it without going through `build_stochastic_context` (e.g. the
    /// opening-tree path) must still see an AR(1) model's zero standard
    /// deviation rejected.
    #[test]
    fn build_inflow_par_rejects_invalid_par() {
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

        let result = build_inflow_par(&system, None);

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
        let rho = 2.0_f64;
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

    /// A reader that sizes the noise vector without the NCS block leaves the
    /// samplers' NCS class offset indexing past the end of a row.
    #[test]
    fn noise_entity_order_counts_every_ncs_including_zero_std() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(10)])
            .hydros(vec![make_hydro(2), make_hydro(1)])
            .stages(vec![make_stage(0, 0, 3)])
            .inflow_models(vec![
                make_inflow_model(1, 0, 30.0, vec![]),
                make_inflow_model(2, 0, 20.0, vec![]),
            ])
            .load_models(vec![
                make_load_model(0, 0, 50.0, 0.0),
                make_load_model(10, 0, 100.0, 10.0),
            ])
            .ncs_models(vec![
                NcsModel {
                    ncs_id: EntityId(20),
                    stage_id: 0,
                    mean: 0.7,
                    std: 0.1,
                },
                NcsModel {
                    ncs_id: EntityId(21),
                    stage_id: 0,
                    mean: 0.8,
                    std: 0.0,
                },
            ])
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();

        let order = noise_entity_order(
            &system,
            &ClassSchemes {
                inflow: None,
                load: None,
                ncs: None,
            },
        );

        assert_eq!(order.dim(), 5, "2 hydros + 1 stochastic load bus + 2 NCS");
        assert_eq!(
            order.entity_order(),
            vec![
                EntityId(1),
                EntityId(2),
                EntityId(10),
                EntityId(20),
                EntityId(21)
            ],
            "blocks concatenate as hydros ++ load buses ++ NCS, each canonically ordered"
        );
    }

    /// A `std_mw = 0.0` load bus is excluded by default (`std_mw > 0.0`
    /// only, unchanged), but included once its class scheme is `External` --
    /// membership becomes external-additive without touching inflow
    /// (all-hydros) or NCS (unfiltered, the C5 invariant above).
    #[test]
    fn noise_entity_order_load_membership_is_external_additive() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(10)])
            .hydros(vec![make_hydro(1)])
            .stages(vec![make_stage(0, 0, 3)])
            .inflow_models(vec![make_inflow_model(1, 0, 30.0, vec![])])
            .load_models(vec![
                make_load_model(0, 0, 50.0, 0.0),
                make_load_model(10, 0, 100.0, 10.0),
            ])
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let default_order = noise_entity_order(
            &system,
            &ClassSchemes {
                inflow: None,
                load: None,
                ncs: None,
            },
        );
        assert_eq!(
            default_order.load_bus_ids,
            vec![EntityId(10)],
            "without an External load scheme, membership stays std_mw > 0 only"
        );

        let external_order = noise_entity_order(
            &system,
            &ClassSchemes {
                inflow: None,
                load: Some(SamplingScheme::External),
                ncs: None,
            },
        );
        assert_eq!(
            external_order.load_bus_ids,
            vec![EntityId(0), EntityId(10)],
            "load_scheme == External admits the std_mw == 0.0 bus alongside the \
             stochastic one"
        );
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
        assert_eq!(
            ctx.opening_tree().opening(0, 0),
            &[99.0_f64, 99.0],
            "opening values must match user-supplied data"
        );

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

    /// AC (Unit Test): the AR-order branch selects derived-vs-seasonal
    /// correctly under `External` — an AR(0) hydro's PAR moments come from
    /// the external samples, while an AR(p > 0) hydro's stay byte-identical
    /// to the seasonal, unmodified build.
    #[test]
    fn external_inflow_override_applies_only_to_ar0_hydros() {
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let stages = vec![make_stage(0, 0, 1), make_stage(1, 1, 1)];
        // Hydro 1: AR(0), seasonal stats deliberately disagree with the
        // external file. Hydro 2: AR(1), stats must stay untouched.
        let inflow_models = vec![
            make_inflow_model(1, 0, 30.0, vec![]),
            make_inflow_model(1, 1, 30.0, vec![]),
            make_inflow_model(2, 0, 30.0, vec![0.4]),
            make_inflow_model(2, 1, 30.0, vec![0.4]),
        ];
        let external_rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id: EntityId(1),
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id: EntityId(1),
                value_m3s: 10.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 1,
                hydro_id: EntityId(1),
                value_m3s: 30.0,
            },
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1, 2]))
            .external_scenarios(external_rows)
            .build()
            .unwrap();

        let seasonal_ctx = build_stochastic_context(
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

        let external_ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        // AR(0) hydro (canonical index 0): derived from the external samples,
        // not the deliberately-disagreeing seasonal stats (mean=30.0 seasonal
        // std, vs. the external file's 200.0 / 10.0 / 30.0 samples).
        assert!((external_ctx.par().deterministic_base(0, 0) - 200.0).abs() < 1e-10);
        assert!(external_ctx.par().sigma(0, 0).abs() < 1e-10);
        assert!((external_ctx.par().deterministic_base(1, 0) - 20.0).abs() < 1e-10);
        assert!((external_ctx.par().sigma(1, 0) - 10.0).abs() < 1e-10);

        // AR(1) hydro (canonical index 1): byte-identical to the seasonal build.
        assert!(
            (external_ctx.par().deterministic_base(0, 1)
                - seasonal_ctx.par().deterministic_base(0, 1))
            .abs()
                < f64::EPSILON
        );
        assert!(
            (external_ctx.par().sigma(0, 1) - seasonal_ctx.par().sigma(0, 1)).abs() < f64::EPSILON
        );
        assert_eq!(
            external_ctx.par().psi_slice(0, 1),
            seasonal_ctx.par().psi_slice(0, 1)
        );
    }

    /// Regression: a GAPPED (non-0-based) declared study-stage id must
    /// resolve to its canonical position — the same resolution cobre-io's
    /// rule-47 validator performs — never be treated as already being that
    /// position. Study stages are declared `2, 5` (not `0, 1`); under the
    /// pre-fix bug, `stage_idx = declared_id` always exceeds `n_stages` here,
    /// so the AR(0)/load overrides would silently never fire and the
    /// (deliberately disagreeing) seasonal stats would survive untouched.
    #[test]
    fn external_derivation_resolves_gapped_stage_ids() {
        let hydro_id = EntityId(1);
        let bus_id = EntityId(10);
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 2, 1), make_stage(1, 5, 1)];
        let inflow_models = vec![
            make_inflow_model(1, 2, 30.0, vec![]),
            make_inflow_model(1, 5, 30.0, vec![]),
        ];
        let load_models = vec![
            make_load_model(10, 2, 999.0, 0.0),
            make_load_model(10, 5, 999.0, 0.0),
        ];
        let external_inflow_rows = vec![
            ExternalScenarioRow {
                stage_id: 2,
                scenario_id: 0,
                hydro_id,
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 5,
                scenario_id: 0,
                hydro_id,
                value_m3s: 10.0,
            },
            ExternalScenarioRow {
                stage_id: 5,
                scenario_id: 1,
                hydro_id,
                value_m3s: 30.0,
            },
        ];
        let external_load_rows = vec![
            ExternalLoadRow {
                stage_id: 2,
                scenario_id: 0,
                bus_id,
                value_mw: 123.0,
            },
            ExternalLoadRow {
                stage_id: 5,
                scenario_id: 0,
                bus_id,
                value_mw: 456.0,
            },
            ExternalLoadRow {
                stage_id: 5,
                scenario_id: 1,
                bus_id,
                value_mw: 456.0,
            },
        ];

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(10)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .correlation(identity_correlation(&[1]))
            .external_scenarios(external_inflow_rows)
            .external_load_scenarios(external_load_rows)
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
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::External),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        // Position 0 <-> declared stage id 2; position 1 <-> declared id 5.
        assert!(
            (ctx.par().deterministic_base(0, 0) - 200.0).abs() < 1e-10,
            "gapped id 2 must resolve to position 0 and derive from its own \
             external sample, not the disagreeing seasonal mean (999.0-shaped \
             sentinel); got {}",
            ctx.par().deterministic_base(0, 0)
        );
        assert!(ctx.par().sigma(0, 0).abs() < 1e-10);
        assert!(
            (ctx.par().deterministic_base(1, 0) - 20.0).abs() < 1e-10,
            "gapped id 5 must resolve to position 1"
        );
        assert!((ctx.par().sigma(1, 0) - 10.0).abs() < 1e-10);

        assert!(
            (ctx.normal().mean(0, 0) - 123.0).abs() < 1e-10,
            "load derivation must resolve gapped id 2 to position 0, not the \
             seasonal mean_mw (999.0)"
        );
        assert!(ctx.normal().std(0, 0).abs() < 1e-10);
        assert!(
            (ctx.normal().mean(1, 0) - 456.0).abs() < 1e-10,
            "gapped id 5 must resolve to position 1"
        );
    }
}
