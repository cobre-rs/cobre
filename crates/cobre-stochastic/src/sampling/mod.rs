//! Scenario sampling schemes.
//!
//! Provides sampling strategies that select which scenarios are simulated
//! during each iteration of a stochastic optimization algorithm.
//!
//! ## Submodules
//!
//! - [`insample`] — `InSample` scheme: draws scenarios uniformly from the
//!   opening tree and fixes them for the full iteration
//!
//! ## Forward Sampler
//!
//! [`ForwardSampler`] is a composite struct that unifies all supported sampling
//! strategies under a single `sample()` dispatch method. It holds three
//! [`ClassSampler`] instances (one per entity class) and applies per-class
//! Cholesky correlation only for `OutOfSample` class samplers. Use
//! [`build_forward_sampler`] to construct the appropriate sampler from a
//! [`SamplingScheme`] and a [`StochasticContext`].
//!
//! ```
//! use cobre_core::scenario::SamplingScheme;
//! use cobre_stochastic::sampling::{ForwardSampler, build_forward_sampler};
//! ```

pub mod class_sampler;
pub mod external;
pub mod historical;
pub mod insample;
pub mod window;

pub use class_sampler::{ClassSampleRequest, ClassSampler};
pub use external::{
    ExternalScenarioLibrary, standardize_external_inflow, standardize_external_load,
    standardize_external_ncs, validate_external_library,
};
pub use historical::{
    HistoricalScenarioLibrary, standardize_historical_windows, validate_historical_library,
};
pub use window::discover_historical_windows;
pub(crate) mod out_of_sample;

use cobre_core::{EntityId, Stage, scenario::SamplingScheme, temporal::NoiseMethod};

use crate::{
    StochasticError, context::StochasticContext, correlation::resolve::DecomposedCorrelation,
    tree::generate::ClassDimensions,
};

// ---------------------------------------------------------------------------
// ForwardNoise
// ---------------------------------------------------------------------------

/// Noise payload returned by [`ForwardSampler::sample`].
///
/// A newtype wrapping a borrowed slice; lifetime tied to the caller-supplied buffer.
#[derive(Debug)]
pub struct ForwardNoise<'b>(pub &'b [f64]);

impl ForwardNoise<'_> {
    /// Return the underlying noise slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.0
    }
}

// ---------------------------------------------------------------------------
// CorrelationRef
// ---------------------------------------------------------------------------

/// Pre-decomposed correlation matrix and entity ordering for one entity class.
///
/// Present (`Some`) only for `OutOfSample` samplers. `InSample`, `Historical`,
/// and `External` samplers produce pre-correlated noise and must not apply it again.
pub struct CorrelationRef<'a> {
    /// Pre-decomposed Cholesky factors for this entity class.
    pub decomposed: &'a DecomposedCorrelation,
    /// Canonical entity ID ordering for the class segment.
    pub entity_order: &'a [EntityId],
}

// ---------------------------------------------------------------------------
// ForwardSampler
// ---------------------------------------------------------------------------

/// Composite forward-pass sampler holding one [`ClassSampler`] per entity class.
///
/// Constructed once per run via [`build_forward_sampler`] and reused
/// across all `(iteration, scenario, stage)` calls without per-call allocation.
///
/// The `sample()` method splits the caller-supplied `noise_buf` into three
/// segments `[hydros | load_buses | ncs]`, delegates to each class sampler's
/// `fill()`, then applies per-class Cholesky correlation where `Some(corr_ref)`
/// is present.
pub struct ForwardSampler<'a> {
    /// Class sampler for inflow (hydro) entities.
    inflow: ClassSampler<'a>,
    /// Class sampler for stochastic load bus entities.
    load: ClassSampler<'a>,
    /// Class sampler for NCS entities.
    ncs: ClassSampler<'a>,
    /// Per-class entity counts that define the buffer split.
    dims: ClassDimensions,
    /// Correlation ref for the inflow class; `None` for pre-correlated sources.
    inflow_correlation: Option<CorrelationRef<'a>>,
    /// Correlation ref for the load class; `None` for pre-correlated sources.
    load_correlation: Option<CorrelationRef<'a>>,
    /// Correlation ref for the NCS class; `None` for pre-correlated sources.
    ncs_correlation: Option<CorrelationRef<'a>>,
}

impl<'a> ForwardSampler<'a> {
    /// Construct a [`ForwardSampler`] from its constituent parts.
    ///
    /// Called by [`build_forward_sampler`] and by tests that need fine-grained
    /// control over per-class sampler selection.
    pub(crate) fn new(
        inflow: ClassSampler<'a>,
        load: ClassSampler<'a>,
        ncs: ClassSampler<'a>,
        dims: ClassDimensions,
        inflow_correlation: Option<CorrelationRef<'a>>,
        load_correlation: Option<CorrelationRef<'a>>,
        ncs_correlation: Option<CorrelationRef<'a>>,
    ) -> Self {
        Self {
            inflow,
            load,
            ncs,
            dims,
            inflow_correlation,
            load_correlation,
            ncs_correlation,
        }
    }
}

impl std::fmt::Debug for ForwardSampler<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForwardSampler")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// SampleRequest
// ---------------------------------------------------------------------------

/// Per-call arguments for [`ForwardSampler::sample`].
///
/// Bundles arguments to keep the `sample()` method signature within budget.
pub struct SampleRequest<'b> {
    /// Training iteration counter (0-based).
    pub iteration: u32,
    /// Global scenario index (includes MPI offset).
    pub scenario: u32,
    /// Stage domain ID used for seed derivation.
    pub stage: u32,
    /// Stage array index used for tree/method lookup.
    pub stage_idx: usize,
    /// Caller-owned buffer for fresh noise output.
    pub noise_buf: &'b mut [f64],
    /// Caller-owned scratch for LHS permutation generation.
    pub perm_scratch: &'b mut [usize],
    /// Total scenario count across all ranks (for LHS stratification).
    pub total_scenarios: u32,
}

impl ForwardSampler<'_> {
    /// Inject pre-study lag values for historical windows into the stage-0 state
    /// vector before the stage loop begins.
    ///
    /// Delegates to each class sampler's `apply_initial_state`. Only
    /// `ClassSampler::Historical` performs any work; all other variants are
    /// no-ops. Load and NCS class samplers are called for consistency but are
    /// always no-ops because neither class has inflow lag state.
    ///
    /// The `lag_offset` is an absolute index into `state` computed by the caller
    /// from the `StageIndexer`.
    pub fn apply_initial_state(
        &self,
        req: &ClassSampleRequest,
        state: &mut [f64],
        lag_offset: usize,
    ) {
        self.inflow.apply_initial_state(req, state, lag_offset);
        // Load and NCS have no lag state; calls are no-ops.
        self.load.apply_initial_state(req, state, 0);
        self.ncs.apply_initial_state(req, state, 0);
    }

    /// Draw noise for a single `(iteration, scenario, stage)` triple.
    ///
    /// Splits `req.noise_buf` into per-class segments `[hydros | load_buses | ncs]`,
    /// calls `fill()` on each class sampler, then applies per-class Cholesky
    /// correlation where `Some(corr_ref)` is present.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::DimensionExceedsCapacity`] — when any `OutOfSample`
    ///   class uses `QmcSobol` and `dim > MAX_SOBOL_DIM`.
    /// - [`StochasticError::InsufficientData`] — when `stage_idx` is out of
    ///   bounds for any per-stage noise methods.
    //
    // Passing SampleRequest by value is intentional: we need owned access
    // to write into req.noise_buf and return a slice borrowing from it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn sample<'b>(&self, req: SampleRequest<'b>) -> Result<ForwardNoise<'b>, StochasticError> {
        let total_dim = self.dims.n_hydros + self.dims.n_load_buses + self.dims.n_ncs;

        // Split the noise buffer into three class segments.
        let (inflow_buf, rest) = req.noise_buf.split_at_mut(self.dims.n_hydros);
        let (load_buf, ncs_buf) = rest.split_at_mut(self.dims.n_load_buses);

        let class_req = ClassSampleRequest {
            iteration: req.iteration,
            scenario: req.scenario,
            stage: req.stage,
            stage_idx: req.stage_idx,
            total_scenarios: req.total_scenarios,
        };

        self.inflow.fill(&class_req, inflow_buf, req.perm_scratch)?;
        self.load.fill(&class_req, load_buf, req.perm_scratch)?;
        self.ncs.fill(&class_req, ncs_buf, req.perm_scratch)?;

        // Apply per-class correlation only where configured (OutOfSample path).
        #[allow(clippy::cast_possible_wrap)]
        if let Some(ref corr) = self.inflow_correlation {
            corr.decomposed.apply_correlation_for_class(
                req.stage as i32,
                inflow_buf,
                corr.entity_order,
                "inflow",
            );
        }
        #[allow(clippy::cast_possible_wrap)]
        if let Some(ref corr) = self.load_correlation {
            corr.decomposed.apply_correlation_for_class(
                req.stage as i32,
                load_buf,
                corr.entity_order,
                "load",
            );
        }
        #[allow(clippy::cast_possible_wrap)]
        if let Some(ref corr) = self.ncs_correlation {
            corr.decomposed.apply_correlation_for_class(
                req.stage as i32,
                ncs_buf,
                corr.entity_order,
                "ncs",
            );
        }

        Ok(ForwardNoise(&req.noise_buf[..total_dim]))
    }
}

// ---------------------------------------------------------------------------
// ForwardSamplerConfig
// ---------------------------------------------------------------------------

/// All parameters needed by [`build_forward_sampler`].
///
/// Bundles the per-class scheme selections, stochastic context, stage list,
/// class dimensions, and optional library references into a single struct to
/// keep the factory function signature within the argument budget.
#[derive(Clone, Copy)]
pub struct ForwardSamplerConfig<'a> {
    /// Per-class sampling scheme selections.
    pub class_schemes: crate::context::ClassSchemes,
    /// Stochastic context providing tree, seeds, correlation, and entity order.
    pub ctx: &'a StochasticContext,
    /// Study stages in index order; required by `OutOfSample` to read per-stage
    /// noise methods.
    pub stages: &'a [Stage],
    /// Per-class entity counts for noise buffer splitting.
    pub dims: ClassDimensions,
    /// Pre-standardized historical inflow windows library.
    ///
    /// Required when `class_schemes.inflow == Some(Historical)`.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library.
    ///
    /// Required when `class_schemes.inflow == Some(External)`.
    pub external_inflow_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library.
    ///
    /// Required when `class_schemes.load == Some(External)`.
    pub external_load_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library.
    ///
    /// Required when `class_schemes.ncs == Some(External)`.
    pub external_ncs_library: Option<&'a ExternalScenarioLibrary>,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build a [`ClassSampler`] for a single entity class from a scheme and its
/// associated parameters.
///
/// `class_name` is used only for error messages (e.g. `"inflow"`, `"load"`,
/// `"ncs"`).
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when:
/// - `OutOfSample` is requested but `forward_seed` is `None`.
/// - `Historical` is requested but `library` is `None`, or the class is not
///   `"inflow"` (historical is only supported for inflow).
/// - `External` is requested but `library` is `None`.
struct ClassSamplerParams<'a, 'b> {
    class_name: &'b str,
    scheme: SamplingScheme,
    offset: usize,
    len: usize,
    forward_seed: Option<u64>,
    noise_methods: &'b [NoiseMethod],
    tree: Option<crate::tree::opening_tree::OpeningTreeView<'a>>,
    base_seed: u64,
    historical_library: Option<&'a HistoricalScenarioLibrary>,
    external_library: Option<&'a ExternalScenarioLibrary>,
}

fn build_class_sampler<'a>(
    p: ClassSamplerParams<'a, '_>,
) -> Result<ClassSampler<'a>, StochasticError> {
    let ClassSamplerParams {
        class_name,
        scheme,
        offset,
        len,
        forward_seed,
        noise_methods,
        tree,
        base_seed,
        historical_library,
        external_library,
    } = p;
    match scheme {
        SamplingScheme::InSample => {
            let tree = tree.ok_or_else(|| StochasticError::MissingScenarioSource {
                scheme: "in_sample".to_string(),
                reason: "opening tree not available for InSample class sampler".to_string(),
            })?;
            Ok(ClassSampler::InSample {
                tree,
                base_seed,
                offset,
                len,
            })
        }
        SamplingScheme::OutOfSample => {
            let forward_seed =
                forward_seed.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: "out_of_sample".to_string(),
                    reason: "no forward_seed configured; set a seed in stages.json for \
                             out-of-sample forward pass noise generation"
                        .to_string(),
                })?;
            Ok(ClassSampler::OutOfSample {
                forward_seed,
                dim: len,
                noise_methods: noise_methods.into(),
            })
        }
        SamplingScheme::Historical => {
            if class_name != "inflow" {
                return Err(StochasticError::MissingScenarioSource {
                    scheme: format!("historical_{class_name}"),
                    reason: format!(
                        "historical replay is only supported for the inflow class; \
                         requested for class '{class_name}'"
                    ),
                });
            }
            let library =
                historical_library.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: "historical".to_string(),
                    reason: "historical replay scheme selected but no historical library \
                             was loaded; provide historical_windows in the study config"
                        .to_string(),
                })?;
            Ok(ClassSampler::Historical { library })
        }
        SamplingScheme::External => {
            let library =
                external_library.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: format!("external_{class_name}"),
                    reason: format!(
                        "external scenario scheme selected for class '{class_name}' but no \
                     external library was loaded; provide the external scenario file"
                    ),
                })?;
            Ok(ClassSampler::External { library })
        }
    }
}

/// Build a composite [`ForwardSampler`] from a [`ForwardSamplerConfig`].
///
/// Constructs three [`ClassSampler`] instances (one per entity class: inflow,
/// load, NCS) and assembles them into a [`ForwardSampler`] with per-class
/// Cholesky correlation refs where applicable.
///
/// A `None` scheme in `config.class_schemes` is treated as `InSample` (the
/// default).
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when:
/// - `OutOfSample` scheme lacks a configured `forward_seed` in `ctx`.
/// - `Historical` is selected for inflow but `historical_library` is `None`.
/// - `Historical` is selected for load or NCS (not supported).
/// - `External` is selected but the corresponding library is `None`.
pub fn build_forward_sampler(
    config: ForwardSamplerConfig<'_>,
) -> Result<ForwardSampler<'_>, StochasticError> {
    let ForwardSamplerConfig {
        class_schemes,
        ctx,
        stages,
        dims,
        historical_library,
        external_inflow_library,
        external_load_library,
        external_ncs_library,
    } = config;

    let inflow_scheme = class_schemes.inflow.unwrap_or(SamplingScheme::InSample);
    let load_scheme = class_schemes.load.unwrap_or(SamplingScheme::InSample);
    let ncs_scheme = class_schemes.ncs.unwrap_or(SamplingScheme::InSample);

    // Pre-compute shared resources only when needed.
    let forward_seed = ctx.forward_seed();
    let base_seed = ctx.base_seed();

    // Build per-stage noise methods once (shared across all OutOfSample classes).
    let noise_methods: Box<[NoiseMethod]> = stages
        .iter()
        .map(|s| s.scenario_config.noise_method)
        .collect::<Vec<_>>()
        .into_boxed_slice();

    // Entity order sliced per-class for per-class correlation refs.
    let entity_order = ctx.entity_order();
    let inflow_order = &entity_order[..dims.n_hydros];
    let load_order = &entity_order[dims.n_hydros..dims.n_hydros + dims.n_load_buses];
    let ncs_order = &entity_order[dims.n_hydros + dims.n_load_buses..];

    let correlation = ctx.correlation();

    let inflow = build_class_sampler(ClassSamplerParams {
        class_name: "inflow",
        scheme: inflow_scheme,
        offset: 0,
        len: dims.n_hydros,
        forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
        historical_library,
        external_library: external_inflow_library,
    })?;

    let load = build_class_sampler(ClassSamplerParams {
        class_name: "load",
        scheme: load_scheme,
        offset: dims.n_hydros,
        len: dims.n_load_buses,
        forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
        historical_library: None,
        external_library: external_load_library,
    })?;

    let ncs = build_class_sampler(ClassSamplerParams {
        class_name: "ncs",
        scheme: ncs_scheme,
        offset: dims.n_hydros + dims.n_load_buses,
        len: dims.n_ncs,
        forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
        historical_library: None,
        external_library: external_ncs_library,
    })?;

    // Apply per-class correlation only for OutOfSample; pre-correlated sources
    // (InSample, Historical, External) must not have correlation applied again.
    let inflow_correlation = if matches!(inflow_scheme, SamplingScheme::OutOfSample) {
        Some(CorrelationRef {
            decomposed: correlation,
            entity_order: inflow_order,
        })
    } else {
        None
    };
    let load_correlation = if matches!(load_scheme, SamplingScheme::OutOfSample) {
        Some(CorrelationRef {
            decomposed: correlation,
            entity_order: load_order,
        })
    } else {
        None
    };
    let ncs_correlation = if matches!(ncs_scheme, SamplingScheme::OutOfSample) {
        Some(CorrelationRef {
            decomposed: correlation,
            entity_order: ncs_order,
        })
    } else {
        None
    };

    Ok(ForwardSampler::new(
        inflow,
        load,
        ncs,
        dims,
        inflow_correlation,
        load_correlation,
        ncs_correlation,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
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

    use super::{ClassSampler, ForwardNoise, ForwardSampler, SampleRequest, build_forward_sampler};
    use crate::{
        StochasticError,
        context::{ClassSchemes, build_stochastic_context},
        tree::generate::ClassDimensions,
        tree::opening_tree::OpeningTree,
    };

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

    fn make_stage(index: usize, id: i32, bf: usize) -> Stage {
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
                branching_factor: bf,
                noise_method: NoiseMethod::Saa,
            },
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
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    fn build_test_ctx(
        forward_seed: Option<u64>,
    ) -> (crate::context::StochasticContext, Vec<Stage>) {
        let hydros = vec![make_hydro(1)];
        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let inflow_models = vec![make_inflow_model(1, 0), make_inflow_model(1, 1)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages.clone())
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();
        let ctx = build_stochastic_context(
            &system,
            42,
            forward_seed,
            &[],
            &[],
            None,
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();
        (ctx, stages)
    }

    /// Build a uniform opening tree for testing.
    fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
        let total = n_stages * openings * dim;
        let data: Vec<f64> = (0_u32..u32::try_from(total).unwrap())
            .map(f64::from)
            .collect();
        OpeningTree::from_parts(data, vec![openings; n_stages], dim)
    }

    // -----------------------------------------------------------------------
    // Factory helper
    // -----------------------------------------------------------------------

    /// Build a `ForwardSamplerConfig` with all three classes set to `scheme`.
    fn all_classes_config<'a>(
        scheme: SamplingScheme,
        ctx: &'a crate::context::StochasticContext,
        stages: &'a [Stage],
    ) -> super::ForwardSamplerConfig<'a> {
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(scheme),
                load: Some(scheme),
                ncs: Some(scheme),
            },
            ctx,
            stages,
            dims,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        }
    }

    // -----------------------------------------------------------------------
    // Factory tests
    // -----------------------------------------------------------------------

    /// AC: all InSample class schemes returns Ok.
    #[test]
    fn test_build_all_in_sample() {
        let (ctx, stages) = build_test_ctx(None);
        let config = all_classes_config(SamplingScheme::InSample, &ctx, &stages);
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for all-InSample but got: {result:?}"
        );
    }

    /// AC: OutOfSample with no forward_seed returns MissingScenarioSource.
    #[test]
    fn test_build_out_of_sample_missing_seed() {
        let (ctx, stages) = build_test_ctx(None);
        let config = all_classes_config(SamplingScheme::OutOfSample, &ctx, &stages);
        let result = build_forward_sampler(config);
        match result {
            Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
                assert!(
                    scheme.contains("out_of_sample"),
                    "expected scheme to contain 'out_of_sample', got: {scheme}"
                );
            }
            other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
        }
    }

    /// AC: OutOfSample with forward_seed returns Ok with correlation refs set.
    #[test]
    fn test_build_out_of_sample_with_seed() {
        let (ctx, stages) = build_test_ctx(Some(99));
        let config = all_classes_config(SamplingScheme::OutOfSample, &ctx, &stages);
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for OutOfSample with seed but got: {result:?}"
        );
    }

    /// AC: Historical inflow with a library returns Ok.
    #[test]
    fn test_build_historical_with_library() {
        use super::HistoricalScenarioLibrary;
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        // 3 windows, 2 stages, 1 hydro, max_order=1.
        let lib =
            HistoricalScenarioLibrary::new(3, stages.len(), n_hydros, 1, vec![2000, 2001, 2002]);
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: Some(&lib),
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for Historical inflow with library, got: {result:?}"
        );
    }

    /// AC: Historical inflow with no library returns MissingScenarioSource.
    #[test]
    fn test_build_historical_missing_library() {
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        match result {
            Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
                assert!(
                    scheme.contains("historical"),
                    "expected scheme to contain 'historical', got: {scheme}"
                );
            }
            other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
        }
    }

    /// AC: External inflow with a library returns Ok.
    #[test]
    fn test_build_external_with_library() {
        use super::ExternalScenarioLibrary;
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        let lib = ExternalScenarioLibrary::new(stages.len(), 10, n_hydros, "inflow");
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: None,
            external_inflow_library: Some(&lib),
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        assert!(
            result.is_ok(),
            "expected Ok for External inflow with library, got: {result:?}"
        );
    }

    /// AC: Historical load returns MissingScenarioSource with scheme "historical_load".
    #[test]
    fn test_build_historical_load_unsupported() {
        let (ctx, stages) = build_test_ctx(None);
        let n_hydros = ctx.dim() - ctx.n_load_buses() - ctx.n_stochastic_ncs();
        let dims = ClassDimensions {
            n_hydros,
            n_load_buses: ctx.n_load_buses(),
            n_ncs: ctx.n_stochastic_ncs(),
        };
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::Historical),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            dims,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        };
        let result = build_forward_sampler(config);
        match result {
            Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
                assert_eq!(
                    scheme, "historical_load",
                    "expected scheme 'historical_load', got: {scheme}"
                );
            }
            other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // ForwardNoise newtype tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_noise_as_slice_newtype() {
        let data = [1.0f64, 2.0, 3.0];
        let noise = ForwardNoise(&data);
        assert_eq!(noise.as_slice(), &data);
    }

    #[test]
    fn test_forward_noise_as_slice() {
        let buf = [4.0f64, 5.0];
        let noise = ForwardNoise(&buf);
        assert_eq!(noise.as_slice(), &buf);
    }

    // -----------------------------------------------------------------------
    // Composite InSample sample() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_in_sample_sample_returns_noise() {
        let (ctx, stages) = build_test_ctx(None);
        let sampler =
            build_forward_sampler(all_classes_config(SamplingScheme::InSample, &ctx, &stages))
                .unwrap();
        let dim = ctx.dim();

        let mut noise_buf = vec![0.0f64; dim];
        let mut perm_scratch = vec![0usize; dim];

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 5,
        });
        let noise = result.expect("expected Ok from InSample sample()");
        assert_eq!(
            noise.as_slice().len(),
            dim,
            "noise slice length {} != dim {dim}",
            noise.as_slice().len()
        );
    }

    #[test]
    fn test_in_sample_sample_is_deterministic() {
        let (ctx, stages) = build_test_ctx(None);
        let sampler =
            build_forward_sampler(all_classes_config(SamplingScheme::InSample, &ctx, &stages))
                .unwrap();
        let dim = ctx.dim();

        let mut buf_a = vec![0.0f64; dim];
        let mut buf_b = vec![0.0f64; dim];
        let mut perm_a = vec![0usize; dim];
        let mut perm_b = vec![0usize; dim];

        let a = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_a,
                perm_scratch: &mut perm_a,
                total_scenarios: 5,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_b,
                perm_scratch: &mut perm_b,
                total_scenarios: 5,
            })
            .unwrap();

        assert_eq!(a.as_slice(), b.as_slice());
    }

    // -----------------------------------------------------------------------
    // New composite tests required by ticket-027
    // -----------------------------------------------------------------------

    /// AC: composite with three `InSample` class samplers fills each segment
    /// from the correct tree region.
    #[test]
    fn test_composite_in_sample_fills_correct_segments() {
        // Tree: 1 stage, 3 openings, dim=5 (2 hydros + 2 load + 1 ncs).
        // Data layout: values 0..14 (sequential).
        let tree = uniform_tree(1, 3, 5);
        let view = tree.view();
        let dims = ClassDimensions {
            n_hydros: 2,
            n_load_buses: 2,
            n_ncs: 1,
        };

        let sampler = ForwardSampler::new(
            ClassSampler::InSample {
                tree: view,
                base_seed: 42,
                offset: 0,
                len: 2,
            },
            ClassSampler::InSample {
                tree: tree.view(),
                base_seed: 42,
                offset: 2,
                len: 2,
            },
            ClassSampler::InSample {
                tree: tree.view(),
                base_seed: 42,
                offset: 4,
                len: 1,
            },
            dims,
            None,
            None,
            None,
        );

        let mut noise_buf = vec![0.0f64; 5];
        let mut perm_scratch = vec![0usize; 10];

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 3,
        });

        let noise = result.expect("expected Ok from composite InSample sample()");
        assert_eq!(
            noise.as_slice().len(),
            5,
            "total noise length must equal total_dim"
        );

        // Verify against direct tree access.
        let (_, full_slice) =
            crate::sampling::insample::sample_forward(&tree.view(), 42, 0, 0, 0, 0);
        assert_eq!(
            noise.as_slice(),
            full_slice,
            "composite InSample must reproduce the full tree slice"
        );
    }

    /// AC: composite with `OutOfSample` inflow and identity correlation
    /// applies correlation to the inflow segment only.
    #[test]
    fn test_composite_out_of_sample_applies_per_class_correlation() {
        // Build a 1-hydro system so we can get a real DecomposedCorrelation.
        let (ctx, stages) = build_test_ctx(Some(99));
        let sampler = build_forward_sampler(all_classes_config(
            SamplingScheme::OutOfSample,
            &ctx,
            &stages,
        ))
        .unwrap();
        let dim = ctx.dim();

        let mut noise_buf = vec![0.0f64; dim];
        let mut perm_scratch = vec![0usize; 5];

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            perm_scratch: &mut perm_scratch,
            total_scenarios: 5,
        });

        let noise = result.expect("expected Ok from OutOfSample sample()");
        // All values must be finite.
        for (i, &v) in noise.as_slice().iter().enumerate() {
            assert!(v.is_finite(), "element[{i}] is not finite: {v}");
        }
        assert_eq!(noise.as_slice().len(), dim);
        let _ = ctx; // suppress unused warning
        let _ = stages;
    }

    /// AC: same inputs produce identical output (determinism).
    #[test]
    fn test_composite_sample_deterministic() {
        let (ctx, stages) = build_test_ctx(Some(77));
        let sampler = build_forward_sampler(all_classes_config(
            SamplingScheme::OutOfSample,
            &ctx,
            &stages,
        ))
        .unwrap();
        let dim = ctx.dim();

        let mut buf_a = vec![0.0f64; dim];
        let mut buf_b = vec![0.0f64; dim];
        let mut perm_a = vec![0usize; 5];
        let mut perm_b = vec![0usize; 5];

        let a = sampler
            .sample(SampleRequest {
                iteration: 3,
                scenario: 7,
                stage: 1,
                stage_idx: 1,
                noise_buf: &mut buf_a,
                perm_scratch: &mut perm_a,
                total_scenarios: 5,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 3,
                scenario: 7,
                stage: 1,
                stage_idx: 1,
                noise_buf: &mut buf_b,
                perm_scratch: &mut perm_b,
                total_scenarios: 5,
            })
            .unwrap();

        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "composite sample() must be deterministic for same inputs"
        );
    }
}
