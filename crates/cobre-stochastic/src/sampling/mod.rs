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
//! [`ForwardSampler`] is an enum that unifies all supported sampling strategies
//! under a single `sample()` dispatch method. Use [`build_forward_sampler`] to
//! construct the appropriate variant from a [`SamplingScheme`] and a
//! [`StochasticContext`].
//!
//! ```
//! use cobre_core::scenario::SamplingScheme;
//! use cobre_stochastic::sampling::{ForwardSampler, build_forward_sampler};
//! ```

pub(crate) mod external;
pub(crate) mod historical;
pub mod insample;
pub(crate) mod out_of_sample;

use cobre_core::{EntityId, Stage, scenario::SamplingScheme, temporal::NoiseMethod};

use crate::{
    StochasticError, context::StochasticContext, correlation::resolve::DecomposedCorrelation,
    tree::opening_tree::OpeningTreeView,
};

// ---------------------------------------------------------------------------
// ForwardNoise
// ---------------------------------------------------------------------------

/// The noise payload returned by [`ForwardSampler::sample`].
///
/// The two lifetime parameters allow variants to borrow from different sources:
/// - `'a` — lifetime of the opening tree (for `TreeSlice`)
/// - `'b` — lifetime of the caller-supplied `noise_buf` (for `FreshNoise`)
#[derive(Debug)]
pub enum ForwardNoise<'a, 'b> {
    /// Noise drawn from the pre-generated opening tree.
    ///
    /// Borrows a slice of the opening tree owned by [`StochasticContext`].
    TreeSlice(&'a [f64]),
    /// Freshly generated noise written into the caller-supplied buffer.
    ///
    /// Borrows the portion of `noise_buf` that was populated by the sampler.
    FreshNoise(&'b [f64]),
}

impl ForwardNoise<'_, '_> {
    /// Return the underlying noise slice regardless of which variant is active.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        match self {
            ForwardNoise::TreeSlice(s) | ForwardNoise::FreshNoise(s) => s,
        }
    }
}

// ---------------------------------------------------------------------------
// ForwardSampler
// ---------------------------------------------------------------------------

/// Unified forward-pass sampler, dispatching over all supported sampling schemes.
///
/// Constructed once per run via [`build_forward_sampler`] and reused
/// across all `(iteration, scenario, stage)` calls without per-call allocation.
pub enum ForwardSampler<'a> {
    /// In-sample scheme: forward pass uses the pre-generated opening tree.
    InSample {
        /// View into the pre-generated opening tree.
        tree: OpeningTreeView<'a>,
        /// Base seed for deterministic scenario selection.
        base_seed: u64,
    },
    /// Out-of-sample scheme: forward pass generates fresh noise on-the-fly.
    OutOfSample {
        /// Seed for the forward-pass noise generator.
        forward_seed: u64,
        /// Noise dimension (hydros + load buses + NCS entities).
        dim: usize,
        /// Per-stage noise generation method.
        noise_methods: Box<[NoiseMethod]>,
        /// Canonical entity ordering for correlation application.
        entity_order: &'a [EntityId],
        /// Pre-decomposed spatial correlation for noise correlation.
        correlation: &'a DecomposedCorrelation,
    },
    /// Historical replay scheme (stub — not yet implemented).
    Historical,
    /// External scenario file scheme (stub — not yet implemented).
    External,
}

impl std::fmt::Debug for ForwardSampler<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardSampler::InSample { base_seed, .. } => f
                .debug_struct("ForwardSampler::InSample")
                .field("base_seed", base_seed)
                .finish_non_exhaustive(),
            ForwardSampler::OutOfSample {
                forward_seed,
                dim,
                noise_methods,
                ..
            } => f
                .debug_struct("ForwardSampler::OutOfSample")
                .field("forward_seed", forward_seed)
                .field("dim", dim)
                .field("noise_methods", noise_methods)
                .finish_non_exhaustive(),
            ForwardSampler::Historical => write!(f, "ForwardSampler::Historical"),
            ForwardSampler::External => write!(f, "ForwardSampler::External"),
        }
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
    /// Caller-owned buffer for fresh noise output (`OutOfSample` path).
    pub noise_buf: &'b mut [f64],
    /// Caller-owned scratch for LHS permutation generation.
    pub perm_scratch: &'b mut [usize],
    /// Total scenario count across all ranks (for LHS stratification).
    pub total_scenarios: u32,
}

impl<'a> ForwardSampler<'a> {
    /// Draw noise for a single `(iteration, scenario, stage)` triple.
    ///
    /// When the `OutOfSample` variant is active and a stage uses
    /// `NoiseMethod::Selective`, the method falls back to SAA and emits a
    /// `tracing::warn!`. This is a known limitation; `Selective` requires
    /// clustering infrastructure that is not available in the forward pass.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::DimensionExceedsCapacity`] — when `OutOfSample`
    ///   uses QmcSobol and `dim > MAX_SOBOL_DIM`.
    /// - [`StochasticError::UnsupportedSamplingScheme`] — for unimplemented
    ///   `Historical` and `External` schemes.
    /// - [`StochasticError::InsufficientData`] — when `stage_idx` is out of
    ///   bounds for the per-stage noise methods.
    //
    // Passing SampleRequest by value is intentional: we need owned access
    // to write into req.noise_buf and return a slice borrowing from it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn sample<'b>(
        &self,
        req: SampleRequest<'b>,
    ) -> Result<ForwardNoise<'a, 'b>, StochasticError> {
        match self {
            ForwardSampler::InSample { tree, base_seed } => {
                let (_idx, slice) = insample::sample_forward(
                    tree,
                    *base_seed,
                    req.iteration,
                    req.scenario,
                    req.stage,
                    req.stage_idx,
                );
                Ok(ForwardNoise::TreeSlice(slice))
            }
            ForwardSampler::OutOfSample {
                forward_seed,
                dim,
                noise_methods,
                entity_order,
                correlation,
            } => {
                let noise_method = noise_methods.get(req.stage_idx).copied().ok_or_else(|| {
                    StochasticError::InsufficientData {
                        context: format!(
                            "stage_idx {} out of bounds for {} noise methods",
                            req.stage_idx,
                            noise_methods.len(),
                        ),
                    }
                })?;
                let spec = out_of_sample::FreshNoiseSpec {
                    forward_seed: *forward_seed,
                    noise_method,
                    iteration: req.iteration,
                    scenario: req.scenario,
                    stage_id: req.stage,
                    dim: *dim,
                    total_scenarios: req.total_scenarios,
                };
                out_of_sample::sample_fresh(
                    spec,
                    req.noise_buf,
                    req.perm_scratch,
                    correlation,
                    entity_order,
                )?;
                Ok(ForwardNoise::FreshNoise(&req.noise_buf[..*dim]))
            }
            ForwardSampler::Historical => Err(StochasticError::UnsupportedSamplingScheme {
                scheme: "historical".to_string(),
                reason: "historical replay is not yet implemented".to_string(),
            }),
            ForwardSampler::External => Err(StochasticError::UnsupportedSamplingScheme {
                scheme: "external".to_string(),
                reason: "external scenario source is not yet implemented".to_string(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build the appropriate [`ForwardSampler`] variant for the given `scheme`.
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when:
/// - `OutOfSample` scheme lacks a configured `forward_seed`
/// - `Historical` or `External` schemes are not yet supported
pub fn build_forward_sampler<'a>(
    scheme: SamplingScheme,
    ctx: &'a StochasticContext,
    stages: &'a [Stage],
) -> Result<ForwardSampler<'a>, StochasticError> {
    match scheme {
        SamplingScheme::InSample => Ok(ForwardSampler::InSample {
            tree: ctx.tree_view(),
            base_seed: ctx.base_seed(),
        }),
        SamplingScheme::OutOfSample => {
            let forward_seed =
                ctx.forward_seed()
                    .ok_or_else(|| StochasticError::MissingScenarioSource {
                        scheme: "out_of_sample".to_string(),
                        reason: "no forward_seed configured; set a seed in stages.json for \
                                 out-of-sample forward pass noise generation"
                            .to_string(),
                    })?;
            let noise_methods: Box<[NoiseMethod]> = stages
                .iter()
                .map(|s| s.scenario_config.noise_method)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            Ok(ForwardSampler::OutOfSample {
                forward_seed,
                dim: ctx.dim(),
                noise_methods,
                entity_order: ctx.entity_order(),
                correlation: ctx.correlation(),
            })
        }
        SamplingScheme::Historical => Err(StochasticError::MissingScenarioSource {
            scheme: "historical".to_string(),
            reason: "historical replay scenario source is not yet supported".to_string(),
        }),
        SamplingScheme::External => Err(StochasticError::MissingScenarioSource {
            scheme: "external".to_string(),
            reason: "external scenario source is not yet supported".to_string(),
        }),
    }
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

    use super::{ForwardNoise, ForwardSampler, SampleRequest, build_forward_sampler};
    use crate::{StochasticError, context::build_stochastic_context};

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
        let ctx = build_stochastic_context(&system, 42, forward_seed, &[], &[], None).unwrap();
        (ctx, stages)
    }

    #[test]
    fn test_build_in_sample_succeeds() {
        let (ctx, stages) = build_test_ctx(None);
        let result = build_forward_sampler(SamplingScheme::InSample, &ctx, &stages);
        assert!(
            result.is_ok(),
            "expected Ok for InSample but got: {:?}",
            result
        );
        assert!(matches!(result.unwrap(), ForwardSampler::InSample { .. }));
    }

    #[test]
    fn test_build_out_of_sample_missing_seed() {
        let (ctx, stages) = build_test_ctx(None);
        let result = build_forward_sampler(SamplingScheme::OutOfSample, &ctx, &stages);
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

    #[test]
    fn test_build_out_of_sample_with_seed_succeeds() {
        let (ctx, stages) = build_test_ctx(Some(99));
        let result = build_forward_sampler(SamplingScheme::OutOfSample, &ctx, &stages);
        assert!(
            result.is_ok(),
            "expected Ok for OutOfSample with seed but got: {:?}",
            result
        );
        assert!(matches!(
            result.unwrap(),
            ForwardSampler::OutOfSample { .. }
        ));
    }

    #[test]
    fn test_build_historical_returns_error() {
        let (ctx, stages) = build_test_ctx(None);
        let result = build_forward_sampler(SamplingScheme::Historical, &ctx, &stages);
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

    #[test]
    fn test_build_external_returns_error() {
        let (ctx, stages) = build_test_ctx(None);
        let result = build_forward_sampler(SamplingScheme::External, &ctx, &stages);
        match result {
            Err(StochasticError::MissingScenarioSource { scheme, .. }) => {
                assert!(
                    scheme.contains("external"),
                    "expected scheme to contain 'external', got: {scheme}"
                );
            }
            other => panic!("expected Err(MissingScenarioSource), got: {other:?}"),
        }
    }

    #[test]
    fn test_in_sample_sample_returns_tree_slice() {
        let (ctx, stages) = build_test_ctx(None);
        let sampler = build_forward_sampler(SamplingScheme::InSample, &ctx, &stages).unwrap();
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
        match result {
            Ok(ForwardNoise::TreeSlice(slice)) => {
                assert_eq!(
                    slice.len(),
                    dim,
                    "tree slice length {len} != dim {dim}",
                    len = slice.len()
                );
            }
            other => panic!("expected Ok(ForwardNoise::TreeSlice), got: {other:?}"),
        }
    }

    #[test]
    fn test_forward_noise_as_slice() {
        // TreeSlice variant
        let data = [1.0f64, 2.0, 3.0];
        let tree_noise: ForwardNoise<'_, '_> = ForwardNoise::TreeSlice(&data);
        assert_eq!(tree_noise.as_slice(), &data);

        // FreshNoise variant
        let buf = [4.0f64, 5.0];
        let fresh_noise: ForwardNoise<'_, '_> = ForwardNoise::FreshNoise(&buf);
        assert_eq!(fresh_noise.as_slice(), &buf);
    }

    #[test]
    fn test_in_sample_sample_is_deterministic() {
        let (ctx, stages) = build_test_ctx(None);
        let sampler = build_forward_sampler(SamplingScheme::InSample, &ctx, &stages).unwrap();
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
}
