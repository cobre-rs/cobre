//! Forward-pass scenario sampler construction.

use cobre_stochastic::context::ClassSchemes;
use cobre_stochastic::{ForwardSampler, ForwardSamplerConfig, build_forward_sampler};

use crate::context::TrainingContext;
use crate::error::SddpError;

/// Build a [`ForwardSampler`] from the sampler-related fields of a
/// [`TrainingContext`], so callers can construct it once and reuse it across all
/// training iterations without repeated heap allocation.
///
/// # Errors
///
/// Propagates any error from [`build_forward_sampler`], such as a missing
/// `OutOfSample` seed or an incompatible library shape.
pub fn build_sampler_from_ctx<'a>(
    ctx: &'a TrainingContext<'a>,
) -> Result<ForwardSampler<'a>, SddpError> {
    let stochastic = ctx.stochastic;
    build_forward_sampler(ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(ctx.inflow_scheme),
            load: Some(ctx.load_scheme),
            ncs: Some(ctx.ncs_scheme),
        },
        ctx: stochastic,
        forward_seed: stochastic.forward_seed(),
        stages: ctx.stages,
        historical_library: ctx.historical_library,
        external_inflow_library: ctx.external_inflow_library,
        external_load_library: ctx.external_load_library,
        external_ncs_library: ctx.external_ncs_library,
    })
    .map_err(SddpError::Stochastic)
}
