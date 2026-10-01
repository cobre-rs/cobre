//! Scenario sampling schemes — strategies that select which scenarios are
//! simulated each iteration.
//!
//! [`ForwardSampler`] is the composite entry point: it holds three
//! [`ClassSampler`] instances (one per entity class) and applies per-class
//! spectral correlation only for `OutOfSample`. Build one with
//! [`build_forward_sampler`].
//!
//! ```
//! use cobre_core::scenario::SamplingScheme;
//! use cobre_stochastic::sampling::{ForwardSampler, build_forward_sampler};
//! ```

use crate::context::ClassSchemes;
use crate::season_cast::{StageCalendar, occurrence_year};

use chrono::{Datelike, Months, NaiveDate};
use std::fmt;
pub mod class_sampler;
mod eta_inversion;
pub mod external;
pub mod historical;
pub mod insample;
pub mod tables;
pub mod window;

pub use class_sampler::{ClassSampleRequest, ClassSampler, select_transition_child};
pub use external::{
    ExternalScenarioLibrary, derive_external_sample_moments, pad_library_to_uniform,
    standardize_external_inflow, standardize_external_load, standardize_external_ncs,
    validate_external_library,
};
pub use historical::{
    HistoricalScenarioLibrary, standardize_historical_windows, validate_historical_library,
};
pub use tables::{ClassNoiseTables, ForwardNoiseTables, NoiseTable};
pub use window::discover_historical_windows;
pub(crate) mod out_of_sample;

use cobre_core::{
    scenario::SamplingScheme,
    temporal::{NoiseMethod, SeasonMap, Stage},
};

use crate::noise::seed::derive_class_forward_seed;
use crate::{
    OpeningTreeView, StochasticError,
    context::StochasticContext,
    correlation::resolve::{DecomposedCorrelation, EntityClass},
    tree::generate::ClassDimensions,
};

// ---------------------------------------------------------------------------
// ForwardNoise
// ---------------------------------------------------------------------------

/// Noise payload returned by [`ForwardSampler::sample`].
#[derive(Debug)]
pub struct ForwardNoise<'b>(&'b [f64]);

impl<'b> ForwardNoise<'b> {
    /// Create a new `ForwardNoise` from a borrowed slice.
    #[must_use]
    pub fn new(data: &'b [f64]) -> Self {
        Self(data)
    }

    /// Return the underlying noise slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.0
    }
}

// ---------------------------------------------------------------------------
// ForwardSampler
// ---------------------------------------------------------------------------

/// Composite forward-pass sampler holding one [`ClassSampler`] per entity class.
///
/// Built once per run via [`build_forward_sampler`] and reused across all
/// `(iteration, scenario, stage)` calls without per-call allocation. The
/// `*_correlation` fields are `Some` only for `OutOfSample`; pre-correlated
/// sources leave them `None`.
pub struct ForwardSampler<'a> {
    /// Class sampler for inflow (hydro) entities.
    inflow: ClassSampler<'a>,
    /// Class sampler for stochastic load bus entities.
    load: ClassSampler<'a>,
    /// Class sampler for NCS entities.
    ncs: ClassSampler<'a>,
    /// Per-class entity counts that define the buffer split.
    dims: ClassDimensions,
    inflow_correlation: Option<&'a DecomposedCorrelation>,
    load_correlation: Option<&'a DecomposedCorrelation>,
    ncs_correlation: Option<&'a DecomposedCorrelation>,
}

impl<'a> ForwardSampler<'a> {
    pub(crate) fn new(
        inflow: ClassSampler<'a>,
        load: ClassSampler<'a>,
        ncs: ClassSampler<'a>,
        dims: ClassDimensions,
        inflow_correlation: Option<&'a DecomposedCorrelation>,
        load_correlation: Option<&'a DecomposedCorrelation>,
        ncs_correlation: Option<&'a DecomposedCorrelation>,
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

impl fmt::Debug for ForwardSampler<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardSampler")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// SampleRequest
// ---------------------------------------------------------------------------

/// Per-call arguments for [`ForwardSampler::sample`].
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
    /// Caller-owned gather/correlate scratch for wide correlation groups.
    pub corr_scratch: &'b mut [f64],
    /// Per-iteration scenario-invariant tables the driver owns, rebuilt once
    /// via [`ForwardSampler::rebuild_noise_tables`].
    pub tables: &'b ForwardNoiseTables,
    /// Total scenario count across all ranks (for LHS stratification).
    pub total_scenarios: u32,
    /// Seed-derivation identifier: stages sharing a `(season_id, year)` bucket
    /// share a `noise_group_id` so their noise draws are identical.
    pub noise_group_id: u32,
    /// Sampled node's Ω sub-range — see [`ClassSampleRequest::node_opening_offset`].
    pub node_opening_offset: usize,
    /// See [`ClassSampleRequest::node_opening_offset`].
    pub node_opening_len: usize,
    /// See [`ClassSampleRequest::pinned_scenario`].
    pub pinned_scenario: Option<usize>,
}

impl ForwardSampler<'_> {
    /// Per-class initial-state hook before the stage-0 solve; currently a no-op
    /// for every class (see [`ClassSampler::apply_initial_state`]) so that
    /// forward, backward, and lower-bound paths consume the same `x_0`.
    ///
    /// `lag_offset` is an absolute index into `state` computed by the caller from
    /// the `StageIndexer`.
    pub fn apply_initial_state(
        &self,
        req: &ClassSampleRequest,
        state: &mut [f64],
        lag_offset: usize,
    ) {
        self.inflow.apply_initial_state(req, state, lag_offset);
        self.load.apply_initial_state(req, state, 0);
        self.ncs.apply_initial_state(req, state, 0);
    }

    /// Draw noise for a single `(iteration, scenario, stage)` triple into the
    /// per-class segments `[hydros | load_buses | ncs]` of `req.noise_buf`.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InsufficientData`] — when `stage_idx` is out of
    ///   bounds for any per-stage noise methods.
    //
    // Passing SampleRequest by value is intentional: we need owned access
    // to write into req.noise_buf and return a slice borrowing from it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn sample<'b>(&self, req: SampleRequest<'b>) -> Result<ForwardNoise<'b>, StochasticError> {
        let total_dim = self.dims.total();

        let (inflow_buf, load_buf, ncs_buf) = self.dims.split_segments_mut(req.noise_buf);

        let class_req = ClassSampleRequest {
            iteration: req.iteration,
            scenario: req.scenario,
            stage: req.stage,
            stage_idx: req.stage_idx,
            total_scenarios: req.total_scenarios,
            noise_group_id: req.noise_group_id,
            node_opening_offset: req.node_opening_offset,
            node_opening_len: req.node_opening_len,
            pinned_scenario: req.pinned_scenario,
        };

        self.inflow
            .fill(&class_req, req.tables.inflow(), inflow_buf)?;
        self.load.fill(&class_req, req.tables.load(), load_buf)?;
        self.ncs.fill(&class_req, req.tables.ncs(), ncs_buf)?;

        // Applied only per class whose correlation ref is set — a pre-correlated
        // source would double-correlate otherwise. All three fields hold the same
        // reference when present, so the profile lookup happens once per draw.
        if let Some(correlation) = self
            .inflow_correlation
            .or(self.load_correlation)
            .or(self.ncs_correlation)
        {
            #[allow(clippy::cast_possible_wrap)]
            let groups = correlation.groups_for_stage(req.stage as i32);
            if self.inflow_correlation.is_some() {
                DecomposedCorrelation::apply_groups_for_class(
                    groups,
                    EntityClass::Inflow,
                    inflow_buf,
                    req.corr_scratch,
                );
            }
            if self.load_correlation.is_some() {
                DecomposedCorrelation::apply_groups_for_class(
                    groups,
                    EntityClass::Load,
                    load_buf,
                    req.corr_scratch,
                );
            }
            if self.ncs_correlation.is_some() {
                DecomposedCorrelation::apply_groups_for_class(
                    groups,
                    EntityClass::Ncs,
                    ncs_buf,
                    req.corr_scratch,
                );
            }
        }

        Ok(ForwardNoise::new(&req.noise_buf[..total_dim]))
    }

    /// Rebuild every scenario-invariant table in `out` for one training
    /// iteration's `(iteration, total_scenarios, noise_group_ids)`, reusing
    /// its buffer capacity across iterations. A class not sampled out of
    /// sample is cleared, and its `table_at` calls return `None`.
    ///
    /// # Errors
    ///
    /// Returns [`StochasticError::DimensionExceedsCapacity`] when an
    /// `OutOfSample` class uses `QmcSobol` with `dim > MAX_SOBOL_DIM`.
    pub fn rebuild_noise_tables(
        &self,
        iteration: u32,
        total_scenarios: u32,
        noise_group_ids: &[u32],
        out: &mut ForwardNoiseTables,
    ) -> Result<(), StochasticError> {
        rebuild_class_tables(
            &self.inflow,
            iteration,
            total_scenarios,
            noise_group_ids,
            &mut out.inflow,
        )?;
        rebuild_class_tables(
            &self.load,
            iteration,
            total_scenarios,
            noise_group_ids,
            &mut out.load,
        )?;
        rebuild_class_tables(
            &self.ncs,
            iteration,
            total_scenarios,
            noise_group_ids,
            &mut out.ncs,
        )?;
        Ok(())
    }
}

/// Refill one class's noise tables from its [`ClassSampler`], clearing them
/// when the class is not sampled out of sample.
///
/// # Errors
///
/// Propagates [`ClassNoiseTables::refill`]'s error.
fn rebuild_class_tables(
    sampler: &ClassSampler<'_>,
    iteration: u32,
    total_scenarios: u32,
    noise_group_ids: &[u32],
    out: &mut ClassNoiseTables,
) -> Result<(), StochasticError> {
    if let ClassSampler::OutOfSample {
        forward_seed,
        dim,
        noise_methods,
    } = sampler
    {
        out.refill(
            *forward_seed,
            *dim,
            iteration,
            total_scenarios,
            noise_group_ids,
            noise_methods,
        )
    } else {
        out.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ForwardSamplerConfig
// ---------------------------------------------------------------------------

/// All parameters needed by [`build_forward_sampler`].
#[derive(Debug, Clone, Copy)]
pub struct ForwardSamplerConfig<'a> {
    /// Per-class sampling scheme selections.
    pub class_schemes: ClassSchemes,
    /// Stochastic context providing tree, seeds, correlation, and entity order.
    pub ctx: &'a StochasticContext,
    /// Study stages in index order; required by `OutOfSample` to read per-stage
    /// noise methods.
    pub stages: &'a [Stage],
    /// Pre-standardized historical inflow windows library, required when
    /// `class_schemes.inflow == Some(Historical)`.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library, required when
    /// `class_schemes.inflow == Some(External)`.
    pub external_inflow_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library, required when
    /// `class_schemes.load == Some(External)`.
    pub external_load_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library, required when
    /// `class_schemes.ncs == Some(External)`.
    pub external_ncs_library: Option<&'a ExternalScenarioLibrary>,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Resolved scenario source for the inflow class — the only class allowed to
/// replay historical windows.
enum InflowSource<'a> {
    InSample,
    OutOfSample,
    Historical(&'a HistoricalScenarioLibrary),
    External(&'a ExternalScenarioLibrary),
}

/// Resolved scenario source for the load and NCS classes. Historical replay
/// is inflow-only, so this set carries no `Historical` variant.
enum ClassSource<'a> {
    InSample,
    OutOfSample,
    External(&'a ExternalScenarioLibrary),
}

/// Inputs to [`build_class_sampler`] for one entity class.
struct ClassSamplerParams<'a, 'b> {
    source: ClassSource<'a>,
    offset: usize,
    len: usize,
    forward_seed: Option<u64>,
    noise_methods: &'b [NoiseMethod],
    tree: Option<OpeningTreeView<'a>>,
    base_seed: u64,
}

/// Build a [`ClassSampler`] for one entity class from its resolved
/// [`ClassSource`].
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when `InSample` lacks
/// an opening tree or `OutOfSample` lacks a `forward_seed`.
fn build_class_sampler<'a>(
    p: ClassSamplerParams<'a, '_>,
) -> Result<ClassSampler<'a>, StochasticError> {
    let ClassSamplerParams {
        source,
        offset,
        len,
        forward_seed,
        noise_methods,
        tree,
        base_seed,
    } = p;
    match source {
        ClassSource::InSample => {
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
        ClassSource::OutOfSample => {
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
        ClassSource::External(library) => Ok(ClassSampler::External { library }),
    }
}

/// Resolve the inflow class's sampling scheme against the historical and
/// external inflow libraries into an [`InflowSource`].
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when `Historical` or
/// `External` is selected but its library was not loaded.
fn resolve_inflow_source<'a>(
    scheme: SamplingScheme,
    historical_library: Option<&'a HistoricalScenarioLibrary>,
    external_library: Option<&'a ExternalScenarioLibrary>,
) -> Result<InflowSource<'a>, StochasticError> {
    match scheme {
        SamplingScheme::InSample => Ok(InflowSource::InSample),
        SamplingScheme::OutOfSample => Ok(InflowSource::OutOfSample),
        SamplingScheme::Historical => {
            let library =
                historical_library.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: "historical".to_string(),
                    reason: "historical replay scheme selected but no historical library \
                             was loaded; provide historical_windows in the study config"
                        .to_string(),
                })?;
            Ok(InflowSource::Historical(library))
        }
        SamplingScheme::External => {
            let library =
                external_library.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: format!("external_{}", EntityClass::Inflow.as_str()),
                    reason: format!(
                        "external scenario scheme selected for class '{}' but no \
                     external library was loaded; provide the external scenario file",
                        EntityClass::Inflow.as_str()
                    ),
                })?;
            Ok(InflowSource::External(library))
        }
    }
}

/// Resolve a non-inflow class's sampling scheme against its external library
/// into a [`ClassSource`]. `ClassSource` carries no `Historical` variant, so
/// selecting `Historical` here always errors.
///
/// # Errors
///
/// Returns [`StochasticError::MissingScenarioSource`] when `Historical` is
/// selected (unsupported outside inflow) or `External` names no loaded
/// library.
fn resolve_class_source(
    scheme: SamplingScheme,
    class: EntityClass,
    external_library: Option<&ExternalScenarioLibrary>,
) -> Result<ClassSource<'_>, StochasticError> {
    match scheme {
        SamplingScheme::InSample => Ok(ClassSource::InSample),
        SamplingScheme::OutOfSample => Ok(ClassSource::OutOfSample),
        SamplingScheme::Historical => Err(StochasticError::MissingScenarioSource {
            scheme: format!("historical_{}", class.as_str()),
            reason: format!(
                "historical replay is only supported for the inflow class; \
                 requested for class '{}'",
                class.as_str()
            ),
        }),
        SamplingScheme::External => {
            let library =
                external_library.ok_or_else(|| StochasticError::MissingScenarioSource {
                    scheme: format!("external_{}", class.as_str()),
                    reason: format!(
                        "external scenario scheme selected for class '{}' but no \
                     external library was loaded; provide the external scenario file",
                        class.as_str()
                    ),
                })?;
            Ok(ClassSource::External(library))
        }
    }
}

/// Emits at most one `tracing::warn!` naming `class` and every stage id whose
/// `noise_method` is `Selective` or `HistoricalResiduals` — methods
/// `fill_uncorrelated` does not implement and instead falls back to SAA.
/// Skips a class whose resolved `scheme` never reaches `fill_uncorrelated`.
fn warn_unsupported_forward_noise_methods(
    class: EntityClass,
    scheme: SamplingScheme,
    stages: &[Stage],
    noise_methods: &[NoiseMethod],
) {
    if scheme != SamplingScheme::OutOfSample {
        return;
    }
    let unsupported: Vec<(i32, NoiseMethod)> = stages
        .iter()
        .zip(noise_methods)
        .filter_map(|(stage, &method)| {
            matches!(
                method,
                NoiseMethod::Selective | NoiseMethod::HistoricalResiduals
            )
            .then_some((stage.id, method))
        })
        .collect();
    if unsupported.is_empty() {
        return;
    }

    let stage_ids: Vec<i32> = unsupported.iter().map(|&(id, _)| id).collect();
    let mut methods: Vec<&'static str> = Vec::new();
    if unsupported
        .iter()
        .any(|&(_, method)| method == NoiseMethod::Selective)
    {
        methods.push("selective");
    }
    if unsupported
        .iter()
        .any(|&(_, method)| method == NoiseMethod::HistoricalResiduals)
    {
        methods.push("historical_residuals");
    }
    let methods = methods.join(" or ");
    let class_str = class.as_str();
    let count = stage_ids.len();

    tracing::warn!(
        "class '{class_str}' has {count} out-of-sample forward stage(s) selecting \
         {methods} noise, not implemented in the forward pass; falling back to the \
         sample-average method for stage id(s) {stage_ids:?}"
    );
}

/// Build a composite [`ForwardSampler`] from a [`ForwardSamplerConfig`].
///
/// A `None` scheme in `config.class_schemes` defaults to `InSample`.
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
        historical_library,
        external_inflow_library,
        external_load_library,
        external_ncs_library,
    } = config;
    let dims = ctx.class_dimensions();

    let inflow_scheme = class_schemes.inflow.unwrap_or(SamplingScheme::InSample);
    let load_scheme = class_schemes.load.unwrap_or(SamplingScheme::InSample);
    let ncs_scheme = class_schemes.ncs.unwrap_or(SamplingScheme::InSample);

    // Inflow keeps the root seed: deriving it too would change every shipped
    // inflow-only deck.
    let inflow_forward_seed = ctx.forward_seed();
    let load_forward_seed =
        inflow_forward_seed.map(|s| derive_class_forward_seed(s, EntityClass::Load));
    let ncs_forward_seed =
        inflow_forward_seed.map(|s| derive_class_forward_seed(s, EntityClass::Ncs));
    let base_seed = ctx.base_seed();

    let noise_methods: Box<[NoiseMethod]> = stages
        .iter()
        .map(|s| s.scenario_config.noise_method)
        .collect();

    let correlation = ctx.correlation();

    warn_unsupported_forward_noise_methods(
        EntityClass::Inflow,
        inflow_scheme,
        stages,
        &noise_methods,
    );
    let hydro_range = dims.hydro_range();
    let build_inflow = |source| {
        build_class_sampler(ClassSamplerParams {
            source,
            offset: hydro_range.start,
            len: hydro_range.len(),
            forward_seed: inflow_forward_seed,
            noise_methods: &noise_methods,
            tree: Some(ctx.tree_view()),
            base_seed,
        })
    };
    let inflow =
        match resolve_inflow_source(inflow_scheme, historical_library, external_inflow_library)? {
            InflowSource::Historical(library) => ClassSampler::Historical { library },
            InflowSource::InSample => build_inflow(ClassSource::InSample)?,
            InflowSource::OutOfSample => build_inflow(ClassSource::OutOfSample)?,
            InflowSource::External(library) => build_inflow(ClassSource::External(library))?,
        };

    warn_unsupported_forward_noise_methods(EntityClass::Load, load_scheme, stages, &noise_methods);
    let load_range = dims.load_bus_range();
    let load = build_class_sampler(ClassSamplerParams {
        source: resolve_class_source(load_scheme, EntityClass::Load, external_load_library)?,
        offset: load_range.start,
        len: load_range.len(),
        forward_seed: load_forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
    })?;

    warn_unsupported_forward_noise_methods(EntityClass::Ncs, ncs_scheme, stages, &noise_methods);
    let ncs_range = dims.ncs_range();
    let ncs = build_class_sampler(ClassSamplerParams {
        source: resolve_class_source(ncs_scheme, EntityClass::Ncs, external_ncs_library)?,
        offset: ncs_range.start,
        len: ncs_range.len(),
        forward_seed: ncs_forward_seed,
        noise_methods: &noise_methods,
        tree: Some(ctx.tree_view()),
        base_seed,
    })?;

    // Correlation refs are set only for OutOfSample; pre-correlated sources must
    // not be correlated again.
    let inflow_correlation =
        matches!(inflow_scheme, SamplingScheme::OutOfSample).then_some(correlation);
    let load_correlation =
        matches!(load_scheme, SamplingScheme::OutOfSample).then_some(correlation);
    let ncs_correlation = matches!(ncs_scheme, SamplingScheme::OutOfSample).then_some(correlation);

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
// Shared helper
// ---------------------------------------------------------------------------

/// Build the observation sequence as `(year_offset, season_id)` pairs.
///
/// The anchor is `stages[0]`; its own resolved year is `y0`, and every other
/// entry's `year_offset` is relative to it — `window_year` is the first study
/// observation's year.
///
/// **Layout.** The study entries come first, one per stage carrying a
/// `season_id`, in stage order. The resolved lag entries follow, for
/// `k = 1..=r` (newest first), where `r <= max_order`: the season-map walk
/// ([`StageCalendar::season_occurrences`]) may truncate before `max_order` on
/// a sparse `Custom` map, and there are no lag entries at all when
/// `stages[0]`'s own season id has no entry in `season_map`.
///
/// **Year.** A study or lag entry whose season id resolves to a
/// [`SeasonDefinition`](cobre_core::temporal::SeasonDefinition) in
/// `season_map` is dated via [`occurrence_year`]; one that does not (a
/// missing def, or `season_map: None`) falls back to its own date's calendar
/// year — the same fallback `None` uses throughout, so a `None` map's lag `k`
/// is `stages[0].start_date` minus `k` calendar months, keyed by that date's
/// `month0()`.
pub(crate) fn build_observation_sequence(
    stages: &[Stage],
    max_order: usize,
    season_map: Option<&SeasonMap>,
) -> Vec<(i32, usize)> {
    let Some(first_stage) = stages.first() else {
        return Vec::new();
    };

    let year_for = |season_id: Option<usize>, start: NaiveDate, end: NaiveDate| -> i32 {
        season_map
            .zip(season_id)
            .and_then(|(map, sid)| {
                map.seasons
                    .iter()
                    .find(|def| def.id == sid)
                    .map(|def| occurrence_year(map, def, start, end))
            })
            .unwrap_or_else(|| start.year())
    };

    let y0 = year_for(
        first_stage.season_id,
        first_stage.start_date,
        first_stage.end_date,
    );

    let mut result: Vec<(i32, usize)> = stages
        .iter()
        .filter_map(|stage| {
            let sid = stage.season_id?;
            let year = year_for(Some(sid), stage.start_date, stage.end_date);
            Some((year - y0, sid))
        })
        .collect();

    match season_map {
        Some(map) => {
            if let Some(def0) = first_stage
                .season_id
                .and_then(|sid| map.seasons.iter().find(|def| def.id == sid))
                && let Some(occurrences) = StageCalendar::new(std::slice::from_ref(first_stage))
                    .season_occurrences(map, def0, max_order)
            {
                for occ in occurrences.iter().skip(1) {
                    let Some(id_k) = map.season_for_date(occ.start) else {
                        break;
                    };
                    let year_k = year_for(Some(id_k), occ.start, occ.end);
                    result.push((year_k - y0, id_k));
                }
            }
        }
        None => {
            for k in 1..=max_order {
                let months = u32::try_from(k).unwrap_or(u32::MAX);
                let Some(d) = first_stage
                    .start_date
                    .checked_sub_months(Months::new(months))
                else {
                    break;
                };
                result.push((d.year() - y0, d.month0() as usize));
            }
        }
    }

    result
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
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };

    use cobre_core::{
        Bus, DeficitSegment, EntityId, Hydro, SystemBuilder,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
            LoadModel, NcsModel, SamplingScheme,
        },
        temporal::{NoiseMethod, ScenarioSourceConfig, Stage},
        test_support::{BusSpec, HydroSpec, StageSpec, single_block},
    };
    use tracing::{Event, Level, Metadata, Subscriber, span};

    #[cfg(debug_assertions)]
    use super::ClassSampleRequest;
    use super::{
        ClassNoiseTables, ClassSampler, ForwardNoise, ForwardNoiseTables, ForwardSampler,
        ForwardSamplerConfig, NoiseTable, SampleRequest, build_forward_sampler,
        rebuild_class_tables,
    };
    use crate::{
        NoisePointSpec, StochasticContext, StochasticError,
        context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context},
        sample_forward,
        test_support::uniform_tree,
        tree::generate::ClassDimensions,
        tree::lhs::{sample_lhs_point, sample_lhs_point_reference},
    };

    /// Records all WARN-level event messages into a shared `Vec<String>`.
    struct WarnRecorder {
        messages: Arc<Mutex<Vec<String>>>,
    }

    impl WarnRecorder {
        fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
            let messages = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    messages: Arc::clone(&messages),
                },
                messages,
            )
        }
    }

    impl Subscriber for WarnRecorder {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            *metadata.level() <= Level::WARN
        }

        fn new_span(&self, _attrs: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

        fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if *event.metadata().level() == Level::WARN {
                struct MessageVisitor(String);
                impl tracing::field::Visit for MessageVisitor {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "message" {
                            self.0 = format!("{value:?}");
                        }
                    }

                    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                        if field.name() == "message" {
                            self.0 = value.to_string();
                        }
                    }
                }
                let mut visitor = MessageVisitor(String::new());
                event.record(&mut visitor);
                self.messages.lock().unwrap().push(visitor.0);
            }
        }

        fn enter(&self, _span: &span::Id) {}

        fn exit(&self, _span: &span::Id) {}
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

    fn make_stage(index: usize, id: i32, bf: usize) -> Stage {
        make_stage_with_method(index, id, bf, NoiseMethod::Saa)
    }

    fn make_stage_with_method(index: usize, id: i32, bf: usize, method: NoiseMethod) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            season_id: Some(0),
            blocks: single_block("SINGLE", 744.0),
            scenario_config: ScenarioSourceConfig {
                branching_factor: bf,
                noise_method: method,
            },
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

    fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
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

    fn build_test_ctx(forward_seed: Option<u64>) -> (StochasticContext, Vec<Stage>) {
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
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();
        (ctx, stages)
    }

    // -----------------------------------------------------------------------
    // Factory helper
    // -----------------------------------------------------------------------

    fn all_classes_config<'a>(
        scheme: SamplingScheme,
        ctx: &'a StochasticContext,
        stages: &'a [Stage],
    ) -> super::ForwardSamplerConfig<'a> {
        super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(scheme),
                load: Some(scheme),
                ncs: Some(scheme),
            },
            ctx,
            stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        }
    }

    /// The `SampleRequest.tables` every `sample()` test call needs.
    fn tables_for(
        sampler: &ForwardSampler<'_>,
        iteration: u32,
        total: u32,
        groups: &[u32],
    ) -> ForwardNoiseTables {
        let mut tables = ForwardNoiseTables::default();
        sampler
            .rebuild_noise_tables(iteration, total, groups, &mut tables)
            .expect("test fixtures never exceed the Sobol dimension cap");
        tables
    }

    // -----------------------------------------------------------------------
    // Factory tests
    // -----------------------------------------------------------------------

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

    #[test]
    fn test_build_historical_with_library() {
        use super::HistoricalScenarioLibrary;
        let (ctx, stages) = build_test_ctx(None);
        let dims = ctx.class_dimensions();
        // 3 windows, 2 stages, 1 hydro, max_order=1.
        let lib = HistoricalScenarioLibrary::new(
            3,
            stages.len(),
            dims.n_hydros,
            1,
            vec![2000, 2001, 2002],
        );
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
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

    /// When both `historical_library` and `external_inflow_library` are
    /// `Some` for a `Historical` inflow scheme, the fold must still pick the
    /// historical library and ignore the external one.
    #[test]
    fn test_build_historical_with_library_ignores_external_library() {
        use super::{ExternalScenarioLibrary, HistoricalScenarioLibrary};
        let (ctx, stages) = build_test_ctx(None);
        let dims = ctx.class_dimensions();
        // A single window makes historical window selection deterministic
        // (hash % 1 == 0) without reaching into ClassSampler's private
        // window-selection helper.
        let mut historical_lib =
            HistoricalScenarioLibrary::new(1, stages.len(), dims.n_hydros, 1, vec![2000]);
        for stage in 0..stages.len() {
            historical_lib.eta_slice_mut(0, stage).fill(7.0);
        }
        let external_lib = ExternalScenarioLibrary::new(
            stages.len(),
            10,
            dims.n_hydros,
            "inflow",
            vec![10usize; stages.len()],
        );
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
            historical_library: Some(&historical_lib),
            external_inflow_library: Some(&external_lib),
            external_load_library: None,
            external_ncs_library: None,
        };
        let sampler = build_forward_sampler(config)
            .expect("Historical inflow with both libraries set must still succeed");

        let mut noise_buf = vec![0.0f64; ctx.dim()];
        let mut corr_scratch = vec![0.0f64; 2 * ctx.dim()];
        let tables = tables_for(&sampler, 0, 5, &[]);
        let noise = sampler
            .sample(SampleRequest {
                iteration: 0,
                scenario: 0,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut noise_buf,
                corr_scratch: &mut corr_scratch,
                total_scenarios: 5,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: ctx.tree_view().n_openings(0),
                pinned_scenario: None,
                tables: &tables,
            })
            .expect("expected Ok noise from Historical inflow");

        assert_eq!(
            noise.as_slice()[0],
            7.0,
            "inflow slot must replay the historical library's value, not the \
             zero-filled external one"
        );
    }

    #[test]
    fn test_build_historical_missing_library() {
        let (ctx, stages) = build_test_ctx(None);
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
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

    #[test]
    fn test_build_external_with_library() {
        use super::ExternalScenarioLibrary;
        let (ctx, stages) = build_test_ctx(None);
        let dims = ctx.class_dimensions();
        let lib = ExternalScenarioLibrary::new(
            stages.len(),
            10,
            dims.n_hydros,
            "inflow",
            vec![10usize; stages.len()],
        );
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
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

    #[test]
    fn test_build_historical_load_unsupported() {
        let (ctx, stages) = build_test_ctx(None);
        let config = super::ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::Historical),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx: &ctx,
            stages: &stages,
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
        let noise = ForwardNoise::new(&data);
        assert_eq!(noise.as_slice(), &data);
    }

    #[test]
    fn test_forward_noise_as_slice() {
        let buf = [4.0f64, 5.0];
        let noise = ForwardNoise::new(&buf);
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
        let mut corr_scratch = vec![0.0f64; 2 * dim];
        let tables = tables_for(&sampler, 0, 5, &[]);

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: ctx.tree_view().n_openings(0),
            pinned_scenario: None,
            tables: &tables,
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
        let mut corr_a = vec![0.0f64; 2 * dim];
        let mut corr_b = vec![0.0f64; 2 * dim];
        let tables = tables_for(&sampler, 1, 5, &[]);

        let a = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_a,
                corr_scratch: &mut corr_a,
                total_scenarios: 5,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: ctx.tree_view().n_openings(0),
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_b,
                corr_scratch: &mut corr_b,
                total_scenarios: 5,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: ctx.tree_view().n_openings(0),
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();

        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn test_composite_in_sample_fills_correct_segments() {
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
        let mut corr_scratch = vec![0.0f64; 2 * 5];
        let tables = tables_for(&sampler, 0, 3, &[]);

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: 3,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: tree.view().n_openings(0),
            pinned_scenario: None,
            tables: &tables,
        });

        let noise = result.expect("expected Ok from composite InSample sample()");
        assert_eq!(
            noise.as_slice().len(),
            5,
            "total noise length must equal total_dim"
        );

        let (_, full_slice) =
            sample_forward(&tree.view(), 42, 0, 0, 0, 0, 0, tree.view().n_openings(0));
        assert_eq!(
            noise.as_slice(),
            full_slice,
            "composite InSample must reproduce the full tree slice"
        );
    }

    #[test]
    fn test_composite_out_of_sample_applies_per_class_correlation() {
        let (ctx, stages) = build_test_ctx(Some(99));
        let sampler = build_forward_sampler(all_classes_config(
            SamplingScheme::OutOfSample,
            &ctx,
            &stages,
        ))
        .unwrap();
        let dim = ctx.dim();

        let mut noise_buf = vec![0.0f64; dim];
        let mut corr_scratch = vec![0.0f64; 2 * dim];
        let tables = tables_for(&sampler, 0, 5, &[]);

        let result = sampler.sample(SampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            noise_buf: &mut noise_buf,
            corr_scratch: &mut corr_scratch,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
            tables: &tables,
        });

        let noise = result.expect("expected Ok from OutOfSample sample()");
        for (i, &v) in noise.as_slice().iter().enumerate() {
            assert!(v.is_finite(), "element[{i}] is not finite: {v}");
        }
        assert_eq!(noise.as_slice().len(), dim);
    }

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
        let mut corr_a = vec![0.0f64; 2 * dim];
        let mut corr_b = vec![0.0f64; 2 * dim];
        let tables = tables_for(&sampler, 3, 5, &[]);

        let a = sampler
            .sample(SampleRequest {
                iteration: 3,
                scenario: 7,
                stage: 1,
                stage_idx: 1,
                noise_buf: &mut buf_a,
                corr_scratch: &mut corr_a,
                total_scenarios: 5,
                noise_group_id: 1,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 3,
                scenario: 7,
                stage: 1,
                stage_idx: 1,
                noise_buf: &mut buf_b,
                corr_scratch: &mut corr_b,
                total_scenarios: 5,
                noise_group_id: 1,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();

        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "composite sample() must be deterministic for same inputs"
        );
    }

    #[test]
    fn test_sample_request_propagates_noise_group_id() {
        let (ctx, stages) = build_test_ctx(Some(42));
        let sampler = build_forward_sampler(all_classes_config(
            SamplingScheme::OutOfSample,
            &ctx,
            &stages,
        ))
        .unwrap();
        let dim = ctx.dim();

        let mut buf_a = vec![0.0f64; dim];
        let mut buf_b = vec![0.0f64; dim];
        let mut corr_a = vec![0.0f64; 2 * dim];
        let mut corr_b = vec![0.0f64; 2 * dim];
        let tables_group7 = tables_for(&sampler, 2, 5, &[7]);

        let a = sampler
            .sample(SampleRequest {
                iteration: 2,
                scenario: 3,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_a,
                corr_scratch: &mut corr_a,
                total_scenarios: 5,
                noise_group_id: 7,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables_group7,
            })
            .unwrap();
        let b = sampler
            .sample(SampleRequest {
                iteration: 2,
                scenario: 3,
                stage: 1,
                stage_idx: 0,
                noise_buf: &mut buf_b,
                corr_scratch: &mut corr_b,
                total_scenarios: 5,
                noise_group_id: 7,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables_group7,
            })
            .unwrap();
        assert_eq!(
            a.as_slice(),
            b.as_slice(),
            "same noise_group_id with different stage must produce identical OutOfSample noise"
        );

        let mut buf_c = vec![0.0f64; dim];
        let mut corr_c = vec![0.0f64; 2 * dim];
        let tables_group8 = tables_for(&sampler, 2, 5, &[8]);
        let c = sampler
            .sample(SampleRequest {
                iteration: 2,
                scenario: 3,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf_c,
                corr_scratch: &mut corr_c,
                total_scenarios: 5,
                noise_group_id: 8,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables_group8,
            })
            .unwrap();
        let any_differ = a.as_slice().iter().zip(c.as_slice()).any(|(x, y)| x != y);
        assert!(
            any_differ,
            "different noise_group_id must produce different OutOfSample noise"
        );
    }

    /// Classes sampled out of sample must not share a noise stream: one seed
    /// for every class makes the load and NCS slots repeat the first inflow
    /// slots bit-for-bit. Inflow keeps the root seed, so its slot is pinned.
    #[test]
    fn test_out_of_sample_classes_draw_distinct_streams() {
        let stages = vec![make_stage(0, 0, 5), make_stage(1, 1, 5)];
        let load_model = |stage_id: i32| LoadModel {
            bus_id: EntityId(0),
            stage_id,
            mean_mw: 100.0,
            std_mw: 10.0,
        };
        let ncs_model = |stage_id: i32| NcsModel {
            ncs_id: EntityId(20),
            stage_id,
            mean: 0.7,
            std: 0.1,
        };
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![make_hydro(1), make_hydro(2)])
            .stages(stages.clone())
            .inflow_models(vec![
                make_inflow_model(1, 0),
                make_inflow_model(1, 1),
                make_inflow_model(2, 0),
                make_inflow_model(2, 1),
            ])
            .load_models(vec![load_model(0), load_model(1)])
            .ncs_models(vec![ncs_model(0), ncs_model(1)])
            .correlation(identity_correlation(&[1, 2]))
            .build()
            .unwrap();
        let ctx = build_stochastic_context(
            &system,
            42,
            Some(99),
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::OutOfSample),
                ncs: Some(SamplingScheme::OutOfSample),
            },
        )
        .unwrap();
        assert_eq!(ctx.n_load_buses(), 1);
        assert_eq!(ctx.n_stochastic_ncs(), 1);
        let sampler = build_forward_sampler(ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::OutOfSample),
                ncs: Some(SamplingScheme::OutOfSample),
            },
            ctx: &ctx,
            stages: &stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        })
        .unwrap();

        let mut buf = vec![0.0f64; ctx.dim()];
        let mut corr = vec![0.0f64; 2 * ctx.dim()];
        let tables = tables_for(&sampler, 1, 5, &[]);
        let noise = sampler
            .sample(SampleRequest {
                iteration: 1,
                scenario: 2,
                stage: 0,
                stage_idx: 0,
                noise_buf: &mut buf,
                corr_scratch: &mut corr,
                total_scenarios: 5,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
        let s = noise.as_slice();
        assert_ne!(
            s[2], s[0],
            "load slot must not repeat the first inflow slot"
        );
        assert_ne!(s[2], s[1]);
        assert_ne!(s[3], s[0], "NCS slot must not repeat the first inflow slot");
        assert_ne!(s[3], s[1]);
        assert_ne!(s[3], s[2], "NCS slot must not repeat the load slot");
        assert_eq!(
            s[0].to_bits(),
            4_608_014_355_120_151_153_u64,
            "inflow draw must keep its root-seed value"
        );
    }

    // -----------------------------------------------------------------------
    // rebuild_noise_tables
    // -----------------------------------------------------------------------

    /// Build a single-hydro, three-stage study whose inflow class is sampled
    /// out of sample with the given per-stage methods and forward seed. Load
    /// and NCS default to `InSample` — this fixture exercises only inflow.
    fn build_inflow_oos_test_ctx(
        methods: [NoiseMethod; 3],
        forward_seed: u64,
    ) -> (StochasticContext, Vec<Stage>) {
        let hydros = vec![make_hydro(1)];
        let stages = vec![
            make_stage_with_method(0, 0, 5, methods[0]),
            make_stage_with_method(1, 1, 5, methods[1]),
            make_stage_with_method(2, 2, 5, methods[2]),
        ];
        let inflow_models = vec![
            make_inflow_model(1, 0),
            make_inflow_model(1, 1),
            make_inflow_model(1, 2),
        ];
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
            Some(forward_seed),
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();
        (ctx, stages)
    }

    fn build_oos_inflow_config<'a>(
        ctx: &'a StochasticContext,
        stages: &'a [Stage],
    ) -> ForwardSamplerConfig<'a> {
        ForwardSamplerConfig {
            class_schemes: ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
            ctx,
            stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
        }
    }

    #[test]
    fn test_rebuild_noise_tables_variant_per_method() {
        let (ctx, stages) = build_inflow_oos_test_ctx(
            [
                NoiseMethod::QmcSobol,
                NoiseMethod::QmcHalton,
                NoiseMethod::Lhs,
            ],
            99,
        );
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 1, 2], &mut tables)
            .expect("single-hydro dim never exceeds the Sobol dimension cap");

        assert!(matches!(
            tables.inflow().table_at(0),
            Some(NoiseTable::Sobol(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(1),
            Some(NoiseTable::Halton(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(2),
            Some(NoiseTable::Lhs(_))
        ));
    }

    #[test]
    fn test_rebuild_noise_tables_dedups_same_group_and_method() {
        let (ctx, stages) =
            build_inflow_oos_test_ctx([NoiseMethod::Lhs, NoiseMethod::Lhs, NoiseMethod::Lhs], 99);
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 0, 1], &mut tables)
            .expect("Lhs never exceeds the Sobol dimension cap");

        let t0 = tables.inflow().table_at(0).expect("stage 0 has a table");
        let t1 = tables.inflow().table_at(1).expect("stage 1 has a table");
        let t2 = tables.inflow().table_at(2).expect("stage 2 has a table");
        assert!(
            std::ptr::eq(t0, t1),
            "stages sharing a (group, method) pair must resolve to the same table"
        );
        assert!(
            !std::ptr::eq(t0, t2),
            "a different noise group must resolve to a distinct table"
        );
    }

    #[test]
    fn test_rebuild_noise_tables_keys_on_group_and_method_pair() {
        let (ctx, stages) = build_inflow_oos_test_ctx(
            [NoiseMethod::Lhs, NoiseMethod::QmcSobol, NoiseMethod::Saa],
            99,
        );
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 0, 0], &mut tables)
            .expect("single-hydro dim never exceeds the Sobol dimension cap");

        assert!(matches!(
            tables.inflow().table_at(0),
            Some(NoiseTable::Lhs(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(1),
            Some(NoiseTable::Sobol(_))
        ));
        assert!(matches!(
            tables.inflow().table_at(2),
            Some(NoiseTable::Direct)
        ));
    }

    #[test]
    fn test_rebuild_noise_tables_lhs_matches_direct_point_function() {
        let (ctx, stages) =
            build_inflow_oos_test_ctx([NoiseMethod::Lhs, NoiseMethod::Lhs, NoiseMethod::Lhs], 99);
        let sampler = build_forward_sampler(build_oos_inflow_config(&ctx, &stages)).unwrap();
        let mut tables = ForwardNoiseTables::default();

        sampler
            .rebuild_noise_tables(0, 8, &[0, 0, 1], &mut tables)
            .expect("Lhs never exceeds the Sobol dimension cap");

        let Some(NoiseTable::Lhs(lhs_ctx)) = tables.inflow().table_at(2) else {
            panic!("expected NoiseTable::Lhs for stage 2");
        };

        let forward_seed = 99; // inflow keeps the root seed unchanged
        for scenario in 0..8u32 {
            let spec = NoisePointSpec {
                sampling_seed: forward_seed,
                iteration: 0,
                scenario,
                stream_id: 1,
                total_scenarios: 8,
                dim: 1,
            };
            let mut precomputed_out = [0.0f64];
            sample_lhs_point(&spec, lhs_ctx, &mut precomputed_out);

            let mut perm = vec![0usize; 8];
            let mut direct_out = [0.0f64];
            sample_lhs_point_reference(&spec, &mut direct_out, &mut perm);

            assert_eq!(
                precomputed_out, direct_out,
                "scenario {scenario}: precomputed LHS table must match the direct point function"
            );
        }
    }

    #[test]
    fn rebuild_class_tables_rejects_an_oversized_sobol_class_and_clears_non_out_of_sample() {
        let mut out = ClassNoiseTables::default();

        let oversized = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 21_202, // one above the crate's Sobol dimension cap (21_201)
            noise_methods: vec![NoiseMethod::QmcSobol].into(),
        };
        match rebuild_class_tables(&oversized, 0, 1, &[0], &mut out) {
            Err(StochasticError::DimensionExceedsCapacity {
                dim,
                max_dim,
                method,
            }) => {
                assert_eq!(dim, 21_202, "dim field");
                assert_eq!(max_dim, 21_201, "max_dim field");
                assert!(
                    method.contains("sobol"),
                    "method must contain 'sobol', got: {method}"
                );
            }
            other => panic!("expected Err(DimensionExceedsCapacity), got {other:?}"),
        }
        assert!(out.table_at(0).is_none());

        let tree = uniform_tree(1, 2, 3);
        let in_sample = ClassSampler::InSample {
            tree: tree.view(),
            base_seed: 42,
            offset: 0,
            len: 2,
        };
        let result = rebuild_class_tables(&in_sample, 0, 1, &[], &mut out);
        assert!(
            result.is_ok(),
            "expected Ok for a class not sampled out of sample, got: {result:?}"
        );
        assert!(out.table_at(0).is_none());
    }

    // -----------------------------------------------------------------------
    // OutOfSample::fill stamp assertions
    // -----------------------------------------------------------------------

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "tables built for")]
    fn out_of_sample_fill_panics_when_tables_stale_for_iteration() {
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 2,
            noise_methods: vec![NoiseMethod::Saa].into_boxed_slice(),
        };
        let mut tables = ClassNoiseTables::default();
        tables
            .refill(1, 2, 0, 4, &[0], &[NoiseMethod::Saa])
            .expect("Saa never exceeds the Sobol dimension cap");

        let req = ClassSampleRequest {
            iteration: 1,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 4,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let mut output = vec![0.0f64; 2];
        let _ = sampler.fill(&req, &tables, &mut output);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "table built for group")]
    fn out_of_sample_fill_panics_when_stage_table_built_for_a_different_group() {
        let methods = [NoiseMethod::Saa, NoiseMethod::Saa, NoiseMethod::Saa];
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 2,
            noise_methods: methods.into(),
        };
        let mut tables = ClassNoiseTables::default();
        tables
            .refill(1, 2, 0, 4, &[0, 0, 1], &methods)
            .expect("Saa never exceeds the Sobol dimension cap");

        let req = ClassSampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 2,
            total_scenarios: 4,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let mut output = vec![0.0f64; 2];
        let _ = sampler.fill(&req, &tables, &mut output);
    }

    // -----------------------------------------------------------------------
    // Unsupported forward noise method warnings
    // -----------------------------------------------------------------------

    /// Build a single-hydro, out-of-sample inflow study whose stages carry the
    /// given `(id, method)` pairs; load and NCS default to `InSample`.
    fn build_inflow_oos_ctx_with_stage_ids(
        stage_specs: &[(i32, NoiseMethod)],
    ) -> (StochasticContext, Vec<Stage>) {
        let hydros = vec![make_hydro(1)];
        let stages: Vec<Stage> = stage_specs
            .iter()
            .enumerate()
            .map(|(idx, &(id, method))| make_stage_with_method(idx, id, 5, method))
            .collect();
        let inflow_models: Vec<InflowModel> = stage_specs
            .iter()
            .map(|&(id, _)| make_inflow_model(1, id))
            .collect();
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages.clone())
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();
        // Selective is unsupported by the opening tree generator regardless of
        // the forward scheme (`generate_stage_raw_noise`), so a study with any
        // Selective stage must supply a pre-built tree to bypass generation.
        let user_tree = uniform_tree(stages.len(), 5, 1);
        let ctx = build_stochastic_context(
            &system,
            42,
            Some(99),
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(user_tree),
                historical_library: None,
                external_scenario_counts: None,
                noise_group_ids: None,
            },
            ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();
        (ctx, stages)
    }

    #[test]
    fn test_warn_once_names_class_and_every_unsupported_stage_id() {
        let (ctx, stages) = build_inflow_oos_ctx_with_stage_ids(&[
            (2, NoiseMethod::Selective),
            (5, NoiseMethod::Selective),
        ]);
        let config = build_oos_inflow_config(&ctx, &stages);

        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            let sampler =
                build_forward_sampler(config).expect("Selective falls back to SAA, not an error");

            {
                let recorded = messages.lock().unwrap();
                assert_eq!(
                    recorded.len(),
                    1,
                    "expected exactly one WARN, got: {recorded:?}"
                );
                let message = &recorded[0];
                assert!(message.contains("inflow"), "message: {message}");
                assert!(message.contains('2'), "message: {message}");
                assert!(message.contains('5'), "message: {message}");
            }

            let dim = ctx.dim();
            let mut noise_buf = vec![0.0f64; dim];
            let mut corr_scratch = vec![0.0f64; 2 * dim];
            let tables = tables_for(&sampler, 0, 5, &[]);
            sampler
                .sample(SampleRequest {
                    iteration: 0,
                    scenario: 0,
                    stage: 2,
                    stage_idx: 0,
                    noise_buf: &mut noise_buf,
                    corr_scratch: &mut corr_scratch,
                    total_scenarios: 5,
                    noise_group_id: 0,
                    node_opening_offset: 0,
                    node_opening_len: ctx.tree_view().n_openings(0),
                    pinned_scenario: None,
                    tables: &tables,
                })
                .expect("Selective fallback sample() must still succeed");
        });

        assert_eq!(
            messages.lock().unwrap().len(),
            1,
            "ForwardSampler::sample must not add a WARN after construction"
        );
    }

    #[test]
    fn test_warn_once_covers_both_unsupported_methods_in_one_class() {
        let (ctx, stages) = build_inflow_oos_ctx_with_stage_ids(&[
            (0, NoiseMethod::Selective),
            (1, NoiseMethod::HistoricalResiduals),
        ]);
        let config = build_oos_inflow_config(&ctx, &stages);

        let (subscriber, messages) = WarnRecorder::new();
        let result =
            tracing::subscriber::with_default(subscriber, || build_forward_sampler(config));

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        assert_eq!(
            messages.lock().unwrap().len(),
            1,
            "a Selective stage and a HistoricalResiduals stage in the same class \
             must still produce exactly one WARN"
        );
    }

    #[test]
    fn test_warn_skips_class_not_sampled_out_of_sample() {
        let hydros = vec![make_hydro(1)];
        let stages = vec![
            make_stage_with_method(0, 0, 5, NoiseMethod::Selective),
            make_stage_with_method(1, 1, 5, NoiseMethod::Selective),
        ];
        let inflow_models = vec![make_inflow_model(1, 0), make_inflow_model(1, 1)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(hydros)
            .stages(stages.clone())
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();
        let user_tree = uniform_tree(stages.len(), 5, 1);
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
        let config = all_classes_config(SamplingScheme::InSample, &ctx, &stages);

        let (subscriber, messages) = WarnRecorder::new();
        let result =
            tracing::subscriber::with_default(subscriber, || build_forward_sampler(config));

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        assert!(
            messages.lock().unwrap().is_empty(),
            "an InSample class must never warn about its stages' noise methods"
        );
    }

    #[test]
    fn test_warn_once_per_class_for_three_out_of_sample_classes() {
        let stages = vec![
            make_stage_with_method(0, 0, 5, NoiseMethod::Selective),
            make_stage_with_method(1, 1, 5, NoiseMethod::Selective),
        ];
        let load_model = |stage_id: i32| LoadModel {
            bus_id: EntityId(0),
            stage_id,
            mean_mw: 100.0,
            std_mw: 10.0,
        };
        let ncs_model = |stage_id: i32| NcsModel {
            ncs_id: EntityId(20),
            stage_id,
            mean: 0.7,
            std: 0.1,
        };
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![make_hydro(1)])
            .stages(stages.clone())
            .inflow_models(vec![make_inflow_model(1, 0), make_inflow_model(1, 1)])
            .load_models(vec![load_model(0), load_model(1)])
            .ncs_models(vec![ncs_model(0), ncs_model(1)])
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();
        let user_tree = uniform_tree(stages.len(), 5, 3);
        let ctx = build_stochastic_context(
            &system,
            42,
            Some(99),
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(user_tree),
                historical_library: None,
                external_scenario_counts: None,
                noise_group_ids: None,
            },
            ClassSchemes {
                inflow: Some(SamplingScheme::OutOfSample),
                load: Some(SamplingScheme::OutOfSample),
                ncs: Some(SamplingScheme::OutOfSample),
            },
        )
        .unwrap();
        assert_eq!(ctx.n_load_buses(), 1);
        assert_eq!(ctx.n_stochastic_ncs(), 1);
        let config = all_classes_config(SamplingScheme::OutOfSample, &ctx, &stages);

        let (subscriber, messages) = WarnRecorder::new();
        let result =
            tracing::subscriber::with_default(subscriber, || build_forward_sampler(config));

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        let recorded = messages.lock().unwrap();
        assert_eq!(
            recorded.len(),
            3,
            "expected exactly three WARNs, got: {recorded:?}"
        );
        assert!(recorded.iter().any(|m| m.contains("inflow")));
        assert!(recorded.iter().any(|m| m.contains("load")));
        assert!(recorded.iter().any(|m| m.contains("ncs")));
    }
}
