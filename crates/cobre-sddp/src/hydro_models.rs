//! Hydro model runtime types for the production function and evaporation pipelines.
//!
//! Provides the preprocessed output types for the hydro model pipeline:
//!
//! - [`ResolvedProductionModel`] and [`ProductionModelSet`] — resolved production
//!   functions (constant productivity or FPHA hyperplanes) indexed by hydro and stage.
//! - [`EvaporationModel`] and [`EvaporationModelSet`] — linearized evaporation
//!   coefficients indexed by hydro plant.
//! - [`HydroModelProvenance`] — tracks the source of each hydro's production and
//!   evaporation model for display and auditing.
//! - [`HydroModelSummary`] — aggregated statistics for display after preprocessing.
//! - [`PrepareHydroModelsResult`] — bundles all pipeline outputs.
//! - [`resolve_production_models`] — resolves per-hydro per-stage production models
//!   from the case directory, returning a [`ProductionModelSet`] and provenance vector.
//!
//! These types live in `cobre-sddp` because they are algorithm-specific (FPHA
//! hyperplane approximation is an SDDP concept). They must not be placed in
//! `cobre-core`.

use std::collections::HashMap;
use std::path::Path;

use cobre_core::{EntityId, System, entities::hydro::HydroGenerationModel};
use cobre_io::extensions::{
    FphaConfig, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig, SelectionMode,
};

use crate::SddpError;
use crate::fpha_fitting::{FphaFitResult, fit_fpha_planes};

// ── Hyperplane types ──────────────────────────────────────────────────────────

/// A single FPHA hyperplane with a pre-scaled intercept.
///
/// The intercept stored here is `gamma_0 * kappa` (the intercept coefficient
/// multiplied by the nominal head factor). The LP builder adds this directly
/// to the right-hand side of the hyperplane inequality constraint without
/// further scaling.
///
/// # Fields
///
/// - `intercept` — pre-scaled intercept (`gamma_0 * kappa`)
/// - `gamma_v` — storage (volume) coefficient
/// - `gamma_q` — turbined-flow coefficient
/// - `gamma_s` — spillage coefficient
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FphaPlane {
    /// Pre-scaled intercept (`gamma_0 * kappa`).
    pub intercept: f64,
    /// Storage (volume) coefficient.
    pub gamma_v: f64,
    /// Turbined-flow coefficient.
    pub gamma_q: f64,
    /// Spillage coefficient.
    pub gamma_s: f64,
}

// ── Resolved production model ─────────────────────────────────────────────────

/// A fully-resolved hydro production model for one (hydro, stage) pair.
///
/// The enum has two variants:
///
/// - [`ConstantProductivity`](ResolvedProductionModel::ConstantProductivity) —
///   generation is modelled as `g = rho * q` with a fixed productivity scalar.
/// - [`Fpha`](ResolvedProductionModel::Fpha) — generation is bounded by `M`
///   linear hyperplane constraints derived from the FPHA approximation.
#[derive(Debug, Clone)]
pub enum ResolvedProductionModel {
    /// Constant productivity: generation = `productivity * turbined_flow`.
    ConstantProductivity {
        /// Productivity coefficient (MW per m³/s).
        productivity: f64,
    },
    /// FPHA hyperplane approximation with `M` linearisation planes.
    Fpha {
        /// Ordered set of hyperplanes; each constrains the generation variable.
        planes: Vec<FphaPlane>,
        /// Penalty cost for turbined flow under the FPHA model (cost/m³/s).
        turbined_cost: f64,
    },
}

// ── Production model set ──────────────────────────────────────────────────────

/// Resolved production models for all (hydro, stage) combinations.
///
/// Indexed as `stage_models[hydro_index][stage_index]`. Access via the
/// [`model`](ProductionModelSet::model) accessor.
///
/// # Layout
///
/// The inner `Vec<ResolvedProductionModel>` at index `h` covers all stages for
/// hydro `h`. Access to a given (hydro, stage) pair is `O(1)`.
#[derive(Debug, Clone)]
pub struct ProductionModelSet {
    /// `stage_models[h][t]` is the resolved production model for hydro `h` at stage `t`.
    stage_models: Vec<Vec<ResolvedProductionModel>>,
    /// Number of hydro plants (outer dimension).
    n_hydros: usize,
    /// Number of stages (inner dimension).
    n_stages: usize,
}

impl ProductionModelSet {
    /// Construct a `ProductionModelSet` from a 2-D grid of models.
    ///
    /// `models` must be indexed as `models[hydro][stage]` and must have
    /// dimensions `n_hydros × n_stages`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `models.len() != n_hydros` or any inner
    /// `Vec` length differs from `n_stages`.
    #[must_use]
    pub fn new(
        models: Vec<Vec<ResolvedProductionModel>>,
        n_hydros: usize,
        n_stages: usize,
    ) -> Self {
        debug_assert_eq!(
            models.len(),
            n_hydros,
            "outer dimension must equal n_hydros"
        );
        debug_assert!(
            models.iter().all(|row| row.len() == n_stages),
            "each hydro's stage vector must have length n_stages"
        );
        Self {
            stage_models: models,
            n_hydros,
            n_stages,
        }
    }

    /// Return the resolved production model for hydro `hydro` at stage `stage`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `hydro >= n_hydros` or `stage >= n_stages`.
    #[must_use]
    pub fn model(&self, hydro: usize, stage: usize) -> &ResolvedProductionModel {
        debug_assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} out of bounds (n_hydros = {})",
            self.n_hydros
        );
        debug_assert!(
            stage < self.n_stages,
            "stage index {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
        &self.stage_models[hydro][stage]
    }

    /// Number of hydro plants.
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.n_hydros
    }

    /// Number of stages.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }
}

// ── Linearized evaporation ────────────────────────────────────────────────────

/// Linearized evaporation coefficients for one (hydro, stage) pair.
///
/// The evaporation volume (hm³) is approximated as:
///
/// ```text
/// evap = k_evap0 + k_evap_v * (V - V_ref)
/// ```
///
/// where `V` is the reservoir volume (hm³) and `V_ref` is the reference volume
/// for each stage stored in [`EvaporationModel::Linearized::reference_volumes_hm3`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearizedEvaporation {
    /// Constant term of the linearized evaporation (hm³).
    pub k_evap0: f64,
    /// Volume-dependent slope of the linearized evaporation (hm³/hm³ = dimensionless).
    pub k_evap_v: f64,
}

// ── Evaporation model ─────────────────────────────────────────────────────────

/// Resolved evaporation model for a single hydro plant.
///
/// The enum has two variants:
///
/// - [`None`](EvaporationModel::None) — evaporation is not modelled for this
///   hydro plant; the LP builder adds no evaporation term.
/// - [`Linearized`](EvaporationModel::Linearized) — per-stage linearized
///   evaporation coefficients derived from reservoir geometry.
#[derive(Debug, Clone)]
pub enum EvaporationModel {
    /// No evaporation for this hydro plant.
    None,
    /// Linearized evaporation with per-stage coefficients.
    Linearized {
        /// Per-stage linearization coefficients; indexed by stage position.
        coefficients: Vec<LinearizedEvaporation>,
        /// Reference storage volumes (hm³) at which the linearization was computed,
        /// one entry per stage. When using the midpoint fallback all entries are
        /// identical; when using user-supplied seasonal volumes each entry reflects
        /// the reference volume for that stage's calendar month.
        reference_volumes_hm3: Vec<f64>,
    },
}

// ── Evaporation model set ─────────────────────────────────────────────────────

/// Evaporation models for all hydro plants, indexed by hydro position.
///
/// Access individual models via [`model`](EvaporationModelSet::model).
/// Use [`has_evaporation`](EvaporationModelSet::has_evaporation) to gate
/// evaporation-related LP setup without iterating the full set.
#[derive(Debug, Clone)]
pub struct EvaporationModelSet {
    /// `models[h]` is the evaporation model for hydro plant at position `h`.
    models: Vec<EvaporationModel>,
}

impl EvaporationModelSet {
    /// Construct an `EvaporationModelSet` from a vector of per-hydro models.
    #[must_use]
    pub fn new(models: Vec<EvaporationModel>) -> Self {
        Self { models }
    }

    /// Return the evaporation model for hydro plant at position `hydro`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `hydro >= models.len()`.
    #[must_use]
    pub fn model(&self, hydro: usize) -> &EvaporationModel {
        debug_assert!(
            hydro < self.models.len(),
            "hydro index {hydro} out of bounds (n_hydros = {})",
            self.models.len()
        );
        &self.models[hydro]
    }

    /// Return `true` if at least one hydro plant has a [`Linearized`](EvaporationModel::Linearized) model.
    #[must_use]
    pub fn has_evaporation(&self) -> bool {
        self.models
            .iter()
            .any(|m| matches!(m, EvaporationModel::Linearized { .. }))
    }

    /// Number of hydro plants in the set.
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.models.len()
    }
}

// ── Provenance types ──────────────────────────────────────────────────────────

/// Source of the production model used for a given hydro plant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionModelSource {
    /// Constant productivity from the entity definition; no geometric data.
    DefaultConstant,
    /// FPHA hyperplanes loaded from a precomputed Parquet file.
    PrecomputedHyperplanes,
    /// FPHA hyperplanes computed from reservoir geometry during preprocessing.
    ComputedFromGeometry,
}

/// Source of the evaporation model used for a given hydro plant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaporationSource {
    /// Evaporation is not included for this hydro plant.
    NotModeled,
    /// Evaporation coefficients were linearized from reservoir geometry.
    LinearizedFromGeometry,
}

/// Source of the reference volume used for evaporation linearization.
///
/// Tracked per hydro plant and included in [`HydroModelProvenance`] for
/// display and auditing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaporationReferenceSource {
    /// User-supplied per-season reference volumes from the entity definition.
    UserSupplied,
    /// Default midpoint: `(min_storage + max_storage) / 2`.
    DefaultMidpoint,
}

/// Provenance record for all hydro plants' production and evaporation models.
///
/// One entry per hydro plant in declaration order (canonical ID order).
#[derive(Debug, Clone)]
pub struct HydroModelProvenance {
    /// `(entity_id, source)` pairs for each hydro's production model.
    pub production_sources: Vec<(EntityId, ProductionModelSource)>,
    /// `(entity_id, source)` pairs for each hydro's evaporation model.
    pub evaporation_sources: Vec<(EntityId, EvaporationSource)>,
    /// `(entity_id, source)` pairs for each hydro's evaporation reference volume.
    ///
    /// For hydros with [`EvaporationSource::NotModeled`], this is set to
    /// [`EvaporationReferenceSource::DefaultMidpoint`] (irrelevant but consistent).
    pub evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)>,
}

// ── Summary types ─────────────────────────────────────────────────────────────

/// Per-hydro detail for FPHA production models.
///
/// Included in [`HydroModelSummary`] for display and auditing. Contains the
/// entity identity, source, and the number of linearisation planes.
#[derive(Debug, Clone)]
pub struct FphaHydroDetail {
    /// Entity identifier of the hydro plant.
    pub hydro_id: EntityId,
    /// Human-readable name of the hydro plant.
    pub name: String,
    /// Source from which the FPHA hyperplanes were obtained.
    pub source: ProductionModelSource,
    /// Number of hyperplanes in the FPHA approximation for this hydro.
    pub n_planes: usize,
}

/// Aggregated summary of the hydro model preprocessing pipeline.
///
/// Produced by the summary builder in the hydro models module and consumed by
/// `cobre-cli` for display. Contains counts for both production and evaporation
/// models.
#[derive(Debug, Clone)]
pub struct HydroModelSummary {
    /// Number of hydro plants using [`ResolvedProductionModel::ConstantProductivity`].
    pub n_constant: usize,
    /// Number of hydro plants using [`ResolvedProductionModel::Fpha`].
    pub n_fpha: usize,
    /// Total number of hyperplanes across all FPHA hydro plants.
    pub total_planes: usize,
    /// Per-hydro detail for each FPHA hydro plant.
    pub fpha_details: Vec<FphaHydroDetail>,
    /// Number of hydro plants with linearized evaporation.
    pub n_evaporation: usize,
    /// Number of hydro plants with no evaporation model.
    pub n_no_evaporation: usize,
    /// Number of hydro plants with evaporation that used user-supplied reference volumes.
    pub n_user_supplied_ref: usize,
    /// Number of hydro plants with evaporation that used the default midpoint reference volume.
    pub n_default_midpoint_ref: usize,
    /// Low-kappa warnings for hydros whose FPHA envelope kappa < 0.95.
    ///
    /// Each entry is `(hydro_name, kappa)`.  An empty vector means all fitted
    /// FPHA envelopes had kappa >= 0.95 (no warnings).
    pub kappa_warnings: Vec<(String, f64)>,
}

// ── Pipeline result ───────────────────────────────────────────────────────────

/// Result of the hydro model preprocessing pipeline.
///
/// Bundles the three outputs so that callers do not need to handle them as
/// separate return values. Consumed by `StudySetup` construction and the
/// hydro model summary builder.
#[derive(Debug)]
pub struct PrepareHydroModelsResult {
    /// Resolved production models for all (hydro, stage) pairs.
    pub production: ProductionModelSet,
    /// Resolved evaporation models for all hydro plants.
    pub evaporation: EvaporationModelSet,
    /// Provenance records for all hydro plants.
    pub provenance: HydroModelProvenance,
    /// Low-kappa warnings collected during computed FPHA fitting.
    ///
    /// Each entry is `(hydro_name, kappa)` for hydros whose fitted FPHA
    /// envelope had kappa < 0.95.  An empty vector means no warnings were
    /// generated.
    pub kappa_warnings: Vec<(String, f64)>,
}

impl PrepareHydroModelsResult {
    /// Build a default result for a system with no FPHA and no evaporation data.
    ///
    /// All hydros receive [`ResolvedProductionModel::ConstantProductivity`] using
    /// the productivity from their entity definition, and [`EvaporationModel::None`].
    /// Provenance is set to [`ProductionModelSource::DefaultConstant`] and
    /// [`EvaporationSource::NotModeled`] for every hydro.
    ///
    /// This factory is used in tests and in entry points where the full
    /// `prepare_hydro_models` pipeline is not available (e.g., non-root MPI ranks
    /// that reconstruct the result independently).
    #[must_use]
    pub fn default_from_system(system: &System) -> Self {
        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
        let n_stages = study_stages.len();
        let n_hydros = system.hydros().len();

        let production_models: Vec<Vec<ResolvedProductionModel>> = system
            .hydros()
            .iter()
            .map(|hydro| {
                let productivity = match &hydro.generation_model {
                    HydroGenerationModel::ConstantProductivity {
                        productivity_mw_per_m3s,
                    }
                    | HydroGenerationModel::LinearizedHead {
                        productivity_mw_per_m3s,
                    } => *productivity_mw_per_m3s,
                    HydroGenerationModel::Fpha => 0.0,
                };
                vec![ResolvedProductionModel::ConstantProductivity { productivity }; n_stages]
            })
            .collect();

        let production = ProductionModelSet::new(production_models, n_hydros, n_stages);

        let evaporation_models: Vec<EvaporationModel> = system
            .hydros()
            .iter()
            .map(|_| EvaporationModel::None)
            .collect();
        let evaporation = EvaporationModelSet::new(evaporation_models);

        let production_sources: Vec<(EntityId, ProductionModelSource)> = system
            .hydros()
            .iter()
            .map(|h| (h.id, ProductionModelSource::DefaultConstant))
            .collect();
        let evaporation_sources: Vec<(EntityId, EvaporationSource)> = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationSource::NotModeled))
            .collect();

        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationReferenceSource::DefaultMidpoint))
            .collect();

        Self {
            production,
            evaporation,
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            kappa_warnings: Vec::new(),
        }
    }
}

// ── Summary builder ───────────────────────────────────────────────────────────

/// Build a [`HydroModelSummary`] from the pipeline result and the system.
///
/// Called after the hydro model preprocessing pipeline completes and before
/// training starts. All fields are derived from the already-validated inputs;
/// construction is infallible.
///
/// # Counting logic
///
/// - `n_constant`: hydros whose production provenance is
///   [`ProductionModelSource::DefaultConstant`].
/// - `n_fpha`: hydros whose production provenance is
///   [`ProductionModelSource::PrecomputedHyperplanes`] or
///   [`ProductionModelSource::ComputedFromGeometry`].
/// - `total_planes`: sum of plane counts from the first study stage across all
///   FPHA hydros. Stages beyond the first use the same hyperplanes in the
///   common case; the first stage is taken as representative for display.
/// - `fpha_details`: one [`FphaHydroDetail`] per FPHA hydro in canonical order.
/// - `n_evaporation`: hydros whose evaporation provenance is
///   [`EvaporationSource::LinearizedFromGeometry`].
/// - `n_no_evaporation`: hydros whose evaporation provenance is
///   [`EvaporationSource::NotModeled`].
#[must_use]
pub fn build_hydro_model_summary(
    result: &PrepareHydroModelsResult,
    system: &System,
) -> HydroModelSummary {
    let mut n_constant = 0usize;
    let mut n_fpha = 0usize;
    let mut total_planes = 0usize;
    let mut fpha_details: Vec<FphaHydroDetail> = Vec::new();

    // Collect study stages to determine plane counts (id >= 0).
    let study_stages: Vec<usize> = system
        .stages()
        .iter()
        .enumerate()
        .filter(|(_, s)| s.id >= 0)
        .map(|(idx, _)| idx)
        .collect();
    // Use stage position 0 (within the study stages) as the representative stage.
    // Production models are indexed by hydro position within `system.hydros()`.
    let representative_stage = 0usize; // index into `study_stages`

    for (hydro_pos, (entity_id, source)) in result.provenance.production_sources.iter().enumerate()
    {
        match source {
            ProductionModelSource::DefaultConstant => {
                n_constant += 1;
            }
            ProductionModelSource::PrecomputedHyperplanes
            | ProductionModelSource::ComputedFromGeometry => {
                n_fpha += 1;

                // Resolve plane count from the representative study stage.
                let n_planes = if study_stages.is_empty() {
                    0
                } else {
                    match result.production.model(hydro_pos, representative_stage) {
                        ResolvedProductionModel::Fpha { planes, .. } => planes.len(),
                        ResolvedProductionModel::ConstantProductivity { .. } => 0,
                    }
                };

                total_planes += n_planes;

                // Look up the hydro name from the system entity list.
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
            EvaporationSource::NotModeled => n_no_evaporation += 1,
        }
    }

    let mut n_user_supplied_ref = 0usize;
    let mut n_default_midpoint_ref = 0usize;

    // Count reference source only for hydros that actually have evaporation.
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
        kappa_warnings: result.kappa_warnings.clone(),
    }
}

// ── Top-level pipeline function ───────────────────────────────────────────────

/// Run the full hydro model preprocessing pipeline for a case directory.
///
/// Composes [`resolve_production_models`] and [`resolve_evaporation_models`]
/// and returns a [`PrepareHydroModelsResult`] bundling all pipeline outputs.
///
/// Called once per entry point (CLI, Python) before constructing `StudySetup`.
/// On MPI setups, this function runs on all ranks independently (each rank has
/// the system via broadcast and can load the optional files from a shared
/// filesystem).
///
/// # Errors
///
/// Propagates errors from [`resolve_production_models`] and
/// [`resolve_evaporation_models`]. See their individual documentation for the
/// full error table.
pub fn prepare_hydro_models(
    system: &System,
    case_dir: &Path,
) -> Result<PrepareHydroModelsResult, SddpError> {
    let (production, production_sources, kappa_warnings) =
        resolve_production_models(system, case_dir)?;
    let (evaporation, evaporation_sources, evaporation_reference_sources) =
        resolve_evaporation_models(system, case_dir)?;

    Ok(PrepareHydroModelsResult {
        production,
        evaporation,
        provenance: HydroModelProvenance {
            production_sources,
            evaporation_sources,
            evaporation_reference_sources,
        },
        kappa_warnings,
    })
}

// ── FPHA production model resolution ─────────────────────────────────────────

/// Return type for [`resolve_production_models`]: the model set, provenance vector,
/// and low-kappa warnings collected during computed FPHA fitting.
type ResolveProductionResult = (
    ProductionModelSet,
    Vec<(EntityId, ProductionModelSource)>,
    Vec<(String, f64)>,
);

/// Resolve per-hydro per-stage production models from the case directory.
///
/// Reads `system/hydro_production_models.json` when present (optional file).
/// If absent, all hydros fall back to the [`HydroGenerationModel`] from their
/// entity definition in `system/hydros.json`. When any hydro is configured as
/// FPHA with `source: "precomputed"`, also loads `system/fpha_hyperplanes.parquet`.
/// When any hydro is configured as FPHA with `source: "computed"`, also loads
/// `system/hydro_geometry.parquet` and runs the FPHA fitting pipeline.
///
/// Returns `(ProductionModelSet, provenance_vec, kappa_warnings)` where the
/// provenance vector records the [`ProductionModelSource`] for each hydro in
/// canonical ID order, and `kappa_warnings` contains `(name, kappa)` pairs for
/// any computed FPHA hydro whose fitted envelope had kappa < 0.95.
///
/// # Model resolution per hydro
///
/// For each hydro in `system.hydros()` (canonical ID order):
///
/// 1. If `hydro_production_models.json` has an entry for this hydro:
///    - `source: "precomputed"` → load hyperplanes from `fpha_hyperplanes.parquet`,
///      scale `gamma_0` by `kappa`, record [`ProductionModelSource::PrecomputedHyperplanes`].
///    - `source: "computed"` → fit hyperplanes from `hydro_geometry.parquet` via the
///      FPHA fitting pipeline, record [`ProductionModelSource::ComputedFromGeometry`].
/// 2. Otherwise, use the [`HydroGenerationModel`] from the entity definition:
///    - [`HydroGenerationModel::ConstantProductivity`] →
///      [`ResolvedProductionModel::ConstantProductivity`].
///    - [`HydroGenerationModel::LinearizedHead`] →
///      [`ResolvedProductionModel::ConstantProductivity`] (uses the productivity field).
///    - [`HydroGenerationModel::Fpha`] without a config entry →
///      [`SddpError::Validation`] (no hyperplane source specified).
///
/// # Errors
///
/// | Condition                                                       | Error variant              |
/// | --------------------------------------------------------------- | -------------------------- |
/// | `Fpha` entity model with no config entry                        | [`SddpError::Validation`]  |
/// | `source: "computed"` with missing tailrace/losses/efficiency    | [`SddpError::Validation`]  |
/// | `source: "computed"` with no geometry rows for the hydro        | [`SddpError::Validation`]  |
/// | FPHA fitting pipeline error                                     | [`SddpError::Validation`]  |
/// | `gamma_v <= 0` for any precomputed hyperplane                   | [`SddpError::Validation`]  |
/// | `gamma_s > 0` for any precomputed hyperplane                    | [`SddpError::Validation`]  |
/// | `gamma_q <= 0` for any precomputed hyperplane                   | [`SddpError::Validation`]  |
/// | `kappa` not in `(0, 1]` for precomputed hyperplane              | [`SddpError::Validation`]  |
/// | Zero hyperplanes for an FPHA hydro at any stage                 | [`SddpError::Validation`]  |
/// | I/O failure loading JSON or Parquet                             | [`SddpError::Io`]          |
pub fn resolve_production_models(
    system: &System,
    case_dir: &Path,
) -> Result<ResolveProductionResult, SddpError> {
    // ── Step 1: check whether the optional config file is present ─────────────
    let mut ctx = cobre_io::ValidationContext::new();
    let manifest = cobre_io::validate_structure(case_dir, &mut ctx);
    let config_path = if manifest.system_hydro_production_models_json {
        Some(case_dir.join("system").join("hydro_production_models.json"))
    } else {
        None
    };

    // ── Step 2: load production model configs ─────────────────────────────────
    let prod_configs: Vec<ProductionModelConfig> =
        cobre_io::extensions::load_production_models(config_path.as_deref())?;

    // ── Step 3: build O(1) lookup: hydro_id → config ─────────────────────────
    let config_map: HashMap<EntityId, &ProductionModelConfig> =
        prod_configs.iter().map(|c| (c.hydro_id, c)).collect();

    // ── Step 4/5/6: load and index precomputed FPHA hyperplane rows ───────────
    let hyperplane_rows = load_precomputed_hyperplanes(&prod_configs, &manifest, case_dir)?;
    let mut hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    for row in &hyperplane_rows {
        hyperplane_map
            .entry((row.hydro_id, row.stage_id))
            .or_default()
            .push(row);
    }

    // ── Step 4b/5b/6b: load and index geometry rows for computed-source hydros ─
    let geometry_rows = load_geometry_rows(&prod_configs, &manifest, case_dir)?;
    let geometry_map = build_geometry_map(&geometry_rows);

    // ── Step 7: collect study stages (id >= 0) in canonical order ────────────
    let study_stages: Vec<&cobre_core::temporal::Stage> =
        system.stages().iter().filter(|s| s.id >= 0).collect();
    let n_stages = study_stages.len();
    let n_hydros = system.hydros().len();

    // ── Step 8: resolve per-hydro per-stage model ─────────────────────────────
    let mut all_models: Vec<Vec<ResolvedProductionModel>> = Vec::with_capacity(n_hydros);
    let mut provenance: Vec<(EntityId, ProductionModelSource)> = Vec::with_capacity(n_hydros);
    let mut export_rows: Vec<cobre_io::FphaHyperplaneRow> = Vec::new();
    let mut kappa_warnings: Vec<(String, f64)> = Vec::new();

    for hydro in system.hydros() {
        let config_entry = config_map.get(&hydro.id).copied();

        let source = determine_source(hydro, config_entry)?;
        provenance.push((hydro.id, source));

        // Fit computed-source planes once per hydro, reuse for each stage.
        let cached_computed_planes: Option<Vec<FphaPlane>> =
            if source == ProductionModelSource::ComputedFromGeometry {
                let fit_result =
                    fit_planes_for_hydro(hydro, config_entry, &geometry_map, &study_stages)?;
                if let Some(kappa) = fit_result.low_kappa_warning {
                    kappa_warnings.push((hydro.name.clone(), kappa));
                }
                for (plane_id, plane) in fit_result.planes.iter().enumerate() {
                    let raw_gamma_0 = plane.intercept / fit_result.kappa;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    export_rows.push(cobre_io::FphaHyperplaneRow {
                        hydro_id: hydro.id,
                        stage_id: None,
                        plane_id: plane_id as i32,
                        gamma_0: raw_gamma_0,
                        gamma_v: plane.gamma_v,
                        gamma_q: plane.gamma_q,
                        gamma_s: plane.gamma_s,
                        kappa: fit_result.kappa,
                        valid_v_min_hm3: None,
                        valid_v_max_hm3: None,
                        valid_q_max_m3s: None,
                    });
                }
                Some(fit_result.planes)
            } else {
                None
            };

        let mut stage_models: Vec<ResolvedProductionModel> = Vec::with_capacity(n_stages);
        for stage in &study_stages {
            let model = resolve_stage_model(
                hydro,
                stage,
                config_entry,
                source,
                &hyperplane_map,
                cached_computed_planes.as_deref(),
            )?;
            stage_models.push(model);
        }

        all_models.push(stage_models);
    }

    // Write exported hyperplane rows for computed-source hydros when any exist.
    if !export_rows.is_empty() {
        let export_path = case_dir
            .join("output")
            .join("hydro_models")
            .join("fpha_hyperplanes.parquet");
        cobre_io::output::write_fpha_hyperplanes(&export_path, &export_rows)
            .map_err(|e| SddpError::Validation(e.to_string()))?;
    }

    let set = ProductionModelSet::new(all_models, n_hydros, n_stages);
    Ok((set, provenance, kappa_warnings))
}

/// Load precomputed FPHA hyperplane rows from disk when any config uses `source: "precomputed"`.
///
/// Returns an empty vector when no precomputed hyperplanes are needed.
fn load_precomputed_hyperplanes(
    prod_configs: &[ProductionModelConfig],
    manifest: &cobre_io::FileManifest,
    case_dir: &Path,
) -> Result<Vec<FphaHyperplaneRow>, SddpError> {
    if !prod_configs.iter().any(config_uses_precomputed_fpha) {
        return Ok(Vec::new());
    }
    let hp_path = if manifest.system_fpha_hyperplanes_parquet {
        Some(case_dir.join("system").join("fpha_hyperplanes.parquet"))
    } else {
        None
    };
    Ok(cobre_io::extensions::load_fpha_hyperplanes(
        hp_path.as_deref(),
    )?)
}

/// Load geometry rows from disk when any config uses `source: "computed"`.
///
/// Returns an empty vector when no computed-source hydros exist.
fn load_geometry_rows(
    prod_configs: &[ProductionModelConfig],
    manifest: &cobre_io::FileManifest,
    case_dir: &Path,
) -> Result<Vec<HydroGeometryRow>, SddpError> {
    if !prod_configs.iter().any(config_uses_computed_fpha) {
        return Ok(Vec::new());
    }
    let geo_path = if manifest.system_hydro_geometry_parquet {
        Some(case_dir.join("system").join("hydro_geometry.parquet"))
    } else {
        None
    };
    Ok(cobre_io::extensions::load_hydro_geometry(
        geo_path.as_deref(),
    )?)
}

/// Build an `O(1)` geometry map: `hydro_id → sorted geometry row references`.
fn build_geometry_map(
    geometry_rows: &[HydroGeometryRow],
) -> HashMap<EntityId, Vec<&HydroGeometryRow>> {
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    for row in geometry_rows {
        geometry_map.entry(row.hydro_id).or_default().push(row);
    }
    for rows in geometry_map.values_mut() {
        rows.sort_by(|a, b| {
            a.volume_hm3
                .partial_cmp(&b.volume_hm3)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    geometry_map
}

/// Fit FPHA planes from geometry for a computed-source hydro.
/// Validates prerequisites (tailrace, losses, efficiency present), then calls
/// `fit_fpha_planes`. Returns planes, kappa, and warnings for caching per hydro.
fn fit_planes_for_hydro(
    hydro: &cobre_core::entities::hydro::Hydro,
    config_entry: Option<&ProductionModelConfig>,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    study_stages: &[&cobre_core::temporal::Stage],
) -> Result<FphaFitResult, SddpError> {
    validate_computed_prerequisites(hydro, geometry_map)?;

    // Use the first study stage as representative for FphaConfig lookup.
    // In the MVP, the fitting result is stage-independent.
    let representative_stage = study_stages.first().ok_or_else(|| {
        SddpError::Validation(format!(
            "hydro {} (id={}) has source: \"computed\" but the system has no study stages",
            hydro.name, hydro.id.0
        ))
    })?;

    let config = config_entry
        .and_then(|c| find_fpha_config_for_stage(c, representative_stage))
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "hydro {} (id={}) has source: \"computed\" but no FphaConfig \
                 was found in hydro_production_models.json",
                hydro.name, hydro.id.0
            ))
        })?;

    // Clone geometry rows from map to satisfy fit_fpha_planes signature.
    let geo_rows_owned: Vec<HydroGeometryRow> = geometry_map
        .get(&hydro.id)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .map(|r| (*r).clone())
        .collect();

    Ok(fit_fpha_planes(&geo_rows_owned, hydro, config)?)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Return `true` if the config entry uses `source: "precomputed"` FPHA in any
/// stage range or season entry.
fn config_uses_precomputed_fpha(config: &ProductionModelConfig) -> bool {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| {
            r.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "precomputed")
        }),
        SelectionMode::Seasonal { seasons, .. } => seasons.iter().any(|s| {
            s.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "precomputed")
        }),
    }
}

/// Return `true` if the config entry uses `source: "computed"` FPHA in any
/// stage range or season entry.
fn config_uses_computed_fpha(config: &ProductionModelConfig) -> bool {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| {
            r.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "computed")
        }),
        SelectionMode::Seasonal { seasons, .. } => seasons.iter().any(|s| {
            s.fpha_config
                .as_ref()
                .is_some_and(|f| f.source == "computed")
        }),
    }
}

/// Extract the [`FphaConfig`] that applies to a given stage from a [`ProductionModelConfig`].
///
/// Returns `None` when no stage range or season entry covers the stage, or when
/// the matched entry has no `fpha_config` field.
fn find_fpha_config_for_stage<'a>(
    config: &'a ProductionModelConfig,
    stage: &cobre_core::temporal::Stage,
) -> Option<&'a FphaConfig> {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => {
            for range in ranges {
                let after_start = stage.id >= range.start_stage_id;
                let before_end = range.end_stage_id.is_none_or(|end| stage.id <= end);
                if after_start && before_end {
                    return range.fpha_config.as_ref();
                }
            }
            None
        }
        SelectionMode::Seasonal {
            default_model: _,
            seasons,
        } => {
            if let Some(season_id) = stage.season_id {
                for season in seasons {
                    if i32::try_from(season_id).is_ok_and(|sid| sid == season.season_id) {
                        return season.fpha_config.as_ref();
                    }
                }
            }
            None
        }
    }
}

/// Validate that a hydro with `source: "computed"` has all required model fields and geometry.
///
/// Checks that `tailrace`, `hydraulic_losses`, and `efficiency` are all `Some`, and
/// that at least one geometry row exists for this hydro in the geometry map.
///
/// # Policy rationale
///
/// Although the production function math can handle `None` for each of these
/// fields (zero tailrace, lossless penstock, 100% efficiency as defaults),
/// requiring all three as `Some` ensures the reservoir geometry was **fully
/// characterized** before committing to the computed FPHA path.  Accepting
/// partial geometry risks producing envelopes that are physically inconsistent
/// with the operator's intent and hard to diagnose after the fact.  Any hydro
/// that genuinely has no tailrace, lossless penstock, or a perfect turbine
/// must declare this explicitly by providing the respective model with an
/// appropriate constant or polynomial value.
///
/// # Errors
///
/// Returns `SddpError::Validation` listing the first missing prerequisite found,
/// including the hydro name and id.
fn validate_computed_prerequisites(
    hydro: &cobre_core::entities::hydro::Hydro,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
) -> Result<(), SddpError> {
    let missing = if hydro.tailrace.is_none() {
        Some("tailrace")
    } else if hydro.hydraulic_losses.is_none() {
        Some("hydraulic_losses")
    } else if hydro.efficiency.is_none() {
        Some("efficiency")
    } else if geometry_map.get(&hydro.id).is_none_or(Vec::is_empty) {
        Some("geometry data")
    } else {
        None
    };

    if let Some(missing_item) = missing {
        return Err(SddpError::Validation(format!(
            "hydro {} (id={}) has source: \"computed\" but is missing {}. \
             Computed FPHA fitting requires tailrace, hydraulic_losses, \
             efficiency, and geometry data.",
            hydro.name, hydro.id.0, missing_item
        )));
    }

    Ok(())
}

/// Determine the [`ProductionModelSource`] for one hydro.
///
/// This checks only the high-level source classification without building the
/// per-stage model data; it is called once per hydro before the per-stage loop.
///
/// The function also rejects unsupported cases early to give clear errors before
/// any expensive Parquet loading occurs.
fn determine_source(
    hydro: &cobre_core::entities::hydro::Hydro,
    config_entry: Option<&ProductionModelConfig>,
) -> Result<ProductionModelSource, SddpError> {
    if let Some(config) = config_entry {
        // Config present: determine from the selection mode.
        // We look at whether any range/season uses "precomputed" or "computed".
        // "computed" is rejected immediately.
        let computed_range = match &config.selection_mode {
            SelectionMode::StageRanges { ranges } => ranges
                .iter()
                .find(|r| {
                    r.fpha_config
                        .as_ref()
                        .is_some_and(|f| f.source == "computed")
                })
                .map(|r| r.model.clone()),
            SelectionMode::Seasonal { seasons, .. } => seasons
                .iter()
                .find(|s| {
                    s.fpha_config
                        .as_ref()
                        .is_some_and(|f| f.source == "computed")
                })
                .map(|s| s.model.clone()),
        };
        if computed_range.is_some() {
            return Ok(ProductionModelSource::ComputedFromGeometry);
        }
        // Only "precomputed" FPHA entries remain.
        let has_fpha = match &config.selection_mode {
            SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| r.model == "fpha"),
            SelectionMode::Seasonal { seasons, .. } => seasons.iter().any(|s| s.model == "fpha"),
        };
        Ok(if has_fpha {
            ProductionModelSource::PrecomputedHyperplanes
        } else {
            ProductionModelSource::DefaultConstant
        })
    } else {
        // No config entry: use HydroGenerationModel from entity.
        match &hydro.generation_model {
            HydroGenerationModel::ConstantProductivity { .. }
            | HydroGenerationModel::LinearizedHead { .. } => {
                Ok(ProductionModelSource::DefaultConstant)
            }
            HydroGenerationModel::Fpha => Err(SddpError::Validation(format!(
                "hydro {} (id={}) has generation_model: \"fpha\" in hydros.json \
                 but no entry in hydro_production_models.json. \
                 Add an entry with source: \"precomputed\" to specify the hyperplane source.",
                hydro.name, hydro.id.0
            ))),
        }
    }
}

/// Resolve the production model for one (hydro, stage) pair.
///
/// `cached_computed_planes` carries planes already fitted by the outer loop
/// when `source == ComputedFromGeometry`. Passing pre-fitted planes avoids
/// running the fitting pipeline once per stage; the outer loop fits once per
/// hydro and clones for every stage via this parameter.
fn resolve_stage_model(
    hydro: &cobre_core::entities::hydro::Hydro,
    stage: &cobre_core::temporal::Stage,
    config_entry: Option<&ProductionModelConfig>,
    source: ProductionModelSource,
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
    cached_computed_planes: Option<&[FphaPlane]>,
) -> Result<ResolvedProductionModel, SddpError> {
    if let Some(config) = config_entry {
        let model_info = find_model_for_stage(config, stage);

        if model_info.as_ref().map(|(name, _)| name.as_str()) == Some("fpha") {
            if source == ProductionModelSource::ComputedFromGeometry {
                // Use the pre-fitted planes from the outer loop cache.
                let planes = cached_computed_planes
                    .ok_or_else(|| {
                        SddpError::Validation(format!(
                            "hydro {} (id={}) is ComputedFromGeometry but no cached planes \
                             were provided to resolve_stage_model",
                            hydro.name, hydro.id.0
                        ))
                    })?
                    .to_vec();
                let turbined_cost = hydro.penalties.fpha_turbined_cost;
                Ok(ResolvedProductionModel::Fpha {
                    planes,
                    turbined_cost,
                })
            } else {
                build_fpha_model(hydro, stage, source, hyperplane_map)
            }
        } else {
            // "constant_productivity" or "linearized_head" from config:
            // use override if provided, otherwise entity productivity.
            let productivity = model_info
                .and_then(|(_, ovr)| ovr)
                .unwrap_or_else(|| entity_productivity(hydro));
            Ok(ResolvedProductionModel::ConstantProductivity { productivity })
        }
    } else {
        // No config entry: use entity generation model.
        match &hydro.generation_model {
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s,
            }
            | HydroGenerationModel::LinearizedHead {
                productivity_mw_per_m3s,
            } => Ok(ResolvedProductionModel::ConstantProductivity {
                productivity: *productivity_mw_per_m3s,
            }),
            // Fpha without config is already rejected by determine_source().
            HydroGenerationModel::Fpha => unreachable!(
                "Fpha entity without config entry should have been rejected in determine_source"
            ),
        }
    }
}

/// Find the model name and optional productivity override for a given stage.
///
/// Returns `None` when the config has no entry covering the given stage (gap in coverage).
/// For `StageRanges`, the match is `start_stage_id <= stage.id <= end_stage_id`.
/// For `Seasonal`, the match is by `season_id == stage.season_id`.
fn find_model_for_stage(
    config: &ProductionModelConfig,
    stage: &cobre_core::temporal::Stage,
) -> Option<(String, Option<f64>)> {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => {
            for range in ranges {
                let after_start = stage.id >= range.start_stage_id;
                let before_end = range.end_stage_id.is_none_or(|end| stage.id <= end);
                if after_start && before_end {
                    return Some((range.model.clone(), range.productivity_override));
                }
            }
            None
        }
        SelectionMode::Seasonal {
            default_model,
            seasons,
        } => {
            if let Some(season_id) = stage.season_id {
                for season in seasons {
                    // season.season_id is i32; stage.season_id is usize.
                    // Convert usize to i32 for comparison to avoid cast_sign_loss.
                    if i32::try_from(season_id).is_ok_and(|sid| sid == season.season_id) {
                        return Some((season.model.clone(), season.productivity_override));
                    }
                }
            }
            // Fall back to default model when no season matches (or no season_id on stage).
            // Default model has no override.
            Some((default_model.clone(), None))
        }
    }
}

/// Return the productivity field from the hydro entity regardless of which
/// `HydroGenerationModel` variant is active.
///
/// Used when a config entry explicitly assigns `"constant_productivity"` or
/// `"linearized_head"` to a stage that would otherwise use the entity model.
fn entity_productivity(hydro: &cobre_core::entities::hydro::Hydro) -> f64 {
    match &hydro.generation_model {
        HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s,
        }
        | HydroGenerationModel::LinearizedHead {
            productivity_mw_per_m3s,
        } => *productivity_mw_per_m3s,
        // Fpha entity model has no productivity scalar — use 0.0 as a safe
        // placeholder; this case is only reached when the config explicitly
        // assigns a non-FPHA model to a stage for an Fpha entity, which is
        // unusual but not an error (the caller chose to override via config).
        HydroGenerationModel::Fpha => 0.0,
    }
}

/// Build an `Fpha` variant `ResolvedProductionModel` for one (hydro, stage) pair.
///
/// Looks up hyperplanes for `(hydro_id, Some(stage.id))` first; falls back to
/// `(hydro_id, None)` when no stage-specific rows exist (global all-stage rows).
/// Validates each hyperplane's coefficients and `kappa`, then constructs
/// [`FphaPlane`] with the pre-scaled intercept `gamma_0 * kappa`.
fn build_fpha_model(
    hydro: &cobre_core::entities::hydro::Hydro,
    stage: &cobre_core::temporal::Stage,
    _source: ProductionModelSource,
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
) -> Result<ResolvedProductionModel, SddpError> {
    // Prefer stage-specific rows; fall back to global (stage_id: None) rows.
    let rows: &[&FphaHyperplaneRow] = hyperplane_map
        .get(&(hydro.id, Some(stage.id)))
        .or_else(|| hyperplane_map.get(&(hydro.id, None)))
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "hydro {} (id={}) is configured as FPHA but has no hyperplane rows \
             in fpha_hyperplanes.parquet for stage {} (and no global all-stage rows).",
                hydro.name, hydro.id.0, stage.id
            ))
        })?;

    if rows.is_empty() {
        return Err(SddpError::Validation(format!(
            "hydro {} (id={}) has zero hyperplane rows for stage {}.",
            hydro.name, hydro.id.0, stage.id
        )));
    }

    let mut planes: Vec<FphaPlane> = Vec::with_capacity(rows.len());
    for row in rows {
        validate_hyperplane_row(hydro, stage, row)?;
        planes.push(FphaPlane {
            intercept: row.gamma_0 * row.kappa,
            gamma_v: row.gamma_v,
            gamma_q: row.gamma_q,
            gamma_s: row.gamma_s,
        });
    }

    let turbined_cost = hydro.penalties.fpha_turbined_cost;
    Ok(ResolvedProductionModel::Fpha {
        planes,
        turbined_cost,
    })
}

/// Validate the physical constraints for one `FphaHyperplaneRow`.
///
/// Returns `Err(SddpError::Validation(...))` when any constraint is violated.
///
/// Constraints:
///
/// - `gamma_v >= 0` — higher storage must not decrease generation; zero is valid
///   for constant-head plants where head does not depend on volume
/// - `gamma_s <= 0` — spillage reduces generation
/// - `gamma_q > 0` — more turbined flow → more generation
/// - `kappa ∈ (0, 1]` — correction factor range
fn validate_hyperplane_row(
    hydro: &cobre_core::entities::hydro::Hydro,
    stage: &cobre_core::temporal::Stage,
    row: &FphaHyperplaneRow,
) -> Result<(), SddpError> {
    let ctx = format!(
        "hydro {} (id={}) plane {} stage {}",
        hydro.name, hydro.id.0, row.plane_id, stage.id
    );

    if row.gamma_v < 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_v must be >= 0 (higher storage must not decrease generation; \
             zero is valid for constant-head plants), got gamma_v = {}",
            row.gamma_v
        )));
    }

    if row.gamma_s > 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_s must be <= 0 (spillage reduces generation), \
             got gamma_s = {}",
            row.gamma_s
        )));
    }

    if row.gamma_q <= 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_q must be > 0 (more turbined flow → more generation), \
             got gamma_q = {}",
            row.gamma_q
        )));
    }

    if row.kappa <= 0.0 || row.kappa > 1.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: kappa must be in (0, 1] (correction factor range), \
             got kappa = {}",
            row.kappa
        )));
    }

    Ok(())
}

// ── Evaporation model resolution ──────────────────────────────────────────────

/// Resolve per-hydro linearized evaporation models from reservoir geometry.
///
/// Scans `system.hydros()` for plants with `evaporation_coefficients_mm` set.
/// When none are found, returns `EvaporationModel::None` for every hydro
/// without touching the filesystem. When any hydros have evaporation
/// coefficients, loads `system/hydro_geometry.parquet` and computes a
/// first-order Taylor linearization around the operating midpoint.
///
/// The linearized evaporation model for stage `t` is:
///
/// ```text
/// Q_ev = k_evap0 + k_evap_v * v
/// ```
///
/// where:
///
/// ```text
/// zeta  = 1.0 / (3.6 * stage_hours)         -- mm·km² → m³/s
/// k_evap_v = zeta * c_ev[month] * dA/dv|_{v_ref}
/// k_evap0  = zeta * c_ev[month] * A(v_ref) - k_evap_v * v_ref
/// ```
///
/// `v_ref = (v_min + v_max) / 2` is the linearization reference volume.
/// `stage_hours` is the sum of all block durations in the stage.
/// `month` is the 0-based calendar month from `stage.season_id` (0 = January).
///
/// # Errors
///
/// | Condition                                                        | Error variant             |
/// | ---------------------------------------------------------------- | ------------------------- |
/// | Hydro has evaporation coefficients but no geometry rows          | [`SddpError::Validation`] |
/// | All geometry `area_km2` values are zero                          | [`SddpError::Validation`] |
/// | Computed `k_evap_v` or `k_evap0` is NaN or infinite             | [`SddpError::Validation`] |
/// | Stage has no `season_id` (cannot map to a month)                 | [`SddpError::Validation`] |
/// | I/O failure loading geometry Parquet                             | [`SddpError::Io`]         |
#[allow(clippy::type_complexity)]
pub fn resolve_evaporation_models(
    system: &System,
    case_dir: &Path,
) -> Result<
    (
        EvaporationModelSet,
        Vec<(EntityId, EvaporationSource)>,
        Vec<(EntityId, EvaporationReferenceSource)>,
    ),
    SddpError,
> {
    // ── Step 1: scan for any hydro that needs evaporation ─────────────────────
    let any_evaporation = system
        .hydros()
        .iter()
        .any(|h| h.evaporation_coefficients_mm.is_some());

    // ── Step 2: early return when no hydro has evaporation ────────────────────
    if !any_evaporation {
        let models = system
            .hydros()
            .iter()
            .map(|_| EvaporationModel::None)
            .collect();
        let provenance = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationSource::NotModeled))
            .collect();
        let reference_sources = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        return Ok((
            EvaporationModelSet::new(models),
            provenance,
            reference_sources,
        ));
    }

    // ── Step 3: load hydro_geometry.parquet ───────────────────────────────────
    let mut ctx = cobre_io::ValidationContext::new();
    let manifest = cobre_io::validate_structure(case_dir, &mut ctx);
    let geometry_path = if manifest.system_hydro_geometry_parquet {
        Some(case_dir.join("system").join("hydro_geometry.parquet"))
    } else {
        None
    };
    let geometry_rows = cobre_io::extensions::load_hydro_geometry(geometry_path.as_deref())?;

    // ── Step 4: group geometry rows by hydro_id ───────────────────────────────
    let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
        HashMap::new();
    for row in &geometry_rows {
        geometry_map.entry(row.hydro_id).or_default().push(row);
    }
    // Sort each hydro's rows by volume_hm3 ascending (should already be sorted,
    // but guarantee it here since we rely on sorted order for interpolation).
    for rows in geometry_map.values_mut() {
        rows.sort_by(|a, b| {
            a.volume_hm3
                .partial_cmp(&b.volume_hm3)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // ── Step 5: collect study stages (id >= 0) ────────────────────────────────
    let study_stages: Vec<&cobre_core::temporal::Stage> =
        system.stages().iter().filter(|s| s.id >= 0).collect();

    // ── Step 6: delegate to core logic (testable without disk I/O) ───────────
    resolve_evaporation_core(system.hydros(), &geometry_map, &study_stages)
}

/// Core evaporation linearization logic, operating on pre-loaded data.
///
/// Separated from [`resolve_evaporation_models`] so that unit tests can exercise
/// the resolution logic without loading files from disk. Takes pre-loaded, grouped
/// geometry rows and the ordered set of study stages.
///
/// # Errors
///
/// Same error conditions as [`resolve_evaporation_models`].
#[allow(clippy::type_complexity, clippy::too_many_lines)]
fn resolve_evaporation_core(
    hydros: &[cobre_core::entities::hydro::Hydro],
    geometry_map: &HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>>,
    study_stages: &[&cobre_core::temporal::Stage],
) -> Result<
    (
        EvaporationModelSet,
        Vec<(EntityId, EvaporationSource)>,
        Vec<(EntityId, EvaporationReferenceSource)>,
    ),
    SddpError,
> {
    let n_stages = study_stages.len();
    let mut all_models: Vec<EvaporationModel> = Vec::with_capacity(hydros.len());
    let mut provenance: Vec<(EntityId, EvaporationSource)> = Vec::with_capacity(hydros.len());
    let mut reference_provenance: Vec<(EntityId, EvaporationReferenceSource)> =
        Vec::with_capacity(hydros.len());

    for hydro in hydros {
        let Some(coefficients_mm) = hydro.evaporation_coefficients_mm else {
            all_models.push(EvaporationModel::None);
            provenance.push((hydro.id, EvaporationSource::NotModeled));
            reference_provenance.push((hydro.id, EvaporationReferenceSource::DefaultMidpoint));
            continue;
        };

        // ── Look up geometry rows ─────────────────────────────────────────────
        let geo_rows: &[&cobre_io::extensions::HydroGeometryRow] =
            geometry_map.get(&hydro.id).map_or(&[], Vec::as_slice);

        if geo_rows.is_empty() {
            return Err(SddpError::Validation(format!(
                "hydro {} (id={}) has evaporation_coefficients_mm but no geometry data \
                 in hydro_geometry.parquet. Evaporation linearization requires \
                 area-volume curve data.",
                hydro.name, hydro.id.0
            )));
        }

        // ── Verify area_km2 values are not all zero ───────────────────────────
        let all_zero_area = geo_rows.iter().all(|r| r.area_km2 == 0.0);
        if all_zero_area {
            return Err(SddpError::Validation(format!(
                "hydro {} (id={}) has evaporation_coefficients_mm but all area_km2 \
                 values in hydro_geometry.parquet are zero. \
                 Evaporation linearization requires non-zero surface area data.",
                hydro.name, hydro.id.0
            )));
        }

        // ── Determine reference volume strategy ───────────────────────────────
        // When the hydro supplies per-season reference volumes, use them per
        // stage (compute A and dA/dv inside the loop). Otherwise fall back to
        // the midpoint once outside the loop.
        let ref_source = if hydro.evaporation_reference_volumes_hm3.is_some() {
            EvaporationReferenceSource::UserSupplied
        } else {
            EvaporationReferenceSource::DefaultMidpoint
        };

        // For the midpoint path, pre-compute v_ref / A(v_ref) / dA/dv once.
        // These are only accessed when evaporation_reference_volumes_hm3 is None,
        // so the values are always initialised before use.
        let midpoint_v = f64::midpoint(hydro.min_storage_hm3, hydro.max_storage_hm3);
        let midpoint_area = if hydro.evaporation_reference_volumes_hm3.is_none() {
            interpolate_area(geo_rows, midpoint_v)
        } else {
            0.0 // unused; per-season path computes per stage
        };
        let midpoint_slope = if hydro.evaporation_reference_volumes_hm3.is_none() {
            area_derivative(geo_rows, midpoint_v)
        } else {
            0.0 // unused; per-season path computes per stage
        };

        // ── Compute per-stage coefficients ────────────────────────────────────
        let mut stage_coefficients: Vec<LinearizedEvaporation> = Vec::with_capacity(n_stages);
        let mut stage_ref_volumes: Vec<f64> = Vec::with_capacity(n_stages);

        for stage in study_stages {
            let month_index = stage.season_id.ok_or_else(|| {
                SddpError::Validation(format!(
                    "stage {} has no season_id and cannot be mapped to a calendar month \
                     for evaporation coefficient lookup (hydro {} id={}). \
                     All study stages must have a season_id for evaporation modeling.",
                    stage.id, hydro.name, hydro.id.0
                ))
            })?;

            if month_index >= 12 {
                return Err(SddpError::Validation(format!(
                    "stage {} has season_id={month_index} which is outside [0, 11]. \
                     Evaporation coefficient arrays have 12 entries (one per calendar month). \
                     (hydro {} id={})",
                    stage.id, hydro.name, hydro.id.0
                )));
            }

            let c_ev = coefficients_mm[month_index];

            // Resolve v_ref, a_ref, da_dv for this stage.
            let (v_ref, a_ref, da_dv) =
                if let Some(ref_vols) = hydro.evaporation_reference_volumes_hm3 {
                    // Per-season path: look up the reference volume for this month.
                    let v = ref_vols[month_index];
                    (
                        v,
                        interpolate_area(geo_rows, v),
                        area_derivative(geo_rows, v),
                    )
                } else {
                    // Midpoint path: use the values pre-computed outside the loop.
                    (midpoint_v, midpoint_area, midpoint_slope)
                };

            // Total stage duration in hours (sum of all block durations).
            let stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();

            let zeta_evap = 1.0 / (3.6 * stage_hours);

            let k_evap_v = zeta_evap * c_ev * da_dv;
            let k_evap0 = zeta_evap * c_ev * a_ref - k_evap_v * v_ref;

            // Validate finiteness (catches degenerate geometry or zero-duration stages).
            if !k_evap_v.is_finite() {
                return Err(SddpError::Validation(format!(
                    "hydro {} (id={}) stage {}: computed k_evap_v = {k_evap_v} is not \
                     finite. Check geometry data for degenerate area-volume curve points.",
                    hydro.name, hydro.id.0, stage.id
                )));
            }
            if !k_evap0.is_finite() {
                return Err(SddpError::Validation(format!(
                    "hydro {} (id={}) stage {}: computed k_evap0 = {k_evap0} is not \
                     finite. Check geometry data for degenerate area-volume curve points.",
                    hydro.name, hydro.id.0, stage.id
                )));
            }

            stage_coefficients.push(LinearizedEvaporation { k_evap0, k_evap_v });
            stage_ref_volumes.push(v_ref);
        }

        all_models.push(EvaporationModel::Linearized {
            coefficients: stage_coefficients,
            reference_volumes_hm3: stage_ref_volumes,
        });
        provenance.push((hydro.id, EvaporationSource::LinearizedFromGeometry));
        reference_provenance.push((hydro.id, ref_source));
    }

    Ok((
        EvaporationModelSet::new(all_models),
        provenance,
        reference_provenance,
    ))
}

// ── Evaporation geometry helpers ──────────────────────────────────────────────

/// Linearly interpolate reservoir surface area at volume `v` from the sorted
/// geometry table.
///
/// When `v` is below the first point or above the last point, returns the
/// area at the first or last point respectively (clamping, not extrapolation).
/// When `v` falls exactly on a geometry point, returns that point's area.
/// Between two points, performs linear interpolation.
///
/// Assumes `geometry` is sorted by `volume_hm3` ascending and non-empty.
/// Returns `0.0` for an empty geometry slice.
fn interpolate_area(geometry: &[&cobre_io::extensions::HydroGeometryRow], v: f64) -> f64 {
    if geometry.is_empty() {
        return 0.0;
    }

    let n = geometry.len();

    // Clamp below the first point.
    if v <= geometry[0].volume_hm3 {
        return geometry[0].area_km2;
    }

    // Clamp above the last point.
    if v >= geometry[n - 1].volume_hm3 {
        return geometry[n - 1].area_km2;
    }

    // Binary search for the interval [i, i+1] that straddles v.
    // We know v is strictly between geometry[0] and geometry[n-1].
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = lo.midpoint(hi);
        if geometry[mid].volume_hm3 <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let v0 = geometry[lo].volume_hm3;
    let v1 = geometry[hi].volume_hm3;
    let a0 = geometry[lo].area_km2;
    let a1 = geometry[hi].area_km2;

    // Linear interpolation: A(v) = A0 + (A1 - A0) * (v - v0) / (v1 - v0).
    // Guard against identical volume points (degenerate geometry).
    let dv = v1 - v0;
    if dv == 0.0 {
        return a0;
    }

    a0 + (a1 - a0) * (v - v0) / dv
}

/// Compute the finite-difference derivative `dA/dv` at volume `v` from the
/// sorted geometry table.
///
/// Uses the slope of the enclosing interval `[i, i+1]`. When `v` is at or
/// below the first point, uses the slope between the first and second points.
/// When `v` is at or above the last point, uses the slope between the last two
/// points. For a single-point geometry, returns `0.0` (no gradient information).
///
/// Assumes `geometry` is sorted by `volume_hm3` ascending.
fn area_derivative(geometry: &[&cobre_io::extensions::HydroGeometryRow], v: f64) -> f64 {
    let n = geometry.len();

    if n < 2 {
        // Single-point or empty geometry: no gradient information.
        return 0.0;
    }

    // Determine the interval to use for the finite difference.
    let (lo, hi) = if v <= geometry[0].volume_hm3 {
        // At or below the first point: use the first interval.
        (0, 1)
    } else if v >= geometry[n - 1].volume_hm3 {
        // At or above the last point: use the last interval.
        (n - 2, n - 1)
    } else {
        // Binary search for the enclosing interval.
        let mut l = 0usize;
        let mut r = n - 1;
        while r - l > 1 {
            let mid = l.midpoint(r);
            if geometry[mid].volume_hm3 <= v {
                l = mid;
            } else {
                r = mid;
            }
        }
        (l, r)
    };

    let dv = geometry[hi].volume_hm3 - geometry[lo].volume_hm3;
    let da = geometry[hi].area_km2 - geometry[lo].area_km2;

    // Guard against identical volume points (degenerate geometry).
    if dv == 0.0 {
        return 0.0;
    }

    da / dv
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
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::{
        Bus, DeficitSegment, EfficiencyModel, EntityId, HydraulicLossesModel, SystemBuilder,
        TailraceModel,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        scenario::CorrelationModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_io::extensions::{
        FphaConfig, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig, SeasonConfig,
        SelectionMode, StageRange,
    };

    use super::{
        EvaporationModel, EvaporationModelSet, EvaporationReferenceSource, EvaporationSource,
        FphaHydroDetail, FphaPlane, HydroModelProvenance, HydroModelSummary, LinearizedEvaporation,
        PrepareHydroModelsResult, ProductionModelSet, ProductionModelSource,
        ResolvedProductionModel, build_fpha_model, build_hydro_model_summary, determine_source,
        find_fpha_config_for_stage, find_model_for_stage, validate_computed_prerequisites,
        validate_hyperplane_row,
    };

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_stage(id: i32) -> Stage {
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
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
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn zero_penalties() -> HydroPenalties {
        HydroPenalties {
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
        }
    }

    fn make_hydro(id: i32, model: HydroGenerationModel) -> cobre_core::entities::hydro::Hydro {
        cobre_core::entities::hydro::Hydro {
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            bus_id: EntityId::from(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 100.0,
            max_storage_hm3: 2000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: model,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
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
        }
    }

    fn valid_row(hydro_id: i32, stage_id: Option<i32>, plane_id: i32) -> FphaHyperplaneRow {
        FphaHyperplaneRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            plane_id,
            gamma_0: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        }
    }

    fn precomputed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId::from(hydro_id),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "fpha".to_string(),
                    fpha_config: Some(FphaConfig {
                        source: "precomputed".to_string(),
                        volume_discretization_points: None,
                        turbine_discretization_points: None,
                        spillage_discretization_points: None,
                        max_planes_per_hydro: None,
                        fitting_window: None,
                    }),
                    productivity_override: None,
                }],
            },
        }
    }

    fn computed_fpha_config(hydro_id: i32) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId::from(hydro_id),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "fpha".to_string(),
                    fpha_config: Some(FphaConfig {
                        source: "computed".to_string(),
                        volume_discretization_points: None,
                        turbine_discretization_points: None,
                        spillage_discretization_points: None,
                        max_planes_per_hydro: None,
                        fitting_window: None,
                    }),
                    productivity_override: None,
                }],
            },
        }
    }

    // ── resolve_production_models unit tests (in-memory, no disk I/O) ─────────

    /// All-constant system with no config file: all hydros → DefaultConstant provenance.
    #[test]
    fn all_constant_no_config_returns_default_constant_provenance() {
        let hydro0 = make_hydro(
            0,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
        );
        let hydro1 = make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.8,
            },
        );

        let stage = make_stage(0);

        // Test determine_source: no config entry, ConstantProductivity entity model.
        let src0 = determine_source(&hydro0, None).expect("should succeed");
        let src1 = determine_source(&hydro1, None).expect("should succeed");
        assert_eq!(src0, ProductionModelSource::DefaultConstant);
        assert_eq!(src1, ProductionModelSource::DefaultConstant);

        // Test resolve_stage_model: constant productivity should come from entity.
        let empty_map = std::collections::HashMap::new();
        let model0 = super::resolve_stage_model(
            &hydro0,
            &stage,
            None,
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
        )
        .expect("should succeed");
        assert!(
            matches!(model0, ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.9).abs() < f64::EPSILON),
            "expected ConstantProductivity 0.9, got {model0:?}"
        );
    }

    /// LinearizedHead entity model without config → ConstantProductivity with the productivity field.
    #[test]
    fn linearized_head_entity_resolves_to_constant_productivity() {
        let hydro = make_hydro(
            0,
            HydroGenerationModel::LinearizedHead {
                productivity_mw_per_m3s: 0.75,
            },
        );
        let stage = make_stage(0);

        let src = determine_source(&hydro, None).expect("should succeed");
        assert_eq!(src, ProductionModelSource::DefaultConstant);

        let empty_map = std::collections::HashMap::new();
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            None,
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
        )
        .expect("should succeed");
        assert!(
            matches!(model, ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.75).abs() < f64::EPSILON),
            "LinearizedHead should resolve to ConstantProductivity 0.75, got {model:?}"
        );
    }

    /// Fpha entity model without config → validation error.
    #[test]
    fn fpha_entity_without_config_entry_returns_validation_error() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let err = determine_source(&hydro, None).expect_err("should fail");
        assert!(
            matches!(err, crate::SddpError::Validation(ref msg) if
                msg.contains("fpha") || msg.contains("no entry") || msg.contains("hydro_production_models")),
            "expected Validation error mentioning missing config entry, got {err:?}"
        );
    }

    /// source: "computed" in config → returns `ComputedFromGeometry` (fitting is now supported).
    #[test]
    fn computed_source_returns_computed_from_geometry() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let config = computed_fpha_config(0);

        let source = determine_source(&hydro, Some(&config)).expect("should succeed");
        assert_eq!(
            source,
            ProductionModelSource::ComputedFromGeometry,
            "expected ComputedFromGeometry, got {source:?}"
        );
    }

    /// Helper: build a minimal hydro with all computed-source prerequisites set.
    fn make_computed_hydro(id: i32) -> cobre_core::entities::hydro::Hydro {
        let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
        hydro.tailrace = Some(TailraceModel::Polynomial {
            coefficients: vec![300.0],
        });
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.02 });
        hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
        hydro
    }

    /// Helper: build a two-point VHA geometry row vector for a hydro.
    fn make_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
        vec![
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 100.0,
                height_m: 400.0,
                area_km2: 10.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 2000.0,
                height_m: 450.0,
                area_km2: 50.0,
            },
        ]
    }

    /// validate_computed_prerequisites: missing tailrace → Validation error with "tailrace" and hydro name.
    #[test]
    fn computed_source_missing_tailrace_returns_validation_error() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        // tailrace is None in make_hydro
        let rows = make_geometry_rows(0);
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
        geometry_map.insert(EntityId::from(0), row_refs);

        let err = validate_computed_prerequisites(&hydro, &geometry_map)
            .expect_err("should fail when tailrace is None");
        let msg = err.to_string();
        assert!(
            msg.contains("tailrace"),
            "error must mention 'tailrace', got: {msg}"
        );
        assert!(
            msg.contains(&hydro.name),
            "error must include hydro name '{}', got: {msg}",
            hydro.name
        );
    }

    /// validate_computed_prerequisites: missing geometry rows → Validation error with "geometry" and hydro name.
    #[test]
    fn computed_source_missing_geometry_returns_validation_error() {
        let hydro = make_computed_hydro(0);
        let empty_geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();

        let err = validate_computed_prerequisites(&hydro, &empty_geometry_map)
            .expect_err("should fail when geometry rows are absent");
        let msg = err.to_string();
        assert!(
            msg.contains("geometry"),
            "error must mention 'geometry', got: {msg}"
        );
        assert!(
            msg.contains(&hydro.name),
            "error must include hydro name '{}', got: {msg}",
            hydro.name
        );
    }

    /// find_fpha_config_for_stage: returns Some(&FphaConfig) when stage is in the range.
    #[test]
    fn find_fpha_config_for_stage_returns_config_in_range() {
        let config = computed_fpha_config(0);
        let stage = make_stage(5);

        let result = find_fpha_config_for_stage(&config, &stage);
        assert!(
            result.is_some(),
            "expected Some(FphaConfig) for stage 5, got None"
        );
        assert_eq!(
            result.expect("just checked is_some").source,
            "computed",
            "expected source 'computed'"
        );
    }

    /// find_fpha_config_for_stage: returns None when no range covers the stage.
    #[test]
    fn find_fpha_config_for_stage_returns_none_outside_range() {
        // Create a config with range [5, 10].
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 5,
                    end_stage_id: Some(10),
                    model: "fpha".to_string(),
                    fpha_config: Some(FphaConfig {
                        source: "computed".to_string(),
                        volume_discretization_points: None,
                        turbine_discretization_points: None,
                        spillage_discretization_points: None,
                        max_planes_per_hydro: None,
                        fitting_window: None,
                    }),
                    productivity_override: None,
                }],
            },
        };

        // Stage 0 is before the range [5, 10].
        let stage = make_stage(0);
        let result = find_fpha_config_for_stage(&config, &stage);
        assert!(
            result.is_none(),
            "expected None for stage 0 (outside range [5,10]), got {result:?}"
        );
    }

    /// kappa = 0.95 → intercept is gamma_0 * kappa.
    #[test]
    fn gamma_0_is_scaled_by_kappa() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: None,
            plane_id: 0,
            gamma_0: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 0.95,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };

        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            vec![&row as &FphaHyperplaneRow],
        );

        let model = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect("should build FPHA model");

        match model {
            ResolvedProductionModel::Fpha { planes, .. } => {
                assert_eq!(planes.len(), 1);
                let expected = 1000.0 * 0.95;
                assert!(
                    (planes[0].intercept - expected).abs() < 1e-10,
                    "intercept must be gamma_0 * kappa = {expected}, got {}",
                    planes[0].intercept
                );
            }
            other => panic!("expected Fpha variant, got {other:?}"),
        }
    }

    /// validate_hyperplane_row rejects negative gamma_v.
    #[test]
    fn validation_rejects_gamma_v_negative() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_v = -0.1; // invalid: must be >= 0

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("gamma_v"),
            "error must mention gamma_v, got: {msg}"
        );
    }

    /// validate_hyperplane_row accepts gamma_v == 0.0 (constant-head plant).
    #[test]
    fn validation_accepts_gamma_v_zero() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_v = 0.0; // valid: constant-head plant

        validate_hyperplane_row(&hydro, &stage, &row)
            .expect("gamma_v = 0.0 must be valid for constant-head plants");
    }

    /// validate_hyperplane_row rejects gamma_s > 0.
    #[test]
    fn validation_rejects_gamma_s_positive() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_s = 0.01;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("gamma_s"),
            "error must mention gamma_s, got: {msg}"
        );
    }

    /// validate_hyperplane_row rejects gamma_q <= 0.
    #[test]
    fn validation_rejects_gamma_q_nonpositive() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.gamma_q = 0.0;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("gamma_q"),
            "error must mention gamma_q, got: {msg}"
        );
    }

    /// validate_hyperplane_row rejects kappa = 0 (must be > 0).
    #[test]
    fn validation_rejects_kappa_zero() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.kappa = 0.0;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("kappa"),
            "error must mention kappa, got: {msg}"
        );
    }

    /// validate_hyperplane_row rejects kappa = 1.5 (must be <= 1).
    #[test]
    fn validation_rejects_kappa_above_one() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let mut row = valid_row(0, None, 0);
        row.kappa = 1.5;

        let err = validate_hyperplane_row(&hydro, &stage, &row).expect_err("should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("kappa"),
            "error must mention kappa, got: {msg}"
        );
    }

    /// Stage-specific hyperplanes (Some(stage_id)) override all-stage (None) rows.
    #[test]
    fn stage_specific_hyperplanes_override_all_stage() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        let global_row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: None,
            plane_id: 0,
            gamma_0: 500.0, // distinct intercept to identify
            gamma_v: 0.001,
            gamma_q: 0.80,
            gamma_s: -0.005,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };
        let stage_row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: Some(0),
            plane_id: 0,
            gamma_0: 900.0, // distinct intercept to identify
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };

        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            vec![&global_row as &FphaHyperplaneRow],
        );
        map.insert(
            (EntityId::from(0), Some(0i32)),
            vec![&stage_row as &FphaHyperplaneRow],
        );

        let model = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect("should succeed");

        match model {
            ResolvedProductionModel::Fpha { planes, .. } => {
                // Stage-specific row has gamma_0 = 900, global has 500; stage-specific wins.
                assert!(
                    (planes[0].intercept - 900.0).abs() < 1e-10,
                    "stage-specific intercept 900 should override global 500, got {}",
                    planes[0].intercept
                );
            }
            other => panic!("expected Fpha variant, got {other:?}"),
        }
    }

    /// All-stage hyperplanes (stage_id: None) are used when no stage-specific rows exist.
    #[test]
    fn all_stage_hyperplanes_used_when_no_stage_specific_rows() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(5); // stage id 5, no stage-specific rows for it

        let global_row = FphaHyperplaneRow {
            hydro_id: EntityId::from(0),
            stage_id: None,
            plane_id: 0,
            gamma_0: 700.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        };

        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            vec![&global_row as &FphaHyperplaneRow],
        );

        let model = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect("should succeed using global rows");

        match model {
            ResolvedProductionModel::Fpha { planes, .. } => {
                assert!(
                    (planes[0].intercept - 700.0).abs() < 1e-10,
                    "expected global intercept 700, got {}",
                    planes[0].intercept
                );
            }
            other => panic!("expected Fpha, got {other:?}"),
        }
    }

    /// Zero hyperplanes for a stage (empty rows) → validation error.
    #[test]
    fn zero_hyperplanes_for_stage_returns_validation_error() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let stage = make_stage(0);

        // Map has the key but an empty rows vec.
        let mut map = std::collections::HashMap::new();
        map.insert(
            (EntityId::from(0), None::<i32>),
            Vec::<&FphaHyperplaneRow>::new(),
        );

        let err = build_fpha_model(
            &hydro,
            &stage,
            ProductionModelSource::PrecomputedHyperplanes,
            &map,
        )
        .expect_err("should fail with zero hyperplanes");

        assert!(
            matches!(err, crate::SddpError::Validation(_)),
            "expected Validation error, got {err:?}"
        );
    }

    /// 4-hydro 12-stage system with 2 FPHA and 2 constant hydros: model(h, s) returns correct variant.
    #[test]
    fn mixed_system_model_returns_correct_variant_for_all_pairs() {
        let n_stages = 12;
        let n_hydros = 4;

        // Hydros 0 and 1 are constant; hydros 2 and 3 are FPHA.
        let mut all_models: Vec<Vec<ResolvedProductionModel>> = Vec::with_capacity(n_hydros);

        for productivity in [0.90f64, 0.91f64] {
            let row: Vec<_> = (0..n_stages)
                .map(|_| ResolvedProductionModel::ConstantProductivity { productivity })
                .collect();
            all_models.push(row);
        }
        for _h in 2..4usize {
            let row: Vec<_> = (0..n_stages)
                .map(|_| ResolvedProductionModel::Fpha {
                    planes: vec![FphaPlane {
                        intercept: 800.0,
                        gamma_v: 0.002,
                        gamma_q: 0.85,
                        gamma_s: -0.01,
                    }],
                    turbined_cost: 0.05,
                })
                .collect();
            all_models.push(row);
        }

        let set = ProductionModelSet::new(all_models, n_hydros, n_stages);

        // Constant hydros (0, 1) at all stages.
        for s in 0..n_stages {
            assert!(
                matches!(
                    set.model(0, s),
                    ResolvedProductionModel::ConstantProductivity { .. }
                ),
                "hydro 0 stage {s} must be ConstantProductivity"
            );
            assert!(
                matches!(
                    set.model(1, s),
                    ResolvedProductionModel::ConstantProductivity { .. }
                ),
                "hydro 1 stage {s} must be ConstantProductivity"
            );
        }
        // FPHA hydros (2, 3) at all stages.
        for s in 0..n_stages {
            assert!(
                matches!(set.model(2, s), ResolvedProductionModel::Fpha { .. }),
                "hydro 2 stage {s} must be Fpha"
            );
            assert!(
                matches!(set.model(3, s), ResolvedProductionModel::Fpha { .. }),
                "hydro 3 stage {s} must be Fpha"
            );
        }
    }

    /// find_model_for_stage: stage_id in range returns fpha model name.
    #[test]
    fn find_model_for_stage_returns_correct_model_name_in_range() {
        let config = precomputed_fpha_config(0);
        let stage = make_stage(3);
        let result = find_model_for_stage(&config, &stage);
        assert_eq!(result.as_ref().map(|(name, _)| name.as_str()), Some("fpha"));
    }

    /// find_model_for_stage: stage_id before start of range returns None.
    #[test]
    fn find_model_for_stage_returns_none_when_before_range_start() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 5,
                    end_stage_id: Some(10),
                    model: "fpha".to_string(),
                    fpha_config: None,
                    productivity_override: None,
                }],
            },
        };
        let stage = make_stage(3); // id 3, before start_stage_id 5
        let result = find_model_for_stage(&config, &stage);
        assert!(
            result.is_none(),
            "stage 3 is before range [5, 10], expected None"
        );
    }

    /// find_model_for_stage: end_stage_id = None covers all stages from start.
    #[test]
    fn find_model_for_stage_open_ended_range_covers_all_stages() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_override: None,
                }],
            },
        };
        for stage_id in [0, 5, 11, 100] {
            let stage = make_stage(stage_id);
            let result = find_model_for_stage(&config, &stage);
            assert_eq!(
                result.as_ref().map(|(name, _)| name.as_str()),
                Some("constant_productivity"),
                "open-ended range must cover stage {stage_id}"
            );
        }
    }

    // ── Productivity override tests ─────────────────────────────────────────

    /// resolve_stage_model uses productivity_override when present.
    #[test]
    fn resolve_stage_model_uses_productivity_override() {
        let hydro = make_hydro(
            0,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
        );
        let stage = make_stage(0);
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_override: Some(0.55),
                }],
            },
        };
        let empty_map = std::collections::HashMap::new();
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
        )
        .expect("should succeed");
        assert!(
            matches!(model, ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.55).abs() < f64::EPSILON),
            "expected ConstantProductivity 0.55 (override), got {model:?}"
        );
    }

    /// resolve_stage_model falls back to entity productivity when override is None.
    #[test]
    fn resolve_stage_model_uses_entity_productivity_when_no_override() {
        let hydro = make_hydro(
            0,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
        );
        let stage = make_stage(0);
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_override: None,
                }],
            },
        };
        let empty_map = std::collections::HashMap::new();
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::DefaultConstant,
            &empty_map,
            None,
        )
        .expect("should succeed");
        assert!(
            matches!(model, ResolvedProductionModel::ConstantProductivity { productivity }
                if (productivity - 0.9).abs() < f64::EPSILON),
            "expected ConstantProductivity 0.9 (entity), got {model:?}"
        );
    }

    /// find_model_for_stage returns override in tuple.
    #[test]
    fn find_model_for_stage_returns_override_in_tuple() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_override: Some(0.75),
                }],
            },
        };
        let stage = make_stage(0);
        let result = find_model_for_stage(&config, &stage);
        assert_eq!(
            result,
            Some(("constant_productivity".to_string(), Some(0.75)))
        );
    }

    /// Seasonal mode: find_model_for_stage returns override for matching season
    /// and None for default model.
    #[test]
    fn find_model_for_stage_seasonal_with_override() {
        let config = ProductionModelConfig {
            hydro_id: EntityId::from(0),
            selection_mode: SelectionMode::Seasonal {
                default_model: "constant_productivity".to_string(),
                seasons: vec![SeasonConfig {
                    season_id: 1,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    productivity_override: Some(0.60),
                }],
            },
        };
        // Stage with matching season_id = 1
        let mut stage_match = make_stage(0);
        stage_match.season_id = Some(1);
        let result = find_model_for_stage(&config, &stage_match);
        assert_eq!(
            result,
            Some(("constant_productivity".to_string(), Some(0.60)))
        );

        // Stage with non-matching season_id → default model, no override
        let mut stage_default = make_stage(0);
        stage_default.season_id = Some(99);
        let result = find_model_for_stage(&config, &stage_default);
        assert_eq!(result, Some(("constant_productivity".to_string(), None)));
    }

    /// precomputed config returns PrecomputedHyperplanes source.
    #[test]
    fn precomputed_config_returns_precomputed_source() {
        let hydro = make_hydro(0, HydroGenerationModel::Fpha);
        let config = precomputed_fpha_config(0);
        let src = determine_source(&hydro, Some(&config)).expect("should succeed");
        assert_eq!(src, ProductionModelSource::PrecomputedHyperplanes);
    }

    // ── Compile-time trait assertions ─────────────────────────────────────────

    #[test]
    fn fpha_plane_is_copy() {
        let plane = FphaPlane {
            intercept: 1.0,
            gamma_v: 0.1,
            gamma_q: 0.2,
            gamma_s: 0.3,
        };
        // Copy: assign to a new binding, then use the original — both must be accessible.
        let plane2 = plane;
        assert_eq!(plane.intercept, plane2.intercept);
    }

    #[test]
    fn linearized_evaporation_is_copy() {
        let coeff = LinearizedEvaporation {
            k_evap0: 0.5,
            k_evap_v: 0.01,
        };
        let coeff2 = coeff;
        assert_eq!(coeff.k_evap0, coeff2.k_evap0);
    }

    #[test]
    fn all_types_implement_debug() {
        let plane = FphaPlane {
            intercept: 1.0,
            gamma_v: 0.1,
            gamma_q: 0.2,
            gamma_s: 0.3,
        };
        let _ = format!("{plane:?}");

        let model_const = ResolvedProductionModel::ConstantProductivity { productivity: 0.95 };
        let _ = format!("{model_const:?}");

        let model_fpha = ResolvedProductionModel::Fpha {
            planes: vec![plane],
            turbined_cost: 0.01,
        };
        let _ = format!("{model_fpha:?}");

        let coeff = LinearizedEvaporation {
            k_evap0: 0.5,
            k_evap_v: 0.01,
        };
        let _ = format!("{coeff:?}");

        let evap_none = EvaporationModel::None;
        let _ = format!("{evap_none:?}");

        let evap_lin = EvaporationModel::Linearized {
            coefficients: vec![coeff],
            reference_volumes_hm3: vec![100.0],
        };
        let _ = format!("{evap_lin:?}");

        let detail = FphaHydroDetail {
            hydro_id: EntityId(1),
            name: "H1".to_string(),
            source: ProductionModelSource::PrecomputedHyperplanes,
            n_planes: 5,
        };
        let _ = format!("{detail:?}");

        let summary = HydroModelSummary {
            n_constant: 3,
            n_fpha: 1,
            total_planes: 5,
            fpha_details: vec![detail],
            n_evaporation: 2,
            n_no_evaporation: 2,
            n_user_supplied_ref: 1,
            n_default_midpoint_ref: 1,
            kappa_warnings: Vec::new(),
        };
        let _ = format!("{summary:?}");

        let prov = HydroModelProvenance {
            production_sources: vec![(EntityId(1), ProductionModelSource::DefaultConstant)],
            evaporation_sources: vec![(EntityId(1), EvaporationSource::NotModeled)],
            evaporation_reference_sources: vec![(
                EntityId(1),
                EvaporationReferenceSource::DefaultMidpoint,
            )],
        };
        let _ = format!("{prov:?}");

        let prod_set = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::ConstantProductivity {
                productivity: 0.95,
            }]],
            1,
            1,
        );
        let evap_set = EvaporationModelSet::new(vec![EvaporationModel::None]);
        let result = PrepareHydroModelsResult {
            production: prod_set,
            evaporation: evap_set,
            provenance: prov,
            kappa_warnings: Vec::new(),
        };
        let _ = format!("{result:?}");
    }

    // ── ProductionModelSet tests ──────────────────────────────────────────────

    #[test]
    fn production_model_set_model_returns_correct_variant() {
        // 2 hydros × 3 stages
        let models = vec![
            vec![
                ResolvedProductionModel::ConstantProductivity { productivity: 0.90 },
                ResolvedProductionModel::ConstantProductivity { productivity: 0.91 },
                ResolvedProductionModel::Fpha {
                    planes: vec![FphaPlane {
                        intercept: 10.0,
                        gamma_v: 0.1,
                        gamma_q: -0.5,
                        gamma_s: -0.2,
                    }],
                    turbined_cost: 0.005,
                },
            ],
            vec![
                ResolvedProductionModel::Fpha {
                    planes: vec![
                        FphaPlane {
                            intercept: 5.0,
                            gamma_v: 0.05,
                            gamma_q: -0.3,
                            gamma_s: -0.1,
                        },
                        FphaPlane {
                            intercept: 8.0,
                            gamma_v: 0.08,
                            gamma_q: -0.4,
                            gamma_s: -0.15,
                        },
                    ],
                    turbined_cost: 0.01,
                },
                ResolvedProductionModel::ConstantProductivity { productivity: 0.80 },
                ResolvedProductionModel::ConstantProductivity { productivity: 0.85 },
            ],
        ];

        let set = ProductionModelSet::new(models, 2, 3);

        // hydro 0, stage 0 → ConstantProductivity 0.90
        assert!(
            matches!(
                set.model(0, 0),
                ResolvedProductionModel::ConstantProductivity { productivity }
                    if (*productivity - 0.90).abs() < f64::EPSILON
            ),
            "model(0, 0) must be ConstantProductivity with productivity 0.90"
        );

        // hydro 0, stage 2 → Fpha with 1 plane
        assert!(
            matches!(set.model(0, 2), ResolvedProductionModel::Fpha { planes, .. } if planes.len() == 1),
            "model(0, 2) must be Fpha with 1 plane"
        );

        // hydro 1, stage 0 → Fpha with 2 planes
        assert!(
            matches!(set.model(1, 0), ResolvedProductionModel::Fpha { planes, .. } if planes.len() == 2),
            "model(1, 0) must be Fpha with 2 planes"
        );

        // hydro 1, stage 2 → ConstantProductivity 0.85
        assert!(
            matches!(
                set.model(1, 2),
                ResolvedProductionModel::ConstantProductivity { productivity }
                    if (*productivity - 0.85).abs() < f64::EPSILON
            ),
            "model(1, 2) must be ConstantProductivity with productivity 0.85"
        );
    }

    #[test]
    #[should_panic(expected = "hydro index 2 out of bounds")]
    fn production_model_set_out_of_bounds_hydro_panics_in_debug() {
        let set = ProductionModelSet::new(
            vec![
                vec![
                    ResolvedProductionModel::ConstantProductivity { productivity: 0.90 },
                    ResolvedProductionModel::ConstantProductivity { productivity: 0.91 },
                    ResolvedProductionModel::ConstantProductivity { productivity: 0.92 },
                ],
                vec![
                    ResolvedProductionModel::ConstantProductivity { productivity: 0.80 },
                    ResolvedProductionModel::ConstantProductivity { productivity: 0.81 },
                    ResolvedProductionModel::ConstantProductivity { productivity: 0.82 },
                ],
            ],
            2,
            3,
        );
        // hydro index 2 is out of bounds for n_hydros = 2 → debug_assert! fires
        let _ = set.model(2, 0);
    }

    // ── EvaporationModelSet tests ─────────────────────────────────────────────

    #[test]
    fn evaporation_model_set_has_evaporation_true_when_any_linearized() {
        let set = EvaporationModelSet::new(vec![
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![
                    LinearizedEvaporation {
                        k_evap0: 0.5,
                        k_evap_v: 0.01,
                    },
                    LinearizedEvaporation {
                        k_evap0: 0.6,
                        k_evap_v: 0.02,
                    },
                ],
                reference_volumes_hm3: vec![200.0, 200.0],
            },
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    k_evap0: 0.3,
                    k_evap_v: 0.005,
                }],
                reference_volumes_hm3: vec![50.0],
            },
        ]);

        assert!(
            set.has_evaporation(),
            "has_evaporation() must return true when at least one hydro is Linearized"
        );
    }

    #[test]
    fn evaporation_model_set_has_evaporation_false_when_all_none() {
        let set = EvaporationModelSet::new(vec![
            EvaporationModel::None,
            EvaporationModel::None,
            EvaporationModel::None,
        ]);

        assert!(
            !set.has_evaporation(),
            "has_evaporation() must return false when all hydros have None"
        );
    }

    #[test]
    fn evaporation_model_set_model_returns_correct_variant() {
        let coeff0 = LinearizedEvaporation {
            k_evap0: 1.0,
            k_evap_v: 0.1,
        };
        let coeff1 = LinearizedEvaporation {
            k_evap0: 2.0,
            k_evap_v: 0.2,
        };

        let set = EvaporationModelSet::new(vec![
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![coeff0, coeff1],
                reference_volumes_hm3: vec![100.0, 100.0],
            },
            EvaporationModel::None,
        ]);

        assert!(
            matches!(set.model(0), EvaporationModel::None),
            "model(0) must be None"
        );
        assert!(
            matches!(
                set.model(1),
                EvaporationModel::Linearized { coefficients, .. } if coefficients.len() == 2
            ),
            "model(1) must be Linearized with 2 coefficients"
        );
        assert!(
            matches!(set.model(2), EvaporationModel::None),
            "model(2) must be None"
        );
    }

    #[test]
    fn evaporation_model_set_empty_has_no_evaporation() {
        let set = EvaporationModelSet::new(vec![]);
        assert!(
            !set.has_evaporation(),
            "has_evaporation() must return false for an empty set"
        );
    }

    // ── interpolate_area unit tests ───────────────────────────────────────────

    /// Helper: build a slice of HydroGeometryRow references for interpolation tests.
    fn make_geo_rows(volume_area: &[(f64, f64)]) -> Vec<cobre_io::extensions::HydroGeometryRow> {
        volume_area
            .iter()
            .map(|&(v, a)| cobre_io::extensions::HydroGeometryRow {
                hydro_id: EntityId::from(1),
                volume_hm3: v,
                height_m: 0.0,
                area_km2: a,
            })
            .collect()
    }

    /// interpolate_area: exact match on the first geometry point returns that area.
    #[test]
    fn interpolate_area_exact_first_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 100.0);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "exact first point: expected 1.0, got {result}"
        );
    }

    /// interpolate_area: exact match on the last geometry point returns that area.
    #[test]
    fn interpolate_area_exact_last_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 300.0);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "exact last point: expected 2.0, got {result}"
        );
    }

    /// interpolate_area: exact match on a middle geometry point returns that area.
    #[test]
    fn interpolate_area_exact_middle_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 200.0);
        assert!(
            (result - 1.5).abs() < 1e-10,
            "exact middle point: expected 1.5, got {result}"
        );
    }

    /// interpolate_area: midpoint between two geometry points is linearly interpolated.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// At v=300, A(300) = 2.0 (exact match). At v=250, A(250) = 1.75 (midpoint of [1.5, 2.0]).
    #[test]
    fn interpolate_area_midpoint_between_two_points() {
        let rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 250.0);
        // Midpoint between (200, 1.5) and (300, 2.0): 1.5 + 0.5 * (2.0 - 1.5) = 1.75
        assert!(
            (result - 1.75).abs() < 1e-10,
            "midpoint: expected 1.75, got {result}"
        );
    }

    /// interpolate_area: volume below first point clamps to first area.
    #[test]
    fn interpolate_area_clamps_below_first_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 50.0);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "below first point: expected clamped area 1.0, got {result}"
        );
    }

    /// interpolate_area: volume above last point clamps to last area.
    #[test]
    fn interpolate_area_clamps_above_last_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 400.0);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "above last point: expected clamped area 2.0, got {result}"
        );
    }

    // ── area_derivative unit tests ────────────────────────────────────────────

    /// area_derivative: correct finite difference between two points spanning v.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// dA/dv at v=300 uses the interval [200, 300]: (2.0 - 1.5) / (300 - 200) = 0.005.
    #[test]
    fn area_derivative_correct_finite_difference() {
        let rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::area_derivative(&refs, 300.0);
        // Interval [200, 300]: (2.0 - 1.5) / (300 - 200) = 0.005
        assert!(
            (result - 0.005).abs() < 1e-10,
            "dA/dv at 300: expected 0.005, got {result}"
        );
    }

    /// area_derivative: single-point geometry returns 0.0.
    #[test]
    fn area_derivative_single_point_returns_zero() {
        let rows = make_geo_rows(&[(200.0, 1.5)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::area_derivative(&refs, 200.0);
        assert!(
            result.abs() < 1e-10,
            "single-point geometry: expected dA/dv = 0.0, got {result}"
        );
    }

    /// area_derivative: at or below the first point uses the first interval.
    #[test]
    fn area_derivative_at_or_below_first_point_uses_first_interval() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        // First interval: (1.5 - 1.0) / (200 - 100) = 0.005
        let result = super::area_derivative(&refs, 50.0);
        assert!(
            (result - 0.005).abs() < 1e-10,
            "below first point: expected first-interval slope 0.005, got {result}"
        );
    }

    /// area_derivative: at or above the last point uses the last interval.
    #[test]
    fn area_derivative_at_or_above_last_point_uses_last_interval() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        // Last interval: (2.0 - 1.5) / (300 - 200) = 0.005
        let result = super::area_derivative(&refs, 400.0);
        assert!(
            (result - 0.005).abs() < 1e-10,
            "above last point: expected last-interval slope 0.005, got {result}"
        );
    }

    // ── resolve_evaporation_models unit tests (in-memory, no disk I/O) ────────

    /// Helper: build a Stage with the given id and season_id (month index).
    fn make_stage_with_month(id: i32, month: usize) -> Stage {
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap_or_default(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap_or_default(),
            season_id: Some(month),
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
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Helper: build a Hydro with the given id and evaporation coefficients.
    fn make_hydro_with_evaporation(
        id: i32,
        min_storage: f64,
        max_storage: f64,
        evap_mm: Option<[f64; 12]>,
    ) -> cobre_core::entities::hydro::Hydro {
        cobre_core::entities::hydro::Hydro {
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            bus_id: EntityId::from(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: min_storage,
            max_storage_hm3: max_storage,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: evap_mm,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        }
    }

    /// resolve_evaporation_models core logic: all-no-evaporation system returns all None
    /// without geometry lookup.
    ///
    /// This test calls the internal core logic directly without loading from disk by
    /// using an empty geometry map.
    #[test]
    fn resolve_evaporation_all_none_when_no_hydro_has_coefficients() {
        let hydros = vec![
            make_hydro_with_evaporation(0, 100.0, 500.0, None),
            make_hydro_with_evaporation(1, 200.0, 1000.0, None),
        ];

        // Build the geometry map (empty, since no hydro needs it).
        let geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&hydros, &geometry_map, &stage_refs)
                .expect("should succeed for all-no-evaporation");

        assert_eq!(models.n_hydros(), 2);
        assert!(
            matches!(models.model(0), EvaporationModel::None),
            "hydro 0 must be None"
        );
        assert!(
            matches!(models.model(1), EvaporationModel::None),
            "hydro 1 must be None"
        );
        assert!(!models.has_evaporation(), "has_evaporation() must be false");
        assert_eq!(provenance.len(), 2);
        assert!(
            provenance
                .iter()
                .all(|(_, src)| *src == EvaporationSource::NotModeled)
        );
    }

    /// resolve_evaporation_models core logic: known geometry + coefficient gives correct k_evap0 and k_evap_v.
    ///
    /// Spec (acceptance criterion 2):
    ///   hydro: v_min=100, v_max=500, evaporation_coefficients_mm=[5.0; 12]
    ///   geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0]
    ///   v_ref = (100 + 500) / 2 = 300
    ///   A(300) = 2.0
    ///   dA/dv|_300 = (2.0 - 1.5) / (300 - 200) = 0.005
    ///   stage: season_id=0 (January), duration=744h
    ///   zeta = 1 / (3.6 * 744) = 1 / 2678.4
    ///   c_ev = 5.0
    ///   k_evap_v = zeta * 5.0 * 0.005
    ///   k_evap0  = zeta * 5.0 * 2.0 - k_evap_v * 300
    #[test]
    fn resolve_evaporation_known_geometry_produces_correct_coefficients() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [make_stage_with_month(0, 0)]; // January
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 1);
        assert_eq!(provenance.len(), 1);
        assert_eq!(provenance[0].1, EvaporationSource::LinearizedFromGeometry);

        match models.model(0) {
            EvaporationModel::Linearized {
                coefficients,
                reference_volumes_hm3,
            } => {
                assert_eq!(
                    reference_volumes_hm3.len(),
                    1,
                    "must have one ref volume per stage"
                );
                assert!(
                    (reference_volumes_hm3[0] - 300.0).abs() < 1e-10,
                    "v_ref must be (100+500)/2 = 300, got {}",
                    reference_volumes_hm3[0]
                );
                assert_eq!(coefficients.len(), 1);

                let v_ref = 300.0_f64;
                let a_ref = 2.0_f64;
                let da_dv = 0.005_f64;
                let c_ev = 5.0_f64;
                let stage_hours = 744.0_f64;
                let zeta = 1.0 / (3.6 * stage_hours);

                let expected_k_evap_v = zeta * c_ev * da_dv;
                let expected_k_evap0 = zeta * c_ev * a_ref - expected_k_evap_v * v_ref;

                let coeff = &coefficients[0];
                assert!(
                    (coeff.k_evap_v - expected_k_evap_v).abs() < 1e-10,
                    "k_evap_v: expected {expected_k_evap_v}, got {}",
                    coeff.k_evap_v
                );
                assert!(
                    (coeff.k_evap0 - expected_k_evap0).abs() < 1e-10,
                    "k_evap0: expected {expected_k_evap0}, got {}",
                    coeff.k_evap0
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_models core logic: negative evaporation coefficients produce valid results.
    ///
    /// Net precipitation (negative c_ev) is physically valid. k_evap_v can be negative.
    #[test]
    fn resolve_evaporation_negative_coefficient_produces_valid_results() {
        let mut evap_mm = [0.0f64; 12];
        evap_mm[0] = -3.0; // net precipitation in January
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [make_stage_with_month(0, 0)]; // January
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("negative evaporation must succeed");

        assert_eq!(provenance[0].1, EvaporationSource::LinearizedFromGeometry);

        match models.model(0) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let coeff = &coefficients[0];
                assert!(
                    coeff.k_evap_v.is_finite(),
                    "k_evap_v must be finite for negative c_ev"
                );
                assert!(
                    coeff.k_evap0.is_finite(),
                    "k_evap0 must be finite for negative c_ev"
                );
                // Negative c_ev with positive dA/dv → negative k_evap_v.
                assert!(
                    coeff.k_evap_v < 0.0,
                    "k_evap_v must be negative for net precipitation scenario"
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_models core logic: hydro with evaporation but no geometry rows
    /// returns SddpError::Validation containing "geometry".
    #[test]
    fn resolve_evaporation_missing_geometry_returns_validation_error() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        // Geometry map has no entry for hydro 0.
        let geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();

        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let err = super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
            .expect_err("missing geometry must return an error");

        match err {
            crate::SddpError::Validation(msg) => {
                assert!(
                    msg.to_lowercase().contains("geometry"),
                    "error message must mention 'geometry', got: {msg}"
                );
            }
            other => panic!("expected Validation error, got {other:?}"),
        }
    }

    /// resolve_evaporation_models core logic: 4 hydros where 2 have evaporation and 2 do not.
    ///
    /// Acceptance criterion 1: returns 2 Linearized and 2 None models.
    #[test]
    fn resolve_evaporation_mixed_system_returns_correct_model_mix() {
        let evap_mm = [5.0f64; 12];
        let hydros = vec![
            make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm)),
            make_hydro_with_evaporation(1, 200.0, 1000.0, None),
            make_hydro_with_evaporation(2, 50.0, 300.0, Some(evap_mm)),
            make_hydro_with_evaporation(3, 300.0, 2000.0, None),
        ];

        let geo_rows_h0 = make_geo_rows(&[(100.0, 1.0), (300.0, 2.0), (500.0, 3.0)]);
        let geo_rows_h2 = make_geo_rows(&[(50.0, 0.5), (175.0, 1.0), (300.0, 1.5)]);

        let refs_h0: Vec<_> = geo_rows_h0.iter().collect();
        let refs_h2: Vec<_> = geo_rows_h2.iter().collect();

        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), refs_h0);
        geometry_map.insert(EntityId::from(2), refs_h2);

        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&hydros, &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 4);
        assert!(
            matches!(models.model(0), EvaporationModel::Linearized { .. }),
            "hydro 0 must be Linearized"
        );
        assert!(
            matches!(models.model(1), EvaporationModel::None),
            "hydro 1 must be None"
        );
        assert!(
            matches!(models.model(2), EvaporationModel::Linearized { .. }),
            "hydro 2 must be Linearized"
        );
        assert!(
            matches!(models.model(3), EvaporationModel::None),
            "hydro 3 must be None"
        );

        // 2 Linearized, 2 NotModeled in provenance.
        let n_linearized = provenance
            .iter()
            .filter(|(_, s)| *s == EvaporationSource::LinearizedFromGeometry)
            .count();
        let n_not_modeled = provenance
            .iter()
            .filter(|(_, s)| *s == EvaporationSource::NotModeled)
            .count();
        assert_eq!(n_linearized, 2, "expected 2 LinearizedFromGeometry");
        assert_eq!(n_not_modeled, 2, "expected 2 NotModeled");
    }

    /// resolve_evaporation_models core logic: NaN/Inf detection from degenerate geometry
    /// (two identical volume points) returns a validation error.
    #[test]
    fn resolve_evaporation_degenerate_geometry_nan_detected() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        // Two rows with same volume but different areas: this is degenerate.
        // With dv=0, area_derivative returns 0.0, so no NaN there.
        // To force NaN we need the scenario where stage_hours=0 (zeta → Inf).
        // Test that scenario instead.
        let mut stage_zero_duration = make_stage_with_month(0, 0);
        stage_zero_duration.blocks = vec![Block {
            index: 0,
            name: "ZERO".to_string(),
            duration_hours: 0.0, // zero duration → zeta = Inf → k_evap_v = Inf
        }];

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let stage_refs = vec![&stage_zero_duration];

        let err = super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
            .expect_err("degenerate geometry (zero duration) must return an error");

        assert!(
            matches!(err, crate::SddpError::Validation(_)),
            "expected Validation error for non-finite coefficients, got {err:?}"
        );
    }

    // ── Per-season reference volume tests ────────────────────────────────────

    /// resolve_evaporation_core: user-supplied per-season reference volumes produce
    /// stage coefficients derived from the month-specific v_ref.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// ref_vols[0] = 200 (January), ref_vols[1] = 400 (February).
    /// Hydro: v_min=100, v_max=500.
    /// Stage 0: season_id=0, 744h. Stage 1: season_id=1, 672h.
    ///
    /// For stage 0 (v_ref=200): A(200)=1.5, dA/dv=(2.0-1.5)/(300-200)=0.005
    /// For stage 1 (v_ref=400): A(400)=2.5, dA/dv=(3.0-2.5)/(500-400)=0.005
    #[test]
    fn resolve_evaporation_per_season_ref_vols_produces_per_stage_coefficients() {
        let mut ref_vols = [0.0f64; 12];
        ref_vols[0] = 200.0; // January
        ref_vols[1] = 400.0; // February

        let mut hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));
        hydro.evaporation_reference_volumes_hm3 = Some(ref_vols);

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Two stages: January (744h) and February (672h).
        let stage_jan = make_stage_with_month(0, 0);
        let mut stage_feb = make_stage_with_month(1, 1);
        stage_feb.blocks = vec![Block {
            index: 0,
            name: "FEB".to_string(),
            duration_hours: 672.0,
        }];
        let stage_refs = vec![&stage_jan, &stage_feb];

        let (models, evap_provenance, ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 1);
        assert_eq!(
            evap_provenance[0].1,
            EvaporationSource::LinearizedFromGeometry
        );
        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::UserSupplied,
            "user-supplied volumes must produce UserSupplied provenance"
        );

        match models.model(0) {
            EvaporationModel::Linearized {
                coefficients,
                reference_volumes_hm3,
            } => {
                assert_eq!(coefficients.len(), 2, "must have 2 stage coefficients");
                assert_eq!(reference_volumes_hm3.len(), 2, "must have 2 ref volumes");

                // Stage 0: v_ref=200
                assert!(
                    (reference_volumes_hm3[0] - 200.0).abs() < 1e-10,
                    "stage 0 ref vol must be 200, got {}",
                    reference_volumes_hm3[0]
                );

                // Stage 1: v_ref=400
                assert!(
                    (reference_volumes_hm3[1] - 400.0).abs() < 1e-10,
                    "stage 1 ref vol must be 400, got {}",
                    reference_volumes_hm3[1]
                );

                // Verify stage 0 coefficients using v_ref=200.
                let c_ev = 5.0_f64;
                let da_dv = 0.005_f64; // same slope in both segments

                let zeta_jan = 1.0 / (3.6 * 744.0);
                let a_jan = 1.5_f64;
                let v_ref_jan = 200.0_f64;
                let expected_k_evap_v_jan = zeta_jan * c_ev * da_dv;
                let expected_k_evap0_jan =
                    zeta_jan * c_ev * a_jan - expected_k_evap_v_jan * v_ref_jan;
                assert!(
                    (coefficients[0].k_evap_v - expected_k_evap_v_jan).abs() < 1e-10,
                    "stage 0 k_evap_v: expected {expected_k_evap_v_jan}, got {}",
                    coefficients[0].k_evap_v
                );
                assert!(
                    (coefficients[0].k_evap0 - expected_k_evap0_jan).abs() < 1e-10,
                    "stage 0 k_evap0: expected {expected_k_evap0_jan}, got {}",
                    coefficients[0].k_evap0
                );

                // Verify stage 1 coefficients using v_ref=400.
                let zeta_feb = 1.0 / (3.6 * 672.0);
                let a_feb = 2.5_f64;
                let v_ref_feb = 400.0_f64;
                let expected_k_evap_v_feb = zeta_feb * c_ev * da_dv;
                let expected_k_evap0_feb =
                    zeta_feb * c_ev * a_feb - expected_k_evap_v_feb * v_ref_feb;
                assert!(
                    (coefficients[1].k_evap_v - expected_k_evap_v_feb).abs() < 1e-10,
                    "stage 1 k_evap_v: expected {expected_k_evap_v_feb}, got {}",
                    coefficients[1].k_evap_v
                );
                assert!(
                    (coefficients[1].k_evap0 - expected_k_evap0_feb).abs() < 1e-10,
                    "stage 1 k_evap0: expected {expected_k_evap0_feb}, got {}",
                    coefficients[1].k_evap0
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_core: None reference volumes produce DefaultMidpoint provenance and
    /// all reference_volumes_hm3 entries equal (v_min + v_max) / 2.
    #[test]
    fn resolve_evaporation_none_ref_vols_produces_default_midpoint_provenance() {
        // `make_hydro_with_evaporation` already sets evaporation_reference_volumes_hm3 = None.
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Two stages with different months (January = 0, June = 5).
        let stage_january = make_stage_with_month(0, 0);
        let stage_june = make_stage_with_month(1, 5);
        let stage_refs = vec![&stage_january, &stage_june];

        let (models, evap_provenance, ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::DefaultMidpoint,
            "None reference volumes must produce DefaultMidpoint provenance"
        );
        assert_eq!(
            evap_provenance[0].1,
            EvaporationSource::LinearizedFromGeometry
        );

        let expected_v_ref = f64::midpoint(100.0, 500.0); // 300.0

        match models.model(0) {
            EvaporationModel::Linearized {
                reference_volumes_hm3,
                ..
            } => {
                assert_eq!(
                    reference_volumes_hm3.len(),
                    2,
                    "must have 2 ref volumes (one per stage)"
                );
                for (s, &v) in reference_volumes_hm3.iter().enumerate() {
                    assert!(
                        (v - expected_v_ref).abs() < 1e-10,
                        "stage {s} ref vol must be midpoint {expected_v_ref}, got {v}"
                    );
                }
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_core: mixed hydro set (one with user-supplied, one without)
    /// produces correct per-hydro provenance.
    #[test]
    fn resolve_evaporation_mixed_ref_vol_provenance() {
        let mut ref_vols = [300.0f64; 12];
        ref_vols[0] = 200.0;

        let mut hydro_with = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));
        hydro_with.evaporation_reference_volumes_hm3 = Some(ref_vols);

        let hydro_without = make_hydro_with_evaporation(1, 100.0, 500.0, Some([5.0f64; 12]));
        // hydro_without.evaporation_reference_volumes_hm3 is already None.

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (300.0, 2.0), (500.0, 3.0)]);
        let refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&cobre_io::extensions::HydroGeometryRow>> =
            HashMap::new();
        geometry_map.insert(EntityId::from(0), refs.clone());
        geometry_map.insert(EntityId::from(1), refs);

        let stage = make_stage_with_month(0, 0);
        let stage_refs = vec![&stage];

        let (_, _, ref_provenance) = super::resolve_evaporation_core(
            &[hydro_with, hydro_without],
            &geometry_map,
            &stage_refs,
        )
        .expect("should succeed");

        assert_eq!(ref_provenance.len(), 2);
        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::UserSupplied,
            "hydro with ref vols must be UserSupplied"
        );
        assert_eq!(
            ref_provenance[1].1,
            EvaporationReferenceSource::DefaultMidpoint,
            "hydro without ref vols must be DefaultMidpoint"
        );
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
            .map(|&id| {
                make_hydro(
                    id,
                    HydroGenerationModel::ConstantProductivity {
                        productivity_mw_per_m3s: 0.95,
                    },
                )
            })
            .collect();
        let system = make_system_for_summary(hydros);

        let n_hydros = hydro_ids.len();
        let n_stages = 1;
        let models: Vec<Vec<ResolvedProductionModel>> = (0..n_hydros)
            .map(|_| vec![ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }])
            .collect();
        let production = ProductionModelSet::new(models, n_hydros, n_stages);
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
                    k_evap0: 1.0,
                    k_evap_v: 0.01,
                }],
                reference_volumes_hm3: vec![200.0],
            },
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    k_evap0: 1.0,
                    k_evap_v: 0.01,
                }],
                reference_volumes_hm3: vec![300.0],
            },
            EvaporationModel::None,
        ];

        let result = PrepareHydroModelsResult {
            production,
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            kappa_warnings: Vec::new(),
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

    // ── build_hydro_model_summary tests ───────────────────────────────────────

    /// Build a minimal single-bus `System` with the given hydros and one study stage.
    ///
    /// Uses bus `EntityId(10)` to match the `make_hydro` helper's `bus_id`.
    fn make_system_for_summary(
        hydros: Vec<cobre_core::entities::hydro::Hydro>,
    ) -> cobre_core::System {
        let bus = Bus {
            id: EntityId(10),
            name: "B10".to_string(),
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
        let production = ProductionModelSet::new(models, n_hydros, n_stages);
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
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            kappa_warnings: Vec::new(),
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
                                turbined_cost: 0.0,
                            }
                        } else {
                            ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }
                        }
                    })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, n_hydros, n_stages);
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
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            kappa_warnings: Vec::new(),
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
        let production = ProductionModelSet::new(models, n_hydros, n_stages);
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
                            k_evap0: 1.0,
                            k_evap_v: 0.01,
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
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            kappa_warnings: Vec::new(),
        }
    }

    /// All-constant system: n_constant = 4, n_fpha = 0, total_planes = 0.
    #[test]
    fn build_hydro_model_summary_all_constant() {
        let hydro_ids = [1i32, 2, 3, 4];
        let hydros = hydro_ids
            .iter()
            .map(|&id| {
                make_hydro(
                    id,
                    HydroGenerationModel::ConstantProductivity {
                        productivity_mw_per_m3s: 0.95,
                    },
                )
            })
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
            .map(|&id| {
                make_hydro(
                    id,
                    HydroGenerationModel::ConstantProductivity {
                        productivity_mw_per_m3s: 0.95,
                    },
                )
            })
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
    #[allow(clippy::too_many_lines)]
    fn build_hydro_model_summary_acceptance_criterion() {
        // IDs 1,2 constant; IDs 3,4 FPHA with 5 planes each.
        // IDs 1,2,3 have evaporation; ID 4 does not.
        let constant_ids = [1i32, 2];
        let fpha_ids = [3i32, 4];
        let all_ids = [1i32, 2, 3, 4];
        let hydros = all_ids
            .iter()
            .map(|&id| {
                make_hydro(
                    id,
                    HydroGenerationModel::ConstantProductivity {
                        productivity_mw_per_m3s: 0.95,
                    },
                )
            })
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
        let n_hydros = sorted.len();

        let models: Vec<Vec<ResolvedProductionModel>> = sorted
            .iter()
            .map(|(_, is_fpha)| {
                (0..n_stages)
                    .map(|_| {
                        if *is_fpha {
                            ResolvedProductionModel::Fpha {
                                planes: vec![fpha_plane; 5],
                                turbined_cost: 0.0,
                            }
                        } else {
                            ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }
                        }
                    })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, n_hydros, n_stages);
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
                            k_evap0: 1.0,
                            k_evap_v: 0.01,
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
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            kappa_warnings: Vec::new(),
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
            .map(|&id| {
                make_hydro(
                    id,
                    HydroGenerationModel::ConstantProductivity {
                        productivity_mw_per_m3s: 0.95,
                    },
                )
            })
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

    // ── Computed-source integration tests ─────────────────────────────────────

    /// Sobradinho-style hydro with all computed prerequisites, matching the known-valid
    /// fixture from `fpha_fitting.rs`. Used for end-to-end computed-source tests.
    fn make_sobradinho_computed_hydro(id: i32) -> cobre_core::entities::hydro::Hydro {
        let mut hydro = make_hydro(id, HydroGenerationModel::Fpha);
        hydro.name = format!("Sobradinho{id}");
        hydro.min_storage_hm3 = 100.0;
        hydro.max_storage_hm3 = 20_000.0;
        hydro.max_turbined_m3s = 500.0;
        hydro.tailrace = Some(TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.001_f64],
        });
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 2.0 });
        hydro.efficiency = Some(EfficiencyModel::Constant { value: 0.92 });
        hydro
    }

    /// Four-point VHA geometry rows spanning volumes 100.0 to 20_000.0 hm³ and heights
    /// 386.5 to 400.0 m. Mirrors the Sobradinho-style fixture used in `fpha_fitting.rs`.
    fn make_sobradinho_geometry_rows(hydro_id: i32) -> Vec<HydroGeometryRow> {
        vec![
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 100.0,
                height_m: 386.5,
                area_km2: 500.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 5_000.0,
                height_m: 392.0,
                area_km2: 800.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 12_000.0,
                height_m: 396.5,
                area_km2: 1_100.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(hydro_id),
                volume_hm3: 20_000.0,
                height_m: 400.0,
                area_km2: 1_400.0,
            },
        ]
    }

    /// Computed-source end-to-end: a hydro with all prerequisites and Sobradinho-style geometry
    /// produces a valid `Fpha` model with 3–10 planes and correct coefficient signs.
    ///
    /// Tests `fit_planes_for_hydro` + `resolve_stage_model` together.
    #[test]
    fn computed_source_end_to_end_produces_valid_fpha_planes() {
        let hydro = make_sobradinho_computed_hydro(0);
        let config = computed_fpha_config(0);
        let stage = make_stage(0);

        let geo_rows = make_sobradinho_geometry_rows(0);
        let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [stage.clone()];
        let stage_refs: Vec<&cobre_core::temporal::Stage> = study_stages.iter().collect();

        // Fit planes once (simulating the outer loop in resolve_production_models).
        let fit_result =
            super::fit_planes_for_hydro(&hydro, Some(&config), &geometry_map, &stage_refs)
                .expect("fit_planes_for_hydro must succeed for valid Sobradinho-style input");
        let planes = &fit_result.planes;

        // Plane count must be within the expected range for default FphaConfig.
        assert!(
            (3..=10).contains(&planes.len()),
            "expected 3–10 planes, got {}",
            planes.len()
        );

        // Coefficient signs must satisfy physical constraints.
        for (idx, plane) in planes.iter().enumerate() {
            assert!(
                plane.gamma_v > 0.0,
                "plane {idx}: gamma_v={} must be > 0",
                plane.gamma_v
            );
            assert!(
                plane.gamma_q > 0.0,
                "plane {idx}: gamma_q={} must be > 0",
                plane.gamma_q
            );
            assert!(
                plane.gamma_s <= 0.0,
                "plane {idx}: gamma_s={} must be <= 0",
                plane.gamma_s
            );
        }

        // Verify resolve_stage_model correctly wraps the cached planes.
        let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();
        let model = super::resolve_stage_model(
            &hydro,
            &stage,
            Some(&config),
            ProductionModelSource::ComputedFromGeometry,
            &empty_hyperplane_map,
            Some(planes),
        )
        .expect("resolve_stage_model must succeed for ComputedFromGeometry with cached planes");

        match model {
            ResolvedProductionModel::Fpha {
                planes: out_planes, ..
            } => {
                assert_eq!(
                    out_planes.len(),
                    planes.len(),
                    "stage model must have the same plane count as the fitted planes"
                );
            }
            other => panic!("expected Fpha variant, got {other:?}"),
        }
    }

    /// Mixed precomputed + computed sources: both hydros resolve to valid `Fpha` models and
    /// provenance is correctly differentiated by source.
    ///
    /// Hydro 0: `source: "precomputed"` with 3 manually-constructed hyperplane rows.
    /// Hydro 1: `source: "computed"` with Sobradinho-style geometry.
    #[test]
    fn mixed_precomputed_and_computed_sources_resolve_correctly() {
        // Hydro 0: precomputed FPHA.
        let hydro0 = make_hydro(0, HydroGenerationModel::Fpha);
        let config0 = precomputed_fpha_config(0);

        let precomp_row_a = valid_row(0, None, 0);
        let precomp_row_b = valid_row(0, None, 1);
        let precomp_row_c = valid_row(0, None, 2);
        let mut hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();
        hyperplane_map.insert(
            (EntityId::from(0), None),
            vec![&precomp_row_a, &precomp_row_b, &precomp_row_c],
        );

        // Hydro 1: computed FPHA.
        let hydro1 = make_sobradinho_computed_hydro(1);
        let config1 = computed_fpha_config(1);

        let geo_rows = make_sobradinho_geometry_rows(1);
        let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(1), geo_refs);

        let stage = make_stage(0);
        let study_stages = [stage.clone()];
        let stage_refs: Vec<&cobre_core::temporal::Stage> = study_stages.iter().collect();

        // Determine sources.
        let src0 = determine_source(&hydro0, Some(&config0)).expect("hydro0 source");
        let src1 = determine_source(&hydro1, Some(&config1)).expect("hydro1 source");
        assert_eq!(
            src0,
            ProductionModelSource::PrecomputedHyperplanes,
            "hydro 0 must be PrecomputedHyperplanes"
        );
        assert_eq!(
            src1,
            ProductionModelSource::ComputedFromGeometry,
            "hydro 1 must be ComputedFromGeometry"
        );

        // Fit computed planes for hydro 1.
        let computed_fit =
            super::fit_planes_for_hydro(&hydro1, Some(&config1), &geometry_map, &stage_refs)
                .expect("fit_planes_for_hydro must succeed for hydro 1");

        // Resolve stage model for hydro 0 (precomputed path).
        let model0 = super::resolve_stage_model(
            &hydro0,
            &stage,
            Some(&config0),
            src0,
            &hyperplane_map,
            None,
        )
        .expect("resolve_stage_model must succeed for hydro 0 (precomputed)");

        // Resolve stage model for hydro 1 (computed path, cached planes).
        let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();
        let model1 = super::resolve_stage_model(
            &hydro1,
            &stage,
            Some(&config1),
            src1,
            &empty_hyperplane_map,
            Some(&computed_fit.planes),
        )
        .expect("resolve_stage_model must succeed for hydro 1 (computed)");

        // Both models must be Fpha.
        assert!(
            matches!(model0, ResolvedProductionModel::Fpha { .. }),
            "hydro 0 must resolve to Fpha, got {model0:?}"
        );
        assert!(
            matches!(model1, ResolvedProductionModel::Fpha { .. }),
            "hydro 1 must resolve to Fpha, got {model1:?}"
        );

        // Provenance in canonical id-sorted order: [(id=0, Precomputed), (id=1, Computed)].
        assert_eq!(
            src0,
            ProductionModelSource::PrecomputedHyperplanes,
            "provenance[0] must be PrecomputedHyperplanes"
        );
        assert_eq!(
            src1,
            ProductionModelSource::ComputedFromGeometry,
            "provenance[1] must be ComputedFromGeometry"
        );
    }

    /// Computed-source all-stages-same: three stages all receive plane vectors with identical
    /// coefficients, confirming that the outer loop fits once and clones for every stage.
    #[test]
    fn computed_source_all_stages_produce_identical_planes() {
        let hydro = make_sobradinho_computed_hydro(0);
        let config = computed_fpha_config(0);

        let geo_rows = make_sobradinho_geometry_rows(0);
        let geo_refs: Vec<&HydroGeometryRow> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Three study stages.
        let stages = [make_stage(0), make_stage(1), make_stage(2)];
        let stage_refs: Vec<&cobre_core::temporal::Stage> = stages.iter().collect();

        // Fit once.
        let cached_fit =
            super::fit_planes_for_hydro(&hydro, Some(&config), &geometry_map, &stage_refs)
                .expect("fit_planes_for_hydro must succeed");

        let empty_hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
            HashMap::new();

        // Resolve for each stage and collect planes.
        let stage_planes: Vec<Vec<FphaPlane>> = stages
            .iter()
            .map(|stage| {
                let model = super::resolve_stage_model(
                    &hydro,
                    stage,
                    Some(&config),
                    ProductionModelSource::ComputedFromGeometry,
                    &empty_hyperplane_map,
                    Some(&cached_fit.planes),
                )
                .expect("resolve_stage_model must succeed");
                match model {
                    ResolvedProductionModel::Fpha { planes, .. } => planes,
                    other => panic!("expected Fpha, got {other:?}"),
                }
            })
            .collect();

        assert_eq!(
            stage_planes.len(),
            3,
            "must have plane vectors for 3 stages"
        );

        // All stages must have the same plane count.
        let expected_count = stage_planes[0].len();
        for (s, planes) in stage_planes.iter().enumerate() {
            assert_eq!(
                planes.len(),
                expected_count,
                "stage {s}: plane count must be {expected_count}, got {}",
                planes.len()
            );
        }

        // Planes must be bitwise-identical across stages (cloned from the same source).
        for (s, planes) in stage_planes.iter().enumerate().skip(1) {
            for (p, plane) in planes.iter().enumerate() {
                assert_eq!(
                    *plane, stage_planes[0][p],
                    "stage {s} plane {p}: must be identical to stage 0 plane {p}"
                );
            }
        }
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
                turbined_cost: 0.0,
            }]],
            1,
            1,
        );

        let result = PrepareHydroModelsResult {
            production,
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
            kappa_warnings: Vec::new(),
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

    /// `validate_computed_prerequisites`: missing `efficiency` returns `SddpError::Validation`
    /// with a message containing both "efficiency" and the hydro name "TestHydro".
    #[test]
    fn computed_source_missing_efficiency_returns_validation_error() {
        let mut hydro = make_computed_hydro(0);
        hydro.name = "TestHydro".to_string();
        hydro.efficiency = None; // remove efficiency to trigger prerequisite failure

        let rows = make_geometry_rows(0);
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
        geometry_map.insert(EntityId::from(0), row_refs);

        let err = validate_computed_prerequisites(&hydro, &geometry_map)
            .expect_err("should fail when efficiency is None");
        let msg = err.to_string();
        assert!(
            msg.contains("efficiency"),
            "error must mention 'efficiency', got: {msg}"
        );
        assert!(
            msg.contains("TestHydro"),
            "error must include hydro name 'TestHydro', got: {msg}"
        );
    }

    /// `validate_computed_prerequisites`: missing `hydraulic_losses` returns `SddpError::Validation`
    /// with a message containing "hydraulic_losses" and the hydro name.
    #[test]
    fn computed_source_missing_losses_returns_validation_error() {
        let mut hydro = make_computed_hydro(0);
        hydro.hydraulic_losses = None; // remove losses to trigger prerequisite failure

        let rows = make_geometry_rows(0);
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        let row_refs: Vec<&HydroGeometryRow> = rows.iter().collect();
        geometry_map.insert(EntityId::from(0), row_refs);

        let err = validate_computed_prerequisites(&hydro, &geometry_map)
            .expect_err("should fail when hydraulic_losses is None");
        let msg = err.to_string();
        assert!(
            msg.contains("hydraulic_losses"),
            "error must mention 'hydraulic_losses', got: {msg}"
        );
        assert!(
            msg.contains(&hydro.name),
            "error must include hydro name '{}', got: {msg}",
            hydro.name
        );
    }
}
