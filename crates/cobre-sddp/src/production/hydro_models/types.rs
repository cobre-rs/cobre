//! Runtime data types for the hydro production and evaporation pipelines.
//!
//! These are the preprocessed output types the LP builder, energy-conversion
//! derivation, and CLI/Python summary display read:
//!
//! - `ResolvedProductionModel` and `ProductionModelSet` — resolved production
//!   functions (constant productivity or FPHA hyperplanes) indexed by hydro and stage.
//! - `EvaporationModel` and `EvaporationModelSet` — linearized evaporation
//!   coefficients indexed by hydro plant.
//! - `HydroModelProvenance` — tracks the source of each hydro's production and
//!   evaporation model for display and auditing.
//! - `HydroModelSummary` — aggregated statistics for display after preprocessing.
//! - `PrepareHydroModelsResult` — bundles all pipeline outputs.
//!
//! These types live in `cobre-sddp` because they are algorithm-specific (FPHA
//! hyperplane approximation is an SDDP concept). They must not be placed in
//! `cobre-core`.

use std::collections::HashMap;

use cobre_core::{EntityId, Hydro, StudyPos, System};
use cobre_io::FphaDeviationPointRow;
use cobre_io::FphaHyperplaneRow;
use cobre_io::HydroGeometryRow;
use serde::{Deserialize, Serialize};

use crate::energy_conversion::HydroEnergyProductivityOverride;

// ── Hyperplane types ──────────────────────────────────────────────────────────

/// A single FPHA hyperplane. The intercept is the pre-scaled `gamma_0 * kappa`,
/// added to the constraint RHS verbatim — the LP builder must not re-scale it.
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
    },
}

// ── Production model set ──────────────────────────────────────────────────────

/// Resolved production models for all (hydro, stage) combinations, indexed
/// `stage_models[hydro][stage]`. Access via [`model`](ProductionModelSet::model).
#[derive(Debug, Clone)]
pub struct ProductionModelSet {
    /// `stage_models[h][t]` is the resolved production model for hydro `h` at stage `t`.
    stage_models: Vec<Vec<ResolvedProductionModel>>,
    /// Number of stages (inner dimension).
    n_stages: usize,
}

impl ProductionModelSet {
    /// Construct a `ProductionModelSet` from `models[hydro][stage]`.
    ///
    /// # Panics
    ///
    /// Panics if `models.len() != hydros.len()` or any inner `Vec` length
    /// differs from `n_stages`.
    #[must_use]
    pub fn new(
        models: Vec<Vec<ResolvedProductionModel>>,
        hydros: &[Hydro],
        n_stages: usize,
    ) -> Self {
        assert_eq!(
            models.len(),
            hydros.len(),
            "outer dimension must equal hydros.len()"
        );
        assert!(
            models.iter().all(|row| row.len() == n_stages),
            "each hydro's stage vector must have length n_stages"
        );
        Self {
            stage_models: models,
            n_stages,
        }
    }

    /// Resolved production model at `(hydro, stage)`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `hydro >= n_hydros` or `stage >= n_stages`.
    #[must_use]
    pub fn model(&self, hydro: usize, stage: usize) -> &ResolvedProductionModel {
        debug_assert!(
            hydro < self.stage_models.len(),
            "hydro index {hydro} out of bounds (n_hydros = {})",
            self.stage_models.len()
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
        self.stage_models.len()
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
/// The stage-averaged evaporation outflow (m³/s) is approximated as:
///
/// ```text
/// evaporation_outflow = intercept_m3s + volume_slope_m3s_per_hm3 * (V - reference_volume)
/// ```
///
/// where `V` is the reservoir volume (hm³) and `reference_volume` is the
/// reference volume for each stage stored in
/// [`EvaporationModel::Linearized::reference_volumes_hm3`]. The coefficients
/// absorb the `1 / (3.6 · stage_hours)` factor that converts the
/// `monthly_evaporation_mm · A(V)` volume per month (mm·km²/month) into the
/// stage-averaged flow consumed by the water balance row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearizedEvaporation {
    /// Constant term of the linearized evaporation outflow (m³/s).
    pub intercept_m3s: f64,
    /// Volume-dependent slope of the linearized evaporation outflow ((m³/s)/hm³).
    pub volume_slope_m3s_per_hm3: f64,
}

// ── Evaporation model ─────────────────────────────────────────────────────────

/// Resolved evaporation model for a single hydro plant.
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

    /// `true` if any hydro plant has a [`Linearized`](EvaporationModel::Linearized) model.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    /// Evaporation coefficients were supplied, but the reservoir has no usable
    /// surface-area data (no geometry rows, or every `area_km2` is zero), so
    /// evaporation is disabled for this hydro and a `tracing::warn!` is emitted
    /// at resolution. Zero surface area yields zero evaporation, so disabling is
    /// the physically-correct degradation rather than a hard error — a new or
    /// being-filled reservoir legitimately may not have an area-volume curve yet.
    DisabledNoArea,
}

/// Source of the reference volume used for evaporation linearization.
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

/// Per-hydro detail for FPHA production models, carried in [`HydroModelSummary`].
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Aggregated production- and evaporation-model counts for `cobre-cli` display.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

// ── FPHA fit deviation ──────────────────────────────────────────────────────────

/// One computed-FPHA fit's deviation magnitudes, one entry per DISTINCT fit
/// (tagged with the first study stage it covers), in canonical `(hydro, stage)`
/// order.
///
/// The magnitudes are plain `f64`s rather than the `pub(crate)`
/// `crate::fpha_fitting::FphaFitDeviation`: that crate-internal type must not cross
/// into `cobre-io`'s generic metadata aggregate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FphaFitDeviationEntry {
    /// Entity id of the hydro plant this fit belongs to.
    pub hydro_id: EntityId,
    /// Domain-level stage id the fitted entry first covers.
    pub stage_id: i32,
    /// Mean of `|fitted − exact|` over the fit grid \[MW\].
    pub mean_abs_mw: f64,
    /// Maximum of `|fitted − exact|` over the fit grid \[MW\].
    pub max_abs_mw: f64,
    /// Mean of the signed residual `fitted − exact` over the fit grid \[MW\].
    pub mean_signed_mw: f64,
    /// `mean_abs_mw` relative to the grid's peak exact generation (dimensionless,
    /// `>= 0`).
    pub relative: f64,
}

// ── Pipeline result ───────────────────────────────────────────────────────────

/// Result of the hydro model preprocessing pipeline, consumed by `StudySetup`
/// construction and the summary builder.
///
/// Every computed-FPHA-only field below is non-empty only when some hydro uses
/// `source: "computed"`; the resolver builds each in canonical `(hydro, stage)`
/// order, so all are declaration-order invariant.
#[derive(Debug)]
pub struct PrepareHydroModelsResult {
    /// Resolved production models for all (hydro, stage) pairs. For non-FPHA
    /// hydros, `productivity` already reflects the
    /// `hydro_energy_productivity.parquet` override — the single source of truth
    /// for downstream consumers.
    pub production: ProductionModelSet,
    /// Per-`(hydro, stage)` override table from `hydro_energy_productivity.parquet`
    /// (empty when absent). Carried because the FPHA derivation in
    /// [`crate::energy_conversion::build_energy_conversion_set`] still needs the raw
    /// override entries to override its VHA + `ρ_esp` derivation; for non-FPHA
    /// hydros the override is already folded into `production`.
    pub productivity_override: HydroEnergyProductivityOverride,
    /// Resolved evaporation models for all hydro plants.
    pub evaporation: EvaporationModelSet,
    /// Provenance records for all hydro plants.
    pub provenance: HydroModelProvenance,
    /// Computed-FPHA hyperplane rows. The CLI/Python entry point writes these to
    /// `hydro_models/fpha_hyperplanes.parquet`; non-root MPI ranks reach no write
    /// site but hold an identical copy via the deterministic preprocessing path.
    pub fpha_export_rows: Vec<FphaHyperplaneRow>,
    /// Per-`(hydro, study-stage)` reference operating volume in absolute hm³
    /// (`(hydro_id, stage_pos, volume_hm3)`). The energy-conversion build and the
    /// FPHA backwater path read this one source.
    pub reference_volumes_hm3: Vec<(EntityId, StudyPos, f64)>,
    /// Per-hydro VHA geometry (volume → forebay height), sorted by ascending
    /// volume. The energy-conversion build derives `ρ_eq` from this + `ρ_esp` for an
    /// FPHA plant with no parquet `equivalent_productivity` override (the parquet
    /// value takes priority). Empty when the case ships no `hydro_geometry`, leaving
    /// the override required.
    pub vha_geometry_by_hydro: HashMap<EntityId, Vec<HydroGeometryRow>>,
    /// Computed-FPHA fit deviations, one entry per distinct fit. The CLI/Python
    /// write site rolls these into the [`cobre_io::DeviationSummary`] in
    /// `training/metadata.json`; an empty vector yields no metadata section.
    pub fpha_fit_deviations: Vec<FphaFitDeviationEntry>,
    /// Per-sampled-point computed-FPHA deviations, one row per `(hydro, stage, V, Q)`
    /// grid point at spillage = 0, already in canonical `(hydro_id, stage_id, grid)`
    /// order — consumers must not re-sort. Empty unless the run opts in via
    /// `config.exports.fpha_deviation_points`; the write site emits
    /// `hydro_models/fpha_deviation_points.parquet` only for a non-empty vector.
    pub fpha_deviation_point_rows: Vec<FphaDeviationPointRow>,
}

impl PrepareHydroModelsResult {
    /// Build a default result for a system with no FPHA and no evaporation data.
    ///
    /// Used in tests and where the full `prepare_hydro_models` pipeline is not
    /// available (e.g. non-root MPI ranks reconstructing the result independently).
    #[must_use]
    pub fn default_from_system(system: &System) -> Self {
        let n_stages = system.stages().iter().filter(|s| s.id >= 0).count();

        let production_models: Vec<Vec<ResolvedProductionModel>> = system
            .hydros()
            .iter()
            .map(|_hydro| {
                // 0.0 placeholder: the real coefficient lives in
                // hydro_production_models.json, which this factory does not read.
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 0.0 }; n_stages]
            })
            .collect();

        let production = ProductionModelSet::new(production_models, system.hydros(), n_stages);

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

        // Resolve through `resolve_reference_volume_hm3(None, ..)` — the single
        // owner of the default fraction — not an inline formula, so the undeclared
        // value stays bit-identical to the JSON-fed path.
        let reference_volumes_hm3: Vec<(EntityId, StudyPos, f64)> = system
            .hydros()
            .iter()
            .flat_map(|hydro| {
                let resolved = super::production::resolve_reference_volume_hm3(
                    None,
                    hydro.min_storage_hm3,
                    hydro.max_storage_hm3,
                );
                (0..n_stages).map(move |stage_index| (hydro.id, StudyPos(stage_index), resolved))
            })
            .collect();

        Self {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation,
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3,
            vha_geometry_by_hydro: HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        }
    }
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
    use cobre_core::EntityId;

    use super::*;
    use crate::HydroEnergyProductivityOverride;
    use crate::test_support::minimal_hydros;

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
                })
                .collect();
            all_models.push(row);
        }

        let set = ProductionModelSet::new(all_models, &minimal_hydros(n_hydros), n_stages);

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
            intercept_m3s: 0.5,
            volume_slope_m3s_per_hm3: 0.01,
        };
        let coeff2 = coeff;
        assert_eq!(coeff.intercept_m3s, coeff2.intercept_m3s);
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
        };
        let _ = format!("{model_fpha:?}");

        let coeff = LinearizedEvaporation {
            intercept_m3s: 0.5,
            volume_slope_m3s_per_hm3: 0.01,
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
            &minimal_hydros(1),
            1,
        );
        let evap_set = EvaporationModelSet::new(vec![EvaporationModel::None]);
        let result = PrepareHydroModelsResult {
            production: prod_set,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: evap_set,
            provenance: prov,
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
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
                },
                ResolvedProductionModel::ConstantProductivity { productivity: 0.80 },
                ResolvedProductionModel::ConstantProductivity { productivity: 0.85 },
            ],
        ];

        let set = ProductionModelSet::new(models, &minimal_hydros(2), 3);

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

    #[cfg(debug_assertions)]
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
            &minimal_hydros(2),
            3,
        );
        // hydro index 2 is out of bounds for n_hydros = 2 → debug_assert! fires
        let _ = set.model(2, 0);
    }

    #[test]
    #[should_panic(expected = "outer dimension must equal hydros.len()")]
    fn production_model_set_new_panics_on_hydro_count_mismatch() {
        let _ = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::ConstantProductivity {
                productivity: 0.9,
            }]],
            &minimal_hydros(2),
            1,
        );
    }

    // ── EvaporationModelSet tests ─────────────────────────────────────────────

    #[test]
    fn evaporation_model_set_has_evaporation_true_when_any_linearized() {
        let set = EvaporationModelSet::new(vec![
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![
                    LinearizedEvaporation {
                        intercept_m3s: 0.5,
                        volume_slope_m3s_per_hm3: 0.01,
                    },
                    LinearizedEvaporation {
                        intercept_m3s: 0.6,
                        volume_slope_m3s_per_hm3: 0.02,
                    },
                ],
                reference_volumes_hm3: vec![200.0, 200.0],
            },
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 0.3,
                    volume_slope_m3s_per_hm3: 0.005,
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
            intercept_m3s: 1.0,
            volume_slope_m3s_per_hm3: 0.1,
        };
        let coeff1 = LinearizedEvaporation {
            intercept_m3s: 2.0,
            volume_slope_m3s_per_hm3: 0.2,
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

    // ── Serde round-trip tests ─────────────────────────────────────────────────

    fn sample_hydro_model_summary() -> HydroModelSummary {
        HydroModelSummary {
            n_constant: 3,
            n_fpha: 2,
            total_planes: 7,
            fpha_details: vec![
                FphaHydroDetail {
                    hydro_id: EntityId(11),
                    name: "Reservoir A".to_string(),
                    source: ProductionModelSource::PrecomputedHyperplanes,
                    n_planes: 4,
                },
                FphaHydroDetail {
                    hydro_id: EntityId(12),
                    name: "Reservoir B".to_string(),
                    source: ProductionModelSource::ComputedFromGeometry,
                    n_planes: 3,
                },
            ],
            n_evaporation: 1,
            n_no_evaporation: 4,
            n_user_supplied_ref: 1,
            n_default_midpoint_ref: 0,
        }
    }

    #[test]
    fn production_model_source_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&ProductionModelSource::DefaultConstant).unwrap(),
            "\"default_constant\""
        );
        assert_eq!(
            serde_json::to_string(&ProductionModelSource::PrecomputedHyperplanes).unwrap(),
            "\"precomputed_hyperplanes\""
        );
        assert_eq!(
            serde_json::to_string(&ProductionModelSource::ComputedFromGeometry).unwrap(),
            "\"computed_from_geometry\""
        );

        for source in [
            ProductionModelSource::DefaultConstant,
            ProductionModelSource::PrecomputedHyperplanes,
            ProductionModelSource::ComputedFromGeometry,
        ] {
            let json = serde_json::to_string(&source).unwrap();
            let back: ProductionModelSource = serde_json::from_str(&json).unwrap();
            assert_eq!(source, back);
        }
    }

    #[test]
    fn entity_id_serializes_as_bare_integer() {
        // EntityId is a transparent newtype: its wire form is a bare integer,
        // exercised here via the FphaHydroDetail field that embeds it.
        let detail = FphaHydroDetail {
            hydro_id: EntityId(42),
            name: "Plant".to_string(),
            source: ProductionModelSource::DefaultConstant,
            n_planes: 1,
        };
        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&detail).unwrap()).unwrap();
        assert_eq!(value["hydro_id"], serde_json::json!(42));
    }

    #[test]
    fn hydro_model_summary_round_trips_through_json() {
        let summary = sample_hydro_model_summary();

        let json = serde_json::to_string(&summary).unwrap();

        // Structural counts and the relocated source qualifier are present.
        assert!(json.contains("\"n_fpha\":2"));
        assert!(json.contains("\"source\":\"precomputed_hyperplanes\""));

        let back: HydroModelSummary = serde_json::from_str(&json).unwrap();

        assert_eq!(back.n_constant, summary.n_constant);
        assert_eq!(back.n_fpha, summary.n_fpha);
        assert_eq!(back.total_planes, summary.total_planes);
        assert_eq!(back.n_evaporation, summary.n_evaporation);
        assert_eq!(back.n_no_evaporation, summary.n_no_evaporation);
        assert_eq!(back.n_user_supplied_ref, summary.n_user_supplied_ref);
        assert_eq!(back.n_default_midpoint_ref, summary.n_default_midpoint_ref);
        assert_eq!(back.fpha_details.len(), summary.fpha_details.len());
        for (got, want) in back.fpha_details.iter().zip(&summary.fpha_details) {
            assert_eq!(got.hydro_id, want.hydro_id);
            assert_eq!(got.name, want.name);
            assert_eq!(got.source, want.source);
            assert_eq!(got.n_planes, want.n_planes);
        }
        assert_eq!(
            back.fpha_details[0].source,
            ProductionModelSource::PrecomputedHyperplanes
        );
    }

    #[test]
    fn hydro_model_summary_serialization_is_deterministic() {
        let summary = sample_hydro_model_summary();
        let first = serde_json::to_string(&summary).unwrap();
        let second = serde_json::to_string(&summary).unwrap();
        assert_eq!(first, second);
    }
}
