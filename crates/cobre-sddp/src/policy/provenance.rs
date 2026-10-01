//! Provenance report types and builder for the SDDP preprocessing pipeline.
//!
//! [`ModelProvenanceReport`] is a JSON-serializable summary of which data source
//! fed each preprocessing role (seasonal stats, AR coefficients, spatial
//! correlation, opening tree).

use std::fmt;

use serde::{Deserialize, Serialize};

use cobre_core::Hydro;
use cobre_io::scenarios::estimation::{EstimationPath, EstimationReport};
use cobre_stochastic::{ComponentProvenance, StochasticProvenance};

use crate::hydro_models::{
    EvaporationReferenceSource, EvaporationSource, HydroModelProvenance, ProductionModelSource,
};

// ── ProvenanceSource ──────────────────────────────────────────────────────────

/// Origin of a single data role in the preprocessing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSource {
    /// Computed from inflow history by the estimation pipeline.
    Estimated,
    /// Loaded from a user-supplied input file.
    UserFile,
    /// Not applicable — either the system has no hydro plants, or this role is
    /// not relevant for the chosen estimation path (e.g., no AR in the
    /// deterministic case).
    #[serde(rename = "n/a")]
    NotApplicable,
}

impl fmt::Display for ProvenanceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Estimated => write!(f, "estimated"),
            Self::UserFile => write!(f, "user_file"),
            Self::NotApplicable => write!(f, "n/a"),
        }
    }
}

// ── ModelProvenanceReport ─────────────────────────────────────────────────────

/// Provenance of the **inflow** model's data sources; the `inflow` sub-section
/// of [`ModelProvenanceReport`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InflowProvenance {
    /// Stable string label of the estimation path taken (from [`EstimationPath::as_str`]).
    pub estimation_path: String,
    /// Origin of seasonal mean/std data used in the inflow model.
    pub seasonal_stats_source: ProvenanceSource,
    /// Origin of the AR lag coefficients.
    pub ar_coefficients_source: ProvenanceSource,
    /// Origin of the spatial correlation decomposition.
    pub correlation_source: ProvenanceSource,
    /// Origin of the noise opening scenario tree.
    pub opening_tree_source: ProvenanceSource,
    /// Number of hydro plants in the system.
    pub n_hydros: usize,
    /// Order selection method used when AR coefficients were estimated
    /// (e.g., `"AIC"`, `"PACF"`). `None` when AR was not estimated.
    pub ar_method: Option<String>,
    /// Maximum AR order across all hydro plants when AR was estimated.
    /// `None` when AR was not estimated.
    pub ar_max_order: Option<usize>,
    /// IDs of hydro plants that fell back to white noise (empty AR,
    /// `residual_std_ratio = 1.0`). Populated only by
    /// [`EstimationPath::PartialEstimation`]; empty otherwise.
    pub white_noise_fallbacks: Vec<i32>,
    /// SipHash-1-3 fingerprint of the derived inflow lag seed
    /// (`DerivedInflowSeeds::lag_values`) that seeded the historical library's
    /// rolling η-inversion chain; `None` when no `HistoricalScenarioLibrary`
    /// was built. A mismatch against a freshly-derived seed means η was
    /// inverted against a different x₀, so replay is no longer exact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub historical_library_seed_digest: Option<u64>,
}

/// Provenance of the **hydro-production** model's data sources (FPHA and
/// evaporation); the `hydro_production` sub-section of [`ModelProvenanceReport`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HydroProductionProvenance {
    /// Count of hydro plants whose FPHA hyperplanes were computed from geometry.
    pub n_fpha_computed_from_geometry: usize,
    /// Count of hydro plants whose FPHA hyperplanes were supplied precomputed.
    pub n_fpha_precomputed_hyperplanes: usize,
    /// Count of hydro plants whose evaporation reference level was user-supplied.
    pub n_evaporation_ref_user_supplied: usize,
    /// Count of hydro plants whose evaporation reference level defaulted to the
    /// storage-curve midpoint.
    pub n_evaporation_ref_default_midpoint: usize,
}

/// Structured summary of which data sources were used in the preprocessing
/// pipeline, split by model, for JSON output.
///
/// # Example
///
/// ```rust
/// use cobre_sddp::{
///     HydroProductionProvenance, InflowProvenance, ModelProvenanceReport, ProvenanceSource,
/// };
///
/// let report = ModelProvenanceReport {
///     inflow: InflowProvenance {
///         estimation_path: "full_estimation".to_string(),
///         seasonal_stats_source: ProvenanceSource::Estimated,
///         ar_coefficients_source: ProvenanceSource::Estimated,
///         correlation_source: ProvenanceSource::Estimated,
///         opening_tree_source: ProvenanceSource::Estimated,
///         n_hydros: 3,
///         ar_method: Some("AIC".to_string()),
///         ar_max_order: Some(2),
///         white_noise_fallbacks: vec![],
///         historical_library_seed_digest: None,
///     },
///     hydro_production: HydroProductionProvenance::default(),
/// };
/// let json = serde_json::to_string_pretty(&report).unwrap();
/// assert!(json.contains("\"full_estimation\""));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvenanceReport {
    /// Provenance of the inflow model's data sources.
    pub inflow: InflowProvenance,
    /// Provenance of the hydro-production model's data sources.
    #[serde(default)]
    pub hydro_production: HydroProductionProvenance,
}

// ── build_provenance_report ───────────────────────────────────────────────────

/// Map a [`ComponentProvenance`] to a [`ProvenanceSource`].
fn component_to_source(cp: ComponentProvenance) -> ProvenanceSource {
    match cp {
        ComponentProvenance::Generated => ProvenanceSource::Estimated,
        ComponentProvenance::UserSupplied => ProvenanceSource::UserFile,
        ComponentProvenance::NotApplicable => ProvenanceSource::NotApplicable,
    }
}

/// Aggregate a [`HydroModelProvenance`] into the four hydro-production counts.
///
/// Evaporation reference counts are restricted to modeled-evaporation hydros:
/// for [`EvaporationSource::NotModeled`] the reference source is a meaningless
/// placeholder. All four counts are declaration-order invariant.
fn aggregate_hydro_production(hp: &HydroModelProvenance) -> HydroProductionProvenance {
    let n_fpha_computed_from_geometry = hp
        .production_sources
        .iter()
        .filter(|(_, source)| matches!(source, ProductionModelSource::ComputedFromGeometry))
        .count();
    let n_fpha_precomputed_hyperplanes = hp
        .production_sources
        .iter()
        .filter(|(_, source)| matches!(source, ProductionModelSource::PrecomputedHyperplanes))
        .count();

    // Zipping (rather than indexing) stays panic-free if the vectors disagree
    // in length.
    let modeled_refs = || {
        hp.evaporation_sources
            .iter()
            .zip(hp.evaporation_reference_sources.iter())
            .filter(|((_, evap_source), _)| {
                matches!(evap_source, EvaporationSource::LinearizedFromGeometry)
            })
            .map(|(_, (_, ref_source))| *ref_source)
    };
    let n_evaporation_ref_user_supplied = modeled_refs()
        .filter(|s| matches!(s, EvaporationReferenceSource::UserSupplied))
        .count();
    let n_evaporation_ref_default_midpoint = modeled_refs()
        .filter(|s| matches!(s, EvaporationReferenceSource::DefaultMidpoint))
        .count();

    HydroProductionProvenance {
        n_fpha_computed_from_geometry,
        n_fpha_precomputed_hyperplanes,
        n_evaporation_ref_user_supplied,
        n_evaporation_ref_default_midpoint,
    }
}

/// Build a [`ModelProvenanceReport`] from preprocessing outputs.
///
/// Infallible: all required inputs are available once
/// [`crate::setup::prepare_stochastic`] returns. The `EstimationPath` → per-role
/// [`ProvenanceSource`] mapping is the match below.
#[must_use]
pub fn build_provenance_report(
    estimation_path: EstimationPath,
    estimation_report: Option<&EstimationReport>,
    provenance: &StochasticProvenance,
    hydros: &[Hydro],
    hydro_provenance: &HydroModelProvenance,
) -> ModelProvenanceReport {
    let (seasonal_stats_source, ar_coefficients_source) = match estimation_path {
        EstimationPath::Deterministic => (
            ProvenanceSource::NotApplicable,
            ProvenanceSource::NotApplicable,
        ),
        EstimationPath::UserStatsWhiteNoise => {
            (ProvenanceSource::UserFile, ProvenanceSource::NotApplicable)
        }
        EstimationPath::UserProvidedNoHistory => {
            (ProvenanceSource::UserFile, ProvenanceSource::UserFile)
        }
        EstimationPath::FullEstimation => {
            (ProvenanceSource::Estimated, ProvenanceSource::Estimated)
        }
        EstimationPath::UserArHistoryStats => {
            (ProvenanceSource::Estimated, ProvenanceSource::UserFile)
        }
        EstimationPath::PartialEstimation => {
            (ProvenanceSource::UserFile, ProvenanceSource::Estimated)
        }
        EstimationPath::UserProvidedAll => (ProvenanceSource::UserFile, ProvenanceSource::UserFile),
    };

    let correlation_source = component_to_source(provenance.correlation);
    let opening_tree_source = component_to_source(provenance.opening_tree);

    let (ar_method, ar_max_order, white_noise_fallbacks) = if let Some(report) = estimation_report {
        let max_order = report
            .entries
            .values()
            .map(|e| e.selected_order as usize)
            .max();
        let fallbacks: Vec<i32> = report.white_noise_fallbacks.iter().map(|id| id.0).collect();
        (Some(report.method.clone()), max_order, fallbacks)
    } else {
        (None, None, vec![])
    };

    ModelProvenanceReport {
        inflow: InflowProvenance {
            estimation_path: estimation_path.as_str().to_owned(),
            seasonal_stats_source,
            ar_coefficients_source,
            correlation_source,
            opening_tree_source,
            n_hydros: hydros.len(),
            ar_method,
            ar_max_order,
            white_noise_fallbacks,
            historical_library_seed_digest: None,
        },
        hydro_production: aggregate_hydro_production(hydro_provenance),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )]

    use std::collections::BTreeMap;

    use cobre_core::EntityId;
    use cobre_io::scenarios::estimation::{EstimationPath, EstimationReport};
    use cobre_stochastic::{ComponentProvenance, StochasticProvenance};

    use crate::hydro_models::{
        EvaporationReferenceSource, EvaporationSource, HydroModelProvenance, ProductionModelSource,
    };
    use crate::test_support::minimal_hydros;
    use cobre_stochastic::par::fitting::HydroEstimationEntry;

    use super::{
        HydroProductionProvenance, ModelProvenanceReport, ProvenanceSource, build_provenance_report,
    };

    // Helper: build a `HydroModelProvenance` from aligned per-hydro source slices.
    //
    // Each hydro is assigned a sequential `EntityId`. The three slices must be
    // equal length; the production source is taken positionally from
    // `production`, and the evaporation/reference sources from `evaporation` and
    // `evap_ref` respectively.
    fn make_hydro_provenance(
        production: &[ProductionModelSource],
        evaporation: &[EvaporationSource],
        evap_ref: &[EvaporationReferenceSource],
    ) -> HydroModelProvenance {
        let id = |i: usize| EntityId(i as i32 + 1);
        HydroModelProvenance {
            production_sources: production
                .iter()
                .copied()
                .enumerate()
                .map(|(i, s)| (id(i), s))
                .collect(),
            evaporation_sources: evaporation
                .iter()
                .copied()
                .enumerate()
                .map(|(i, s)| (id(i), s))
                .collect(),
            evaporation_reference_sources: evap_ref
                .iter()
                .copied()
                .enumerate()
                .map(|(i, s)| (id(i), s))
                .collect(),
        }
    }

    // Helper: an all-`DefaultConstant`, all-`NotModeled` provenance of `n` hydros,
    // for inflow-focused tests where the hydro counts must stay zero.
    fn empty_hydro_provenance(n: usize) -> HydroModelProvenance {
        make_hydro_provenance(
            &vec![ProductionModelSource::DefaultConstant; n],
            &vec![EvaporationSource::NotModeled; n],
            &vec![EvaporationReferenceSource::DefaultMidpoint; n],
        )
    }

    // Helper: StochasticProvenance with all Generated.
    fn prov_all_generated() -> StochasticProvenance {
        StochasticProvenance {
            opening_tree: ComponentProvenance::Generated,
            correlation: ComponentProvenance::Generated,
            inflow_model: ComponentProvenance::Generated,
            inflow_scheme: None,
            load_scheme: None,
            ncs_scheme: None,
        }
    }

    // Helper: StochasticProvenance for a deterministic (no-entity) system.
    fn prov_not_applicable() -> StochasticProvenance {
        StochasticProvenance {
            opening_tree: ComponentProvenance::NotApplicable,
            correlation: ComponentProvenance::NotApplicable,
            inflow_model: ComponentProvenance::NotApplicable,
            inflow_scheme: None,
            load_scheme: None,
            ncs_scheme: None,
        }
    }

    // Helper: StochasticProvenance with user-supplied tree.
    fn prov_user_tree() -> StochasticProvenance {
        StochasticProvenance {
            opening_tree: ComponentProvenance::UserSupplied,
            correlation: ComponentProvenance::Generated,
            inflow_model: ComponentProvenance::Generated,
            inflow_scheme: None,
            load_scheme: None,
            ncs_scheme: None,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn make_estimation_report(method: &str, orders: &[u32], fallbacks: &[i32]) -> EstimationReport {
        let entries: BTreeMap<EntityId, HydroEstimationEntry> = orders
            .iter()
            .enumerate()
            .map(|(i, &order)| {
                (
                    EntityId(i as i32 + 1),
                    HydroEstimationEntry {
                        selected_order: order,
                        coefficients: vec![],
                        contribution_reductions: vec![],
                    },
                )
            })
            .collect();
        EstimationReport {
            entries,
            method: method.to_owned(),
            white_noise_fallbacks: fallbacks.iter().map(|&id| EntityId(id)).collect(),
            std_ratio_warnings: Vec::new(),
        }
    }

    // ── Path mapping tests ────────────────────────────────────────────────────

    #[test]
    fn build_provenance_report_omits_historical_digest_by_default() {
        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_all_generated(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        assert!(
            report.inflow.historical_library_seed_digest.is_none(),
            "builder must leave historical_library_seed_digest unset; \
             callers populate it from setup.inputs.scenario_libraries.training.historical \
             when the historical scheme is active"
        );
    }

    #[test]
    fn historical_digest_field_round_trips_through_json() {
        let mut report = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_all_generated(),
            &minimal_hydros(1),
            &empty_hydro_provenance(1),
        );
        let digest: u64 = 0xDEAD_BEEF_CAFE_F00D;
        report.inflow.historical_library_seed_digest = Some(digest);
        let json = serde_json::to_string(&report).unwrap();
        assert!(
            json.contains("historical_library_seed_digest"),
            "JSON must surface the digest field when populated: {json}"
        );
        assert!(
            json.contains(&digest.to_string()),
            "JSON must serialize the digest as a decimal u64 ({digest}); got: {json}"
        );
    }

    #[test]
    fn historical_digest_field_omitted_when_none() {
        let report = build_provenance_report(
            EstimationPath::Deterministic,
            None,
            &prov_not_applicable(),
            &minimal_hydros(0),
            &empty_hydro_provenance(0),
        );
        let json = serde_json::to_string(&report).unwrap();
        assert!(
            !json.contains("historical_library_seed_digest"),
            "JSON must NOT include digest field when None (Option::is_none skip): {json}"
        );
    }

    #[test]
    fn deterministic_path_both_na() {
        let report = build_provenance_report(
            EstimationPath::Deterministic,
            None,
            &prov_not_applicable(),
            &minimal_hydros(0),
            &empty_hydro_provenance(0),
        );
        assert!(
            matches!(
                report.inflow.seasonal_stats_source,
                ProvenanceSource::NotApplicable
            ),
            "seasonal_stats_source must be NotApplicable for Deterministic"
        );
        assert!(
            matches!(
                report.inflow.ar_coefficients_source,
                ProvenanceSource::NotApplicable
            ),
            "ar_coefficients_source must be NotApplicable for Deterministic"
        );
        assert!(report.inflow.ar_method.is_none(), "ar_method must be None");
        assert!(
            report.inflow.ar_max_order.is_none(),
            "ar_max_order must be None"
        );
        assert_eq!(report.inflow.estimation_path, "deterministic");
    }

    #[test]
    fn user_stats_white_noise_path() {
        let report = build_provenance_report(
            EstimationPath::UserStatsWhiteNoise,
            None,
            &prov_all_generated(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        assert!(
            matches!(
                report.inflow.seasonal_stats_source,
                ProvenanceSource::UserFile
            ),
            "seasonal_stats_source must be UserFile for UserStatsWhiteNoise"
        );
        assert!(
            matches!(
                report.inflow.ar_coefficients_source,
                ProvenanceSource::NotApplicable
            ),
            "ar_coefficients_source must be NotApplicable for UserStatsWhiteNoise"
        );
        assert_eq!(report.inflow.estimation_path, "user_stats_white_noise");
    }

    #[test]
    fn user_provided_no_history_path() {
        let report = build_provenance_report(
            EstimationPath::UserProvidedNoHistory,
            None,
            &prov_all_generated(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        assert!(
            matches!(
                report.inflow.seasonal_stats_source,
                ProvenanceSource::UserFile
            ),
            "seasonal_stats_source must be UserFile for UserProvidedNoHistory"
        );
        assert!(
            matches!(
                report.inflow.ar_coefficients_source,
                ProvenanceSource::UserFile
            ),
            "ar_coefficients_source must be UserFile for UserProvidedNoHistory"
        );
        assert_eq!(report.inflow.estimation_path, "user_provided_no_history");
    }

    #[test]
    fn full_estimation_path() {
        let er = make_estimation_report("AIC", &[2, 3], &[]);
        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            Some(&er),
            &prov_all_generated(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        assert!(
            matches!(
                report.inflow.seasonal_stats_source,
                ProvenanceSource::Estimated
            ),
            "seasonal_stats_source must be Estimated for FullEstimation"
        );
        assert!(
            matches!(
                report.inflow.ar_coefficients_source,
                ProvenanceSource::Estimated
            ),
            "ar_coefficients_source must be Estimated for FullEstimation"
        );
        assert_eq!(report.inflow.ar_method.as_deref(), Some("AIC"));
        assert_eq!(report.inflow.ar_max_order, Some(3));
        assert_eq!(report.inflow.estimation_path, "full_estimation");
    }

    #[test]
    fn user_ar_history_stats_path() {
        let er = make_estimation_report("PACF", &[1], &[]);
        let report = build_provenance_report(
            EstimationPath::UserArHistoryStats,
            Some(&er),
            &prov_all_generated(),
            &minimal_hydros(1),
            &empty_hydro_provenance(1),
        );
        assert!(
            matches!(
                report.inflow.seasonal_stats_source,
                ProvenanceSource::Estimated
            ),
            "seasonal_stats_source must be Estimated for UserArHistoryStats"
        );
        assert!(
            matches!(
                report.inflow.ar_coefficients_source,
                ProvenanceSource::UserFile
            ),
            "ar_coefficients_source must be UserFile for UserArHistoryStats"
        );
        assert_eq!(report.inflow.estimation_path, "user_ar_history_stats");
    }

    #[test]
    fn partial_estimation_path() {
        let er = make_estimation_report("AIC", &[2], &[5, 7]);
        let report = build_provenance_report(
            EstimationPath::PartialEstimation,
            Some(&er),
            &prov_all_generated(),
            &minimal_hydros(3),
            &empty_hydro_provenance(3),
        );
        assert!(
            matches!(
                report.inflow.seasonal_stats_source,
                ProvenanceSource::UserFile
            ),
            "seasonal_stats_source must be UserFile for PartialEstimation"
        );
        assert!(
            matches!(
                report.inflow.ar_coefficients_source,
                ProvenanceSource::Estimated
            ),
            "ar_coefficients_source must be Estimated for PartialEstimation"
        );
        assert_eq!(report.inflow.white_noise_fallbacks, vec![5, 7]);
        assert_eq!(report.inflow.estimation_path, "partial_estimation");
    }

    #[test]
    fn user_provided_all_path() {
        let report = build_provenance_report(
            EstimationPath::UserProvidedAll,
            None,
            &prov_all_generated(),
            &minimal_hydros(4),
            &empty_hydro_provenance(4),
        );
        assert!(
            matches!(
                report.inflow.seasonal_stats_source,
                ProvenanceSource::UserFile
            ),
            "seasonal_stats_source must be UserFile for UserProvidedAll"
        );
        assert!(
            matches!(
                report.inflow.ar_coefficients_source,
                ProvenanceSource::UserFile
            ),
            "ar_coefficients_source must be UserFile for UserProvidedAll"
        );
        assert!(
            report.inflow.ar_method.is_none(),
            "ar_method must be None when no report"
        );
        assert_eq!(report.inflow.estimation_path, "user_provided_all");
    }

    // ── ComponentProvenance mapping tests ─────────────────────────────────────

    #[test]
    fn user_supplied_tree_maps_to_user_file() {
        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_user_tree(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        assert!(
            matches!(
                report.inflow.opening_tree_source,
                ProvenanceSource::UserFile
            ),
            "UserSupplied opening tree must map to UserFile"
        );
        assert!(
            matches!(
                report.inflow.correlation_source,
                ProvenanceSource::Estimated
            ),
            "Generated correlation must map to Estimated"
        );
    }

    // ── JSON serialization tests ──────────────────────────────────────────────

    #[test]
    fn full_estimation_json_round_trip() {
        let er = make_estimation_report("AIC", &[2], &[]);
        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            Some(&er),
            &prov_all_generated(),
            &minimal_hydros(1),
            &empty_hydro_provenance(1),
        );
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(
            json.contains("\"full_estimation\""),
            "JSON must contain estimation_path value"
        );
        assert!(
            json.contains("\"estimated\""),
            "JSON must contain estimated source"
        );
        // Verify it parses as a valid JSON object with expected keys.
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["inflow"]["estimation_path"], "full_estimation");
        assert_eq!(value["inflow"]["seasonal_stats_source"], "estimated");
        assert_eq!(value["inflow"]["ar_coefficients_source"], "estimated");
    }

    #[test]
    fn deterministic_json_na_variant() {
        let report = build_provenance_report(
            EstimationPath::Deterministic,
            None,
            &prov_not_applicable(),
            &minimal_hydros(0),
            &empty_hydro_provenance(0),
        );
        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value["inflow"]["seasonal_stats_source"], "n/a",
            "NotApplicable must serialize as \"n/a\""
        );
        assert_eq!(value["inflow"]["ar_coefficients_source"], "n/a");
    }

    // ── ProvenanceSource Display tests ────────────────────────────────────────

    #[test]
    fn provenance_source_display() {
        assert_eq!(ProvenanceSource::Estimated.to_string(), "estimated");
        assert_eq!(ProvenanceSource::UserFile.to_string(), "user_file");
        assert_eq!(ProvenanceSource::NotApplicable.to_string(), "n/a");
    }

    // ── white_noise_fallbacks propagation ─────────────────────────────────────

    #[test]
    fn white_noise_fallbacks_propagated_as_raw_ids() {
        let er = make_estimation_report("AIC", &[1, 2], &[3, 7]);
        let report = build_provenance_report(
            EstimationPath::PartialEstimation,
            Some(&er),
            &prov_all_generated(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        assert_eq!(
            report.inflow.white_noise_fallbacks,
            vec![3, 7],
            "white_noise_fallbacks must carry raw i32 IDs"
        );
    }

    #[test]
    fn no_estimation_report_yields_empty_fallbacks() {
        let report = build_provenance_report(
            EstimationPath::Deterministic,
            None,
            &prov_not_applicable(),
            &minimal_hydros(0),
            &empty_hydro_provenance(0),
        );
        assert!(
            report.inflow.white_noise_fallbacks.is_empty(),
            "white_noise_fallbacks must be empty when no estimation_report"
        );
    }

    // ── Nested report shape / deserialization tests ───────────────────────────

    #[test]
    fn nested_report_serializes_inflow_and_hydro_production_keys() {
        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_all_generated(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        let json = serde_json::to_string(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            value.get("inflow").is_some(),
            "JSON must contain a top-level \"inflow\" object: {json}"
        );
        assert!(
            value.get("hydro_production").is_some(),
            "JSON must contain a top-level \"hydro_production\" object: {json}"
        );
        let hp = &value["hydro_production"];
        assert_eq!(hp["n_fpha_computed_from_geometry"], 0);
        assert_eq!(hp["n_fpha_precomputed_hyperplanes"], 0);
        assert_eq!(hp["n_evaporation_ref_user_supplied"], 0);
        assert_eq!(hp["n_evaporation_ref_default_midpoint"], 0);
    }

    #[test]
    fn report_deserializes_from_json() {
        let original = build_provenance_report(
            EstimationPath::FullEstimation,
            Some(&make_estimation_report("AIC", &[2, 3], &[])),
            &prov_all_generated(),
            &minimal_hydros(2),
            &empty_hydro_provenance(2),
        );
        let json = serde_json::to_string(&original).unwrap();
        let parsed: ModelProvenanceReport = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.inflow.estimation_path,
            original.inflow.estimation_path
        );
        assert!(matches!(
            parsed.inflow.seasonal_stats_source,
            ProvenanceSource::Estimated
        ));
        assert_eq!(
            parsed.hydro_production.n_fpha_computed_from_geometry,
            original.hydro_production.n_fpha_computed_from_geometry
        );
        assert_eq!(
            parsed.hydro_production.n_fpha_precomputed_hyperplanes,
            original.hydro_production.n_fpha_precomputed_hyperplanes
        );
        assert_eq!(
            parsed.hydro_production.n_evaporation_ref_user_supplied,
            original.hydro_production.n_evaporation_ref_user_supplied
        );
        assert_eq!(
            parsed.hydro_production.n_evaporation_ref_default_midpoint,
            original.hydro_production.n_evaporation_ref_default_midpoint
        );
    }

    #[test]
    fn report_deserializes_when_hydro_production_omitted() {
        let json = r#"{
            "inflow": {
                "estimation_path": "full_estimation",
                "seasonal_stats_source": "estimated",
                "ar_coefficients_source": "estimated",
                "correlation_source": "estimated",
                "opening_tree_source": "estimated",
                "n_hydros": 2,
                "ar_method": "AIC",
                "ar_max_order": 3,
                "white_noise_fallbacks": []
            }
        }"#;
        let parsed: ModelProvenanceReport = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.inflow.estimation_path, "full_estimation");
        let default = HydroProductionProvenance::default();
        assert_eq!(
            parsed.hydro_production.n_fpha_computed_from_geometry,
            default.n_fpha_computed_from_geometry
        );
        assert_eq!(
            parsed.hydro_production.n_fpha_precomputed_hyperplanes,
            default.n_fpha_precomputed_hyperplanes
        );
        assert_eq!(
            parsed.hydro_production.n_evaporation_ref_user_supplied,
            default.n_evaporation_ref_user_supplied
        );
        assert_eq!(
            parsed.hydro_production.n_evaporation_ref_default_midpoint,
            default.n_evaporation_ref_default_midpoint
        );
    }

    // ── hydro-production aggregation tests ─────────────────────────────────────

    #[test]
    fn hydro_production_counts_fpha_sources() {
        // 2 ComputedFromGeometry + 3 PrecomputedHyperplanes (plus the evaporation
        // arrays are not modeled, so they contribute zero reference counts).
        let production = [
            ProductionModelSource::ComputedFromGeometry,
            ProductionModelSource::PrecomputedHyperplanes,
            ProductionModelSource::ComputedFromGeometry,
            ProductionModelSource::PrecomputedHyperplanes,
            ProductionModelSource::PrecomputedHyperplanes,
        ];
        let evaporation = [EvaporationSource::NotModeled; 5];
        let evap_ref = [EvaporationReferenceSource::DefaultMidpoint; 5];
        let hp = make_hydro_provenance(&production, &evaporation, &evap_ref);

        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_all_generated(),
            &minimal_hydros(5),
            &hp,
        );
        assert_eq!(
            report.hydro_production.n_fpha_computed_from_geometry, 2,
            "two ComputedFromGeometry production sources must be tallied"
        );
        assert_eq!(
            report.hydro_production.n_fpha_precomputed_hyperplanes, 3,
            "three PrecomputedHyperplanes production sources must be tallied"
        );
    }

    #[test]
    fn hydro_production_excludes_not_modeled_evaporation_refs() {
        // Hydro A: LinearizedFromGeometry + UserSupplied -> counted.
        // Hydro B: NotModeled + DefaultMidpoint placeholder -> excluded.
        let production = [
            ProductionModelSource::DefaultConstant,
            ProductionModelSource::DefaultConstant,
        ];
        let evaporation = [
            EvaporationSource::LinearizedFromGeometry,
            EvaporationSource::NotModeled,
        ];
        let evap_ref = [
            EvaporationReferenceSource::UserSupplied,
            EvaporationReferenceSource::DefaultMidpoint,
        ];
        let hp = make_hydro_provenance(&production, &evaporation, &evap_ref);

        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_all_generated(),
            &minimal_hydros(2),
            &hp,
        );
        assert_eq!(
            report.hydro_production.n_evaporation_ref_user_supplied, 1,
            "the modeled hydro's UserSupplied reference must be counted"
        );
        assert_eq!(
            report.hydro_production.n_evaporation_ref_default_midpoint, 0,
            "the NotModeled hydro's DefaultMidpoint placeholder must be excluded"
        );
    }

    #[test]
    fn hydro_production_counts_order_invariant() {
        let production_a = [
            ProductionModelSource::ComputedFromGeometry,
            ProductionModelSource::PrecomputedHyperplanes,
            ProductionModelSource::DefaultConstant,
        ];
        let evaporation_a = [
            EvaporationSource::LinearizedFromGeometry,
            EvaporationSource::NotModeled,
            EvaporationSource::LinearizedFromGeometry,
        ];
        let evap_ref_a = [
            EvaporationReferenceSource::UserSupplied,
            EvaporationReferenceSource::DefaultMidpoint,
            EvaporationReferenceSource::DefaultMidpoint,
        ];
        let hp_a = make_hydro_provenance(&production_a, &evaporation_a, &evap_ref_a);

        // Same three hydros, permuted (reverse order). Counts must be identical.
        let production_b = [
            ProductionModelSource::DefaultConstant,
            ProductionModelSource::PrecomputedHyperplanes,
            ProductionModelSource::ComputedFromGeometry,
        ];
        let evaporation_b = [
            EvaporationSource::LinearizedFromGeometry,
            EvaporationSource::NotModeled,
            EvaporationSource::LinearizedFromGeometry,
        ];
        let evap_ref_b = [
            EvaporationReferenceSource::DefaultMidpoint,
            EvaporationReferenceSource::DefaultMidpoint,
            EvaporationReferenceSource::UserSupplied,
        ];
        let hp_b = make_hydro_provenance(&production_b, &evaporation_b, &evap_ref_b);

        let report_a = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_all_generated(),
            &minimal_hydros(3),
            &hp_a,
        );
        let report_b = build_provenance_report(
            EstimationPath::FullEstimation,
            None,
            &prov_all_generated(),
            &minimal_hydros(3),
            &hp_b,
        );
        assert_eq!(
            report_a.hydro_production.n_fpha_computed_from_geometry,
            report_b.hydro_production.n_fpha_computed_from_geometry,
            "computed-from-geometry count must be order invariant"
        );
        assert_eq!(
            report_a.hydro_production.n_fpha_precomputed_hyperplanes,
            report_b.hydro_production.n_fpha_precomputed_hyperplanes,
            "precomputed-hyperplanes count must be order invariant"
        );
        assert_eq!(
            report_a.hydro_production.n_evaporation_ref_user_supplied,
            report_b.hydro_production.n_evaporation_ref_user_supplied,
            "user-supplied reference count must be order invariant"
        );
        assert_eq!(
            report_a.hydro_production.n_evaporation_ref_default_midpoint,
            report_b.hydro_production.n_evaporation_ref_default_midpoint,
            "default-midpoint reference count must be order invariant"
        );
    }
}
