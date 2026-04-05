//! Provenance report types and builder for the SDDP preprocessing pipeline.
//!
//! [`ModelProvenanceReport`] is a JSON-serializable summary of which data
//! sources were used for each role in the stochastic preprocessing pipeline:
//! seasonal statistics, AR coefficients, spatial correlation, and the opening
//! scenario tree.
//!
//! [`build_provenance_report`] constructs a [`ModelProvenanceReport`] from
//! the outputs already available after [`crate::setup::prepare_stochastic`]
//! returns.

use std::fmt;

use serde::Serialize;

use cobre_stochastic::{ComponentProvenance, StochasticProvenance};

use crate::estimation::{EstimationPath, EstimationReport};

// ── ProvenanceSource ──────────────────────────────────────────────────────────

/// Origin of a single data role in the preprocessing pipeline.
///
/// Used by [`ModelProvenanceReport`] to describe where seasonal stats, AR
/// coefficients, spatial correlation, and the opening tree came from.
#[derive(Debug, Clone, Serialize)]
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

/// Structured summary of which data sources were used in the stochastic
/// preprocessing pipeline.
///
/// Intended for JSON output via `serde_json::to_writer_pretty`. Each field
/// records the origin of one data role. Callers (CLI, Python bindings) pass
/// this struct directly to `serde_json` without additional transformation.
///
/// # Example
///
/// ```rust
/// use cobre_sddp::{ModelProvenanceReport, ProvenanceSource};
///
/// let report = ModelProvenanceReport {
///     estimation_path: "full_estimation".to_string(),
///     seasonal_stats_source: ProvenanceSource::Estimated,
///     ar_coefficients_source: ProvenanceSource::Estimated,
///     correlation_source: ProvenanceSource::Estimated,
///     opening_tree_source: ProvenanceSource::Estimated,
///     n_hydros: 3,
///     ar_method: Some("AIC".to_string()),
///     ar_max_order: Some(2),
///     white_noise_fallbacks: vec![],
/// };
/// let json = serde_json::to_string_pretty(&report).unwrap();
/// assert!(json.contains("\"full_estimation\""));
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct ModelProvenanceReport {
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

/// Build a [`ModelProvenanceReport`] from preprocessing outputs.
///
/// This function is infallible: all required inputs are guaranteed to be
/// available after [`crate::setup::prepare_stochastic`] returns.
///
/// The mapping from [`EstimationPath`] to per-role [`ProvenanceSource`] is:
///
/// | Path                  | Seasonal stats | AR coefficients |
/// |-----------------------|---------------|-----------------|
/// | `Deterministic`       | N/A           | N/A             |
/// | `UserStatsWhiteNoise` | UserFile      | N/A             |
/// | `UserProvidedNoHistory` | UserFile    | UserFile        |
/// | `FullEstimation`      | Estimated     | Estimated       |
/// | `UserArHistoryStats`  | Estimated     | UserFile        |
/// | `PartialEstimation`   | UserFile      | Estimated       |
/// | `UserProvidedAll`     | UserFile      | UserFile        |
///
/// Correlation and opening-tree sources are derived from the
/// [`StochasticProvenance`] embedded in the stochastic context.
///
/// When `estimation_report` is `Some`, `ar_method` and `ar_max_order` are
/// populated from its fields; otherwise both are `None`.
#[must_use]
pub fn build_provenance_report(
    estimation_path: EstimationPath,
    estimation_report: Option<&EstimationReport>,
    provenance: &StochasticProvenance,
    n_hydros: usize,
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
        estimation_path: estimation_path.as_str().to_owned(),
        seasonal_stats_source,
        ar_coefficients_source,
        correlation_source,
        opening_tree_source,
        n_hydros,
        ar_method,
        ar_max_order,
        white_noise_fallbacks,
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
    use cobre_stochastic::{ComponentProvenance, StochasticProvenance};

    use crate::estimation::{EstimationPath, EstimationReport, HydroEstimationEntry};

    use super::{ProvenanceSource, build_provenance_report};

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
        }
    }

    // ── Path mapping tests ────────────────────────────────────────────────────

    #[test]
    fn deterministic_path_both_na() {
        let report = build_provenance_report(
            EstimationPath::Deterministic,
            None,
            &prov_not_applicable(),
            0,
        );
        assert!(
            matches!(
                report.seasonal_stats_source,
                ProvenanceSource::NotApplicable
            ),
            "seasonal_stats_source must be NotApplicable for Deterministic"
        );
        assert!(
            matches!(
                report.ar_coefficients_source,
                ProvenanceSource::NotApplicable
            ),
            "ar_coefficients_source must be NotApplicable for Deterministic"
        );
        assert!(report.ar_method.is_none(), "ar_method must be None");
        assert!(report.ar_max_order.is_none(), "ar_max_order must be None");
        assert_eq!(report.estimation_path, "deterministic");
    }

    #[test]
    fn user_stats_white_noise_path() {
        let report = build_provenance_report(
            EstimationPath::UserStatsWhiteNoise,
            None,
            &prov_all_generated(),
            2,
        );
        assert!(
            matches!(report.seasonal_stats_source, ProvenanceSource::UserFile),
            "seasonal_stats_source must be UserFile for UserStatsWhiteNoise"
        );
        assert!(
            matches!(
                report.ar_coefficients_source,
                ProvenanceSource::NotApplicable
            ),
            "ar_coefficients_source must be NotApplicable for UserStatsWhiteNoise"
        );
        assert_eq!(report.estimation_path, "user_stats_white_noise");
    }

    #[test]
    fn user_provided_no_history_path() {
        let report = build_provenance_report(
            EstimationPath::UserProvidedNoHistory,
            None,
            &prov_all_generated(),
            2,
        );
        assert!(
            matches!(report.seasonal_stats_source, ProvenanceSource::UserFile),
            "seasonal_stats_source must be UserFile for UserProvidedNoHistory"
        );
        assert!(
            matches!(report.ar_coefficients_source, ProvenanceSource::UserFile),
            "ar_coefficients_source must be UserFile for UserProvidedNoHistory"
        );
        assert_eq!(report.estimation_path, "user_provided_no_history");
    }

    #[test]
    fn full_estimation_path() {
        let er = make_estimation_report("AIC", &[2, 3], &[]);
        let report = build_provenance_report(
            EstimationPath::FullEstimation,
            Some(&er),
            &prov_all_generated(),
            2,
        );
        assert!(
            matches!(report.seasonal_stats_source, ProvenanceSource::Estimated),
            "seasonal_stats_source must be Estimated for FullEstimation"
        );
        assert!(
            matches!(report.ar_coefficients_source, ProvenanceSource::Estimated),
            "ar_coefficients_source must be Estimated for FullEstimation"
        );
        assert_eq!(report.ar_method.as_deref(), Some("AIC"));
        assert_eq!(report.ar_max_order, Some(3));
        assert_eq!(report.estimation_path, "full_estimation");
    }

    #[test]
    fn user_ar_history_stats_path() {
        let er = make_estimation_report("PACF", &[1], &[]);
        let report = build_provenance_report(
            EstimationPath::UserArHistoryStats,
            Some(&er),
            &prov_all_generated(),
            1,
        );
        assert!(
            matches!(report.seasonal_stats_source, ProvenanceSource::Estimated),
            "seasonal_stats_source must be Estimated for UserArHistoryStats"
        );
        assert!(
            matches!(report.ar_coefficients_source, ProvenanceSource::UserFile),
            "ar_coefficients_source must be UserFile for UserArHistoryStats"
        );
        assert_eq!(report.estimation_path, "user_ar_history_stats");
    }

    #[test]
    fn partial_estimation_path() {
        let er = make_estimation_report("AIC", &[2], &[5, 7]);
        let report = build_provenance_report(
            EstimationPath::PartialEstimation,
            Some(&er),
            &prov_all_generated(),
            3,
        );
        assert!(
            matches!(report.seasonal_stats_source, ProvenanceSource::UserFile),
            "seasonal_stats_source must be UserFile for PartialEstimation"
        );
        assert!(
            matches!(report.ar_coefficients_source, ProvenanceSource::Estimated),
            "ar_coefficients_source must be Estimated for PartialEstimation"
        );
        assert_eq!(report.white_noise_fallbacks, vec![5, 7]);
        assert_eq!(report.estimation_path, "partial_estimation");
    }

    #[test]
    fn user_provided_all_path() {
        let report = build_provenance_report(
            EstimationPath::UserProvidedAll,
            None,
            &prov_all_generated(),
            4,
        );
        assert!(
            matches!(report.seasonal_stats_source, ProvenanceSource::UserFile),
            "seasonal_stats_source must be UserFile for UserProvidedAll"
        );
        assert!(
            matches!(report.ar_coefficients_source, ProvenanceSource::UserFile),
            "ar_coefficients_source must be UserFile for UserProvidedAll"
        );
        assert!(
            report.ar_method.is_none(),
            "ar_method must be None when no report"
        );
        assert_eq!(report.estimation_path, "user_provided_all");
    }

    // ── ComponentProvenance mapping tests ─────────────────────────────────────

    #[test]
    fn user_supplied_tree_maps_to_user_file() {
        let report =
            build_provenance_report(EstimationPath::FullEstimation, None, &prov_user_tree(), 2);
        assert!(
            matches!(report.opening_tree_source, ProvenanceSource::UserFile),
            "UserSupplied opening tree must map to UserFile"
        );
        assert!(
            matches!(report.correlation_source, ProvenanceSource::Estimated),
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
            1,
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
        assert_eq!(value["estimation_path"], "full_estimation");
        assert_eq!(value["seasonal_stats_source"], "estimated");
        assert_eq!(value["ar_coefficients_source"], "estimated");
    }

    #[test]
    fn deterministic_json_na_variant() {
        let report = build_provenance_report(
            EstimationPath::Deterministic,
            None,
            &prov_not_applicable(),
            0,
        );
        let json = serde_json::to_string_pretty(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value["seasonal_stats_source"], "n/a",
            "NotApplicable must serialize as \"n/a\""
        );
        assert_eq!(value["ar_coefficients_source"], "n/a");
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
            2,
        );
        assert_eq!(
            report.white_noise_fallbacks,
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
            0,
        );
        assert!(
            report.white_noise_fallbacks.is_empty(),
            "white_noise_fallbacks must be empty when no estimation_report"
        );
    }
}
