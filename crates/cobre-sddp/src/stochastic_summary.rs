//! Stochastic preprocessing summary types and builder.
//!
//! Provides [`StochasticSummary`], [`ArOrderSummary`], and [`StochasticSource`] for
//! reporting the outcome of the stochastic preprocessing pipeline, plus the
//! [`build_stochastic_summary`] constructor that derives all fields from a validated
//! [`cobre_stochastic::StochasticContext`].
//!
//! These types live in `cobre-sddp` so that they can be reused by Python bindings and
//! other callers without pulling in CLI-specific display dependencies. Display/formatting
//! methods that use `console::style` remain in `cobre-cli`.

use cobre_core::{scenario::SamplingScheme, System};
use cobre_io::output::{FittingReductionEntry, FittingReport, HydroFittingEntry};
use cobre_io::scenarios::{InflowArCoefficientRow, InflowSeasonalStatsRow};
use cobre_stochastic::{ComponentProvenance, StochasticContext};

use crate::EstimationReport;

// ── Public types ─────────────────────────────────────────────────────────────

/// Source of stochastic data for a given component.
#[derive(Debug, Clone)]
pub enum StochasticSource {
    /// Data was estimated from historical records.
    Estimated,
    /// Data was loaded from user-supplied files.
    Loaded,
    /// No data available (component not modeled).
    None,
}

/// Summary of AR order selection across hydro plants.
#[derive(Debug, Clone)]
pub struct ArOrderSummary {
    /// Method used for order selection (e.g., `"AIC"`, `"fixed"`).
    pub method: String,
    /// Count of hydros at each AR order. Index = order, value = count.
    ///
    /// For example, `[0, 3, 2]` means 0 hydros at order 0, 3 at order 1,
    /// 2 at order 2.
    pub order_counts: Vec<usize>,
    /// Minimum AR order across all hydros.
    pub min_order: usize,
    /// Maximum AR order across all hydros.
    pub max_order: usize,
    /// Number of hydro plants included in the summary.
    pub n_hydros: usize,
}

impl ArOrderSummary {
    /// Render a compact human-readable string describing the AR order distribution.
    ///
    /// Three tiers based on the number of hydro plants:
    ///
    /// - **≤10 hydros**: compact distribution, e.g. `"AIC (3x order-1, 2x order-2)"`.
    /// - **11–30 hydros**: range format, e.g. `"AIC (orders 1-4, 15 hydros)"`.
    /// - **31+ hydros**: histogram format, e.g. `"AIC (order 1: 12, order 2: 8, 31 hydros)"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::stochastic_summary::ArOrderSummary;
    ///
    /// let s = ArOrderSummary {
    ///     method: "AIC".into(),
    ///     order_counts: vec![0, 3, 2],
    ///     min_order: 1,
    ///     max_order: 2,
    ///     n_hydros: 5,
    /// };
    /// let text = s.display_string();
    /// assert!(text.contains("3x order-1"));
    /// assert!(text.contains("2x order-2"));
    /// ```
    #[must_use]
    pub fn display_string(&self) -> String {
        if self.n_hydros <= 10 {
            let parts: Vec<String> = self
                .order_counts
                .iter()
                .enumerate()
                .filter(|&(_, count)| *count > 0)
                .map(|(order, count)| format!("{count}x order-{order}"))
                .collect();
            format!("{} ({})", self.method, parts.join(", "))
        } else if self.n_hydros <= 30 {
            format!(
                "{} (orders {}-{}, {} hydros)",
                self.method, self.min_order, self.max_order, self.n_hydros
            )
        } else {
            let parts: Vec<String> = self
                .order_counts
                .iter()
                .enumerate()
                .filter(|&(_, count)| *count > 0)
                .map(|(order, count)| format!("order {order}: {count}"))
                .collect();
            format!(
                "{} ({}, {} hydros)",
                self.method,
                parts.join(", "),
                self.n_hydros
            )
        }
    }
}

/// Summary of the stochastic preprocessing pipeline for display.
#[derive(Debug, Clone)]
pub struct StochasticSummary {
    /// Source of inflow seasonal statistics.
    pub inflow_source: StochasticSource,
    /// Number of hydro plants in the system.
    pub n_hydros: usize,
    /// Number of seasons in the PAR model.
    pub n_seasons: usize,
    /// AR order summary (`None` if no hydros or no AR model).
    pub ar_summary: Option<ArOrderSummary>,
    /// Source of correlation data.
    pub correlation_source: StochasticSource,
    /// Dimension of the correlation matrix (e.g., `"5x5"`).
    pub correlation_dim: Option<String>,
    /// Source of the opening tree.
    pub opening_tree_source: StochasticSource,
    /// Number of openings at each stage.
    pub openings_per_stage: Vec<usize>,
    /// Number of stages in the stochastic context.
    pub n_stages: usize,
    /// Number of buses with stochastic load noise.
    pub n_load_buses: usize,
    /// Number of stochastic NCS entities in the noise dimension.
    pub n_stochastic_ncs: usize,
    /// Sampling scheme for the inflow entity class (`None` when not configured).
    pub inflow_scheme: Option<SamplingScheme>,
    /// Sampling scheme for the load entity class (`None` when not configured).
    pub load_scheme: Option<SamplingScheme>,
    /// Sampling scheme for the NCS entity class (`None` when not configured).
    pub ncs_scheme: Option<SamplingScheme>,
    /// Random seed used for noise generation.
    pub seed: u64,
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Build the AR order summary from the estimation report or loaded inflow models.
///
/// Returns `None` when `n_hydros == 0`.
fn build_ar_order_summary(
    system: &System,
    estimation_report: Option<&EstimationReport>,
    n_hydros: usize,
) -> Option<ArOrderSummary> {
    if n_hydros == 0 {
        return None;
    }

    let (method, orders): (String, Vec<usize>) = if let Some(report) = estimation_report {
        let orders: Vec<usize> = report
            .entries
            .values()
            .map(|entry| entry.selected_order as usize)
            .collect();
        (report.method.clone(), orders)
    } else {
        // Derive from loaded inflow models: use max AR coefficient length per hydro.
        let orders: Vec<usize> = system
            .hydros()
            .iter()
            .map(|h| {
                system
                    .inflow_models()
                    .iter()
                    .filter(|m| m.hydro_id == h.id)
                    .map(|m| m.ar_coefficients.len())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        ("fixed".to_string(), orders)
    };

    let min_order = orders.iter().copied().min().unwrap_or(0);
    let max_order = orders.iter().copied().max().unwrap_or(0);

    let mut order_counts = vec![0usize; max_order + 1];
    for &ord in &orders {
        order_counts[ord] += 1;
    }

    Some(ArOrderSummary {
        method,
        order_counts,
        min_order,
        max_order,
        n_hydros,
    })
}

// ── Builder function ──────────────────────────────────────────────────────────

/// Build a [`StochasticSummary`] from the system, stochastic context, and estimation report.
///
/// Called after [`cobre_stochastic::build_stochastic_context`] returns and before training
/// starts. All fields are derived from the already-validated inputs; construction is
/// infallible.
///
/// # Source detection
///
/// - **Inflow source**: `Estimated` when `estimation_report` is `Some(_)` — the estimation
///   pipeline ran. `Loaded` when hydros are present but no report. `None` when no hydros.
/// - **Correlation source**: determined by `stochastic.provenance().correlation`. When
///   `Generated`, source follows estimation report presence (`Estimated` or `Loaded`).
///   `UserSupplied` maps to `Loaded`. `NotApplicable` maps to `None`.
/// - **Opening tree source**: determined by `stochastic.provenance().opening_tree` directly.
///   `UserSupplied` → `Loaded`, `Generated` → `Estimated`, `NotApplicable` → `None`.
#[must_use]
pub fn build_stochastic_summary(
    system: &System,
    stochastic: &StochasticContext,
    estimation_report: Option<&EstimationReport>,
    seed: u64,
) -> StochasticSummary {
    let n_hydros = system.hydros().len();

    // Determine inflow source from estimation report presence.
    let inflow_source = if estimation_report.is_some() {
        StochasticSource::Estimated
    } else if n_hydros > 0 {
        StochasticSource::Loaded
    } else {
        StochasticSource::None
    };

    // Count distinct stage_id values across all hydros' inflow models.
    let n_seasons = if n_hydros > 0 {
        let mut stage_ids: Vec<i32> = system.inflow_models().iter().map(|m| m.stage_id).collect();
        stage_ids.sort_unstable();
        stage_ids.dedup();
        stage_ids.len()
    } else {
        0
    };

    // Build AR order summary from estimation report or inflow model coefficients.
    let ar_summary = build_ar_order_summary(system, estimation_report, n_hydros);

    // Correlation source from provenance — replaces the heuristic that mirrored
    // inflow source. When the correlation was Generated from system data, its
    // provenance still cannot distinguish estimated vs loaded on its own; the
    // estimation report presence is the tiebreaker.
    let correlation_source = match stochastic.provenance().correlation {
        ComponentProvenance::Generated => {
            // Correlation was decomposed from system data. If estimation ran,
            // the correlation matrix was estimated; otherwise it was loaded.
            if estimation_report.is_some() {
                StochasticSource::Estimated
            } else {
                StochasticSource::Loaded
            }
        }
        ComponentProvenance::UserSupplied => StochasticSource::Loaded,
        ComponentProvenance::NotApplicable => StochasticSource::None,
    };

    // Correlation dimension spans all correlated entities: hydros + load buses + NCS.
    // `stochastic.dim()` returns `n_hydros + n_load_buses + n_stochastic_ncs`, which is
    // the full noise dimension that the spectral decomposition operates on.
    let n_correlated = stochastic.dim();
    let correlation_dim = if n_correlated > 0 {
        Some(format!("{n_correlated}x{n_correlated}"))
    } else {
        None
    };

    // Opening tree source from provenance. The old heuristic (based on opening count)
    // could not distinguish a user-supplied tree with one opening/stage from a
    // generated tree. Provenance records this unambiguously.
    let opening_tree_source = match stochastic.provenance().opening_tree {
        ComponentProvenance::UserSupplied => StochasticSource::Loaded,
        ComponentProvenance::Generated => StochasticSource::Estimated,
        ComponentProvenance::NotApplicable => StochasticSource::None,
    };

    let opening_tree = stochastic.opening_tree();
    let openings_per_stage: Vec<usize> = opening_tree.openings_per_stage_slice().to_vec();
    let n_stages = stochastic.n_stages();
    let n_load_buses = stochastic.n_load_buses();
    let n_stochastic_ncs = stochastic.n_stochastic_ncs();
    let provenance = stochastic.provenance();

    StochasticSummary {
        inflow_source,
        n_hydros,
        n_seasons,
        ar_summary,
        correlation_source,
        correlation_dim,
        opening_tree_source,
        openings_per_stage,
        n_stages,
        n_load_buses,
        n_stochastic_ncs,
        inflow_scheme: provenance.inflow_scheme,
        load_scheme: provenance.load_scheme,
        ncs_scheme: provenance.ncs_scheme,
        seed,
    }
}

// ── Stochastic artifact conversion helpers ────────────────────────────────────

/// Convert [`EstimationReport`] to [`FittingReport`] with entity IDs as strings.
#[must_use]
pub fn estimation_report_to_fitting_report(report: &EstimationReport) -> FittingReport {
    let hydros = report
        .entries
        .iter()
        .map(
            |(id, entry): (
                &cobre_core::EntityId,
                &crate::estimation::HydroEstimationEntry,
            )| {
                (
                    id.0.to_string(),
                    HydroFittingEntry {
                        selected_order: entry.selected_order,
                        coefficients: entry.coefficients.clone(),
                        contribution_reductions: entry
                            .contribution_reductions
                            .iter()
                            .map(|r| FittingReductionEntry {
                                season_id: r.season_id,
                                original_order: r.original_order,
                                reduced_order: r.reduced_order,
                                contributions: r.contributions.clone(),
                                reason: r.reason.as_str().to_string(),
                            })
                            .collect(),
                    },
                )
            },
        )
        .collect();
    FittingReport { hydros }
}

/// Convert inflow models to seasonal stats rows (one row per model).
#[must_use]
pub fn inflow_models_to_stats_rows(
    models: &[cobre_core::scenario::InflowModel],
) -> Vec<InflowSeasonalStatsRow> {
    models
        .iter()
        .map(|m| InflowSeasonalStatsRow {
            hydro_id: m.hydro_id,
            stage_id: m.stage_id,
            mean_m3s: m.mean_m3s,
            std_m3s: m.std_m3s,
        })
        .collect()
}

/// Convert inflow models to AR coefficient rows (one row per lag).
///
/// Each model's `ar_coefficients` expands into rows with 1-based lag indices.
/// White-noise models (AR order 0) produce no rows.
#[must_use]
pub fn inflow_models_to_ar_rows(
    models: &[cobre_core::scenario::InflowModel],
) -> Vec<InflowArCoefficientRow> {
    models
        .iter()
        .flat_map(|m| {
            m.ar_coefficients
                .iter()
                .enumerate()
                .map(move |(i, &coefficient)| {
                    // AR order is bounded by a small integer (typical range 1-12);
                    // the cast from usize to i32 is safe in practice.
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    let lag = (i + 1) as i32;
                    InflowArCoefficientRow {
                        hydro_id: m.hydro_id,
                        stage_id: m.stage_id,
                        lag,
                        coefficient,
                        residual_std_ratio: m.residual_std_ratio,
                    }
                })
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::NaiveDate;
    use cobre_core::{
        entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        scenario::{CorrelationModel, InflowModel, SamplingScheme},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
        Bus, DeficitSegment, EntityId, SystemBuilder,
    };
    use cobre_stochastic::{build_stochastic_context, ClassSchemes, OpeningTreeInputs};

    use super::{
        build_stochastic_summary, estimation_report_to_fitting_report, inflow_models_to_ar_rows,
        inflow_models_to_stats_rows, StochasticSource,
    };
    use crate::estimation::HydroEstimationEntry;
    use crate::EstimationReport;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
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
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.95,
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

    fn make_stage(idx: usize, id: i32) -> Stage {
        Stage {
            index: idx,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(id.unsigned_abs() as usize),
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
        }
    }

    fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![0.5, 0.3],
            residual_std_ratio: 0.8,
        }
    }

    fn identity_correlation(entity_ids: &[i32]) -> cobre_core::scenario::CorrelationModel {
        use cobre_core::scenario::{CorrelationEntity, CorrelationGroup, CorrelationProfile};
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
        cobre_core::scenario::CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    /// Build a minimal `System` with one hydro, one bus, two study stages,
    /// and two `InflowModel` entries (one per stage, AR order 2).
    fn make_system_with_hydro() -> cobre_core::System {
        let stages = vec![make_stage(0, 0), make_stage(1, 1)];
        let inflow_models = vec![make_inflow_model(10, 0), make_inflow_model(10, 1)];

        SystemBuilder::new()
            .buses(vec![make_bus(1)])
            .hydros(vec![make_hydro(10)])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(CorrelationModel::default())
            .build()
            .unwrap()
    }

    // ── estimation_report_to_fitting_report tests ─────────────────────────────

    #[test]
    fn estimation_report_to_fitting_report_two_hydros() {
        let mut entries = BTreeMap::new();
        entries.insert(
            EntityId(1),
            HydroEstimationEntry {
                selected_order: 3,
                coefficients: vec![vec![0.4, -0.1, 0.05], vec![0.3, -0.08, 0.04]],
                contribution_reductions: Vec::new(),
            },
        );
        entries.insert(
            EntityId(5),
            HydroEstimationEntry {
                selected_order: 2,
                coefficients: vec![vec![0.6, -0.2]],
                contribution_reductions: Vec::new(),
            },
        );
        let report = EstimationReport {
            entries,
            method: "AIC".to_string(),
            white_noise_fallbacks: Vec::new(),
            lag_scale_warnings: Vec::new(),
            std_ratio_warnings: Vec::new(),
        };
        let fitting = estimation_report_to_fitting_report(&report);

        assert_eq!(
            fitting.hydros.len(),
            2,
            "FittingReport must contain exactly 2 hydro entries"
        );

        let h1 = fitting.hydros.get("1").unwrap();
        assert_eq!(h1.selected_order, 3);
        assert_eq!(h1.coefficients.len(), 2);

        let h5 = fitting.hydros.get("5").unwrap();
        assert_eq!(h5.selected_order, 2);
        assert_eq!(h5.coefficients, vec![vec![0.6, -0.2]]);
    }

    // ── inflow_models_to_stats_rows tests ─────────────────────────────────────

    #[test]
    fn inflow_models_to_stats_rows_field_values() {
        let models = vec![
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: 150.0,
                std_m3s: 30.0,
                ar_coefficients: vec![0.5],
                residual_std_ratio: 0.87,
            },
            InflowModel {
                hydro_id: EntityId(2),
                stage_id: 1,
                mean_m3s: 200.0,
                std_m3s: 40.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];
        let rows = inflow_models_to_stats_rows(&models);

        assert_eq!(rows.len(), 2, "must produce one row per model");
        assert_eq!(rows[0].hydro_id, EntityId(1));
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].mean_m3s, 150.0);
        assert_eq!(rows[0].std_m3s, 30.0);
        assert_eq!(rows[1].hydro_id, EntityId(2));
        assert_eq!(rows[1].mean_m3s, 200.0);
    }

    // ── inflow_models_to_ar_rows tests ────────────────────────────────────────

    #[test]
    fn inflow_models_to_ar_rows_lag_numbering_and_count() {
        let models = vec![
            InflowModel {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.4, -0.1, 0.05],
                residual_std_ratio: 0.92,
            },
            InflowModel {
                hydro_id: EntityId(2),
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 15.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            },
        ];
        let rows = inflow_models_to_ar_rows(&models);

        // 3 rows for hydro 1 (order 3), 0 rows for hydro 2 (order 0).
        assert_eq!(rows.len(), 3, "must produce 3 rows total (3 + 0)");

        assert_eq!(rows[0].hydro_id, EntityId(1));
        assert_eq!(rows[0].lag, 1, "first lag must be 1 (1-based)");
        assert_eq!(rows[0].coefficient, 0.4);
        assert_eq!(rows[0].residual_std_ratio, 0.92);

        assert_eq!(rows[1].lag, 2);
        assert_eq!(rows[1].coefficient, -0.1);

        assert_eq!(rows[2].lag, 3);
        assert_eq!(rows[2].coefficient, 0.05);
    }

    // ── build_stochastic_summary tests ───────────────────────────────────────

    #[test]
    fn build_stochastic_summary_loaded_source_when_no_estimation_report() {
        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(
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
        let summary = build_stochastic_summary(&system, &stochastic, None, 42);

        assert!(
            matches!(summary.inflow_source, StochasticSource::Loaded),
            "inflow_source must be Loaded when no estimation report is present"
        );
        assert_eq!(summary.n_hydros, 1, "n_hydros must be 1");
        assert_eq!(
            summary.n_seasons, 2,
            "n_seasons must be 2 (stage 0 and stage 1)"
        );
        assert_eq!(summary.seed, 42, "seed must be 42");
    }

    #[test]
    fn build_stochastic_summary_estimated_source_with_estimation_report() {
        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(
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

        let mut entries = BTreeMap::new();
        entries.insert(
            EntityId(10),
            HydroEstimationEntry {
                selected_order: 2,
                coefficients: vec![vec![0.5, 0.3], vec![0.4, 0.2]],
                contribution_reductions: Vec::new(),
            },
        );
        let report = EstimationReport {
            entries,
            method: "AIC".to_string(),
            white_noise_fallbacks: Vec::new(),
            lag_scale_warnings: Vec::new(),
            std_ratio_warnings: Vec::new(),
        };

        let summary = build_stochastic_summary(&system, &stochastic, Some(&report), 7);

        assert!(
            matches!(summary.inflow_source, StochasticSource::Estimated),
            "inflow_source must be Estimated when estimation report is present"
        );
        // With CorrelationModel::default() (empty profiles), provenance is NotApplicable → None.
        assert!(
            matches!(summary.correlation_source, StochasticSource::None),
            "correlation_source must be None for empty correlation model, got {:?}",
            summary.correlation_source
        );
        let ar = summary.ar_summary.as_ref().unwrap();
        assert_eq!(ar.method, "AIC", "AR method must be AIC");
        assert_eq!(ar.max_order, 2, "max AR order must be 2");
        assert_eq!(summary.seed, 7, "seed must be 7");
    }

    #[test]
    fn build_stochastic_summary_no_hydros_yields_none_source() {
        let stages = vec![make_stage(0, 0)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(1)])
            .stages(stages)
            .correlation(CorrelationModel::default())
            .build()
            .unwrap();

        let stochastic = build_stochastic_context(
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
        let summary = build_stochastic_summary(&system, &stochastic, None, 0);

        assert!(
            matches!(summary.inflow_source, StochasticSource::None),
            "inflow_source must be None when there are no hydros"
        );
        assert_eq!(summary.n_hydros, 0, "n_hydros must be 0");
        assert!(
            summary.ar_summary.is_none(),
            "ar_summary must be None with no hydros"
        );
    }

    #[test]
    fn build_stochastic_summary_stages_and_load_buses() {
        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(
            &system,
            1,
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
        let summary = build_stochastic_summary(&system, &stochastic, None, 1);

        assert_eq!(
            summary.n_stages, 2,
            "n_stages must match stochastic context"
        );
        assert_eq!(
            summary.n_load_buses, 0,
            "n_load_buses must be 0 (no stochastic load)"
        );
    }

    // ── Provenance-based opening tree source tests ────────────────────────────

    #[test]
    fn opening_tree_source_user_supplied() {
        use cobre_stochastic::context::OpeningTree;

        let system = make_system_with_hydro();
        // 2 stages × 2 openings × 1 dim = 4 entries
        let user_tree = OpeningTree::from_parts(vec![1.0_f64; 2 * 2], vec![2, 2], 1);
        let stochastic = build_stochastic_context(
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
        let summary = build_stochastic_summary(&system, &stochastic, None, 42);

        assert!(
            matches!(summary.opening_tree_source, StochasticSource::Loaded),
            "user-supplied opening tree must produce opening_tree_source == Loaded, got {:?}",
            summary.opening_tree_source
        );
    }

    #[test]
    fn opening_tree_source_generated() {
        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(
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
        let summary = build_stochastic_summary(&system, &stochastic, None, 42);

        assert!(
            matches!(summary.opening_tree_source, StochasticSource::Estimated),
            "generated opening tree must produce opening_tree_source == Estimated, got {:?}",
            summary.opening_tree_source
        );
    }

    #[test]
    fn correlation_source_estimated_when_estimation_ran() {
        let stages = vec![make_stage(0, 0), make_stage(1, 1)];
        let inflow_models = vec![make_inflow_model(10, 0), make_inflow_model(10, 1)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(1)])
            .hydros(vec![make_hydro(10)])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[10]))
            .build()
            .unwrap();

        let stochastic = build_stochastic_context(
            &system,
            1,
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

        let mut entries = BTreeMap::new();
        entries.insert(
            EntityId(10),
            HydroEstimationEntry {
                selected_order: 1,
                coefficients: vec![vec![0.4]],
                contribution_reductions: Vec::new(),
            },
        );
        let report = EstimationReport {
            entries,
            method: "AIC".to_string(),
            white_noise_fallbacks: Vec::new(),
            lag_scale_warnings: Vec::new(),
            std_ratio_warnings: Vec::new(),
        };

        let summary = build_stochastic_summary(&system, &stochastic, Some(&report), 1);

        assert!(
            matches!(summary.correlation_source, StochasticSource::Estimated),
            "Generated correlation + estimation report must produce Estimated, got {:?}",
            summary.correlation_source
        );
    }

    #[test]
    fn correlation_source_loaded_when_no_estimation() {
        let stages = vec![make_stage(0, 0), make_stage(1, 1)];
        let inflow_models = vec![make_inflow_model(10, 0), make_inflow_model(10, 1)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(1)])
            .hydros(vec![make_hydro(10)])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[10]))
            .build()
            .unwrap();

        let stochastic = build_stochastic_context(
            &system,
            1,
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
        let summary = build_stochastic_summary(&system, &stochastic, None, 1);

        assert!(
            matches!(summary.correlation_source, StochasticSource::Loaded),
            "Generated correlation + no estimation report must produce Loaded, got {:?}",
            summary.correlation_source
        );
    }

    #[test]
    fn correlation_source_none_when_empty() {
        let stages = vec![make_stage(0, 0)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(1)])
            .stages(stages)
            .correlation(CorrelationModel::default())
            .build()
            .unwrap();

        let stochastic = build_stochastic_context(
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
        let summary = build_stochastic_summary(&system, &stochastic, None, 0);

        assert!(
            matches!(summary.correlation_source, StochasticSource::None),
            "NotApplicable correlation must produce None, got {:?}",
            summary.correlation_source
        );
    }

    // ── build_stochastic_summary field and correlation_dim tests ─────────────

    /// Verify that the new per-class scheme fields are populated from provenance
    /// and that `correlation_dim` reflects the full noise dimension (not just
    /// `n_hydros`).
    #[test]
    fn build_stochastic_summary_new_fields_and_correlation_dim() {
        let system = make_system_with_hydro();
        let stochastic = build_stochastic_context(
            &system,
            99,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::OutOfSample),
                ncs: Some(SamplingScheme::Historical),
            },
        )
        .unwrap();

        let summary = build_stochastic_summary(&system, &stochastic, None, 99);

        // Per-class scheme fields must reflect what was passed to the context.
        assert_eq!(
            summary.inflow_scheme,
            Some(SamplingScheme::InSample),
            "inflow_scheme must be InSample"
        );
        assert_eq!(
            summary.load_scheme,
            Some(SamplingScheme::OutOfSample),
            "load_scheme must be OutOfSample"
        );
        assert_eq!(
            summary.ncs_scheme,
            Some(SamplingScheme::Historical),
            "ncs_scheme must be Historical"
        );

        // n_stochastic_ncs must be 0 for a hydro-only system.
        assert_eq!(
            summary.n_stochastic_ncs, 0,
            "n_stochastic_ncs must be 0 when no NCS entities"
        );

        // correlation_dim must be derived from stochastic.dim(), not just n_hydros.
        // For this single-hydro system with no load buses and no NCS: dim == 1.
        let expected_dim = stochastic.dim();
        let expected_str = format!("{expected_dim}x{expected_dim}");
        // correlation_dim is Some("NxN") when dim > 0, None when dim == 0.
        // The dimension reflects the noise vector size, not the correlation model presence.
        assert_eq!(
            summary.correlation_dim,
            if stochastic.dim() > 0 {
                Some(expected_str)
            } else {
                None
            },
            "correlation_dim must be derived from stochastic.dim()"
        );
    }

    /// Verify that `correlation_dim` for a system with a real correlation model
    /// uses `dim()` (hydros + load buses + NCS), not just `n_hydros`.
    #[test]
    fn build_stochastic_summary_correlation_dim_uses_full_dim() {
        let stages = vec![make_stage(0, 0), make_stage(1, 1)];
        let inflow_models = vec![make_inflow_model(10, 0), make_inflow_model(10, 1)];
        let system = SystemBuilder::new()
            .buses(vec![make_bus(1)])
            .hydros(vec![make_hydro(10)])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[10]))
            .build()
            .unwrap();

        let stochastic = build_stochastic_context(
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

        let summary = build_stochastic_summary(&system, &stochastic, None, 0);

        // One hydro, zero load buses, zero NCS: dim == 1.
        let dim = stochastic.dim();
        assert_eq!(dim, 1, "expected dim == 1 for single-hydro system");
        assert_eq!(
            summary.correlation_dim,
            Some("1x1".to_string()),
            "correlation_dim must be '1x1' for a single-hydro system (dim==1)"
        );
    }

    // ── Reduction reason tests ─────────────────────────────────────────────────

    #[test]
    fn fitting_report_includes_reason_strings() {
        use crate::estimation::{
            ContributionReduction, EstimationReport, HydroEstimationEntry, ReductionReason,
        };

        let mut entries = BTreeMap::new();
        entries.insert(
            EntityId(1),
            HydroEstimationEntry {
                selected_order: 0,
                coefficients: vec![vec![], vec![]],
                contribution_reductions: vec![
                    ContributionReduction {
                        season_id: 0,
                        original_order: 2,
                        reduced_order: 0,
                        contributions: Vec::new(),
                        reason: ReductionReason::Phi1Negative,
                    },
                    ContributionReduction {
                        season_id: 1,
                        original_order: 3,
                        reduced_order: 1,
                        contributions: vec![0.3, -0.1],
                        reason: ReductionReason::NegativeContribution,
                    },
                ],
            },
        );
        entries.insert(
            EntityId(2),
            HydroEstimationEntry {
                selected_order: 0,
                coefficients: vec![vec![]],
                contribution_reductions: vec![ContributionReduction {
                    season_id: 0,
                    original_order: 1,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::MagnitudeBound,
                }],
            },
        );

        let report = EstimationReport {
            entries,
            method: "PACF".to_string(),
            white_noise_fallbacks: Vec::new(),
            lag_scale_warnings: Vec::new(),
            std_ratio_warnings: Vec::new(),
        };

        let fitting = estimation_report_to_fitting_report(&report);

        // Hydro 1 should have 2 reductions with correct reason strings.
        let h1 = fitting.hydros.get("1").unwrap();
        assert_eq!(h1.contribution_reductions.len(), 2);
        assert_eq!(h1.contribution_reductions[0].reason, "phi1_negative");
        assert_eq!(
            h1.contribution_reductions[1].reason,
            "negative_contribution"
        );

        // Hydro 2 should have 1 reduction with magnitude_bound.
        let h2 = fitting.hydros.get("2").unwrap();
        assert_eq!(h2.contribution_reductions.len(), 1);
        assert_eq!(h2.contribution_reductions[0].reason, "magnitude_bound");
    }

    #[test]
    fn estimation_report_tracks_all_reductions() {
        use crate::estimation::{build_estimation_report, ContributionReduction, ReductionReason};
        use std::collections::HashMap;

        let estimates = vec![
            cobre_stochastic::par::fitting::ArCoefficientEstimate {
                hydro_id: EntityId(1),
                season_id: 0,
                coefficients: Vec::new(),
                residual_std_ratio: 1.0,
            },
            cobre_stochastic::par::fitting::ArCoefficientEstimate {
                hydro_id: EntityId(1),
                season_id: 1,
                coefficients: vec![0.4],
                residual_std_ratio: 0.9,
            },
        ];

        let mut reductions: HashMap<EntityId, Vec<ContributionReduction>> = HashMap::new();
        reductions.insert(
            EntityId(1),
            vec![
                ContributionReduction {
                    season_id: 0,
                    original_order: 2,
                    reduced_order: 0,
                    contributions: Vec::new(),
                    reason: ReductionReason::Phi1Negative,
                },
                ContributionReduction {
                    season_id: 1,
                    original_order: 3,
                    reduced_order: 1,
                    contributions: vec![0.3, -0.2],
                    reason: ReductionReason::NegativeContribution,
                },
            ],
        );

        let report = build_estimation_report(&estimates, 2, &reductions, "PACF");

        let entry = report.entries.get(&EntityId(1)).unwrap();
        assert_eq!(entry.contribution_reductions.len(), 2);
        assert_eq!(
            entry.contribution_reductions[0].reason,
            ReductionReason::Phi1Negative
        );
        assert_eq!(
            entry.contribution_reductions[1].reason,
            ReductionReason::NegativeContribution
        );
    }
}
