//! Time series estimation configuration types for `config.json → estimation`.

use serde::{Deserialize, Deserializer, Serialize};

/// Order selection criterion for autoregressive model fitting.
///
/// Controls how the lag order is chosen when fitting a time series model.
/// Two variants are accepted:
///
/// - `"pacf"` — classical periodic Yule-Walker with PACF-based order
///   selection. Default.
/// - `"pacf_annual"` — extends `"pacf"` with an annual component (PAR(p)-A),
///   adding one extra coefficient ψ per (entity, season) that multiplies
///   the rolling 12-month average of past observations.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum OrderSelectionMethod {
    /// Periodic Yule-Walker partial autocorrelation method (PACF).
    #[default]
    Pacf,
    /// Periodic Yule-Walker order selection augmented with an annual component.
    ///
    /// When selected, the estimation pipeline performs four steps beyond the
    /// classical `pacf` path:
    ///
    /// 1. **Extended Yule-Walker fitting** — the system is augmented with a
    ///    cross-correlation term between the current-season inflow and the
    ///    rolling 12-month average, yielding the annual coefficient ψ
    ///    alongside the classical AR coefficients.
    /// 2. **Annual-stats computation** — per-season sample mean μ^A and
    ///    Bessel-corrected standard deviation σ^A of the rolling 12-month
    ///    average are computed for each hydro plant.
    /// 3. **Parquet emission** — the triple (ψ, μ^A, σ^A) is written to
    ///    `inflow_annual_component.parquet` in the output directory.
    /// 4. **Widened LP lag stride** — the noise-column layout in the LP is
    ///    extended to accommodate the annual term alongside the classical lags.
    PacfAnnual,
}

impl<'de> serde::Deserialize<'de> for OrderSelectionMethod {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "pacf" => Ok(Self::Pacf),
            "pacf_annual" => Ok(Self::PacfAnnual),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["pacf", "pacf_annual"],
            )),
        }
    }
}

/// Time series estimation settings (`config.json → estimation`).
///
/// Controls automatic parameter estimation when historical inflow data is
/// provided without explicit model statistics or coefficients.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EstimationConfig {
    /// Maximum lag order considered during autoregressive model fitting.
    pub max_order: u32,

    /// Order selection criterion. Accepts `"pacf"` (classical PACF, default)
    /// or `"pacf_annual"` (PACF augmented with an annual component, PAR(p)-A).
    pub order_selection: OrderSelectionMethod,

    /// Minimum number of observations required per (entity, season) group
    /// to proceed with estimation. Groups below this threshold are skipped.
    pub min_observations_per_season: u32,

    /// Maximum allowed absolute magnitude for any AR coefficient.
    ///
    /// When set, any (entity, season) pair with `|coefficient| > threshold`
    /// is immediately reduced to order 0 before the contribution analysis
    /// runs. This acts as a fast-path safety net for the most extreme
    /// explosive models. Absent by default (disabled; contribution analysis
    /// is the primary guard).
    #[serde(default)]
    pub max_coefficient_magnitude: Option<f64>,
}

impl Default for EstimationConfig {
    fn default() -> Self {
        Self {
            max_order: 6,
            order_selection: OrderSelectionMethod::Pacf,
            min_observations_per_season: 30,
            max_coefficient_magnitude: None,
        }
    }
}
