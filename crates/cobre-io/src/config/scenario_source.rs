//! Raw serde types for scenario source configuration in `config.json`.
//!
//! These are intermediate deserialization types. Conversion to the canonical
//! [`cobre_core::scenario::ScenarioSource`] is performed by the helpers in
//! `config/mod.rs`.

use serde::{Deserialize, Deserializer, Serialize};

/// Per-class scenario source configuration in `config.json`.
///
/// Used by `training.scenario_source` and `simulation.scenario_source`.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RawScenarioSourceConfig {
    /// Required when any class uses the `out_of_sample` or `external` scheme.
    #[serde(default)]
    pub seed: Option<i64>,

    /// Historical year pool. Absent: the years are discovered at validation time.
    #[serde(default)]
    pub historical_years: Option<RawHistoricalYearsConfig>,

    /// Inflow class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub inflow: Option<RawClassConfigEntry>,

    /// Load class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub load: Option<RawClassConfigEntry>,

    /// NCS class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub ncs: Option<RawClassConfigEntry>,

    /// Where a stage's openings come from. Absent defaults to `generated`,
    /// preserving generation-sourced openings.
    #[serde(default)]
    pub openings: Option<Openings>,
}

/// Where a stage's openings originate (`config.json` scenario-source `source`).
///
/// The tag key is `source` and the field is `openings`, deliberately distinct
/// from the per-class `scheme` key: `openings.source` selects where a stage's
/// openings come from, while `scheme` selects how a class's noise is modelled.
/// A single word would otherwise carry both axes. An absent `openings`
/// declaration is equivalent to `generated`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Openings {
    /// Openings come from noise generation (the default when `openings` is absent).
    Generated {},
    /// Openings come from the conventional `scenarios/noise_openings.parquet` file.
    File {},
}

/// A single per-class scenario scheme in `config.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RawClassConfigEntry {
    /// Forward-pass scenario scheme for this class.
    pub scheme: RawSamplingScheme,
}

/// Per-class forward-pass scenario scheme (`config.json` scenario-source `scheme`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum RawSamplingScheme {
    /// Reuse the backward-pass opening tree for the forward pass.
    InSample,
    /// Draw fresh noise from the same distribution with an independent seed.
    OutOfSample,
    /// Draw from an externally supplied scenario file.
    External,
    /// Replay historical realisations.
    Historical,
}

impl<'de> Deserialize<'de> for RawSamplingScheme {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "in_sample" => Ok(Self::InSample),
            "out_of_sample" => Ok(Self::OutOfSample),
            "external" => Ok(Self::External),
            "historical" => Ok(Self::Historical),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["in_sample", "out_of_sample", "external", "historical"],
            )),
        }
    }
}

/// `historical_years` in `config.json`: an array of years, such as
/// `[1940, 1953, 1971]`, or an inclusive range, such as
/// `{"from": 1940, "to": 2010}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum RawHistoricalYearsConfig {
    // Untagged: `List` stays first, or a two-year array would load as a range.
    /// Explicit list of year integers.
    List(Vec<i32>),
    /// Inclusive range shorthand.
    Range(HistoricalYearRange),
}

/// Inclusive `{"from", "to"}` range form of `historical_years`.
// A named struct, not an inline variant: `deny_unknown_fields` is unenforced
// on an untagged enum's inline struct variant, and the range form must reject
// a stray key instead of silently dropping it.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct HistoricalYearRange {
    /// First year (inclusive).
    pub from: i32,
    /// Last year (inclusive).
    pub to: i32,
}
