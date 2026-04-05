//! Scenario pipeline raw data types — PAR model parameters, load statistics,
//! and correlation model.
//!
//! This module defines the clarity-first data containers for the raw scenario
//! pipeline parameters loaded from input files. These are the data types stored
//! in [`System`](crate::System) and passed to downstream crates for processing.
//!
//! ## Dual-nature design
//!
//! Following the dual-nature design principle (internal-structures.md §1.1),
//! this module holds only the **raw input-facing types**:
//!
//! - [`InflowModel`] — PAR(p) parameters per (hydro, stage). AR coefficients
//!   are stored **standardized by seasonal std** (dimensionless ψ\*), and
//!   `residual_std_ratio` (`σ_m` / `s_m`) captures the remaining variance not
//!   explained by the AR model. Downstream crates recover the runtime residual
//!   std as `std_m3s * residual_std_ratio`.
//! - [`LoadModel`] — seasonal load statistics per (bus, stage)
//! - [`CorrelationModel`] — named correlation profiles with entity groups
//!   and correlation matrices
//!
//! Performance-adapted views (`PrecomputedPar`, Cholesky-decomposed matrices)
//! belong in downstream solver crates (`cobre-stochastic`).
//!
//! ## Declaration-order invariance
//!
//! [`CorrelationModel::profiles`] uses [`BTreeMap`] to preserve deterministic
//! ordering of named profiles, ensuring bit-for-bit identical behaviour
//! regardless of the order in which profiles appear in `correlation.json`.
//!
//! Source: `inflow_seasonal_stats.parquet`, `inflow_ar_coefficients.parquet`,
//! `load_seasonal_stats.parquet`, `correlation.json`.
//! See [internal-structures.md §14](../specs/data-model/internal-structures.md)
//! and [Input Scenarios §2–5](../specs/data-model/input-scenarios.md).

use std::collections::BTreeMap;

use crate::EntityId;

/// Forward-pass noise source for multi-stage optimization solvers.
///
/// Determines where the forward-pass scenario realisations come from.
/// This is orthogonal to [`NoiseMethod`](crate::temporal::NoiseMethod),
/// which controls how the opening tree is generated during the backward
/// pass. `SamplingScheme` selects the *source* of forward-pass noise;
/// `NoiseMethod` selects the *algorithm* used to produce backward-pass
/// openings.
///
/// See [Input Scenarios §1.8](input-scenarios.md) for the full catalog.
///
/// # Examples
///
/// ```
/// use cobre_core::scenario::SamplingScheme;
///
/// let scheme = SamplingScheme::InSample;
/// // SamplingScheme is Copy
/// let copy = scheme;
/// assert_eq!(scheme, copy);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplingScheme {
    /// Forward pass uses the same opening tree generated for the backward pass.
    /// This is the default for the minimal viable solver.
    InSample,
    /// Forward pass generates fresh noise on-the-fly from the same distribution
    /// as the opening tree, using an independent seed.
    OutOfSample,
    /// Forward pass draws from an externally supplied scenario file.
    External,
    /// Forward pass replays historical inflow realisations in sequence or at random.
    Historical,
}

// ScenarioSource (SS14 top-level config)

/// Top-level scenario source configuration, parsed from `stages.json`.
///
/// Groups the sampling scheme and random seed that govern how forward-pass
/// scenarios are produced. Populated during case loading by `cobre-io` from
/// the `scenario_source` field in `stages.json`. Distinct from
/// [`ScenarioSourceConfig`](crate::temporal::ScenarioSourceConfig),
/// which also holds the branching factor (`num_scenarios`).
///
/// Each entity class (inflow, load, NCS) independently specifies its
/// forward-pass noise source via a dedicated `SamplingScheme` field.
/// The `seed` and `historical_years` fields are shared across all classes.
///
/// See [Input Scenarios §1.4, §1.8](input-scenarios.md).
///
/// # Examples
///
/// ```
/// use cobre_core::scenario::{SamplingScheme, ScenarioSource};
///
/// let source = ScenarioSource {
///     inflow_scheme: SamplingScheme::InSample,
///     load_scheme: SamplingScheme::OutOfSample,
///     ncs_scheme: SamplingScheme::InSample,
///     seed: Some(42),
///     historical_years: None,
/// };
/// assert_eq!(source.inflow_scheme, SamplingScheme::InSample);
/// assert_eq!(source.load_scheme, SamplingScheme::OutOfSample);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScenarioSource {
    /// Noise source used during the inflow forward pass.
    pub inflow_scheme: SamplingScheme,

    /// Noise source used during the load forward pass.
    pub load_scheme: SamplingScheme,

    /// Noise source used during the NCS (non-controllable source) forward pass.
    pub ncs_scheme: SamplingScheme,

    /// Random seed for reproducible opening tree generation.
    /// `None` means non-deterministic (OS entropy).
    pub seed: Option<i64>,

    /// Historical year pool for [`SamplingScheme::Historical`] inflow sampling.
    /// When `None`, all valid windows are auto-discovered at validation time.
    pub historical_years: Option<HistoricalYears>,
}

// HistoricalYears (SS14 — year pool for Historical sampling)

/// Specifies which historical years to draw from when using
/// [`SamplingScheme::Historical`] sampling.
///
/// Preserves user intent (list vs range) so that validation and error messages
/// can reference the original specification form. Expansion into a concrete
/// year list is deferred to `cobre-io` validation (Tier 1) and Epic 04 library
/// construction.
///
/// When absent (represented as `Option<HistoricalYears>::None` at the
/// `ScenarioSource` level), all valid windows are auto-discovered at
/// validation time.
///
/// # Examples
///
/// ```
/// use cobre_core::scenario::HistoricalYears;
///
/// // Explicit list of years
/// let list = HistoricalYears::List(vec![1940, 1953, 1971]);
/// assert!(matches!(list, HistoricalYears::List(_)));
///
/// // Inclusive range shorthand
/// let range = HistoricalYears::Range { from: 1940, to: 2010 };
/// assert!(matches!(range, HistoricalYears::Range { from: 1940, to: 2010 }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HistoricalYears {
    /// Explicit list of historical years (e.g., `[1940, 1953, 1971]`).
    List(Vec<i32>),

    /// Inclusive range shorthand (e.g., years 1940 through 2010).
    /// `from` and `to` are both inclusive. Validation of `from <= to`
    /// is performed by `cobre-io` (ticket-014 Tier 1 validation).
    Range {
        /// First year of the range (inclusive).
        from: i32,
        /// Last year of the range (inclusive).
        to: i32,
    },
}

impl HistoricalYears {
    /// Expand the year specification into a concrete sorted list.
    ///
    /// - `List` — returns the years as-is (caller order is preserved).
    /// - `Range` — expands the inclusive range `[from, to]` into a full list.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::scenario::HistoricalYears;
    ///
    /// let list = HistoricalYears::List(vec![1995, 2000, 2005]);
    /// assert_eq!(list.to_years(), vec![1995, 2000, 2005]);
    ///
    /// let range = HistoricalYears::Range { from: 2000, to: 2003 };
    /// assert_eq!(range.to_years(), vec![2000, 2001, 2002, 2003]);
    /// ```
    #[must_use]
    pub fn to_years(&self) -> Vec<i32> {
        match self {
            HistoricalYears::List(years) => years.clone(),
            HistoricalYears::Range { from, to } => (*from..=*to).collect(),
        }
    }
}

// InflowModel (SS14 — per hydro, per stage)

/// Raw PAR(p) model parameters for a single (hydro, stage) pair.
///
/// Stores the seasonal mean, standard deviation, and standardized AR lag
/// coefficients loaded from `inflow_seasonal_stats.parquet` and
/// `inflow_ar_coefficients.parquet`. These are the raw input-facing values.
///
/// AR coefficients are stored **standardized by seasonal std** (dimensionless ψ\*,
/// direct Yule-Walker output). The `residual_std_ratio` field carries the ratio
/// `σ_m` / `s_m` so that downstream crates can recover the runtime residual std as
/// `std_m3s * residual_std_ratio` without re-deriving it from the coefficients.
///
/// The performance-adapted view (`PrecomputedPar`) is built from these
/// parameters once at solver initialisation and belongs to downstream solver crates.
///
/// ## Declaration-order invariance
///
/// The `System` holds a `Vec<InflowModel>` sorted by `(hydro_id, stage_id)`.
/// All processing must iterate in that canonical order.
///
/// See [internal-structures.md §14](../specs/data-model/internal-structures.md)
/// and [PAR Inflow Model §7](../math/par-inflow-model.md).
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel};
///
/// let model = InflowModel {
///     hydro_id: EntityId(1),
///     stage_id: 3,
///     mean_m3s: 150.0,
///     std_m3s: 30.0,
///     ar_coefficients: vec![0.45, 0.22],
///     residual_std_ratio: 0.85,
/// };
/// assert_eq!(model.ar_order(), 2);
/// assert_eq!(model.ar_coefficients.len(), 2);
/// assert!((model.residual_std_ratio - 0.85).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InflowModel {
    /// Hydro plant this model belongs to.
    pub hydro_id: EntityId,

    /// Stage (0-based index within `System::stages`) this model applies to.
    pub stage_id: i32,

    /// Seasonal mean inflow μ in m³/s.
    pub mean_m3s: f64,

    /// Seasonal standard deviation `s_m` in m³/s (seasonal sample std).
    pub std_m3s: f64,

    /// AR lag coefficients [ψ\*₁, ψ\*₂, …, ψ\*ₚ] standardized by seasonal std
    /// (dimensionless). These are the direct Yule-Walker output. Length is the
    /// AR order p. Empty when p == 0 (white noise).
    pub ar_coefficients: Vec<f64>,

    /// Ratio of residual standard deviation to seasonal standard deviation
    /// (`σ_m` / `s_m`). Dimensionless, in (0, 1]. The runtime residual std is
    /// `std_m3s * residual_std_ratio`. When `ar_coefficients` is empty
    /// (white noise), this is 1.0 (the AR model explains nothing).
    pub residual_std_ratio: f64,
}

impl InflowModel {
    /// AR model order p (number of lags). Zero means white-noise inflow.
    #[must_use]
    pub fn ar_order(&self) -> usize {
        self.ar_coefficients.len()
    }
}

// LoadModel (SS14 — per bus, per stage)

/// Raw load seasonal statistics for a single (bus, stage) pair.
///
/// Stores the mean and standard deviation of load demand loaded from
/// `load_seasonal_stats.parquet`. Load typically has no AR structure,
/// so no lag coefficients are stored here.
///
/// The `System` holds a `Vec<LoadModel>` sorted by `(bus_id, stage_id)`.
///
/// See [internal-structures.md §14](../specs/data-model/internal-structures.md).
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::LoadModel};
///
/// let model = LoadModel {
///     bus_id: EntityId(5),
///     stage_id: 0,
///     mean_mw: 320.5,
///     std_mw: 45.0,
/// };
/// assert_eq!(model.mean_mw, 320.5);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LoadModel {
    /// Bus this load model belongs to.
    pub bus_id: EntityId,

    /// Stage (0-based index within `System::stages`) this model applies to.
    pub stage_id: i32,

    /// Seasonal mean load demand in MW.
    pub mean_mw: f64,

    /// Seasonal standard deviation of load demand in MW.
    pub std_mw: f64,
}

// NcsModel (per NCS entity, per stage)

/// Per-stage normal noise model parameters for a non-controllable source.
///
/// Loaded from `scenarios/non_controllable_stats.parquet`. Each row provides
/// the mean and standard deviation of the stochastic availability factor for
/// one NCS entity at one stage. The scenario pipeline uses these parameters
/// to generate per-scenario availability realisations.
///
/// The noise model is: `A_r = max_gen * clamp(mean + std * epsilon, 0, 1)`,
/// where `epsilon ~ N(0,1)` and `mean`, `std` are dimensionless availability
/// factors in `[0, 1]`.
///
/// The `System` holds a `Vec<NcsModel>` sorted by `(ncs_id, stage_id)`.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::NcsModel};
///
/// let model = NcsModel {
///     ncs_id: EntityId(3),
///     stage_id: 0,
///     mean: 0.5,
///     std: 0.1,
/// };
/// assert_eq!(model.mean, 0.5);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NcsModel {
    /// NCS entity identifier matching `NonControllableSource.id`.
    pub ncs_id: EntityId,

    /// Stage (0-based index within `System::stages`) this model applies to.
    pub stage_id: i32,

    /// Mean availability factor [dimensionless, in `[0, 1]`].
    pub mean: f64,

    /// Standard deviation of the availability factor [dimensionless, >= 0].
    pub std: f64,
}

// InflowHistoryRow (SS2.4 — raw historical observation)

/// A single row from `scenarios/inflow_history.parquet`.
///
/// Carries one historical inflow observation for a (hydro, date) pair.
/// These rows constitute the raw historical record used by PAR(p) fitting
/// routines in `cobre-stochastic` and by the historical scenario library
/// constructed during solver setup.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowHistoryRow};
/// use chrono::NaiveDate;
///
/// let row = InflowHistoryRow {
///     hydro_id: EntityId::from(1),
///     date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
///     value_m3s: 500.0,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(1));
/// assert_eq!(row.value_m3s, 500.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InflowHistoryRow {
    /// Hydro plant this observation belongs to.
    pub hydro_id: EntityId,
    /// Date of the observation (timezone-free calendar date).
    pub date: chrono::NaiveDate,
    /// Mean inflow for this observation period in m³/s. Must be finite.
    pub value_m3s: f64,
}

// ExternalScenarioRow (SS2.5 — pre-computed external scenario value)

/// A single row from `scenarios/external_inflow_scenarios.parquet`.
///
/// Each row defines the pre-computed inflow value for one (stage, scenario, hydro)
/// triple. Used when [`SamplingScheme::External`] is active.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::ExternalScenarioRow};
///
/// let row = ExternalScenarioRow {
///     stage_id: 0,
///     scenario_id: 2,
///     hydro_id: EntityId::from(5),
///     value_m3s: 320.5,
/// };
/// assert_eq!(row.scenario_id, 2);
/// assert!((row.value_m3s - 320.5).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExternalScenarioRow {
    /// Stage index (0-based within `System::stages`).
    pub stage_id: i32,

    /// Scenario index (0-based). Must be >= 0.
    pub scenario_id: i32,

    /// Hydro plant this inflow value belongs to.
    pub hydro_id: EntityId,

    /// Pre-computed inflow value in m³/s. Must be finite.
    pub value_m3s: f64,
}

// ExternalLoadRow (E2 — pre-computed external load scenario value)

/// A single row from `scenarios/external_load_scenarios.parquet`.
///
/// Each row defines the pre-computed load value for one (stage, scenario, bus)
/// triple. Used when [`SamplingScheme::External`] is active for load variables.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::ExternalLoadRow};
///
/// let row = ExternalLoadRow {
///     stage_id: 0,
///     scenario_id: 2,
///     bus_id: EntityId::from(3),
///     value_mw: 150.0,
/// };
/// assert_eq!(row.scenario_id, 2);
/// assert!((row.value_mw - 150.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExternalLoadRow {
    /// Stage index (0-based within `System::stages`).
    pub stage_id: i32,

    /// Scenario index (0-based). Must be >= 0.
    pub scenario_id: i32,

    /// Bus this load value belongs to.
    pub bus_id: EntityId,

    /// Pre-computed load value in MW. Must be finite.
    pub value_mw: f64,
}

// ExternalNcsRow (E2 — pre-computed external NCS scenario value)

/// A single row from `scenarios/external_ncs_scenarios.parquet`.
///
/// Each row defines the pre-computed dimensionless availability factor for one
/// (stage, scenario, ncs) triple. Used when [`SamplingScheme::External`] is
/// active for NCS availability variables.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::ExternalNcsRow};
///
/// let row = ExternalNcsRow {
///     stage_id: 1,
///     scenario_id: 0,
///     ncs_id: EntityId::from(7),
///     value: 0.85,
/// };
/// assert_eq!(row.ncs_id, EntityId::from(7));
/// assert!((row.value - 0.85).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExternalNcsRow {
    /// Stage index (0-based within `System::stages`).
    pub stage_id: i32,

    /// Scenario index (0-based). Must be >= 0.
    pub scenario_id: i32,

    /// NCS source this availability factor belongs to.
    pub ncs_id: EntityId,

    /// Pre-computed dimensionless availability factor. Must be finite.
    pub value: f64,
}

// CorrelationEntity

/// A single entity reference within a correlation group.
///
/// `entity_type` is a string tag that identifies the kind of stochastic
/// variable. Valid values are:
///
/// - `"inflow"` — hydro inflow series (entity ID matches `Hydro.id`)
/// - `"load"` — stochastic load demand (entity ID matches `Bus.id`)
/// - `"ncs"` — non-controllable source availability (entity ID matches
///   `NonControllableSource.id`)
///
/// Using `String` rather than an enum preserves forward compatibility when
/// additional entity types are added without a breaking schema change.
///
/// See [Input Scenarios §5](input-scenarios.md).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelationEntity {
    /// Entity type tag: `"inflow"`, `"load"`, or `"ncs"`.
    pub entity_type: String,

    /// Entity identifier matching the corresponding entity's `id` field.
    pub id: EntityId,
}

// CorrelationGroup

/// A named group of correlated entities and their correlation matrix.
///
/// `matrix` is a symmetric positive-semi-definite matrix stored in
/// row-major order as `Vec<Vec<f64>>`. `matrix[i][j]` is the correlation
/// coefficient between `entities[i]` and `entities[j]`.
/// `matrix.len()` must equal `entities.len()`.
///
/// Cholesky decomposition of the matrix is NOT performed here; that
/// belongs to `cobre-stochastic`.
///
/// See [Input Scenarios §5](input-scenarios.md).
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::{CorrelationEntity, CorrelationGroup}};
///
/// let group = CorrelationGroup {
///     name: "Southeast".to_string(),
///     entities: vec![
///         CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(1) },
///         CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(2) },
///     ],
///     matrix: vec![
///         vec![1.0, 0.8],
///         vec![0.8, 1.0],
///     ],
/// };
/// assert_eq!(group.matrix.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelationGroup {
    /// Human-readable group label (e.g., `"Southeast"`, `"North"`).
    pub name: String,

    /// Ordered list of entities whose correlation is captured by `matrix`.
    pub entities: Vec<CorrelationEntity>,

    /// Symmetric correlation matrix in row-major order.
    /// `matrix[i][j]` = correlation between `entities[i]` and `entities[j]`.
    /// Diagonal entries must be 1.0. Shape: `entities.len() × entities.len()`.
    pub matrix: Vec<Vec<f64>>,
}

// CorrelationProfile

/// A named correlation profile containing one or more correlation groups.
///
/// A profile groups correlated entities into disjoint [`CorrelationGroup`]s.
/// Entities in different groups are treated as uncorrelated. Profiles are
/// stored in [`CorrelationModel::profiles`] keyed by profile name.
///
/// See [Input Scenarios §5](input-scenarios.md).
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::{CorrelationEntity, CorrelationGroup, CorrelationProfile}};
///
/// let profile = CorrelationProfile {
///     groups: vec![CorrelationGroup {
///         name: "All".to_string(),
///         entities: vec![
///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(1) },
///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(2) },
///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(3) },
///         ],
///         matrix: vec![
///             vec![1.0, 0.0, 0.0],
///             vec![0.0, 1.0, 0.0],
///             vec![0.0, 0.0, 1.0],
///         ],
///     }],
/// };
/// assert_eq!(profile.groups[0].matrix.len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelationProfile {
    /// Disjoint groups of correlated entities within this profile.
    pub groups: Vec<CorrelationGroup>,
}

// CorrelationScheduleEntry

/// Maps a stage to its active correlation profile name.
///
/// When [`CorrelationModel::schedule`] is non-empty, each stage that
/// requires a non-default correlation profile has an entry here. Stages
/// without an entry use the profile named `"default"` if present, or the
/// sole profile if only one profile exists.
///
/// See [Input Scenarios §5](input-scenarios.md).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelationScheduleEntry {
    /// Stage index (0-based within `System::stages`) this entry applies to.
    pub stage_id: i32,

    /// Name of the correlation profile active for this stage.
    /// Must match a key in [`CorrelationModel::profiles`].
    pub profile_name: String,
}

// CorrelationModel

/// Top-level correlation configuration for the scenario pipeline.
///
/// Holds all named correlation profiles and the optional stage-to-profile
/// schedule. When `schedule` is empty, the solver uses a single profile
/// (typically named `"default"`) for all stages.
///
/// `profiles` uses [`BTreeMap`] rather than [`HashMap`](std::collections::HashMap) to preserve
/// deterministic iteration order, satisfying the declaration-order
/// invariance requirement (design-principles.md §3).
///
/// `method` is always `"cholesky"` for the minimal viable solver but stored
/// as a `String` for forward compatibility with future decomposition methods.
///
/// Source: `correlation.json`.
/// See [Input Scenarios §5](input-scenarios.md) and
/// [internal-structures.md §14](../specs/data-model/internal-structures.md).
///
/// # Examples
///
/// ```
/// use std::collections::BTreeMap;
/// use cobre_core::{EntityId, scenario::{
///     CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
///     CorrelationScheduleEntry,
/// }};
///
/// let mut profiles = BTreeMap::new();
/// profiles.insert("default".to_string(), CorrelationProfile {
///     groups: vec![CorrelationGroup {
///         name: "All".to_string(),
///         entities: vec![
///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(1) },
///         ],
///         matrix: vec![vec![1.0]],
///     }],
/// });
///
/// let model = CorrelationModel {
///     method: "cholesky".to_string(),
///     profiles,
///     schedule: vec![],
/// };
/// assert!(model.profiles.contains_key("default"));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelationModel {
    /// Decomposition method. Always `"cholesky"` for the minimal viable solver.
    /// Stored as `String` for forward compatibility.
    pub method: String,

    /// Named correlation profiles keyed by profile name.
    /// `BTreeMap` for deterministic ordering (declaration-order invariance).
    pub profiles: BTreeMap<String, CorrelationProfile>,

    /// Stage-to-profile schedule. Empty when a single profile applies to
    /// all stages.
    pub schedule: Vec<CorrelationScheduleEntry>,
}

impl Default for ScenarioSource {
    fn default() -> Self {
        Self {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        }
    }
}

impl Default for CorrelationModel {
    fn default() -> Self {
        Self {
            method: "cholesky".to_string(),
            profiles: BTreeMap::new(),
            schedule: Vec::new(),
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    #[cfg(feature = "serde")]
    use super::ScenarioSource;
    use super::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
        CorrelationScheduleEntry, InflowModel, NcsModel, SamplingScheme,
    };
    use crate::EntityId;

    #[test]
    fn test_inflow_model_construction() {
        let model = InflowModel {
            hydro_id: EntityId(7),
            stage_id: 11,
            mean_m3s: 250.0,
            std_m3s: 55.0,
            ar_coefficients: vec![0.5, 0.2, 0.1],
            residual_std_ratio: 0.85,
        };

        assert_eq!(model.hydro_id, EntityId(7));
        assert_eq!(model.stage_id, 11);
        assert_eq!(model.mean_m3s, 250.0);
        assert_eq!(model.std_m3s, 55.0);
        assert_eq!(model.ar_order(), 3);
        assert_eq!(model.ar_coefficients, vec![0.5, 0.2, 0.1]);
        assert_eq!(model.ar_coefficients.len(), model.ar_order());
        assert!((model.residual_std_ratio - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inflow_model_ar_order_method() {
        // Empty coefficients: ar_order() == 0 (white noise)
        let white_noise = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
        };
        assert_eq!(white_noise.ar_order(), 0);

        // Two coefficients: ar_order() == 2
        let par2 = InflowModel {
            hydro_id: EntityId(2),
            stage_id: 1,
            mean_m3s: 200.0,
            std_m3s: 20.0,
            ar_coefficients: vec![0.45, 0.22],
            residual_std_ratio: 0.85,
        };
        assert_eq!(par2.ar_order(), 2);
    }

    #[test]
    fn test_correlation_model_construction() {
        let make_profile = |entity_ids: &[i32]| {
            let entities: Vec<CorrelationEntity> = entity_ids
                .iter()
                .map(|&id| CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(id),
                })
                .collect();
            let n = entities.len();
            let matrix: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect();
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "group_a".to_string(),
                    entities,
                    matrix,
                }],
            }
        };

        let mut profiles = BTreeMap::new();
        profiles.insert("wet".to_string(), make_profile(&[1, 2, 3]));
        profiles.insert("dry".to_string(), make_profile(&[1, 2]));

        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![
                CorrelationScheduleEntry {
                    stage_id: 0,
                    profile_name: "wet".to_string(),
                },
                CorrelationScheduleEntry {
                    stage_id: 6,
                    profile_name: "dry".to_string(),
                },
            ],
        };

        // Two profiles present
        assert_eq!(model.profiles.len(), 2);

        // BTreeMap ordering is alphabetical: "dry" before "wet"
        let mut profile_iter = model.profiles.keys();
        assert_eq!(profile_iter.next().unwrap(), "dry");
        assert_eq!(profile_iter.next().unwrap(), "wet");

        // Profile lookup by name
        assert!(model.profiles.contains_key("wet"));
        assert!(model.profiles.contains_key("dry"));

        // Matrix dimensions match entity count
        let wet = &model.profiles["wet"];
        assert_eq!(wet.groups[0].matrix.len(), 3);

        let dry = &model.profiles["dry"];
        assert_eq!(dry.groups[0].matrix.len(), 2);

        // Schedule entries
        assert_eq!(model.schedule.len(), 2);
        assert_eq!(model.schedule[0].profile_name, "wet");
        assert_eq!(model.schedule[1].profile_name, "dry");
    }

    #[test]
    fn test_sampling_scheme_copy() {
        let original = SamplingScheme::InSample;
        let copied = original;
        assert_eq!(original, copied);

        let original_oos = SamplingScheme::OutOfSample;
        let copied_oos = original_oos;
        assert_eq!(original_oos, copied_oos);

        let original_ext = SamplingScheme::External;
        let copied_ext = original_ext;
        assert_eq!(original_ext, copied_ext);

        let original_hist = SamplingScheme::Historical;
        let copied_hist = original_hist;
        assert_eq!(original_hist, copied_hist);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_scenario_source_serde_roundtrip() {
        use super::HistoricalYears;

        // All three schemes set to different values with seed and no historical_years
        let source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::OutOfSample,
            ncs_scheme: SamplingScheme::External,
            seed: Some(12345),
            historical_years: None,
        };
        let json = serde_json::to_string(&source).unwrap();
        let deserialized: ScenarioSource = serde_json::from_str(&json).unwrap();
        assert_eq!(source, deserialized);

        // Historical inflow with historical_years list
        let source_hist = ScenarioSource {
            inflow_scheme: SamplingScheme::Historical,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: Some(7),
            historical_years: Some(HistoricalYears::List(vec![1990, 2000, 2010])),
        };
        let json_hist = serde_json::to_string(&source_hist).unwrap();
        let deserialized_hist: ScenarioSource = serde_json::from_str(&json_hist).unwrap();
        assert_eq!(source_hist, deserialized_hist);

        // Historical inflow with historical_years range and no seed
        let source_range = ScenarioSource {
            inflow_scheme: SamplingScheme::Historical,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: Some(HistoricalYears::Range {
                from: 1940,
                to: 2010,
            }),
        };
        let json_range = serde_json::to_string(&source_range).unwrap();
        let deserialized_range: ScenarioSource = serde_json::from_str(&json_range).unwrap();
        assert_eq!(source_range, deserialized_range);

        // All InSample, no seed, no historical_years (default-like)
        let source_default = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };
        let json_default = serde_json::to_string(&source_default).unwrap();
        let deserialized_default: ScenarioSource = serde_json::from_str(&json_default).unwrap();
        assert_eq!(source_default, deserialized_default);
    }

    #[test]
    fn test_scenario_source_default() {
        let source = ScenarioSource::default();
        assert_eq!(source.inflow_scheme, SamplingScheme::InSample);
        assert_eq!(source.load_scheme, SamplingScheme::InSample);
        assert_eq!(source.ncs_scheme, SamplingScheme::InSample);
        assert!(source.seed.is_none());
        assert!(source.historical_years.is_none());
    }

    #[test]
    fn test_historical_years_list_construction() {
        use super::HistoricalYears;
        let years = HistoricalYears::List(vec![1940, 1953, 1971]);
        match &years {
            HistoricalYears::List(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0], 1940);
                assert_eq!(v[1], 1953);
                assert_eq!(v[2], 1971);
            }
            HistoricalYears::Range { .. } => panic!("expected List variant"),
        }
    }

    #[test]
    fn test_historical_years_range_construction() {
        use super::HistoricalYears;
        let years = HistoricalYears::Range {
            from: 1940,
            to: 2010,
        };
        match years {
            HistoricalYears::Range { from, to } => {
                assert_eq!(from, 1940);
                assert_eq!(to, 2010);
            }
            HistoricalYears::List(_) => panic!("expected Range variant"),
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_historical_years_list_serde_roundtrip() {
        use super::HistoricalYears;
        let years = HistoricalYears::List(vec![1940, 1953, 1971]);
        let json = serde_json::to_string(&years).unwrap();
        let deserialized: HistoricalYears = serde_json::from_str(&json).unwrap();
        assert_eq!(years, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_historical_years_range_serde_roundtrip() {
        use super::HistoricalYears;
        let years = HistoricalYears::Range {
            from: 1940,
            to: 2010,
        };
        let json = serde_json::to_string(&years).unwrap();
        let deserialized: HistoricalYears = serde_json::from_str(&json).unwrap();
        assert_eq!(years, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_inflow_model_serde_roundtrip() {
        let model = InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 150.0,
            std_m3s: 30.0,
            ar_coefficients: vec![0.45, 0.22],
            residual_std_ratio: 0.85,
        };
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: InflowModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, deserialized);
        assert!((deserialized.residual_std_ratio - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_model_construction() {
        let model = NcsModel {
            ncs_id: EntityId(3),
            stage_id: 0,
            mean: 0.5,
            std: 0.1,
        };

        assert_eq!(model.ncs_id, EntityId(3));
        assert_eq!(model.stage_id, 0);
        assert_eq!(model.mean, 0.5);
        assert_eq!(model.std, 0.1);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ncs_model_serde_roundtrip() {
        let model = NcsModel {
            ncs_id: EntityId(5),
            stage_id: 2,
            mean: 0.75,
            std: 0.15,
        };
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: NcsModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, deserialized);
    }

    #[test]
    fn test_correlation_model_identity_matrix_access() {
        let identity = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "all_hydros".to_string(),
                    entities: vec![
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(1),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(2),
                        },
                        CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(3),
                        },
                    ],
                    matrix: identity,
                }],
            },
        );
        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };

        // AC: model.profiles["default"].groups[0].matrix.len() == 3
        assert_eq!(model.profiles["default"].groups[0].matrix.len(), 3);
    }
}
