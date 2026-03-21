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

// ---------------------------------------------------------------------------
// SamplingScheme (SS14 scenario source)
// ---------------------------------------------------------------------------

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
    /// Forward pass draws from an externally supplied scenario file.
    External,
    /// Forward pass replays historical inflow realisations in sequence or at random.
    Historical,
}

// ---------------------------------------------------------------------------
// ExternalSelectionMode
// ---------------------------------------------------------------------------

/// Scenario selection mode when [`SamplingScheme::External`] is active.
///
/// Controls whether external scenarios are replayed sequentially (useful for
/// deterministic replay of a fixed test set) or drawn at random (useful for
/// Monte Carlo evaluation with a large external library).
///
/// See [Input Scenarios §1.8](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExternalSelectionMode {
    /// Scenarios are drawn uniformly at random from the external library.
    Random,
    /// Scenarios are replayed in file order, cycling when the end is reached.
    Sequential,
}

// ---------------------------------------------------------------------------
// ScenarioSource (SS14 top-level config)
// ---------------------------------------------------------------------------

/// Top-level scenario source configuration, parsed from `stages.json`.
///
/// Groups the sampling scheme, random seed, and external selection mode
/// that govern how forward-pass scenarios are produced. Populated during
/// case loading by `cobre-io` from the `scenario_source` field in
/// `stages.json`. Distinct from [`ScenarioSourceConfig`](crate::temporal::ScenarioSourceConfig),
/// which also holds the branching factor (`num_scenarios`).
///
/// See [Input Scenarios §1.4, §1.8](input-scenarios.md).
///
/// # Examples
///
/// ```
/// use cobre_core::scenario::{SamplingScheme, ScenarioSource};
///
/// let source = ScenarioSource {
///     sampling_scheme: SamplingScheme::InSample,
///     seed: Some(42),
///     selection_mode: None,
/// };
/// assert_eq!(source.sampling_scheme, SamplingScheme::InSample);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScenarioSource {
    /// Noise source used during the forward pass.
    pub sampling_scheme: SamplingScheme,

    /// Random seed for reproducible opening tree generation.
    /// `None` means non-deterministic (OS entropy).
    pub seed: Option<i64>,

    /// Selection mode when `sampling_scheme` is [`SamplingScheme::External`].
    /// `None` for `InSample` and `Historical` schemes.
    pub selection_mode: Option<ExternalSelectionMode>,
}

// ---------------------------------------------------------------------------
// InflowModel (SS14 — per hydro, per stage)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// LoadModel (SS14 — per bus, per stage)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// NcsModel (per NCS entity, per stage)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CorrelationEntity
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CorrelationGroup
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CorrelationProfile
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CorrelationScheduleEntry
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CorrelationModel
// ---------------------------------------------------------------------------

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
            sampling_scheme: SamplingScheme::InSample,
            seed: None,
            selection_mode: None,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
        CorrelationScheduleEntry, InflowModel, NcsModel, SamplingScheme,
    };
    #[cfg(feature = "serde")]
    use super::{ExternalSelectionMode, ScenarioSource};
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
        // InSample with seed
        let source = ScenarioSource {
            sampling_scheme: SamplingScheme::InSample,
            seed: Some(12345),
            selection_mode: None,
        };
        let json = serde_json::to_string(&source).unwrap();
        let deserialized: ScenarioSource = serde_json::from_str(&json).unwrap();
        assert_eq!(source, deserialized);

        // External with selection mode
        let source_ext = ScenarioSource {
            sampling_scheme: SamplingScheme::External,
            seed: Some(99),
            selection_mode: Some(ExternalSelectionMode::Sequential),
        };
        let json_ext = serde_json::to_string(&source_ext).unwrap();
        let deserialized_ext: ScenarioSource = serde_json::from_str(&json_ext).unwrap();
        assert_eq!(source_ext, deserialized_ext);

        // Historical without seed
        let source_hist = ScenarioSource {
            sampling_scheme: SamplingScheme::Historical,
            seed: None,
            selection_mode: None,
        };
        let json_hist = serde_json::to_string(&source_hist).unwrap();
        let deserialized_hist: ScenarioSource = serde_json::from_str(&json_hist).unwrap();
        assert_eq!(source_hist, deserialized_hist);
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
