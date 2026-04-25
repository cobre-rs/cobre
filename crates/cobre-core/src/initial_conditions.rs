//! Initial conditions for the optimization study.
//!
//! [`InitialConditions`] holds the reservoir storage levels and past inflow
//! values at the start of the study. Two storage arrays are kept separate
//! because filling hydros can have an initial volume below dead storage
//! (`min_storage_hm3`), which is not a valid operating level for regular hydros.
//!
//! See `internal-structures.md §16` and `input-constraints.md §1` for the
//! full specification including validation rules.
//!
//! # Examples
//!
//! ```
//! use cobre_core::{EntityId, InitialConditions, HydroStorage, HydroPastInflows};
//!
//! let ic = InitialConditions {
//!     storage: vec![
//!         HydroStorage { hydro_id: EntityId(0), value_hm3: 15_000.0 },
//!         HydroStorage { hydro_id: EntityId(1), value_hm3:  8_500.0 },
//!     ],
//!     filling_storage: vec![
//!         HydroStorage { hydro_id: EntityId(10), value_hm3: 200.0 },
//!     ],
//!     past_inflows: vec![
//!         HydroPastInflows { hydro_id: EntityId(0), values_m3s: vec![600.0, 500.0], season_ids: None },
//!     ],
//!     recent_observations: vec![],
//! };
//!
//! assert_eq!(ic.storage.len(), 2);
//! assert_eq!(ic.filling_storage.len(), 1);
//! assert_eq!(ic.past_inflows.len(), 1);
//! assert_eq!(ic.recent_observations.len(), 0);
//! ```

use chrono::NaiveDate;

use crate::EntityId;

/// Initial storage volume for a single hydro plant.
///
/// For operating hydros, `value_hm3` must be within
/// `[min_storage_hm3, max_storage_hm3]` (validated by `cobre-io`).
/// For filling hydros (present in [`InitialConditions::filling_storage`]),
/// `value_hm3` must be within `[0.0, min_storage_hm3]`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStorage {
    /// Hydro plant identifier. Must reference a hydro entity in the system.
    pub hydro_id: EntityId,
    /// Reservoir volume at the start of the study, in hm³.
    pub value_hm3: f64,
}

/// Past inflow values for PAR(p) lag initialization for a single hydro plant.
///
/// Each entry provides the most-recent inflow history for one hydro plant,
/// ordered from most recent (lag 1) to oldest (lag p). The length of
/// `values_m3s` must be >= the hydro's PAR order.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroPastInflows {
    /// Hydro plant identifier. Must reference a hydro entity in the system.
    pub hydro_id: EntityId,
    /// Past inflow values in m³/s, ordered from most recent (index 0 = lag 1)
    /// to oldest (index p-1 = lag p).
    pub values_m3s: Vec<f64>,
    /// Optional season IDs corresponding to each lag entry in `values_m3s`.
    ///
    /// When present, `season_ids.len()` must equal `values_m3s.len()`. Each
    /// element is a season ID (as defined in `season_definitions`) identifying
    /// the temporal period of the corresponding lag entry. When absent, no
    /// temporal validation of lag entries is performed.
    ///
    /// In JSON: the field is optional (`serde(default)` fills `None` when the
    /// key is absent). Backward-compatible with existing JSON files.
    #[cfg_attr(feature = "serde", serde(default))]
    pub season_ids: Option<Vec<u32>>,
}

/// Observed inflow for a single hydro plant over a specific date range.
///
/// Used to seed the lag accumulator when a study begins mid-season. Each entry
/// represents the average inflow (in m³/s) observed between `start_date`
/// (inclusive) and `end_date` (exclusive) for one hydro. Multiple entries per
/// hydro are allowed for rolling revisions with several observed weeks.
///
/// Date ranges for the same hydro must not overlap; adjacent ranges
/// (`start_date == previous end_date`) are accepted.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecentObservation {
    /// Hydro plant identifier. Must reference a hydro entity in the system.
    pub hydro_id: EntityId,
    /// Start of the observation period (inclusive).
    pub start_date: NaiveDate,
    /// End of the observation period (exclusive). Must be after `start_date`.
    pub end_date: NaiveDate,
    /// Average inflow observed during the period, in m³/s. Must be finite and
    /// non-negative.
    pub value_m3s: f64,
}

/// Initial system state at the start of the optimization study.
///
/// Produced by parsing `initial_conditions.json` (in `cobre-io`) and stored
/// inside [`crate::System`]. All arrays are sorted by `hydro_id` after
/// loading to satisfy the declaration-order invariance requirement.
///
/// A hydro must appear in exactly one of the two storage arrays, never both.
/// Hydros with a `filling` configuration belong in [`filling_storage`]; all
/// other hydros (including late-entry hydros) belong in
/// [`storage`](InitialConditions::storage).
///
/// [`filling_storage`]: InitialConditions::filling_storage
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InitialConditions {
    /// Initial storage for operating hydros, in hm³ per hydro.
    pub storage: Vec<HydroStorage>,
    /// Initial storage for filling hydros (below dead volume), in hm³ per hydro.
    pub filling_storage: Vec<HydroStorage>,
    /// Past inflow values for PAR(p) lag initialization, in m³/s per hydro.
    ///
    /// For each hydro, `values_m3s[0]` is the most recent past inflow (lag 1)
    /// and `values_m3s[p-1]` is the oldest (lag p). Absent when lag
    /// initialization is not required (no PAR models or `inflow_lags: false`).
    ///
    /// In JSON: the field is optional on input (`serde(default)` fills an empty
    /// `Vec` when the key is absent). The field is always emitted on output —
    /// omitting it would break postcard round-trips used by MPI broadcast.
    #[cfg_attr(feature = "serde", serde(default))]
    pub past_inflows: Vec<HydroPastInflows>,
    /// Observed inflow data for partial periods before the study start.
    ///
    /// Used to seed the lag accumulator when a study begins mid-season (i.e.,
    /// before the first lag-period boundary). Each entry covers one hydro over
    /// a specific date range. Sorted by `(hydro_id, start_date)` after loading.
    ///
    /// In JSON: the field is optional (`serde(default)` fills an empty `Vec`
    /// when the key is absent). Backward-compatible with existing JSON files.
    #[cfg_attr(feature = "serde", serde(default))]
    pub recent_observations: Vec<RecentObservation>,
}

impl Default for InitialConditions {
    /// Returns an empty `InitialConditions` (no hydros).
    fn default() -> Self {
        Self {
            storage: Vec::new(),
            filling_storage: Vec::new(),
            past_inflows: Vec::new(),
            recent_observations: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_conditions_construction() {
        let ic = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(0),
                    value_hm3: 15_000.0,
                },
                HydroStorage {
                    hydro_id: EntityId(1),
                    value_hm3: 8_500.0,
                },
            ],
            filling_storage: vec![HydroStorage {
                hydro_id: EntityId(10),
                value_hm3: 200.0,
            }],
            past_inflows: vec![HydroPastInflows {
                hydro_id: EntityId(0),
                values_m3s: vec![600.0, 500.0],
                season_ids: None,
            }],
            recent_observations: vec![],
        };

        assert_eq!(ic.storage.len(), 2);
        assert_eq!(ic.filling_storage.len(), 1);
        assert_eq!(ic.past_inflows.len(), 1);
        assert_eq!(ic.storage[0].hydro_id, EntityId(0));
        assert_eq!(ic.storage[0].value_hm3, 15_000.0);
        assert_eq!(ic.storage[1].hydro_id, EntityId(1));
        assert_eq!(ic.filling_storage[0].hydro_id, EntityId(10));
        assert_eq!(ic.filling_storage[0].value_hm3, 200.0);
        assert_eq!(ic.past_inflows[0].hydro_id, EntityId(0));
        assert_eq!(ic.past_inflows[0].values_m3s, vec![600.0, 500.0]);
        assert!(ic.recent_observations.is_empty());
    }

    #[test]
    fn test_initial_conditions_default_is_empty() {
        let ic = InitialConditions::default();
        assert!(ic.storage.is_empty());
        assert!(ic.filling_storage.is_empty());
        assert!(ic.past_inflows.is_empty());
        assert!(ic.recent_observations.is_empty());
    }

    #[test]
    fn test_hydro_storage_clone() {
        let hs = HydroStorage {
            hydro_id: EntityId(5),
            value_hm3: 1_234.5,
        };
        let cloned = hs.clone();
        assert_eq!(hs, cloned);
        assert_eq!(cloned.hydro_id, EntityId(5));
        assert_eq!(cloned.value_hm3, 1_234.5);
    }

    #[test]
    fn test_hydro_past_inflows_clone() {
        let hpi = HydroPastInflows {
            hydro_id: EntityId(3),
            values_m3s: vec![300.0, 200.0, 100.0],
            season_ids: None,
        };
        let cloned = hpi.clone();
        assert_eq!(hpi, cloned);
        assert_eq!(cloned.hydro_id, EntityId(3));
        assert_eq!(cloned.values_m3s, vec![300.0, 200.0, 100.0]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_roundtrip() {
        let ic = InitialConditions {
            storage: vec![
                HydroStorage {
                    hydro_id: EntityId(0),
                    value_hm3: 15_000.0,
                },
                HydroStorage {
                    hydro_id: EntityId(1),
                    value_hm3: 8_500.0,
                },
            ],
            filling_storage: vec![HydroStorage {
                hydro_id: EntityId(10),
                value_hm3: 200.0,
            }],
            past_inflows: vec![HydroPastInflows {
                hydro_id: EntityId(0),
                values_m3s: vec![600.0, 500.0],
                season_ids: None,
            }],
            recent_observations: vec![],
        };

        let json = serde_json::to_string(&ic).unwrap();
        let deserialized: InitialConditions = serde_json::from_str(&json).unwrap();
        assert_eq!(ic, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_roundtrip_empty_past_inflows() {
        // Empty past_inflows is always serialized as [] (never omitted) to keep
        // postcard round-trips used by MPI broadcast working correctly.
        let ic = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            filling_storage: vec![],
            past_inflows: vec![],
            recent_observations: vec![],
        };

        let json = serde_json::to_string(&ic).unwrap();
        let deserialized: InitialConditions = serde_json::from_str(&json).unwrap();
        assert_eq!(ic, deserialized);
        // Verify the field round-trips correctly (may or may not be present in JSON).
        assert_eq!(deserialized.past_inflows.len(), 0);
    }

    #[test]
    fn test_recent_observation_construction_and_clone() {
        let obs = RecentObservation {
            hydro_id: EntityId(2),
            start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
            value_m3s: 500.0,
        };
        let cloned = obs.clone();
        assert_eq!(obs, cloned);
        assert_eq!(cloned.hydro_id, EntityId(2));
        assert_eq!(cloned.value_m3s, 500.0);
    }

    #[test]
    fn test_initial_conditions_construction_with_recent_observations() {
        let ic = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            filling_storage: vec![],
            past_inflows: vec![],
            recent_observations: vec![RecentObservation {
                hydro_id: EntityId(0),
                start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                    .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                    .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                value_m3s: 500.0,
            }],
        };
        assert_eq!(ic.recent_observations.len(), 1);
        assert_eq!(ic.recent_observations[0].hydro_id, EntityId(0));
        assert_eq!(ic.recent_observations[0].value_m3s, 500.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_roundtrip_with_recent_observations() {
        let ic = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 1_000.0,
            }],
            filling_storage: vec![],
            past_inflows: vec![],
            recent_observations: vec![
                RecentObservation {
                    hydro_id: EntityId(0),
                    start_date: NaiveDate::from_ymd_opt(2026, 4, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_m3s: 500.0,
                },
                RecentObservation {
                    hydro_id: EntityId(0),
                    start_date: NaiveDate::from_ymd_opt(2026, 4, 4)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    end_date: NaiveDate::from_ymd_opt(2026, 4, 11)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    value_m3s: 480.0,
                },
            ],
        };
        let json = serde_json::to_string(&ic).unwrap();
        let deserialized: InitialConditions = serde_json::from_str(&json).unwrap();
        assert_eq!(ic, deserialized);
        assert_eq!(deserialized.recent_observations.len(), 2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_initial_conditions_serde_default_recent_observations_absent() {
        // When recent_observations is absent from JSON, it defaults to an empty Vec.
        let json = r#"{"storage":[],"filling_storage":[]}"#;
        let ic: InitialConditions = serde_json::from_str(json).unwrap();
        assert!(ic.recent_observations.is_empty());
    }

    #[test]
    fn test_hydro_past_inflows_with_season_ids() {
        let hpi = HydroPastInflows {
            hydro_id: EntityId(5),
            values_m3s: vec![600.0, 500.0],
            season_ids: Some(vec![3, 2]),
        };
        assert_eq!(hpi.hydro_id, EntityId(5));
        assert_eq!(hpi.values_m3s, vec![600.0, 500.0]);
        assert_eq!(hpi.season_ids, Some(vec![3, 2]));
        // Clone preserves season_ids
        let cloned = hpi.clone();
        assert_eq!(cloned, hpi);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hydro_past_inflows_serde_roundtrip_with_season_ids() {
        let hpi = HydroPastInflows {
            hydro_id: EntityId(5),
            values_m3s: vec![600.0, 500.0],
            season_ids: Some(vec![3, 2]),
        };
        let json = serde_json::to_string(&hpi).unwrap();
        let deserialized: HydroPastInflows = serde_json::from_str(&json).unwrap();
        assert_eq!(hpi, deserialized);
        assert_eq!(deserialized.season_ids, Some(vec![3, 2]));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hydro_past_inflows_serde_default_season_ids_absent() {
        // When season_ids is absent from JSON, it defaults to None.
        let json = r#"{"hydro_id":0,"values_m3s":[600.0,500.0]}"#;
        let hpi: HydroPastInflows = serde_json::from_str(json).unwrap();
        assert_eq!(hpi.hydro_id, EntityId(0));
        assert_eq!(hpi.values_m3s, vec![600.0, 500.0]);
        assert_eq!(hpi.season_ids, None);
    }
}
