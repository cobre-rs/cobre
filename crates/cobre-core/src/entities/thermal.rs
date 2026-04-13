//! Thermal plant entity — generation with MW bounds and cost.
//!
//! A `Thermal` represents a thermal (combustion, nuclear, etc.) power plant.
//! Thermal plants have MW generation bounds and a scalar marginal cost used to
//! compute the objective contribution in each stage LP.

use crate::EntityId;

/// Gás Natural Liquefeito (GNL) configuration for staged dispatch anticipation.
///
/// GNL models the commitment and start-up lag of thermal units that require
/// advance scheduling over multiple stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GnlConfig {
    /// Number of stages of dispatch anticipation.
    pub lag_stages: i32,
}

/// Thermal power plant with a scalar marginal cost.
///
/// A `Thermal` contributes generation variables and cost objective terms to each
/// stage LP. Generation is bounded between `min_generation_mw` and
/// `max_generation_mw`. The cost is `cost_per_mwh` multiplied by the block
/// duration in hours.
///
/// Source: system/thermals.json. See Input System Entities SS1.9.5.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Thermal {
    /// Unique thermal plant identifier.
    pub id: EntityId,
    /// Human-readable plant name.
    pub name: String,
    /// Bus to which this plant's generation is injected.
    pub bus_id: EntityId,
    /// Stage index when the plant enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Marginal cost of generation \[$/`MWh`\].
    pub cost_per_mwh: f64,
    /// Minimum electrical generation (minimum stable load) \[MW\].
    pub min_generation_mw: f64,
    /// Maximum electrical generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// GNL dispatch anticipation configuration. None = no anticipation lag.
    pub gnl_config: Option<GnlConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_construction() {
        let thermal = Thermal {
            id: EntityId::from(1),
            name: "Angra 1".to_string(),
            bus_id: EntityId::from(10),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 657.0,
            gnl_config: None,
        };

        assert_eq!(thermal.id, EntityId::from(1));
        assert_eq!(thermal.name, "Angra 1");
        assert_eq!(thermal.bus_id, EntityId::from(10));
        assert_eq!(thermal.entry_stage_id, None);
        assert_eq!(thermal.exit_stage_id, None);
        assert!((thermal.cost_per_mwh - 50.0).abs() < f64::EPSILON);
        assert!((thermal.min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((thermal.max_generation_mw - 657.0).abs() < f64::EPSILON);
        assert_eq!(thermal.gnl_config, None);
    }

    #[test]
    fn test_thermal_with_gnl() {
        let thermal = Thermal {
            id: EntityId::from(2),
            name: "Pecém I".to_string(),
            bus_id: EntityId::from(20),
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            cost_per_mwh: 120.0,
            min_generation_mw: 100.0,
            max_generation_mw: 360.0,
            gnl_config: Some(GnlConfig { lag_stages: 2 }),
        };

        assert_eq!(thermal.gnl_config, Some(GnlConfig { lag_stages: 2 }));
        assert_eq!(thermal.entry_stage_id, Some(1));
        assert_eq!(thermal.exit_stage_id, Some(120));
    }

    #[test]
    fn test_thermal_without_gnl() {
        let thermal = Thermal {
            id: EntityId::from(3),
            name: "Candiota".to_string(),
            bus_id: EntityId::from(5),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 60.0,
            min_generation_mw: 0.0,
            max_generation_mw: 446.0,
            gnl_config: None,
        };

        assert_eq!(thermal.gnl_config, None);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_thermal_serde_roundtrip() {
        let thermal = Thermal {
            id: EntityId::from(2),
            name: "Pecém I".to_string(),
            bus_id: EntityId::from(20),
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            cost_per_mwh: 80.0,
            min_generation_mw: 100.0,
            max_generation_mw: 360.0,
            gnl_config: Some(GnlConfig { lag_stages: 2 }),
        };
        let json = serde_json::to_string(&thermal).unwrap();
        let deserialized: Thermal = serde_json::from_str(&json).unwrap();
        assert_eq!(thermal, deserialized);
    }
}
