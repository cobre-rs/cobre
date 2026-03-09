//! Pumping station entity — water transfer consuming electrical power.
//!
//! A `PumpingStation` transfers water between hydro reservoirs while consuming
//! electrical power from the network. This entity is a NO-OP stub:
//! the type exists in the registry but contributes zero LP variables or constraints.

use crate::EntityId;

/// Pumping station that transfers water between hydro reservoirs.
///
/// A `PumpingStation` withdraws water from a source hydro reservoir and injects
/// it into a destination hydro reservoir, consuming electrical power from a bus
/// in the process. In the minimal viable solver this entity is data-complete but
/// contributes no LP variables or constraints.
///
/// Source: `system/pumping_stations.json`. See Input System Entities SS1.9.6.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PumpingStation {
    /// Unique pumping station identifier.
    pub id: EntityId,
    /// Human-readable pumping station name.
    pub name: String,
    /// Bus from which electrical power is consumed.
    pub bus_id: EntityId,
    /// Hydro plant from whose reservoir water is extracted.
    pub source_hydro_id: EntityId,
    /// Hydro plant into whose reservoir water is injected.
    pub destination_hydro_id: EntityId,
    /// Stage index when the station enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the station is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Power consumption rate per unit of pumped flow \[MW/(m³/s)\].
    pub consumption_mw_per_m3s: f64,
    /// Minimum pumped flow \[m³/s\].
    pub min_flow_m3s: f64,
    /// Maximum pumped flow (installed pump capacity) \[m³/s\].
    pub max_flow_m3s: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pumping_station_construction() {
        let station = PumpingStation {
            id: EntityId::from(1),
            name: "Bombeamento Serra da Mesa".to_string(),
            bus_id: EntityId::from(10),
            source_hydro_id: EntityId::from(3),
            destination_hydro_id: EntityId::from(5),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 150.0,
        };

        assert_eq!(station.id, EntityId::from(1));
        assert_eq!(station.name, "Bombeamento Serra da Mesa");
        assert_eq!(station.bus_id, EntityId::from(10));
        assert_eq!(station.source_hydro_id, EntityId::from(3));
        assert_eq!(station.destination_hydro_id, EntityId::from(5));
        assert_eq!(station.entry_stage_id, None);
        assert_eq!(station.exit_stage_id, None);
        assert!((station.consumption_mw_per_m3s - 0.5).abs() < f64::EPSILON);
        assert!((station.min_flow_m3s - 0.0).abs() < f64::EPSILON);
        assert!((station.max_flow_m3s - 150.0).abs() < f64::EPSILON);
    }
}
