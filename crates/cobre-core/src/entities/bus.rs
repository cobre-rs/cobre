//! Bus entity — an electrical network node with power balance constraint.
//!
//! A `Bus` represents a node in the transmission network. Each bus has an
//! associated power balance constraint that must be satisfied at every stage
//! and block. Pre-resolved deficit segments are stored on the bus after loading.

use crate::EntityId;

/// A single segment of the piecewise-linear deficit cost curve.
///
/// Segments are cumulative: the first `depth_mw` MW of deficit costs `cost_per_mwh`,
/// the next segment's `depth_mw` MW costs that segment's `cost_per_mwh`, and so on.
/// The final segment has `depth_mw` = None (extends to infinity).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeficitSegment {
    /// MW of deficit covered by this segment \[MW\]. None for the final unbounded segment.
    pub depth_mw: Option<f64>,
    /// Cost per `MWh` of deficit in this segment \[$/`MWh`\].
    pub cost_per_mwh: f64,
}

/// Electrical network node where energy balance is maintained.
///
/// Buses represent aggregation points in the transmission network -- regional
/// subsystems, substations, or any user-defined grouping. Each bus has a
/// piecewise-linear deficit cost curve that ensures LP feasibility when demand
/// cannot be met.
///
/// Source: system/buses.json. See Input System Entities SS1.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bus {
    /// Unique bus identifier.
    pub id: EntityId,
    /// Human-readable bus name.
    pub name: String,
    /// Pre-resolved piecewise-linear deficit cost segments.
    /// Segments are ordered by ascending cost. The final segment has `depth_mw` = None
    /// (unbounded) to ensure LP feasibility.
    pub deficit_segments: Vec<DeficitSegment>,
    /// Cost per `MWh` for surplus generation absorption \[$/`MWh`\].
    pub excess_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_construction() {
        let bus = Bus {
            id: EntityId::from(1),
            name: "Bus A".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: Some(200.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            excess_cost: 1.0,
        };

        assert_eq!(bus.id, EntityId::from(1));
        assert_eq!(bus.name, "Bus A");
        assert_eq!(bus.deficit_segments.len(), 3);
        assert_eq!(bus.deficit_segments[2].depth_mw, None);
        assert_eq!(bus.excess_cost, 1.0);
    }

    #[test]
    fn test_deficit_segment_unbounded() {
        let segment = DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 9999.0,
        };

        assert_eq!(segment.depth_mw, None);
        assert_eq!(segment.cost_per_mwh, 9999.0);
    }

    #[test]
    fn test_bus_equality() {
        let bus_a = Bus {
            id: EntityId::from(1),
            name: "Bus A".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 1.0,
        };

        let bus_b = bus_a.clone();
        assert_eq!(bus_a, bus_b);

        let bus_c = Bus {
            id: EntityId::from(2),
            ..bus_a.clone()
        };

        assert_ne!(bus_a, bus_c);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_bus_serde_roundtrip() {
        let bus = Bus {
            id: EntityId::from(1),
            name: "Main Bus".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: Some(200.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            excess_cost: 1.5,
        };
        let json = serde_json::to_string(&bus).unwrap();
        let deserialized: Bus = serde_json::from_str(&json).unwrap();
        assert_eq!(bus, deserialized);
    }
}
