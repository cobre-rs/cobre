//! Line entity — a transmission interconnection between two buses.
//!
//! A `Line` represents a transmission line (or interconnection) between a source
//! bus and a target bus. Lines have MW capacity bounds that limit power flow.

use crate::EntityId;

/// Transmission interconnection between two buses.
///
/// Lines allow bidirectional power transfer subject to capacity limits and
/// transmission losses. Line flow is a hard constraint (no slack variables) --
/// the `exchange_cost` is a regularization penalty, not a violation penalty.
///
/// Source: system/lines.json. See Input System Entities SS2.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Line {
    /// Unique line identifier.
    pub id: EntityId,
    /// Human-readable line name.
    pub name: String,
    /// Source bus for direct flow direction.
    pub source_bus_id: EntityId,
    /// Target bus for direct flow direction.
    pub target_bus_id: EntityId,
    /// Stage when line enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage when line is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Maximum flow from source to target \[MW\]. Hard bound.
    pub direct_capacity_mw: f64,
    /// Maximum flow from target to source \[MW\]. Hard bound.
    pub reverse_capacity_mw: f64,
    /// Transmission losses as percentage (e.g., 2.5 means 2.5%).
    pub losses_percent: f64,
    /// Regularization cost per `MWh` exchanged \[$/`MWh`\].
    pub exchange_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_construction() {
        let line = Line {
            id: EntityId::from(1),
            name: "Line A-B".to_string(),
            source_bus_id: EntityId::from(10),
            target_bus_id: EntityId::from(20),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 500.0,
            reverse_capacity_mw: 400.0,
            losses_percent: 1.5,
            exchange_cost: 0.5,
        };

        assert_eq!(line.id, EntityId::from(1));
        assert_eq!(line.name, "Line A-B");
        assert_eq!(line.source_bus_id, EntityId::from(10));
        assert_eq!(line.target_bus_id, EntityId::from(20));
        assert_eq!(line.entry_stage_id, None);
        assert_eq!(line.exit_stage_id, None);
        assert_eq!(line.direct_capacity_mw, 500.0);
        assert_eq!(line.reverse_capacity_mw, 400.0);
        assert_eq!(line.losses_percent, 1.5);
        assert_eq!(line.exchange_cost, 0.5);
    }

    #[test]
    fn test_line_lifecycle_always() {
        let line = Line {
            id: EntityId::from(2),
            name: "Permanent Line".to_string(),
            source_bus_id: EntityId::from(1),
            target_bus_id: EntityId::from(2),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        };

        assert_eq!(line.entry_stage_id, None);
        assert_eq!(line.exit_stage_id, None);
    }

    #[test]
    fn test_line_lifecycle_bounded() {
        let line = Line {
            id: EntityId::from(3),
            name: "Temporary Line".to_string(),
            source_bus_id: EntityId::from(1),
            target_bus_id: EntityId::from(2),
            entry_stage_id: Some(5),
            exit_stage_id: Some(120),
            direct_capacity_mw: 200.0,
            reverse_capacity_mw: 200.0,
            losses_percent: 2.5,
            exchange_cost: 1.0,
        };

        assert_eq!(line.entry_stage_id, Some(5));
        assert_eq!(line.exit_stage_id, Some(120));
    }

    #[test]
    fn test_line_equality() {
        let line_a = Line {
            id: EntityId::from(1),
            name: "Line A-B".to_string(),
            source_bus_id: EntityId::from(10),
            target_bus_id: EntityId::from(20),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 500.0,
            reverse_capacity_mw: 400.0,
            losses_percent: 1.5,
            exchange_cost: 0.5,
        };

        let line_b = line_a.clone();
        assert_eq!(line_a, line_b);

        let line_c = Line {
            id: EntityId::from(99),
            ..line_b
        };

        assert_ne!(line_a, line_c);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_line_serde_roundtrip() {
        let line = Line {
            id: EntityId::from(3),
            name: "Temporary Line".to_string(),
            source_bus_id: EntityId::from(1),
            target_bus_id: EntityId::from(2),
            entry_stage_id: Some(5),
            exit_stage_id: Some(120),
            direct_capacity_mw: 200.0,
            reverse_capacity_mw: 200.0,
            losses_percent: 2.5,
            exchange_cost: 1.0,
        };
        let json = serde_json::to_string(&line).unwrap();
        let deserialized: Line = serde_json::from_str(&json).unwrap();
        assert_eq!(line, deserialized);
    }
}
