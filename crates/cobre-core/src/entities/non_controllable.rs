//! Non-controllable generation source entity — intermittent wind/solar.
//!
//! A `NonControllableSource` represents intermittent generation (wind, solar, run-of-river)
//! that cannot be dispatched. Curtailment may be optionally allowed. This entity is a
//! NO-OP stub in Phase 1: the type exists in the registry but contributes zero LP
//! variables or constraints.

use crate::EntityId;

/// Intermittent generation source that cannot be dispatched.
///
/// A `NonControllableSource` injects all available generation into the network.
/// If curtailment is permitted, excess generation can be curtailed at a cost of
/// `curtailment_cost` per `MWh`. In the minimal viable solver this entity is
/// data-complete but contributes no LP variables or constraints.
///
/// Source: `system/non_controllable.json`. See Input System Entities SS1.9.8.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NonControllableSource {
    /// Unique source identifier.
    pub id: EntityId,
    /// Human-readable source name.
    pub name: String,
    /// Bus to which this source's generation is injected.
    pub bus_id: EntityId,
    /// Stage index when the source enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the source is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Maximum generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// Resolved cost per `MWh` of curtailed generation \[$/`MWh`\].
    ///
    /// This is a resolved field — defaults are applied during loading so this
    /// value is always ready for LP construction without further lookup.
    pub curtailment_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_controllable_construction() {
        let source = NonControllableSource {
            id: EntityId::from(1),
            name: "Eólica Caetité".to_string(),
            bus_id: EntityId::from(7),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 300.0,
            curtailment_cost: 0.01,
        };

        assert_eq!(source.id, EntityId::from(1));
        assert_eq!(source.name, "Eólica Caetité");
        assert_eq!(source.bus_id, EntityId::from(7));
        assert_eq!(source.entry_stage_id, None);
        assert_eq!(source.exit_stage_id, None);
        assert!((source.max_generation_mw - 300.0).abs() < f64::EPSILON);
        assert!((source.curtailment_cost - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_non_controllable_curtailment_cost() {
        let source = NonControllableSource {
            id: EntityId::from(2),
            name: "Solar Pirapora".to_string(),
            bus_id: EntityId::from(3),
            entry_stage_id: Some(12),
            exit_stage_id: None,
            max_generation_mw: 400.0,
            curtailment_cost: 5.0,
        };

        assert!((source.curtailment_cost - 5.0).abs() < f64::EPSILON);
    }
}
