//! Error types produced during `System` construction and validation.
//!
//! [`ValidationError`] is returned by the system builder when entity cross-references
//! are invalid, duplicate IDs are detected, topology is malformed, or penalty
//! configuration is invalid.

use core::fmt;

use crate::EntityId;

/// Errors produced during System construction and validation.
///
/// Returned by the `System` builder when loading and validating entity collections.
/// Each variant captures enough context to pinpoint the invalid input without
/// requiring the caller to re-inspect the data.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ValidationError};
///
/// let err = ValidationError::DuplicateId {
///     entity_type: "Bus",
///     id: EntityId(1),
/// };
/// assert!(err.to_string().contains("Bus"));
/// ```
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// A cross-reference field (e.g., `bus_id`, `downstream_id`) refers to
    /// an entity ID that does not exist in the system.
    InvalidReference {
        /// The entity type containing the invalid reference.
        source_entity_type: &'static str,
        /// The ID of the entity containing the invalid reference.
        source_id: EntityId,
        /// The name of the field containing the invalid reference.
        field_name: &'static str,
        /// The referenced ID that does not exist.
        referenced_id: EntityId,
        /// The entity type that was expected.
        expected_type: &'static str,
    },
    /// Duplicate entity ID within a single entity collection.
    DuplicateId {
        /// The entity type with the duplicated ID.
        entity_type: &'static str,
        /// The duplicated entity ID.
        id: EntityId,
    },
    /// The hydro cascade contains a cycle.
    CascadeCycle {
        /// IDs of hydros forming the cycle.
        cycle_ids: Vec<EntityId>,
    },
    /// A hydro's filling configuration is invalid.
    InvalidFillingConfig {
        /// The hydro whose filling configuration is invalid.
        hydro_id: EntityId,
        /// Human-readable explanation of why the configuration is invalid.
        reason: String,
    },
    /// A bus has no connections (no lines, generators, or loads).
    DisconnectedBus {
        /// The ID of the disconnected bus.
        bus_id: EntityId,
    },
    /// Entity-level penalty value is invalid (e.g., negative cost).
    InvalidPenalty {
        /// The entity type with the invalid penalty.
        entity_type: &'static str,
        /// The ID of the entity with the invalid penalty.
        entity_id: EntityId,
        /// The name of the penalty field that is invalid.
        field_name: &'static str,
        /// Human-readable explanation of why the penalty is invalid.
        reason: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReference {
                source_entity_type,
                source_id,
                field_name,
                referenced_id,
                expected_type,
            } => write!(
                f,
                "{source_entity_type} with id {source_id} has invalid cross-reference \
                 in field '{field_name}': referenced {expected_type} id {referenced_id} does not exist"
            ),
            Self::DuplicateId { entity_type, id } => {
                write!(f, "duplicate {entity_type} id: {id}")
            }
            Self::CascadeCycle { cycle_ids } => {
                let ids = cycle_ids
                    .iter()
                    .map(EntityId::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "hydro cascade contains a cycle: [{ids}]")
            }
            Self::InvalidFillingConfig { hydro_id, reason } => {
                write!(
                    f,
                    "hydro {hydro_id} has invalid filling configuration: {reason}"
                )
            }
            Self::DisconnectedBus { bus_id } => {
                write!(f, "bus {bus_id} is disconnected (no lines, generators, or loads)")
            }
            Self::InvalidPenalty {
                entity_type,
                entity_id,
                field_name,
                reason,
            } => write!(
                f,
                "{entity_type} with id {entity_id} has invalid penalty in field '{field_name}': {reason}"
            ),
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::ValidationError;
    use crate::EntityId;

    #[test]
    fn test_display_invalid_reference() {
        let err = ValidationError::InvalidReference {
            source_entity_type: "Hydro",
            source_id: EntityId(3),
            field_name: "bus_id",
            referenced_id: EntityId(99),
            expected_type: "Bus",
        };
        let msg = err.to_string();
        assert!(msg.contains("Hydro"), "missing source entity type: {msg}");
        assert!(msg.contains("bus_id"), "missing field name: {msg}");
        assert!(msg.contains("99"), "missing referenced id: {msg}");
    }

    #[test]
    fn test_display_duplicate_id() {
        let err = ValidationError::DuplicateId {
            entity_type: "Thermal",
            id: EntityId(5),
        };
        let msg = err.to_string();
        assert!(msg.contains("Thermal"), "missing entity type: {msg}");
        assert!(msg.contains('5'), "missing id: {msg}");
    }

    #[test]
    fn test_display_cascade_cycle() {
        let err = ValidationError::CascadeCycle {
            cycle_ids: vec![EntityId(1), EntityId(2), EntityId(3)],
        };
        let msg = err.to_string();
        assert!(msg.contains('1'), "missing id 1: {msg}");
        assert!(msg.contains('2'), "missing id 2: {msg}");
        assert!(msg.contains('3'), "missing id 3: {msg}");
    }

    #[test]
    fn test_error_trait() {
        let err = ValidationError::DisconnectedBus {
            bus_id: EntityId(7),
        };
        // Verify ValidationError can be used as &dyn std::error::Error
        let _: &dyn std::error::Error = &err;
    }
}
