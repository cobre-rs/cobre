//! Postcard serialization helpers for MPI broadcast of [`System`].
//!
//! Cobre uses `postcard` (not `bincode`) for MPI serialization (see CLAUDE.md hard rules).
//! These helpers serialize a [`System`] to a compact byte buffer for broadcast
//! and deserialize it on worker ranks, rebuilding the O(1) lookup indices that
//! are skipped during serialization (per spec SS6.2).
//!
//! # Usage
//!
//! On rank 0, load the case and serialize:
//!
//! ```rust,ignore
//! let system = cobre_io::load_case(&path)?;
//! let bytes = cobre_io::serialize_system(&system)?;
//! // broadcast bytes via MPI ...
//! ```
//!
//! On worker ranks, deserialize after receiving:
//!
//! ```rust,ignore
//! // ... receive bytes via MPI
//! let system = cobre_io::deserialize_system(&bytes)?;
//! // system.bus(id) works immediately — indices are rebuilt
//! ```

use cobre_core::System;

use crate::LoadError;

/// Serialize a [`System`] to a postcard byte buffer for MPI broadcast.
///
/// The returned `Vec<u8>` is suitable for broadcasting over MPI. The recipient
/// must call [`deserialize_system`] to reconstruct the [`System`] with working
/// O(1) lookup indices.
///
/// # Errors
///
/// Returns [`LoadError::ParseError`] with path `"<broadcast>"` if postcard
/// encounters an unsupported type during serialization. This should not occur
/// in practice given the types used in [`System`].
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_io::serialize_system;
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "Main Bus".to_string(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 500.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();
/// let bytes = serialize_system(&system).unwrap();
/// assert!(!bytes.is_empty());
/// ```
pub fn serialize_system(system: &System) -> Result<Vec<u8>, LoadError> {
    postcard::to_allocvec(system)
        .map_err(|e| LoadError::parse("<broadcast>", format!("postcard serialization failed: {e}")))
}

/// Deserialize a [`System`] from a postcard byte buffer received via MPI broadcast.
///
/// Calls [`System::rebuild_indices`] after deserialization so that O(1) entity
/// lookups (e.g., `system.bus(id)`) work immediately on the returned value.
///
/// # Errors
///
/// Returns [`LoadError::ParseError`] with path `"<broadcast>"` if the byte slice
/// is corrupted, truncated, or not a valid postcard encoding of [`System`].
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_io::{deserialize_system, serialize_system};
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "Main Bus".to_string(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 500.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();
/// let bytes = serialize_system(&system).unwrap();
/// let restored = deserialize_system(&bytes).unwrap();
/// assert_eq!(restored.n_buses(), 1);
/// assert!(restored.bus(EntityId(1)).is_some());
/// ```
pub fn deserialize_system(bytes: &[u8]) -> Result<System, LoadError> {
    let mut system: System = postcard::from_bytes(bytes).map_err(|e| {
        LoadError::parse(
            "<broadcast>",
            format!("postcard deserialization failed: {e}"),
        )
    })?;
    system.rebuild_indices();
    Ok(system)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use cobre_core::{
        Bus, DeficitSegment, EntityId, Hydro, HydroGenerationModel, HydroPenalties, SystemBuilder,
        Thermal, ThermalCostSegment,
    };

    fn minimal_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus {id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn minimal_thermal(id: i32, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("Thermal {id}"),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_segments: vec![ThermalCostSegment {
                capacity_mw: 100.0,
                cost_per_mwh: 50.0,
            }],
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            gnl_config: None,
        }
    }

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
        }
    }

    fn minimal_hydro(id: i32, bus_id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId(bus_id),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 200.0,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        }
    }

    #[test]
    fn test_round_trip_minimal_system() {
        let bus = minimal_bus(1);
        let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();

        let bytes = serialize_system(&system).unwrap();
        assert!(!bytes.is_empty());

        let restored = deserialize_system(&bytes).unwrap();

        assert_eq!(restored.n_buses(), system.n_buses());
        assert!(restored.bus(EntityId(1)).is_some());
    }

    #[test]
    fn test_round_trip_populated_system() {
        let buses = vec![minimal_bus(1), minimal_bus(2)];
        let thermals = vec![minimal_thermal(1, 1), minimal_thermal(2, 2)];
        let hydros = vec![minimal_hydro(1, 1)];

        let system = SystemBuilder::new()
            .buses(buses)
            .thermals(thermals)
            .hydros(hydros)
            .build()
            .unwrap();

        let bytes = serialize_system(&system).unwrap();
        let restored = deserialize_system(&bytes).unwrap();

        // Verify all entity counts match
        assert_eq!(restored.n_buses(), system.n_buses());
        assert_eq!(restored.n_thermals(), system.n_thermals());
        assert_eq!(restored.n_hydros(), system.n_hydros());

        // Verify O(1) lookups work for all entity types
        assert!(restored.bus(EntityId(1)).is_some());
        assert!(restored.bus(EntityId(2)).is_some());
        assert!(restored.thermal(EntityId(1)).is_some());
        assert!(restored.thermal(EntityId(2)).is_some());
        assert!(restored.hydro(EntityId(1)).is_some());

        // Verify structural equality
        assert_eq!(restored, system);
    }

    #[test]
    fn test_deserialize_corrupted_bytes() {
        let result = deserialize_system(&[0u8; 4]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("<broadcast>"));
        assert!(matches!(err, LoadError::ParseError { .. }));
    }

    #[test]
    fn test_deserialize_empty_bytes() {
        let result = deserialize_system(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LoadError::ParseError { .. }));
        assert!(err.to_string().contains("<broadcast>"));
    }

    #[test]
    fn test_serialized_size_reasonable() {
        let bus = minimal_bus(1);
        let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();
        let bytes = serialize_system(&system).unwrap();
        assert!(bytes.len() < 1024);
    }
}
