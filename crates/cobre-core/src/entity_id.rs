//! Strongly-typed entity identifier used across all entity collections.
//!
//! [`EntityId`] wraps an `i32` (the type used in JSON input schemas) and
//! prevents accidental confusion between entity IDs and collection indices.

use core::fmt;

/// Strongly-typed entity identifier.
///
/// Wraps the `i32` identifier from JSON input files. The newtype pattern prevents
/// accidental confusion between entity IDs and collection indices (`usize`), which
/// is a common source of bugs in systems with both ID-based lookup and index-based
/// access. `EntityId` is used as the key in `HashMap<EntityId, usize>` lookup tables
/// and as the value in cross-reference fields (e.g., `Hydro::bus_id`, `Line::source_bus_id`).
///
/// Why `i32` and not `String`: All JSON entity schemas use integer IDs (`i32`). Integer
/// keys are cheaper to hash, compare, and copy than strings — important because
/// `EntityId` appears in every lookup table and cross-reference field. If a future
/// input format requires string IDs, the newtype boundary isolates the change to
/// `EntityId`'s internal representation and its `From`/`Into` impls.
///
/// # Examples
///
/// ```
/// use cobre_core::EntityId;
///
/// let id: EntityId = EntityId::from(42);
/// assert_eq!(id.to_string(), "42");
///
/// let raw: i32 = i32::from(id);
/// assert_eq!(raw, 42);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EntityId(pub i32);

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i32> for EntityId {
    fn from(value: i32) -> Self {
        Self(value)
    }
}

impl From<EntityId> for i32 {
    fn from(id: EntityId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use core::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    use super::EntityId;

    #[test]
    fn test_equality() {
        let a = EntityId(1);
        let b = EntityId(1);
        let c = EntityId(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_copy() {
        let a = EntityId(10);
        let b = a;
        assert_eq!(a, b);
        assert_eq!(a.0, 10);
    }

    #[test]
    fn test_hash_consistency() {
        let mut hasher_a = DefaultHasher::new();
        EntityId(99).hash(&mut hasher_a);
        let hash_a = hasher_a.finish();

        let mut hasher_b = DefaultHasher::new();
        EntityId(99).hash(&mut hasher_b);
        let hash_b = hasher_b.finish();

        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn test_display() {
        assert_eq!(EntityId(42).to_string(), "42");
        assert_eq!(EntityId(0).to_string(), "0");
        assert_eq!(EntityId(-1).to_string(), "-1");
    }

    #[test]
    fn test_from_i32() {
        let id = EntityId::from(5);
        assert_eq!(id, EntityId(5));
    }

    #[test]
    fn test_into_i32() {
        let raw: i32 = i32::from(EntityId(7));
        assert_eq!(raw, 7);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_entity_id_serde_roundtrip() {
        let id = EntityId(42);
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: EntityId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
