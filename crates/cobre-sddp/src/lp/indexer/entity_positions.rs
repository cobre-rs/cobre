//! [`EntityPositions`]: the single entity-id -> slot-index owner for every
//! position-addressed entity family.

use std::collections::BTreeMap;

use cobre_core::{EntityId, System};

/// Canonical `EntityId -> slot` maps for hydros, thermals, lines, buses,
/// pumping stations, and energy contracts, each in the family's own
/// canonical (id-sorted) slice order.
pub(crate) struct EntityPositions {
    hydro: BTreeMap<EntityId, usize>,
    thermal: BTreeMap<EntityId, usize>,
    line: BTreeMap<EntityId, usize>,
    bus: BTreeMap<EntityId, usize>,
    pumping: BTreeMap<EntityId, usize>,
    contract: BTreeMap<EntityId, usize>,
}

impl EntityPositions {
    /// Builds every position map from `system`'s own canonical slices.
    ///
    /// Also holds the two resolved-bounds station/contract count
    /// cross-checks: a divergence would otherwise silently reserve the wrong
    /// number of pumping-flow or contract columns.
    #[must_use]
    pub(crate) fn build(system: &System) -> Self {
        let pumping_stations = system.pumping_stations();
        debug_assert_eq!(
            pumping_stations.len(),
            system.bounds().n_pumping(),
            "pumping_stations.len() ({}) != bounds.n_pumping() ({}): resolved-bounds \
             station count disagrees with the entity slice",
            pumping_stations.len(),
            system.bounds().n_pumping()
        );
        let contracts = system.contracts();
        debug_assert_eq!(
            contracts.len(),
            system.bounds().n_contracts(),
            "contracts.len() ({}) != bounds.n_contracts() ({}): resolved-bounds \
             contract count disagrees with the entity slice",
            contracts.len(),
            system.bounds().n_contracts()
        );
        Self::from_slices(
            system.hydros().iter().map(|h| h.id),
            system.thermals().iter().map(|t| t.id),
            system.lines().iter().map(|l| l.id),
            system.buses().iter().map(|b| b.id),
            pumping_stations.iter().map(|p| p.id),
            contracts.iter().map(|c| c.id),
        )
    }

    /// Builds the six maps from id iterators alone, bypassing the
    /// resolved-bounds cross-checks [`Self::build`] additionally holds
    /// (unavailable to a hand-crafted test fixture with no backing
    /// [`System`]). [`Self::build`] delegates here once its own checks pass.
    pub(crate) fn from_slices(
        hydros: impl IntoIterator<Item = EntityId>,
        thermals: impl IntoIterator<Item = EntityId>,
        lines: impl IntoIterator<Item = EntityId>,
        buses: impl IntoIterator<Item = EntityId>,
        pumping_stations: impl IntoIterator<Item = EntityId>,
        contracts: impl IntoIterator<Item = EntityId>,
    ) -> Self {
        fn indexed(ids: impl IntoIterator<Item = EntityId>) -> BTreeMap<EntityId, usize> {
            ids.into_iter().enumerate().map(|(i, id)| (id, i)).collect()
        }
        Self {
            hydro: indexed(hydros),
            thermal: indexed(thermals),
            line: indexed(lines),
            bus: indexed(buses),
            pumping: indexed(pumping_stations),
            contract: indexed(contracts),
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn hydro(&self, id: EntityId) -> Option<usize> {
        self.hydro.get(&id).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn thermal(&self, id: EntityId) -> Option<usize> {
        self.thermal.get(&id).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn line(&self, id: EntityId) -> Option<usize> {
        self.line.get(&id).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn bus(&self, id: EntityId) -> Option<usize> {
        self.bus.get(&id).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn pumping(&self, id: EntityId) -> Option<usize> {
        self.pumping.get(&id).copied()
    }

    #[inline]
    #[must_use]
    pub(crate) fn contract(&self, id: EntityId) -> Option<usize> {
        self.contract.get(&id).copied()
    }
}

#[cfg(test)]
mod tests {
    use cobre_core::EntityId;

    use super::EntityPositions;

    #[test]
    fn positions_follow_slice_order() {
        let positions = EntityPositions::from_slices(
            [EntityId(3), EntityId(1)],
            [EntityId(10)],
            [],
            [EntityId(5), EntityId(6)],
            [],
            [],
        );
        assert_eq!(positions.hydro(EntityId(3)), Some(0));
        assert_eq!(positions.hydro(EntityId(1)), Some(1));
        assert_eq!(positions.thermal(EntityId(10)), Some(0));
        assert_eq!(positions.bus(EntityId(5)), Some(0));
        assert_eq!(positions.bus(EntityId(6)), Some(1));
    }

    #[test]
    fn unknown_id_is_none() {
        let positions = EntityPositions::from_slices([EntityId(1)], [], [], [], [], []);
        assert_eq!(positions.hydro(EntityId(99)), None);
        assert_eq!(positions.thermal(EntityId(1)), None);
        assert_eq!(positions.line(EntityId(1)), None);
        assert_eq!(positions.bus(EntityId(1)), None);
        assert_eq!(positions.pumping(EntityId(1)), None);
        assert_eq!(positions.contract(EntityId(1)), None);
    }
}
