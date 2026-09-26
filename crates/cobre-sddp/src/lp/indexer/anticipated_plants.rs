//! [`AnticipatedPlants`]: the study-scope set of thermals carrying a declared
//! `anticipated_config`.

use cobre_core::Thermal;

use super::{AnticipatedLocal, ThermalSys};

/// The anticipated-plant set. Anticipated-local position `i` is the study's
/// `i`-th anticipated plant — the order every per-plant vector
/// (`AnticipatedResolution::per_plant`, `RingResidue::plant`, …) shares.
#[derive(Debug, Clone, Default)]
pub struct AnticipatedPlants {
    thermals: Vec<ThermalSys>,
}

impl AnticipatedPlants {
    /// Keep the thermals with a declared `anticipated_config`, in canonical
    /// `thermals` order — the crate's sole membership predicate for this set.
    #[must_use]
    pub fn build(thermals: &[Thermal]) -> Self {
        Self {
            thermals: thermals
                .iter()
                .enumerate()
                .filter(|(_, thermal)| thermal.anticipated_config.is_some())
                .map(|(t_idx, _)| ThermalSys::new(t_idx))
                .collect(),
        }
    }

    /// Number of anticipated plants.
    #[inline]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.thermals.len()
    }

    /// The set's thermals, in anticipated-local order.
    pub(crate) fn thermals(&self) -> impl Iterator<Item = ThermalSys> + '_ {
        self.thermals.iter().copied()
    }

    /// Plant `local`'s system position.
    #[inline]
    #[must_use]
    pub(crate) fn thermal_of(&self, local: AnticipatedLocal) -> ThermalSys {
        self.thermals[local.get()]
    }

    /// Test-only seam: builds a set from `thermals` in the given order,
    /// bypassing `build`'s canonical-order guarantee. Never reachable outside
    /// the crate's own unit-test builds.
    #[cfg(test)]
    pub(crate) fn from_positions_for_test(thermals: Vec<ThermalSys>) -> Self {
        Self { thermals }
    }
}

#[cfg(test)]
mod tests {
    use super::AnticipatedPlants;
    use crate::indexer::{AnticipatedLocal, ThermalSys};
    use crate::test_support::anticipated_plants_at;

    #[test]
    fn build_keeps_only_anticipated_thermals_in_canonical_order() {
        let plants = anticipated_plants_at(&[1, 3]);
        assert_eq!(plants.len(), 2);
        assert_eq!(
            plants.thermals().collect::<Vec<_>>(),
            vec![ThermalSys::new(1), ThermalSys::new(3)]
        );
        assert_eq!(
            plants.thermal_of(AnticipatedLocal::new(1)),
            ThermalSys::new(3)
        );
    }

    #[test]
    fn default_is_the_empty_set() {
        assert_eq!(AnticipatedPlants::default().len(), 0);
        assert_eq!(AnticipatedPlants::build(&[]).len(), 0);
    }
}
