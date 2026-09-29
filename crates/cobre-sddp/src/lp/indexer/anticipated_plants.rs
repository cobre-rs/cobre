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
    windows: Vec<(Option<i32>, Option<i32>)>,
    local_by_thermal: Vec<Option<AnticipatedLocal>>,
}

impl AnticipatedPlants {
    /// Keep the thermals with a declared `anticipated_config`, in canonical
    /// `thermals` order — the crate's sole membership predicate for this set.
    #[must_use]
    pub fn build(thermals: &[Thermal]) -> Self {
        let mut anticipated = Vec::new();
        let mut windows = Vec::new();
        let mut local_by_thermal = vec![None; thermals.len()];
        for (t_idx, thermal) in thermals.iter().enumerate() {
            if thermal.anticipated_config.is_none() {
                continue;
            }
            local_by_thermal[t_idx] = Some(AnticipatedLocal::new(anticipated.len()));
            anticipated.push(ThermalSys::new(t_idx));
            windows.push((thermal.entry_stage_id, thermal.exit_stage_id));
        }
        Self {
            thermals: anticipated,
            windows,
            local_by_thermal,
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

    /// Per-plant `(entry_stage_id, exit_stage_id)` commissioning window,
    /// anticipated-local order.
    #[inline]
    #[must_use]
    pub(crate) fn windows(&self) -> &[(Option<i32>, Option<i32>)] {
        &self.windows
    }

    /// Thermal `t`'s anticipated-local position, or `None` when `t` is not
    /// anticipated.
    #[inline]
    #[must_use]
    pub(crate) fn local_of(&self, t: ThermalSys) -> Option<AnticipatedLocal> {
        self.local_by_thermal.get(t.get()).copied().flatten()
    }

    /// Test-only seam: builds a set from `thermals` and their `windows` in the
    /// given order, bypassing `build`'s canonical-order guarantee. Never
    /// reachable outside the crate's own unit-test builds.
    #[cfg(test)]
    pub(crate) fn from_positions_for_test(
        thermals: Vec<ThermalSys>,
        windows: Vec<(Option<i32>, Option<i32>)>,
    ) -> Self {
        let mut local_by_thermal =
            vec![None; thermals.iter().map(|t| t.get()).max().map_or(0, |m| m + 1)];
        for (local, t) in thermals.iter().enumerate() {
            local_by_thermal[t.get()] = Some(AnticipatedLocal::new(local));
        }
        Self {
            thermals,
            windows,
            local_by_thermal,
        }
    }
}

#[cfg(test)]
mod tests {
    use cobre_core::{AnticipatedConfig, EntityId, Thermal};

    use super::AnticipatedPlants;
    use crate::indexer::{AnticipatedLocal, ThermalSys};
    use crate::test_support::anticipated_plants_at;

    fn thermal_with_window(
        id: i32,
        anticipated_config: Option<AnticipatedConfig>,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: String::new(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(0),
            entry_stage_id,
            exit_stage_id,
            cost_per_mwh: 0.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config,
        }
    }

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

    #[test]
    fn local_of_agrees_with_thermal_of() {
        let plants = anticipated_plants_at(&[1, 3]);
        for local in 0..plants.len() {
            let local = AnticipatedLocal::new(local);
            assert_eq!(plants.local_of(plants.thermal_of(local)), Some(local));
        }
        assert_eq!(plants.local_of(ThermalSys::new(0)), None);
        assert_eq!(plants.local_of(ThermalSys::new(2)), None);
        assert_eq!(plants.local_of(ThermalSys::new(99)), None);
    }

    #[test]
    fn windows_follow_the_thermal_declaration() {
        let thermals = vec![
            thermal_with_window(0, None, None, None),
            thermal_with_window(1, Some(AnticipatedConfig::LeadStages(1)), Some(2), Some(5)),
            thermal_with_window(2, Some(AnticipatedConfig::LeadStages(1)), None, Some(9)),
        ];
        let plants = AnticipatedPlants::build(&thermals);
        assert_eq!(plants.windows(), [(Some(2), Some(5)), (None, Some(9))]);
    }
}
