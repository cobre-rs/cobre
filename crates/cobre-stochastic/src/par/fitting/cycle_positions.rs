//! Season relabeling at the fitting boundary: the fitting kernels take a
//! season's lags at `(m + n − lag % n) % n`, which names its predecessors only
//! when `m` is the season's position in the season cycle.

use std::collections::HashMap;

use cobre_core::temporal::{SeasonCycles, SeasonDefinition, SeasonMap, Stage};

use super::{ArCoefficientEstimate, ContributionReduction, EstimationReport, HydroEstimationEntry};

/// Raw season ids and their positions in a single-level season cycle whose
/// ids are not already its positions.
///
/// Only single-level maps qualify: their definitions never overlap, so
/// first-match `season_for_date` on the relabeled map resolves every date to
/// the same season.
pub(super) struct CyclePositions {
    raw_of: Vec<usize>,
    position_of: Vec<Option<usize>>,
}

impl CyclePositions {
    pub(super) fn new(season_map: Option<&SeasonMap>) -> Option<Self> {
        let season_map = season_map?;
        let cycles = SeasonCycles::new(season_map);
        let mut raw_of = vec![0; season_map.seasons.len()];
        for def in &season_map.seasons {
            let Some((0, position)) = cycles.position(def.id) else {
                return None;
            };
            *raw_of.get_mut(position)? = def.id;
        }
        if raw_of
            .iter()
            .enumerate()
            .all(|(position, &raw)| position == raw)
        {
            return None;
        }
        let mut position_of = vec![None; raw_of.iter().max().map_or(0, |&raw| raw + 1)];
        for (position, &raw) in raw_of.iter().enumerate() {
            position_of[raw] = Some(position);
        }
        Some(Self {
            raw_of,
            position_of,
        })
    }

    pub(super) fn position(&self, raw: usize) -> Option<usize> {
        self.position_of.get(raw).copied().flatten()
    }

    pub(super) fn raw(&self, position: usize) -> Option<usize> {
        self.raw_of.get(position).copied()
    }

    pub(super) fn raw_ids(&self) -> &[usize] {
        &self.raw_of
    }

    pub(super) fn relabel_stages(&self, stages: &[Stage]) -> Vec<Stage> {
        stages
            .iter()
            .map(|stage| Stage {
                season_id: stage.season_id.and_then(|raw| self.position(raw)),
                ..stage.clone()
            })
            .collect()
    }

    pub(super) fn relabel_season_map(&self, season_map: &SeasonMap) -> SeasonMap {
        let mut seasons: Vec<SeasonDefinition> = season_map
            .seasons
            .iter()
            .filter_map(|def| {
                Some(SeasonDefinition {
                    id: self.position(def.id)?,
                    ..def.clone()
                })
            })
            .collect();
        seasons.sort_by_key(|def| def.id);
        SeasonMap {
            cycle_type: season_map.cycle_type,
            seasons,
        }
    }

    pub(super) fn estimates_to_positions(
        &self,
        estimates: &[ArCoefficientEstimate],
    ) -> Vec<ArCoefficientEstimate> {
        estimates
            .iter()
            .filter_map(|estimate| {
                Some(ArCoefficientEstimate {
                    season_id: self.position(estimate.season_id)?,
                    ..estimate.clone()
                })
            })
            .collect()
    }

    pub(super) fn estimates_to_raw(
        &self,
        estimates: Vec<ArCoefficientEstimate>,
    ) -> Vec<ArCoefficientEstimate> {
        estimates
            .into_iter()
            .filter_map(|estimate| {
                Some(ArCoefficientEstimate {
                    season_id: self.raw(estimate.season_id)?,
                    ..estimate
                })
            })
            .collect()
    }

    pub(super) fn report_to_raw(&self, report: EstimationReport) -> EstimationReport {
        let entries = report
            .entries
            .into_iter()
            .map(|(hydro_id, entry)| (hydro_id, self.entry_to_raw(entry)))
            .collect();
        EstimationReport { entries, ..report }
    }

    /// Re-indexes `coefficients` by raw id, empty at ids that name no season.
    fn entry_to_raw(&self, entry: HydroEstimationEntry) -> HydroEstimationEntry {
        let len = (0..entry.coefficients.len())
            .filter_map(|position| self.raw(position))
            .max()
            .map_or(0, |raw| raw + 1);
        let mut coefficients = vec![Vec::new(); len];
        for (position, season_coefficients) in entry.coefficients.into_iter().enumerate() {
            if let Some(raw) = self.raw(position) {
                coefficients[raw] = season_coefficients;
            }
        }
        let contribution_reductions = entry
            .contribution_reductions
            .into_iter()
            .filter_map(|reduction| {
                Some(ContributionReduction {
                    season_id: self.raw(reduction.season_id)?,
                    ..reduction
                })
            })
            .collect();
        HydroEstimationEntry {
            selected_order: entry.selected_order,
            coefficients,
            contribution_reductions,
        }
    }

    pub(super) fn seasons_to_raw<T>(&self, by_position: HashMap<usize, T>) -> HashMap<usize, T> {
        by_position
            .into_iter()
            .filter_map(|(position, value)| Some((self.raw(position)?, value)))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
pub(super) mod twin_fixtures {
    use chrono::NaiveDate;
    use cobre_core::test_support::{StageSpec, date, make_stage};
    use cobre_core::{EntityId, SeasonDefinition, SeasonMap, Stage};

    use crate::test_support::{quarterly_season_map, sparse_ring_season_map};

    /// A map whose ids are not their calendar positions and its twin over the
    /// same calendar spans with ids `0..n` in calendar order.
    pub(crate) struct TwinMaps {
        pub(crate) map: SeasonMap,
        pub(crate) twin: SeasonMap,
        /// `(map id, twin id)` per season, in calendar order.
        pub(crate) ids: Vec<(usize, usize)>,
    }

    impl TwinMaps {
        pub(crate) fn map_ids(&self) -> Vec<usize> {
            self.ids.iter().map(|&(id, _)| id).collect()
        }

        pub(crate) fn twin_ids(&self) -> Vec<usize> {
            self.ids.iter().map(|&(_, twin_id)| twin_id).collect()
        }

        pub(crate) fn twin_of(&self, id: usize) -> Option<usize> {
            self.ids
                .iter()
                .find(|&&(map_id, _)| map_id == id)
                .map(|&(_, twin_id)| twin_id)
        }
    }

    /// `quarterly_season_map()` renumbered `[2 Jan–Mar, 0 Apr–Jun, 3 Jul–Sep, 1 Oct–Dec]`.
    pub(crate) fn permuted_quarterly_season_map() -> SeasonMap {
        let mut season_map = quarterly_season_map();
        for (def, id) in season_map.seasons.iter_mut().zip([2, 0, 3, 1]) {
            def.id = id;
        }
        season_map.seasons.sort_by_key(|def| def.id);
        season_map
    }

    pub(crate) fn sparse_ring_twins() -> TwinMaps {
        let map = sparse_ring_season_map();
        let mut twin = map.clone();
        for (id, def) in twin.seasons.iter_mut().enumerate() {
            def.id = id;
        }
        TwinMaps {
            map,
            twin,
            ids: vec![(0, 0), (1, 1), (2, 2), (12, 3), (13, 4)],
        }
    }

    pub(crate) fn permuted_quarterly_twins() -> TwinMaps {
        TwinMaps {
            map: permuted_quarterly_season_map(),
            twin: quarterly_season_map(),
            ids: vec![(2, 0), (0, 1), (3, 2), (1, 3)],
        }
    }

    fn definition(season_map: &SeasonMap, id: usize) -> &SeasonDefinition {
        season_map
            .seasons
            .iter()
            .find(|def| def.id == id)
            .expect("the fixture defines every calendar id")
    }

    /// One stage per season in 2050, in calendar order, with stage id = the
    /// season's calendar index, so a map and its twin share stage ids.
    pub(crate) fn calendar_stages(season_map: &SeasonMap, calendar_ids: &[usize]) -> Vec<Stage> {
        (0_i32..)
            .zip(calendar_ids)
            .map(|(stage_id, &season_id)| {
                let def = definition(season_map, season_id);
                let last_month = def.month_end.unwrap_or(def.month_start);
                let end_date = if last_month == 12 {
                    date(2051, 1, 1)
                } else {
                    date(2050, last_month + 1, 1)
                };
                make_stage(StageSpec {
                    id: stage_id,
                    start_date: date(2050, def.month_start, def.day_start.unwrap_or(1)),
                    end_date,
                    season_id: Some(season_id),
                    ..StageSpec::default()
                })
            })
            .collect()
    }

    /// `n_years` of history from January 1990, one observation per hydro dated
    /// at each season's start: a periodic AR(2) chain stepped in calendar order,
    /// whose noise is shared in part across hydros. Depends only on calendar
    /// order, so a map and its twin get the same history.
    pub(crate) fn calendar_history(
        season_map: &SeasonMap,
        calendar_ids: &[usize],
        hydro_ids: &[EntityId],
        n_years: i32,
    ) -> Vec<(EntityId, NaiveDate, f64)> {
        let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
        let mut uniform = move || {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(lcg >> 32).unwrap_or(u32::MAX)) / f64::from(u32::MAX) - 0.5
        };
        let starts: Vec<(u32, u32)> = calendar_ids
            .iter()
            .map(|&id| {
                let def = definition(season_map, id);
                (def.month_start, def.day_start.unwrap_or(1))
            })
            .collect();
        let mut chains = vec![[0.0_f64; 2]; hydro_ids.len()];
        let mut history = Vec::new();
        for year in 0..n_years {
            for (position, &(month, day)) in (0_u32..).zip(&starts) {
                let shared = uniform();
                for ((chain, &hydro_id), scale) in chains.iter_mut().zip(hydro_ids).zip(1_u32..) {
                    let noise = 0.6 * shared + 0.8 * uniform();
                    let z = (0.3 + 0.04 * f64::from(position)) * chain[0] + 0.45 * chain[1] + noise;
                    *chain = [z, chain[0]];
                    let value = 100.0 * f64::from(scale)
                        + 20.0 * f64::from(position)
                        + (8.0 + 2.0 * f64::from(position)) * z;
                    history.push((hydro_id, date(1990 + year, month, day), value));
                }
            }
        }
        history.sort_by_key(|&(hydro_id, date, _)| (hydro_id, date));
        history
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::CyclePositions;
    use super::twin_fixtures::permuted_quarterly_season_map;
    use crate::test_support::{
        MonthlyLabels, monthly_quarterly_season_map, monthly_season_map, quarterly_season_map,
        sparse_ring_season_map, weekly_season_map,
    };

    #[test]
    fn cycle_positions_relabel_only_maps_whose_ids_are_not_their_calendar_positions() {
        for season_map in [
            monthly_season_map(MonthlyLabels::ZeroBased),
            weekly_season_map(),
            quarterly_season_map(),
            monthly_quarterly_season_map(),
        ] {
            assert!(CyclePositions::new(Some(&season_map)).is_none());
        }
        assert!(CyclePositions::new(None).is_none());

        for (season_map, positions) in [
            (
                sparse_ring_season_map(),
                vec![(0, 0), (1, 1), (2, 2), (12, 3), (13, 4)],
            ),
            (
                permuted_quarterly_season_map(),
                vec![(2, 0), (0, 1), (3, 2), (1, 3)],
            ),
        ] {
            let cycle_positions = CyclePositions::new(Some(&season_map)).unwrap();
            for &(raw, position) in &positions {
                assert_eq!(cycle_positions.position(raw), Some(position), "raw {raw}");
                assert_eq!(
                    cycle_positions.raw(position),
                    Some(raw),
                    "position {position}"
                );
            }
            assert_eq!(cycle_positions.raw(positions.len()), None);

            let relabeled = cycle_positions.relabel_season_map(&season_map);
            assert_eq!(relabeled.cycle_type, season_map.cycle_type);
            for (position, def) in relabeled.seasons.iter().enumerate() {
                assert_eq!(def.id, position);
                let raw = cycle_positions.raw(position).unwrap();
                let original = season_map.seasons.iter().find(|d| d.id == raw).unwrap();
                assert_eq!(def.month_start, original.month_start);
                assert_eq!(def.label, original.label);
            }
        }
        assert_eq!(
            CyclePositions::new(Some(&sparse_ring_season_map()))
                .unwrap()
                .position(3),
            None
        );
    }
}
