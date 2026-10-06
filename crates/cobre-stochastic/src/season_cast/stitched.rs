//! Stage-id to season-id lookup that stitches the study stages to the
//! pre-study lags recorded by the backward season walk.

use cobre_core::temporal::{SeasonCycles, SeasonMap, Stage};

use super::{previous_occurrence, season_period_window};

/// Season of every study stage and of every lag that precedes the first one.
///
/// Lag `k` of the first study stage has stage id `first_id - k`. Lag seasons
/// are the ids the backward season walk records for the `k`-th previous
/// occurrence, so they follow the calendar even when season ids are sparse or
/// a weekly year has 53 weeks.
///
/// Declared pre-study stages (negative ids) are not consulted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StitchedSeasonMap {
    anchor_id: i32,
    study: Vec<(i32, usize)>,
    lags: Vec<usize>,
}

impl StitchedSeasonMap {
    /// Build the lookup from the study `stages` (negative ids are ignored),
    /// walking back at most `max_lag` steps from the lowest-id stage that
    /// carries a season.
    ///
    /// The walk stops early where a step cannot resolve; later lags are then
    /// absent.
    #[must_use]
    pub fn build(stages: &[Stage], season_map: &SeasonMap, max_lag: usize) -> Self {
        let mut study: Vec<(i32, usize)> = stages
            .iter()
            .filter(|s| s.id >= 0)
            .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
            .collect();
        study.sort_unstable_by_key(|&(id, _)| id);

        let anchor = stages
            .iter()
            .filter(|s| s.id >= 0 && s.season_id.is_some())
            .min_by_key(|s| s.id);

        let mut lags = Vec::new();
        if let Some(anchor) = anchor
            && let Some(sid) = anchor.season_id
            && let Some(def) = season_map.seasons.iter().find(|d| d.id == sid)
        {
            let cycles = SeasonCycles::new(season_map);
            let mut window = season_period_window(season_map, def, anchor);
            let mut current = sid;
            for k in 1..=max_lag {
                if i32::try_from(k)
                    .ok()
                    .and_then(|k| anchor.id.checked_sub(k))
                    .is_none()
                {
                    break;
                }
                let Some((id, previous)) =
                    previous_occurrence(season_map, &cycles, current, &window)
                else {
                    break;
                };
                lags.push(id);
                current = id;
                window = previous;
            }
        }

        Self {
            anchor_id: anchor.map_or(0, |a| a.id),
            study,
            lags,
        }
    }

    /// Season id of `stage_id`: the walk's lag season for an id below the first
    /// study stage, the declared season for a study id, `None` otherwise.
    #[must_use]
    pub fn season_of(&self, stage_id: i32) -> Option<usize> {
        if stage_id < self.anchor_id {
            let index = self.anchor_id.checked_sub(stage_id)?.checked_sub(1)?;
            return self.lags.get(usize::try_from(index).ok()?).copied();
        }
        self.study
            .binary_search_by_key(&stage_id, |&(id, _)| id)
            .ok()
            .map(|i| self.study[i].1)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonMap, Stage, StageRiskConfig,
        StageStateConfig,
    };

    use super::StitchedSeasonMap;
    use crate::season_cast::{nth_previous_occurrence, season_period_window};
    use crate::test_support::{
        MonthlyLabels, monthly_quarterly_season_map, monthly_season_map, sparse_ring_season_map,
        weekly_season_map,
    };

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn study_stage(id: i32, start: NaiveDate, end: NaiveDate, season: Option<usize>) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: start,
            end_date: end,
            season_id: season,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: f64::from(u32::try_from((end - start).num_days()).unwrap()) * 24.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn month_stage(id: i32, year: i32, month: u32, season: Option<usize>) -> Stage {
        let end = if month == 12 {
            date(year + 1, 1, 1)
        } else {
            date(year, month + 1, 1)
        };
        study_stage(id, date(year, month, 1), end, season)
    }

    fn walk_oracle(season_map: &SeasonMap, anchor: &Stage, k: usize) -> Option<usize> {
        let def = season_map
            .seasons
            .iter()
            .find(|d| Some(d.id) == anchor.season_id)?;
        let in_progress = season_period_window(season_map, def, anchor);
        let occurrence = nth_previous_occurrence(season_map, def, &in_progress, k)?;
        season_map.season_for_date(occurrence.start)
    }

    fn lag_window(season_map: &SeasonMap, anchor: &Stage, k: usize) -> (NaiveDate, NaiveDate) {
        let def = season_map
            .seasons
            .iter()
            .find(|d| Some(d.id) == anchor.season_id)
            .unwrap();
        let in_progress = season_period_window(season_map, def, anchor);
        let occurrence = nth_previous_occurrence(season_map, def, &in_progress, k).unwrap();
        (occurrence.start, occurrence.end)
    }

    #[test]
    fn stitched_map_matches_the_calendar_walk_on_a_uniform_monthly_map() {
        let season_map = monthly_season_map(MonthlyLabels::ZeroBased);
        let stages = [
            month_stage(0, 2026, 3, Some(2)),
            month_stage(1, 2026, 4, Some(3)),
            month_stage(2, 2026, 5, Some(4)),
        ];

        let stitched = StitchedSeasonMap::build(&stages, &season_map, 24);

        for (id, season) in [(0, 2), (1, 3), (2, 4)] {
            assert_eq!(stitched.season_of(id), Some(season));
        }
        for k in 1..=24_i32 {
            let modulo = usize::try_from((2 - k).rem_euclid(12)).unwrap();
            assert_eq!(stitched.season_of(-k), Some(modulo), "lag {k}");
            let walk = walk_oracle(&season_map, &stages[0], usize::try_from(k).unwrap());
            assert_eq!(stitched.season_of(-k), walk, "lag {k}");
        }
    }

    #[test]
    fn stitched_map_folds_iso_week_53_into_season_51_on_a_weekly_map() {
        let season_map = weekly_season_map();
        let stages = [
            study_stage(0, date(2021, 1, 4), date(2021, 1, 11), Some(0)),
            study_stage(1, date(2021, 1, 11), date(2021, 1, 18), Some(1)),
        ];

        let stitched = StitchedSeasonMap::build(&stages, &season_map, 3);

        let lags = [-1, -2, -3].map(|id| stitched.season_of(id));
        assert_eq!(lags, [Some(51), Some(51), Some(50)]);
        for (k, lag) in (1..=3).zip(lags) {
            assert_eq!(lag, walk_oracle(&season_map, &stages[0], k), "lag {k}");
        }
        let modulo = [1_i32, 2, 3].map(|k| Some(usize::try_from((-k).rem_euclid(52)).unwrap()));
        assert_eq!(modulo, [Some(51), Some(50), Some(49)]);
        assert_ne!(lags, modulo);
    }

    #[test]
    fn stitched_map_walks_ring_predecessors_on_a_sparse_custom_map() {
        let season_map = sparse_ring_season_map();
        let stages = [
            month_stage(0, 2026, 1, Some(0)),
            month_stage(1, 2026, 2, Some(1)),
            month_stage(2, 2026, 3, Some(2)),
        ];

        let stitched = StitchedSeasonMap::build(&stages, &season_map, 6);

        let lags = [-1, -2, -3, -4, -5, -6].map(|id| stitched.season_of(id));
        assert_eq!(
            lags,
            [Some(13), Some(12), Some(2), Some(1), Some(0), Some(13)]
        );
        for (k, lag) in (1..=6).zip(lags) {
            assert_eq!(lag, walk_oracle(&season_map, &stages[0], k), "lag {k}");
        }
        let modulo =
            [1_i32, 2, 3, 4, 5, 6].map(|k| Some(usize::try_from((-k).rem_euclid(5)).unwrap()));
        assert_ne!(lags, modulo);
    }

    #[test]
    fn stitched_map_steps_a_monthly_anchor_back_through_months_on_a_layered_map() {
        let season_map = monthly_quarterly_season_map();
        let stages = [month_stage(0, 2024, 1, Some(0))];

        let stitched = StitchedSeasonMap::build(&stages, &season_map, 4);

        assert_eq!(
            [-1, -2, -3, -4].map(|id| stitched.season_of(id)),
            [Some(11), Some(10), Some(9), Some(8)]
        );
        assert_eq!(
            lag_window(&season_map, &stages[0], 1),
            (date(2023, 12, 1), date(2024, 1, 1))
        );
    }

    #[test]
    fn stitched_map_steps_a_quarterly_anchor_along_the_quarterly_cycle() {
        let season_map = monthly_quarterly_season_map();
        let stages = [study_stage(
            0,
            date(2024, 7, 1),
            date(2024, 10, 1),
            Some(12),
        )];

        let stitched = StitchedSeasonMap::build(&stages, &season_map, 4);

        assert_eq!(
            [-1, -2, -3, -4].map(|id| stitched.season_of(id)),
            [Some(15), Some(14), Some(13), Some(12)]
        );
        assert_eq!(
            lag_window(&season_map, &stages[0], 1),
            (date(2024, 4, 1), date(2024, 7, 1))
        );
        assert_eq!(
            lag_window(&season_map, &stages[0], 4),
            (date(2023, 7, 1), date(2023, 10, 1))
        );
    }

    #[test]
    fn stitched_map_ignores_declared_pre_study_stages_and_unresolvable_anchors() {
        let season_map = monthly_season_map(MonthlyLabels::ZeroBased);
        let stages = [
            month_stage(-1, 2025, 12, Some(7)),
            month_stage(0, 2026, 1, Some(0)),
            month_stage(2, 2026, 3, Some(2)),
        ];

        let stitched = StitchedSeasonMap::build(&stages, &season_map, 3);

        assert_eq!(stitched.season_of(-1), Some(11));
        assert_eq!(stitched.season_of(-3), Some(9));
        assert_eq!(stitched.season_of(0), Some(0));
        assert_eq!(stitched.season_of(2), Some(2));
        assert_eq!(stitched.season_of(1), None);
        assert_eq!(stitched.season_of(3), None);
        assert_eq!(stitched.season_of(-4), None);

        let unresolvable =
            StitchedSeasonMap::build(&[month_stage(0, 2026, 1, Some(99))], &season_map, 3);
        assert_eq!(unresolvable.season_of(-1), None);
        assert_eq!(unresolvable.season_of(0), Some(99));

        let empty = StitchedSeasonMap::build(&[], &season_map, 3);
        for id in [-2, -1, 0, 1] {
            assert_eq!(empty.season_of(id), None);
        }
    }
}
