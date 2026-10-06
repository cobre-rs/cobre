//! Test-only fixture builders for this crate's `OpeningTree`, `SeasonMap` and
//! `InflowModel` shapes, shared across this crate's unit and integration
//! tests and reachable from downstream crates via the `test-support`
//! feature. Compiles only under `cfg(test)` or that feature; must never be
//! enabled in a production build.

use cobre_core::scenario::AnnualComponent;
use cobre_core::{EntityId, InflowModel, SeasonCycleType, SeasonDefinition, SeasonMap};

use crate::OpeningTree;

/// A uniformly-valued [`OpeningTree`] with `n_stages` stages, each holding
/// `openings` openings of dimension `dim`: values `0.0, 1.0, 2.0, ..` filled
/// row-major over (stage, opening, dim).
///
/// # Panics
///
/// Never for the stage/opening/dim triples the fixtures below pass.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: every call site below passes a triple whose product fits in
// `u32`, so `u32::try_from` cannot return `Err`.
pub fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
    let total = n_stages * openings * dim;
    let data: Vec<f64> = (0_u32..u32::try_from(total).expect("total fits in u32"))
        .map(f64::from)
        .collect();
    OpeningTree::from_parts(data, vec![openings; n_stages], dim)
}

/// Which of the three live monthly-season label spellings
/// [`monthly_season_map`] produces.
#[derive(Debug, Clone, Copy)]
pub enum MonthlyLabels {
    /// `Month1`..`Month12`.
    OneBased,
    /// `Month0`..`Month11`.
    ZeroBased,
    /// `Month01`..`Month12`.
    ZeroPadded,
}

/// A monthly [`SeasonMap`] (12 seasons, `id` 0..11, `month_start` 1..12,
/// [`SeasonCycleType::Monthly`]), labelled per `labels`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
// Rationale: `i` is bounded to 0..12 by the loop range, so the cast to
// `u32` cannot truncate.
pub fn monthly_season_map(labels: MonthlyLabels) -> SeasonMap {
    let seasons: Vec<SeasonDefinition> = (0_usize..12)
        .map(|i| SeasonDefinition {
            id: i,
            label: match labels {
                MonthlyLabels::OneBased => format!("Month{}", i + 1),
                MonthlyLabels::ZeroBased => format!("Month{i}"),
                MonthlyLabels::ZeroPadded => format!("Month{:02}", i + 1),
            },
            month_start: (i as u32) + 1,
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons,
    }
}

/// A quarterly [`SeasonMap`] (`Q1`..`Q4`, [`SeasonCycleType::Custom`]).
#[must_use]
pub fn quarterly_season_map() -> SeasonMap {
    SeasonMap {
        cycle_type: SeasonCycleType::Custom,
        seasons: vec![
            SeasonDefinition {
                id: 0,
                label: "Q1".to_string(),
                month_start: 1,
                day_start: Some(1),
                month_end: Some(3),
                day_end: Some(31),
            },
            SeasonDefinition {
                id: 1,
                label: "Q2".to_string(),
                month_start: 4,
                day_start: Some(1),
                month_end: Some(6),
                day_end: Some(30),
            },
            SeasonDefinition {
                id: 2,
                label: "Q3".to_string(),
                month_start: 7,
                day_start: Some(1),
                month_end: Some(9),
                day_end: Some(30),
            },
            SeasonDefinition {
                id: 3,
                label: "Q4".to_string(),
                month_start: 10,
                day_start: Some(1),
                month_end: Some(12),
                day_end: Some(31),
            },
        ],
    }
}

/// A sparse-id ring [`SeasonMap`] (`Custom`): ids `0` Jan, `1` Feb, `2` Mar,
/// `12` Apr-Jun, `13` Jul-Sep.
#[must_use]
pub fn sparse_ring_season_map() -> SeasonMap {
    let def = |id: usize, month_start: u32, month_end: Option<u32>| SeasonDefinition {
        id,
        label: format!("S{id}"),
        month_start,
        day_start: None,
        month_end,
        day_end: None,
    };
    SeasonMap {
        cycle_type: SeasonCycleType::Custom,
        seasons: vec![
            def(0, 1, None),
            def(1, 2, None),
            def(2, 3, None),
            def(12, 4, Some(6)),
            def(13, 7, Some(9)),
        ],
    }
}

/// D30's layered `Custom` [`SeasonMap`]: months 0–11, then Q3 = 12, Q4 = 13, Q1 = 14, Q2 = 15.
#[must_use]
pub fn monthly_quarterly_season_map() -> SeasonMap {
    const MONTHS: [(&str, u32); 12] = [
        ("January", 31),
        ("February", 28),
        ("March", 31),
        ("April", 30),
        ("May", 31),
        ("June", 30),
        ("July", 31),
        ("August", 31),
        ("September", 30),
        ("October", 31),
        ("November", 30),
        ("December", 31),
    ];
    let def =
        |id: usize, label: &str, month_start: u32, month_end: u32, day_end: u32| SeasonDefinition {
            id,
            label: label.to_string(),
            month_start,
            day_start: Some(1),
            month_end: Some(month_end),
            day_end: Some(day_end),
        };
    let months = MONTHS
        .into_iter()
        .zip(1_u32..)
        .enumerate()
        .map(|(id, ((label, day_end), month))| def(id, label, month, month, day_end));
    let quarters = [
        def(12, "Q3", 7, 9, 30),
        def(13, "Q4", 10, 12, 31),
        def(14, "Q1", 1, 3, 31),
        def(15, "Q2", 4, 6, 30),
    ];
    SeasonMap {
        cycle_type: SeasonCycleType::Custom,
        seasons: months.chain(quarters).collect(),
    }
}

/// A weekly [`SeasonMap`] (52 seasons, `Week1`..`Week52`,
/// [`SeasonCycleType::Weekly`]).
#[must_use]
pub fn weekly_season_map() -> SeasonMap {
    let seasons: Vec<SeasonDefinition> = (0..52u32)
        .map(|i| SeasonDefinition {
            id: i as usize,
            label: format!("Week{}", i + 1),
            month_start: 1,
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: SeasonCycleType::Weekly,
        seasons,
    }
}

/// Fixture fields for [`make_inflow_model`].
#[derive(Debug, Clone)]
pub struct InflowModelSpec {
    /// Hydro plant this model belongs to.
    pub hydro_id: i32,
    /// Declared study-stage id this model applies to.
    pub stage_id: i32,
    /// Seasonal mean inflow in m³/s.
    pub mean_m3s: f64,
    /// Seasonal standard deviation in m³/s.
    pub std_m3s: f64,
    /// AR lag coefficients, standardized by seasonal std.
    pub ar_coefficients: Vec<f64>,
    /// Ratio of residual standard deviation to seasonal standard deviation.
    pub residual_std_ratio: f64,
    /// Optional annual component; `None` selects the classical PAR(p) model.
    pub annual: Option<AnnualComponent>,
}

impl Default for InflowModelSpec {
    fn default() -> Self {
        Self {
            hydro_id: 0,
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: Vec::new(),
            residual_std_ratio: 1.0,
            annual: None,
        }
    }
}

/// Build an [`InflowModel`] from `spec`.
#[must_use]
pub fn make_inflow_model(
    InflowModelSpec {
        hydro_id,
        stage_id,
        mean_m3s,
        std_m3s,
        ar_coefficients,
        residual_std_ratio,
        annual,
    }: InflowModelSpec,
) -> InflowModel {
    InflowModel {
        hydro_id: EntityId(hydro_id),
        stage_id,
        mean_m3s,
        std_m3s,
        ar_coefficients,
        residual_std_ratio,
        annual,
    }
}
