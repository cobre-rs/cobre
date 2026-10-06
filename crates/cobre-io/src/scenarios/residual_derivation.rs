//! Post-assembly derivation of `residual_std_ratio` via the periodic-ACF closure.
//!
//! [`populate_derived_residual_ratios`] runs immediately after
//! [`crate::scenarios::assembly::assemble_inflow_models`] at all four production
//! call sites (`pipeline.rs`, and `scenarios::estimation`'s `run_estimation`,
//! `run_partial_estimation`, `run_user_ar_estimation`). The standardized AR
//! coefficients (`ψ*`) plus the unit-marginal-variance contract pin every
//! residual std ratio `r_m` (see `cobre_stochastic::par::closure` module docs); this
//! function overwrites each [`InflowModel::residual_std_ratio`] with that
//! closure-derived value, making the closure authoritative for both file-loaded and
//! internally-fitted models (assembly writes only a placeholder into the field).
//!
//! [`resolve_stage_seasons`] builds the `(stage_to_season, n_seasons)` pair every
//! call site needs, shared so this crate's several season-indexed consumers (this
//! module's four wirings, plus `validation::semantic::scenarios`'s stationarity
//! gate) don't each re-derive it — and, more importantly, so a season-density fix
//! lands once for all of them.

use std::collections::{BTreeMap, HashMap, HashSet};

use cobre_core::EntityId;
use cobre_core::scenario::InflowModel;
use cobre_core::temporal::{SeasonCycles, SeasonMap, Stage};
use cobre_stochastic::par::{
    AnnualParams, derive_residual_std_ratios, derive_residual_std_ratios_annual,
};

use crate::LoadError;

/// Overwrites every model's `residual_std_ratio` with the closure-derived value for
/// its `(hydro_id, season)`.
///
/// Groups `models` by `hydro_id` (ascending), then within each hydro by season (via
/// `stage_to_season`), taking one representative model per season for its
/// `ar_coefficients` (`ψ*`), AR order, `std_m3s` (`s_m`), and `annual` component —
/// stages sharing a season carry identical `ψ*`, so any representative is
/// equivalent. Calls [`derive_residual_std_ratios_annual`] when any season of the
/// hydro carries an annual component, else [`derive_residual_std_ratios`], then
/// writes the resulting `r[season]` back onto every model of that hydro.
///
/// A model whose `ar_order() == 0` and `annual.is_none()` but whose stage has no
/// entry in `stage_to_season` is set to `residual_std_ratio = 1.0` directly (the
/// closure's own order-0 short-circuit) without needing a season.
///
/// # Errors
///
/// - [`LoadError::ConstraintError`] — a model has `ar_order() > 0` or a `Some`
///   annual component but its stage has no entry in `stage_to_season`.
/// - [`LoadError::ConstraintError`] — the closure system is singular for a hydro
///   (`derive_residual_std_ratios`/`_annual` returned `None`).
/// - [`LoadError::ConstraintError`] — a derived ratio is non-finite: the
///   (effective) coefficients imply a non-stationary process. User-supplied
///   coefficients are rejected earlier by the semantic stationarity gate; this
///   guard is what covers the internally-estimated paths, which reach the
///   closure without that gate.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use cobre_core::EntityId;
/// use cobre_core::scenario::InflowModel;
/// use cobre_io::scenarios::residual_derivation::populate_derived_residual_ratios;
///
/// let mut models = vec![InflowModel {
///     hydro_id: EntityId(1),
///     stage_id: 0,
///     mean_m3s: 100.0,
///     std_m3s: 20.0,
///     ar_coefficients: vec![0.5],
///     residual_std_ratio: 0.42, // placeholder; overwritten by the derivation
///     annual: None,
/// }];
/// let stage_to_season: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
///
/// populate_derived_residual_ratios(&mut models, &stage_to_season, 1).unwrap();
///
/// let expected = (1.0_f64 - 0.25).sqrt();
/// assert!((models[0].residual_std_ratio - expected).abs() < 1e-12);
/// ```
#[allow(clippy::implicit_hasher)]
pub fn populate_derived_residual_ratios(
    models: &mut [InflowModel],
    stage_to_season: &HashMap<i32, usize>,
    n_seasons: usize,
) -> Result<(), LoadError> {
    let mut by_hydro: BTreeMap<EntityId, Vec<usize>> = BTreeMap::new();
    for (idx, model) in models.iter().enumerate() {
        by_hydro.entry(model.hydro_id).or_default().push(idx);
    }

    for (hydro_id, indices) in by_hydro {
        let mut orders = vec![0_usize; n_seasons];
        let mut psi_by_season = vec![Vec::new(); n_seasons];
        let mut seasonal_std = vec![0.0_f64; n_seasons];
        let mut annual: Vec<Option<AnnualParams>> = vec![None; n_seasons];
        let mut season_seen = vec![false; n_seasons];

        for &idx in &indices {
            let model = &models[idx];
            match stage_to_season.get(&model.stage_id) {
                Some(&season) => {
                    if !season_seen[season] {
                        season_seen[season] = true;
                        orders[season] = model.ar_order();
                        psi_by_season[season].clone_from(&model.ar_coefficients);
                        seasonal_std[season] = model.std_m3s;
                        annual[season] = model.annual.as_ref().map(|component| AnnualParams {
                            coefficient: component.coefficient,
                            sigma_a: component.std_m3s,
                        });
                    }
                }
                None if model.ar_order() > 0 || model.annual.is_some() => {
                    return Err(LoadError::ConstraintError {
                        description: format!(
                            "hydro_id={hydro_id} stage_id={} has AR order > 0 or an \
                             annual component but no resolvable season; cannot \
                             derive residual_std_ratio",
                            model.stage_id
                        ),
                    });
                }
                None => {}
            }
        }

        let has_annual = annual.iter().any(Option::is_some);
        let derived = if has_annual {
            derive_residual_std_ratios_annual(
                &psi_by_season,
                &orders,
                &annual,
                &seasonal_std,
                n_seasons,
            )
        } else {
            derive_residual_std_ratios(&psi_by_season, &orders, n_seasons)
        }
        .ok_or_else(|| LoadError::ConstraintError {
            description: format!("residual_std_ratio closure is singular for hydro_id={hydro_id}"),
        })?;

        if let Some(season) = derived.iter().position(|r| !r.is_finite()) {
            return Err(LoadError::ConstraintError {
                description: format!(
                    "derived residual_std_ratio is non-finite for hydro_id={hydro_id} \
                     season={season}: the coefficients imply a non-stationary process \
                     (implied residual variance 1 - sum(psi*rho) is negative)"
                ),
            });
        }

        for &idx in &indices {
            let model = &mut models[idx];
            model.residual_std_ratio = match stage_to_season.get(&model.stage_id) {
                Some(&season) => derived[season],
                None => 1.0,
            };
        }
    }

    Ok(())
}

/// Builds the `(stage_to_season, n_seasons)` pair every
/// [`populate_derived_residual_ratios`] call site needs from its local `stages`
/// slice.
///
/// `stage_to_season`'s values are **dense** 0-based ordinals — the rank of each
/// raw `season_id` among the distinct ids — never the raw `season_id` itself.
/// This is required because [`populate_derived_residual_ratios`]
/// (and every other caller of this map) indexes fixed-size `Vec`s of length
/// `n_seasons` by season; a season-definitions cycle with sparse or
/// non-contiguous ids (e.g. `Weekly` season ids `21`/`26` for a 2-season cycle)
/// would otherwise produce an out-of-bounds raw id used as an array index. This
/// mirrors `PrecomputedPar::build`'s `stage_to_season` *shape* (a
/// `HashMap<i32, usize>` keyed by `stage.id`), but not its raw values —
/// `precompute.rs`'s own arrays are indexed by stage position, not by season, so
/// it never needed densification; every caller of this shared helper does.
///
/// Prefers the registered `season_map`'s declared ids for both the ordinal
/// assignment and `n_seasons`. A single-resolution map ranks them by calendar
/// order ([`SeasonCycles::position`]), so the closure's cyclic predecessor of
/// an ordinal is its calendar predecessor; a multi-resolution map ranks them by
/// ascending id. Falls back to the distinct `season_id`s actually used by
/// `stages`, dense-indexed in ascending order, when no `season_map` is
/// registered — a `System` may carry per-stage `season_id`s without one.
///
/// A stage whose `season_id` does not appear among the resolved ids (a dangling
/// reference to an undeclared season — rejected upstream by semantic validation
/// in the production pipeline) is omitted from `stage_to_season`, surfacing as
/// an unresolvable season to callers rather than panicking.
#[must_use]
pub fn resolve_stage_seasons(
    stages: &[Stage],
    season_map: Option<&SeasonMap>,
) -> (HashMap<i32, usize>, usize) {
    let (dense_index, n_seasons) = season_dense_index(stages, season_map);

    let stage_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| {
            let raw = s.season_id?;
            dense_index.get(&raw).map(|&idx| (s.id, idx))
        })
        .collect();

    (stage_to_season, n_seasons)
}

/// The raw-season-id-to-dense-ordinal ranking `resolve_stage_seasons` assigns,
/// factored out so [`crate::scenarios::estimation::resolve_model_stage_seasons`]
/// can rank a synthesized season id the same way without re-deriving stages'
/// season coverage.
pub(crate) fn season_dense_index(
    stages: &[Stage],
    season_map: Option<&SeasonMap>,
) -> (HashMap<usize, usize>, usize) {
    let mut raw_ids: Vec<usize> = season_map.map_or_else(
        || {
            stages
                .iter()
                .filter_map(|s| s.season_id)
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        },
        |sm| sm.seasons.iter().map(|s| s.id).collect(),
    );
    raw_ids.sort_unstable();
    raw_ids.dedup();
    if let Some(cycles) = season_map.map(SeasonCycles::new)
        && raw_ids
            .iter()
            .all(|&raw| matches!(cycles.position(raw), Some((0, _))))
    {
        raw_ids.sort_by_key(|&raw| cycles.position(raw));
    }
    let n_seasons = raw_ids.len();

    let dense_index = raw_ids
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| (raw, idx))
        .collect();

    (dense_index, n_seasons)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use cobre_core::scenario::AnnualComponent;

    fn model(
        hydro_id: i32,
        stage_id: i32,
        ar_coefficients: Vec<f64>,
        residual_std_ratio: f64,
        std_m3s: f64,
        annual: Option<AnnualComponent>,
    ) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s,
            ar_coefficients,
            residual_std_ratio,
            annual,
        }
    }

    /// Minimal `Stage` carrying only `id`/`season_id` as load-bearing fields —
    /// mirrors `closure.rs`'s `par_a_stage` fixture construction.
    fn stage_with_season(index: usize, id: i32, season_id: usize) -> Stage {
        use chrono::NaiveDate;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };

        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// A 2-season `Weekly` `SeasonMap` with sparse, non-contiguous ids `21`/`26`
    /// — reproduces a downstream solver-integration fixture whose `Weekly`
    /// season definitions covered only weeks 21/26, which exposed the
    /// raw-season-id-as-array-index panic.
    fn sparse_two_season_map() -> SeasonMap {
        use cobre_core::temporal::{SeasonCycleType, SeasonDefinition};

        SeasonMap {
            cycle_type: SeasonCycleType::Weekly,
            seasons: vec![
                SeasonDefinition {
                    id: 21,
                    label: "W22".to_string(),
                    month_start: 1,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                },
                SeasonDefinition {
                    id: 26,
                    label: "W27".to_string(),
                    month_start: 1,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                },
            ],
        }
    }

    /// Uniform AR(1) order across every season decouples the periodic closure
    /// (each season's `implied_periodic_acf` row only ever hits the `j == kp`
    /// branch), so `r_m = sqrt(1 - psi_m^2)` exactly — an analytically known
    /// closure value, giving an exact (not approximate) "previously stored"
    /// oracle to compare against. Two stages share season 0 to also exercise
    /// "every stage of a hydro gets its season's r".
    #[test]
    fn derived_ratio_uniform_matches_stored() {
        let psi = [0.5_f64, 0.3_f64];
        let stored: Vec<f64> = psi.iter().map(|&p| (1.0 - p * p).sqrt()).collect();

        let mut models = vec![
            model(1, 0, vec![psi[0]], stored[0], 20.0, None),
            model(1, 1, vec![psi[1]], stored[1], 22.0, None),
            model(1, 2, vec![psi[0]], stored[0], 20.0, None),
        ];
        let stage_to_season: HashMap<i32, usize> = [(0, 0), (1, 1), (2, 0)].into_iter().collect();

        populate_derived_residual_ratios(&mut models, &stage_to_season, 2).unwrap();

        for (m, expected) in [(0_usize, stored[0]), (1, stored[1]), (2, stored[0])] {
            let gap = (models[m].residual_std_ratio - expected).abs();
            assert!(
                gap < 1e-12,
                "model[{m}]: derived={}, stored={expected}, gap={gap:e}",
                models[m].residual_std_ratio
            );
        }
    }

    /// Reproduces the `t2_mixed_orders_gate_passes` fixture bit-for-bit
    /// (`orders = [3, 1, 2, 1]` against the periodic Yule-Walker fit of a
    /// plausible sample ACF table) — the closure-derived `r` and the YW-fitted
    /// `r` are known (from that test) to differ only in season 0 (the
    /// max-order season), by a gap in `[1e-5, 1e-3)`, and to agree elsewhere to
    /// better than `1e-9`. Here the YW-fitted values stand in for the
    /// "previously stored" column value that assembly used to trust.
    #[test]
    fn derived_ratio_mixed_order_shifts() {
        let psi: [Vec<f64>; 4] = [
            vec![
                0.398_915_659_532_620_8,
                0.062_802_463_704_355_48,
                -0.014_634_714_422_504_587,
            ],
            vec![0.35],
            vec![0.294_017_094_017_094, 0.017_094_017_094_017_092],
            vec![0.38],
        ];
        let stored = [
            0.905_780_151_456_293,
            0.936_749_699_759_759_7,
            0.953_804_796_456_586_1,
            0.924_986_486_387_774_3,
        ];

        let mut models: Vec<InflowModel> = (0..4_usize)
            .map(|season| {
                model(
                    1,
                    i32::try_from(season).unwrap(),
                    psi[season].clone(),
                    stored[season],
                    20.0,
                    None,
                )
            })
            .collect();
        let stage_to_season: HashMap<i32, usize> = (0..4_usize)
            .map(|s| (i32::try_from(s).unwrap(), s))
            .collect();

        populate_derived_residual_ratios(&mut models, &stage_to_season, 4).unwrap();

        let gap0 = (models[0].residual_std_ratio - stored[0]).abs();
        assert!(
            (1e-5..1e-3).contains(&gap0),
            "season 0: expected a ~1e-4-scale gap from the stored value, got {gap0:e}"
        );
        for season in 1..4 {
            let gap = (models[season].residual_std_ratio - stored[season]).abs();
            assert!(
                gap < 1e-9,
                "season {season}: expected the derived value to match stored, got gap {gap:e}"
            );
        }
    }

    /// The populated `r` for a PAR-A hydro must equal
    /// `derive_residual_std_ratios_annual` called with the same per-season
    /// inputs the function assembles internally — pinning that
    /// `populate_derived_residual_ratios` routes to the annual closure the
    /// moment any season carries an annual component.
    #[test]
    fn derived_ratio_par_a_uses_annual_closure() {
        let psi_by_season: Vec<Vec<f64>> = vec![vec![0.3], vec![], vec![0.2], vec![]];
        let orders = vec![1, 0, 1, 0];
        let seasonal_std = vec![20.0, 18.0, 22.0, 19.0];
        let annual_params: Vec<Option<AnnualParams>> = vec![
            Some(AnnualParams {
                coefficient: 0.4,
                sigma_a: 15.0,
            }),
            None,
            None,
            None,
        ];

        let expected = derive_residual_std_ratios_annual(
            &psi_by_season,
            &orders,
            &annual_params,
            &seasonal_std,
            4,
        )
        .expect("annual closure solves for this fixture");

        let mut models = vec![
            model(
                1,
                0,
                vec![0.3],
                0.99,
                20.0,
                Some(AnnualComponent {
                    coefficient: 0.4,
                    mean_m3s: 100.0,
                    std_m3s: 15.0,
                }),
            ),
            model(1, 1, vec![], 1.0, 18.0, None),
            model(1, 2, vec![0.2], 0.98, 22.0, None),
            model(1, 3, vec![], 1.0, 19.0, None),
        ];
        let stage_to_season: HashMap<i32, usize> = (0..4_usize)
            .map(|s| (i32::try_from(s).unwrap(), s))
            .collect();

        populate_derived_residual_ratios(&mut models, &stage_to_season, 4).unwrap();

        for (season, expected_r) in expected.iter().enumerate() {
            let gap = (models[season].residual_std_ratio - expected_r).abs();
            assert!(
                gap < 1e-12,
                "season {season}: populated={}, expected={expected_r}, gap={gap:e}",
                models[season].residual_std_ratio
            );
        }
    }

    /// A non-stationary coefficient set (uniform AR(1) with `|psi| > 1`, so the
    /// closure yields `1 - psi^2 < 0` and a NaN sqrt) must hard-error, never
    /// silently write NaN into `residual_std_ratio`. This is the guard for the
    /// internally-estimated paths, which reach the derivation without the
    /// semantic stationarity gate (that gate only sees user-supplied
    /// coefficient files).
    #[test]
    fn derived_ratio_non_stationary_errors_not_nan() {
        let mut models = vec![model(9, 0, vec![1.2], 0.9, 20.0, None)];
        let stage_to_season: HashMap<i32, usize> = [(0, 0)].into_iter().collect();

        let err = populate_derived_residual_ratios(&mut models, &stage_to_season, 1).unwrap_err();
        match err {
            LoadError::ConstraintError { description } => {
                assert!(
                    description.contains("non-finite") && description.contains("hydro_id=9"),
                    "error should name the failure class and the hydro, got: {description}"
                );
            }
            other => panic!("expected ConstraintError, got: {other:?}"),
        }
        assert!(
            (models[0].residual_std_ratio - 0.9).abs() < 1e-15,
            "the placeholder must be left untouched on error, got {}",
            models[0].residual_std_ratio
        );
    }

    /// An order-bearing model whose stage is absent from `stage_to_season`
    /// must error, naming the hydro — mirrors `precompute.rs`'s own
    /// "AR order > 0 requires a season_id" gate.
    #[test]
    fn derived_ratio_missing_season_errors() {
        let mut models = vec![model(7, 0, vec![0.5], 0.9, 20.0, None)];
        let stage_to_season: HashMap<i32, usize> = HashMap::new();

        let err = populate_derived_residual_ratios(&mut models, &stage_to_season, 1).unwrap_err();
        match err {
            LoadError::ConstraintError { description } => {
                assert!(
                    description.contains("hydro_id=7"),
                    "error should name the hydro, got: {description}"
                );
            }
            other => panic!("expected ConstraintError, got: {other:?}"),
        }
    }

    /// [`resolve_stage_seasons`] must densify sparse/non-contiguous raw
    /// `season_id`s (e.g. a `Weekly` cycle's `21`/`26`) to `0..n_seasons` —
    /// pins a downstream regression where a `Weekly`-cycle case with 2
    /// declared seasons (ids 21 and 26) panicked with "index out of bounds:
    /// the len is 2 but the index is 21" because the raw id was previously
    /// used as the array index directly. Both the `season_map`-present and the
    /// no-`season_map` fallback paths must densify identically.
    #[test]
    fn resolve_stage_seasons_densifies_sparse_ids() {
        let stages = vec![stage_with_season(0, 0, 21), stage_with_season(1, 1, 26)];
        let season_map = sparse_two_season_map();

        let (with_map, n_with_map) = resolve_stage_seasons(&stages, Some(&season_map));
        assert_eq!(
            n_with_map, 2,
            "n_seasons must equal the declared season count"
        );
        assert_eq!(
            with_map.get(&0),
            Some(&0),
            "raw id 21 must densify to index 0"
        );
        assert_eq!(
            with_map.get(&1),
            Some(&1),
            "raw id 26 must densify to index 1"
        );

        let (without_map, n_without_map) = resolve_stage_seasons(&stages, None);
        assert_eq!(
            n_without_map, 2,
            "no-season_map fallback must also count 2 distinct season_ids"
        );
        assert_eq!(
            without_map.get(&0),
            Some(&0),
            "no-season_map fallback must densify raw id 21 to index 0"
        );
        assert_eq!(
            without_map.get(&1),
            Some(&1),
            "no-season_map fallback must densify raw id 26 to index 1"
        );
    }

    /// End-to-end regression for the sparse-season-id panic:
    /// [`populate_derived_residual_ratios`] must succeed (not index-panic) on
    /// a hydro whose two stages carry sparse, non-contiguous season_ids, and
    /// must assign each stage its own season's closure-derived `r` — proven
    /// against the exact analytic AR(1) closure value (order-1 decouples
    /// per-season exactly, see `derived_ratio_uniform_matches_stored`), not
    /// the other season's value.
    #[test]
    fn derived_ratio_sparse_season_ids_regression() {
        let stages = vec![stage_with_season(0, 0, 21), stage_with_season(1, 1, 26)];
        let season_map = sparse_two_season_map();
        let (stage_to_season, n_seasons) = resolve_stage_seasons(&stages, Some(&season_map));

        let psi = [0.4_f64, 0.25_f64];
        let expected: Vec<f64> = psi.iter().map(|&p| (1.0 - p * p).sqrt()).collect();

        let mut models = vec![
            model(1, 0, vec![psi[0]], 0.5, 20.0, None),
            model(1, 1, vec![psi[1]], 0.5, 22.0, None),
        ];

        populate_derived_residual_ratios(&mut models, &stage_to_season, n_seasons)
            .expect("derivation must succeed for sparse season ids, not panic");

        let gap0 = (models[0].residual_std_ratio - expected[0]).abs();
        let gap1 = (models[1].residual_std_ratio - expected[1]).abs();
        assert!(
            gap0 < 1e-12,
            "stage 0 (season 21->0): expected {}, got {}, gap={gap0:e}",
            expected[0],
            models[0].residual_std_ratio
        );
        assert!(
            gap1 < 1e-12,
            "stage 1 (season 26->1): expected {}, got {}, gap={gap1:e}",
            expected[1],
            models[1].residual_std_ratio
        );
    }

    /// Every season is AR(2), so each season's ratio depends on its cycle
    /// predecessor's implied autocorrelation.
    #[test]
    fn derived_residual_ratios_follow_the_calendar_cycle_when_season_ids_are_out_of_order() {
        const OUT_OF_ORDER_IDS: [usize; 4] = [2, 0, 3, 1];
        let calendar_ordered = cobre_stochastic::test_support::quarterly_season_map();
        let mut out_of_order = calendar_ordered.clone();
        for (def, id) in out_of_order.seasons.iter_mut().zip(OUT_OF_ORDER_IDS) {
            def.id = id;
        }
        out_of_order.seasons.sort_by_key(|def| def.id);

        let psi_by_quarter: Vec<Vec<f64>> = vec![
            vec![0.5, 0.2],
            vec![0.3, 0.1],
            vec![0.6, -0.15],
            vec![0.4, 0.25],
        ];
        assert!(cobre_stochastic::par::check_stationarity(&psi_by_quarter, &[2; 4], 4).is_ok());

        let stages_with = |season_ids: [usize; 4]| -> Vec<Stage> {
            (0_usize..)
                .zip(0_i32..)
                .zip(season_ids)
                .map(|((index, id), season_id)| stage_with_season(index, id, season_id))
                .collect()
        };
        let ratios_under = |stage_to_season: &HashMap<i32, usize>, n_seasons: usize| -> Vec<u64> {
            let mut models: Vec<InflowModel> = (0_i32..)
                .zip(&psi_by_quarter)
                .map(|(stage_id, psi)| model(1, stage_id, psi.clone(), 0.5, 20.0, None))
                .collect();
            populate_derived_residual_ratios(&mut models, stage_to_season, n_seasons).unwrap();
            models
                .iter()
                .map(|m| m.residual_std_ratio.to_bits())
                .collect()
        };

        let (stage_to_season, n_seasons) =
            resolve_stage_seasons(&stages_with(OUT_OF_ORDER_IDS), Some(&out_of_order));
        let (twin_stage_to_season, twin_n_seasons) =
            resolve_stage_seasons(&stages_with([0, 1, 2, 3]), Some(&calendar_ordered));
        let calendar_ratios = ratios_under(&twin_stage_to_season, twin_n_seasons);
        assert_eq!(ratios_under(&stage_to_season, n_seasons), calendar_ratios);

        let id_ranked: HashMap<i32, usize> = (0_i32..).zip(OUT_OF_ORDER_IDS).collect();
        let id_ranked_ratios = ratios_under(&id_ranked, 4);
        let quarters_with_another_predecessor: Vec<usize> = (0..4)
            .filter(|&quarter| {
                let calendar_predecessor = OUT_OF_ORDER_IDS[(quarter + 3) % 4];
                let id_predecessor = (OUT_OF_ORDER_IDS[quarter] + 3) % 4;
                calendar_predecessor != id_predecessor
            })
            .collect();
        assert!(quarters_with_another_predecessor.len() >= 2);
        for quarter in quarters_with_another_predecessor {
            assert_ne!(
                id_ranked_ratios[quarter], calendar_ratios[quarter],
                "quarter {quarter} must tell the two rankings apart"
            );
        }
    }
}
