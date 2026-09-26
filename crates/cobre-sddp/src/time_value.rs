//! Present-value discounting of stage costs and the post-study delivery
//! calendar.

use cobre_core::{EntityId, HorizonGraph, PostStudyStages, PostStudyThermalBound, Stage, System};
use cobre_stochastic::season_cast::post_study_calendar_stages;

use crate::lp::indexer::{AnticipatedLocal, AnticipatedPlants};

/// Compute per-stage one-step discount factors from study stages and a policy graph.
///
/// `discount_factors[t] = 1 / (1 + r_t)^(Dt / 365.25)` where `r_t` is the annual
/// discount rate for stage `t` (`HorizonGraph::stage_discount_rate_overrides` keyed
/// by `Stage::id`, else the global `annual_discount_rate`) and `Dt` is the stage
/// duration in days. When `rate == 0.0`, the factor is `1.0` (no discounting).
pub(crate) fn compute_per_stage_discount_factors(
    study_stages: &[&Stage],
    pg: &HorizonGraph,
) -> Vec<f64> {
    study_stages
        .iter()
        .map(|stage| {
            let rate = pg
                .stage_discount_rate_overrides
                .get(&stage.id)
                .copied()
                .unwrap_or(pg.annual_discount_rate);
            if rate == 0.0 {
                1.0
            } else {
                let dt_days = f64::from(
                    i32::try_from((stage.end_date - stage.start_date).num_days())
                        .unwrap_or(i32::MAX),
                );
                1.0 / (1.0 + rate).powf(dt_days / 365.25)
            }
        })
        .collect()
}

/// Compute cumulative discount factors from per-stage one-step factors.
///
/// Length is `per_stage.len()` exactly: the strict anticipated-decision predicate
/// (`stage_idx + K_i < n_stages`) keeps every delivery lookup within
/// `[0, n_stages)`, so no boundary-stage entry is needed.
pub(crate) fn compute_cumulative_discount_factors(per_stage: &[f64]) -> Vec<f64> {
    let n = per_stage.len();
    let mut cumulative = vec![1.0; n];
    for t in 1..n {
        cumulative[t] = cumulative[t - 1] * per_stage[t - 1];
    }
    cumulative
}

/// Per-`(thermal, post-study stage)` cost/bounds lookup — `PostStudyStages::
/// thermal_bounds` verbatim, never rebuilt into a nondeterministic-
/// iteration-order map.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PostStudyThermalLookup {
    bounds: Vec<PostStudyThermalBound>,
}

impl PostStudyThermalLookup {
    fn new(bounds: Vec<PostStudyThermalBound>) -> Self {
        debug_assert!(
            bounds.is_sorted_by_key(|b| (b.thermal_id, b.post_study_stage_index)),
            "PostStudyStages::thermal_bounds must already be canonically sorted by \
             (thermal_id, post_study_stage_index) — the cobre-io parser's own invariant"
        );
        Self { bounds }
    }

    /// `(cost_per_mwh, min_mw, max_mw)` declared for `(thermal_id,
    /// post_study_stage_index)`; `None` when undeclared.
    #[must_use]
    pub(crate) fn lookup(
        &self,
        thermal_id: EntityId,
        post_study_stage_index: usize,
    ) -> Option<(f64, f64, f64)> {
        self.bounds
            .binary_search_by_key(&(thermal_id, post_study_stage_index), |b| {
                (b.thermal_id, b.post_study_stage_index)
            })
            .ok()
            .map(|i| {
                let b = &self.bounds[i];
                (b.cost_per_mwh, b.min_mw, b.max_mw)
            })
    }

    /// The one field, for the canonical byte-encoding snapshot — the
    /// no-`..` destructure fails to compile the moment a field is added.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn canonical_fields(&self) -> &[PostStudyThermalBound] {
        let Self { bounds } = self;
        bounds
    }
}

/// Setup-side resolved post-study boundary artifacts (`System::
/// post_study_stages`), built once so the LP builder
/// (`TemplateBuildCtx`/`StageLayout`) and `policy_export` read them without
/// re-deriving the calendar walk, the discount continuation, or the
/// per-thermal lookup. Every field is empty without `post_study_stages` —
/// inert: a study with no post-horizon commitment leaves the rest of setup
/// unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PostStudyResolved {
    /// Post-study stage `j`'s own duration in hours
    /// (`PostStudyStage::duration_hours` verbatim).
    pub(crate) total_hours: Vec<f64>,
    /// Cumulative discount factor continued past the study horizon — the exact
    /// values [`compute_cumulative_discount_factors`] would hold for these
    /// stages had the horizon been extended to cover them (the study's last
    /// cumulative factor bridged by the last study stage's own one-step
    /// factor, then multiplied stage-by-stage).
    pub(crate) cumulative_discount_factors: Vec<f64>,
    /// Per-`(thermal, post-study stage)` cost/bounds lookup.
    pub(crate) thermal_bounds: PostStudyThermalLookup,
    /// Dense row-major `[anticipated_local][post_study_stage]` projection of
    /// [`PostStudyThermalLookup::lookup`] — one cell per anticipated plant times
    /// post-study stage, `None` where the deck declares none. `anticipated_local`
    /// MUST be [`crate::indexer::AnticipatedPlants`]'s canonical order; a
    /// mismatched order silently prices one plant's post-study commitment with
    /// another's fuel cost. Never index this directly — read it
    /// only through [`Self::anticipated_bound`], and never rebuild it into a
    /// `Vec<Vec<_>>` (a per-plant allocation) or an `EntityId`-keyed map (a
    /// nondeterministic-iteration-order read).
    anticipated_bounds: Vec<Option<(f64, f64, f64)>>,
    /// Row stride of [`Self::anticipated_bounds`] — the post-study stage count,
    /// `total_hours.len()`.
    anticipated_bounds_stride: usize,
}

/// Return type of [`PostStudyResolved::canonical_fields`] — a `type` alias
/// rather than a struct, since the fields are consumed positionally by the
/// one encoder call site.
#[cfg(any(test, feature = "test-support"))]
type PostStudyResolvedCanonicalFields<'a> = (
    &'a [f64],
    &'a [f64],
    &'a PostStudyThermalLookup,
    &'a [Option<(f64, f64, f64)>],
    usize,
);

impl PostStudyResolved {
    /// `(cost_per_mwh, min_mw, max_mw)` declared for anticipated-local plant
    /// `local_idx` at post-study stage `post_study_stage`, or `None` when the
    /// deck declares no cell there — never a panic, including on an empty table
    /// (`PostStudyResolved::default()`) or an out-of-range `local_idx`/
    /// `post_study_stage`.
    #[must_use]
    pub(crate) fn anticipated_bound(
        &self,
        local_idx: AnticipatedLocal,
        post_study_stage: usize,
    ) -> Option<(f64, f64, f64)> {
        if post_study_stage >= self.anticipated_bounds_stride {
            return None;
        }
        self.anticipated_bounds
            .get(local_idx.get() * self.anticipated_bounds_stride + post_study_stage)
            .copied()
            .flatten()
    }

    /// Every field, in declaration order, for the canonical byte-encoding
    /// snapshot — the no-`..` destructure fails to compile the moment a field
    /// is added, so the digest cannot silently drop it.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn canonical_fields(&self) -> PostStudyResolvedCanonicalFields<'_> {
        let Self {
            total_hours,
            cumulative_discount_factors,
            thermal_bounds,
            anticipated_bounds,
            anticipated_bounds_stride,
        } = self;
        (
            total_hours,
            cumulative_discount_factors,
            thermal_bounds,
            anticipated_bounds,
            *anticipated_bounds_stride,
        )
    }
}

/// Resolve `System::post_study_stages` into the setup-side artifacts:
/// post-study `total_hours`, the discount continuation, the per-thermal
/// cost/bounds lookup, and its dense anticipated-local projection. `None`/empty
/// `post_study` returns [`PostStudyResolved::default`] — inert.
///
/// `anticipated_thermal_ids` is the anticipated plants' `EntityId`s in
/// anticipated-local order — `resolve_state_layout`'s own
/// [`crate::indexer::AnticipatedPlants`] order, handed in rather than
/// re-derived here, since a second derivation could silently diverge from it.
///
/// `last_real_cumulative` and `last_real_per_stage` are the study's own last
/// cumulative and per-stage discount factors — `StageTemplates::
/// cumulative_discount_factors`/`StageTemplates::discount_factors`'s last
/// entries, or (`build_stage_templates`'s own `TemplateBuildCtx` build) the
/// identical values computed from the same
/// `compute_per_stage_discount_factors`/`compute_cumulative_discount_factors`
/// pair before those output slices exist. The first post-study cumulative
/// factor bridges the horizon by the LAST STUDY stage's own one-step factor
/// (`last_real_cumulative * last_real_per_stage`), NEVER the first post-study
/// stage's (`* per_stage_post[0]`): the continuation must equal what
/// `cumulative_discount_factors` would hold had the horizon been extended to
/// cover the post-study stages.
pub(crate) fn resolve_post_study_artifacts(
    post_study: Option<&PostStudyStages>,
    anticipated_thermal_ids: &[EntityId],
    pg: &HorizonGraph,
    last_real_cumulative: f64,
    last_real_per_stage: f64,
) -> PostStudyResolved {
    let Some(post_study) = post_study else {
        return PostStudyResolved::default();
    };
    if post_study.stages.is_empty() {
        return PostStudyResolved::default();
    }

    let total_hours: Vec<f64> = post_study.stages.iter().map(|s| s.duration_hours).collect();

    let calendar_stages = post_study_calendar_stages(&post_study.stages);
    // `PostStudyStage` declares no rate-override field (unlike a dispatched
    // `Stage`); a `HorizonGraph` carrying only `annual_discount_rate` keeps this
    // call from resolving a synthetic post-study stage id against a REAL study
    // stage's override in `pg.stage_discount_rate_overrides`.
    let rate_graph = HorizonGraph {
        annual_discount_rate: pg.annual_discount_rate,
        ..HorizonGraph::default()
    };
    let calendar_stage_refs: Vec<&Stage> = calendar_stages.iter().collect();
    let per_stage_post = compute_per_stage_discount_factors(&calendar_stage_refs, &rate_graph);

    let mut cumulative_discount_factors = Vec::with_capacity(per_stage_post.len());
    let mut cumulative = last_real_cumulative * last_real_per_stage;
    for &factor in &per_stage_post {
        cumulative_discount_factors.push(cumulative);
        cumulative *= factor;
    }

    let thermal_bounds = PostStudyThermalLookup::new(post_study.thermal_bounds.clone());

    // Dense [anticipated_local][post_study_stage] projection of `thermal_bounds`,
    // built once here so the ring fill (`fill_anticipated_columns`) never
    // reconstructs an `EntityId` from an anticipated-local index or re-searches
    // `thermal_bounds` per fill call.
    let anticipated_bounds_stride = total_hours.len();
    let mut anticipated_bounds =
        Vec::with_capacity(anticipated_thermal_ids.len() * anticipated_bounds_stride);
    for &thermal_id in anticipated_thermal_ids {
        for post_study_stage in 0..anticipated_bounds_stride {
            anticipated_bounds.push(thermal_bounds.lookup(thermal_id, post_study_stage));
        }
    }
    debug_assert_eq!(
        anticipated_bounds.len(),
        anticipated_thermal_ids.len() * anticipated_bounds_stride,
        "PostStudyResolved.anticipated_bounds row count must equal \
         anticipated_thermal_ids.len()"
    );

    PostStudyResolved {
        total_hours,
        cumulative_discount_factors,
        thermal_bounds,
        anticipated_bounds,
        anticipated_bounds_stride,
    }
}

/// The cumulative discount and delivery calendar of every delivery stage
/// (study stages, then post-study stages), `D(0) == 1.0`.
#[derive(Debug, Clone)]
pub(crate) struct TimeValue {
    /// Study one-step discount factors, length `n_study_stages`.
    discount_factors: Vec<f64>,
    delivery_cumulative_discount_factors: Vec<f64>,
    delivery_total_hours: Vec<f64>,
    delivery_stage_ids: Vec<i32>,
    post_study: PostStudyResolved,
}

impl TimeValue {
    /// Resolve the whole delivery calendar directly from `system`: derives the
    /// study stages (`id >= 0`) and the anticipated thermal ids (projected from
    /// `anticipated_plants`, [`crate::indexer::AnticipatedPlants`]'s canonical
    /// order), the post-study calendar and the policy graph, then delegates to
    /// [`Self::resolve`]. `study_total_hours` is supplied by the caller
    /// (`BlockClock`) rather than derived here — `block_clock` is not on this
    /// module's import allowlist.
    pub(crate) fn from_system(
        system: &System,
        anticipated_plants: &AnticipatedPlants,
        study_total_hours: &[f64],
    ) -> Self {
        let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
        let anticipated_thermal_ids: Vec<EntityId> = anticipated_plants
            .thermals()
            .map(|t| system.thermals()[t.get()].id)
            .collect();
        Self::resolve(
            &study_stages,
            study_total_hours,
            system.post_study_stages(),
            &anticipated_thermal_ids,
            system.policy_graph(),
        )
    }

    /// Resolve the whole delivery calendar: the study's own one-step and
    /// cumulative discount factors, [`resolve_post_study_artifacts`]'s
    /// post-study continuation, and the concatenated delivery hours,
    /// cumulative-discount and synthetic-id vectors.
    fn resolve(
        study_stages: &[&Stage],
        study_total_hours: &[f64],
        post_study: Option<&PostStudyStages>,
        anticipated_thermal_ids: &[EntityId],
        pg: &HorizonGraph,
    ) -> Self {
        let discount_factors = compute_per_stage_discount_factors(study_stages, pg);
        let cumulative = compute_cumulative_discount_factors(&discount_factors);
        debug_assert_eq!(
            cumulative.len(),
            study_stages.len(),
            "cumulative_discount_factors length must equal n_study_stages"
        );

        let post_study_resolved = resolve_post_study_artifacts(
            post_study,
            anticipated_thermal_ids,
            pg,
            cumulative.last().copied().unwrap_or(1.0),
            discount_factors.last().copied().unwrap_or(1.0),
        );

        let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();

        // Concatenate rather than recompute: `resolve_post_study_artifacts` already
        // establishes that the post-study half continues the study recurrence, so a
        // second derivation would risk diverging from it.
        let delivery_total_hours: Vec<f64> = study_total_hours
            .iter()
            .copied()
            .chain(post_study_resolved.total_hours.iter().copied())
            .collect();
        let delivery_cumulative_discount_factors: Vec<f64> = cumulative
            .iter()
            .copied()
            .chain(
                post_study_resolved
                    .cumulative_discount_factors
                    .iter()
                    .copied(),
            )
            .collect();
        // Synthetic continuation from `study_stage_ids.last()`, never
        // `post_study_calendar_stages`'s own `Stage::id`: those restart at `0`
        // and would make a post-study delivery compare as an early study stage.
        let n_post = post_study_resolved.total_hours.len();
        let next_delivery_id = study_stage_ids.last().map_or(0, |&last| last + 1);
        let end_delivery_id =
            next_delivery_id.saturating_add(i32::try_from(n_post).unwrap_or(i32::MAX));
        let delivery_stage_ids: Vec<i32> = study_stage_ids
            .iter()
            .copied()
            .chain(next_delivery_id..end_delivery_id)
            .collect();

        let n_delivery = study_stage_ids.len() + n_post;
        debug_assert_eq!(
            delivery_total_hours.len(),
            n_delivery,
            "delivery_total_hours length must equal n_study_stages + n_post"
        );
        debug_assert_eq!(
            delivery_cumulative_discount_factors.len(),
            n_delivery,
            "delivery_cumulative_discount_factors length must equal n_study_stages + n_post"
        );
        debug_assert_eq!(
            delivery_stage_ids.len(),
            n_delivery,
            "delivery_stage_ids length must equal n_study_stages + n_post"
        );
        debug_assert!(
            delivery_stage_ids.windows(2).all(|w| w[0] < w[1]),
            "delivery_stage_ids must be strictly increasing — commissioning_active's \
             monotonicity depends on it"
        );

        Self {
            discount_factors,
            delivery_cumulative_discount_factors,
            delivery_total_hours,
            delivery_stage_ids,
            post_study: post_study_resolved,
        }
    }

    /// Test/fixture constructor: carries the literal delivery vectors over
    /// verbatim, with no post-study derivation.
    #[cfg(any(test, feature = "test-support"))]
    #[allow(clippy::float_cmp)]
    pub(crate) fn from_parts(
        discount_factors: Vec<f64>,
        delivery_cumulative_discount_factors: Vec<f64>,
        delivery_total_hours: Vec<f64>,
        delivery_stage_ids: Vec<i32>,
        post_study: PostStudyResolved,
    ) -> Self {
        debug_assert!(
            delivery_cumulative_discount_factors
                .first()
                .is_none_or(|&d| d == 1.0),
            "TimeValue::from_parts: the first delivery stage's cumulative discount must be 1.0"
        );
        debug_assert_eq!(
            delivery_cumulative_discount_factors.len(),
            delivery_total_hours.len(),
            "TimeValue::from_parts: delivery_cumulative_discount_factors and \
             delivery_total_hours must have equal length"
        );
        debug_assert_eq!(
            delivery_cumulative_discount_factors.len(),
            delivery_stage_ids.len(),
            "TimeValue::from_parts: delivery_cumulative_discount_factors and \
             delivery_stage_ids must have equal length"
        );
        Self {
            discount_factors,
            delivery_cumulative_discount_factors,
            delivery_total_hours,
            delivery_stage_ids,
            post_study,
        }
    }

    /// Study one-step discount factors, length `n_study_stages`.
    pub(crate) fn discount_factors(&self) -> &[f64] {
        &self.discount_factors
    }

    /// Study-only cumulative discount factors, length `n_study_stages` — the
    /// delivery vector's own prefix, bit-identical to
    /// `compute_cumulative_discount_factors(discount_factors())` because the
    /// delivery vector starts with that exact result.
    pub(crate) fn cumulative_discount_factors(&self) -> &[f64] {
        &self.delivery_cumulative_discount_factors[..self.discount_factors.len()]
    }

    /// Σ `block.duration_hours` at DELIVERY stage `delivery`.
    pub(crate) fn delivery_total_hours(&self, delivery: usize) -> f64 {
        self.delivery_total_hours[delivery]
    }

    /// `study_stage_ids` continued past the horizon by a synthetic id
    /// sequence, length `n_study_stages + n_post`; indexed by DELIVERY stage.
    pub(crate) fn delivery_stage_ids(&self) -> &[i32] {
        &self.delivery_stage_ids
    }

    /// Resolved post-study boundary artifacts.
    pub(crate) fn post_study(&self) -> &PostStudyResolved {
        &self.post_study
    }

    /// The discount of a cost delivered at `delivery`, in the units of stage
    /// `decision`: `D(delivery) / D(decision)`.
    pub(crate) fn relative_delivery_discount(&self, decision: usize, delivery: usize) -> f64 {
        self.delivery_cumulative_discount_factors[delivery]
            / self.delivery_cumulative_discount_factors[decision]
    }

    /// Every field, in declaration order, for the canonical byte-encoding
    /// snapshot — the no-`..` destructure fails to compile the moment a field
    /// is added, so the digest cannot silently drop it.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn canonical_fields(&self) -> (&[f64], &[f64], &[f64], &[i32], &PostStudyResolved) {
        let Self {
            discount_factors,
            delivery_cumulative_discount_factors,
            delivery_total_hours,
            delivery_stage_ids,
            post_study,
        } = self;
        (
            discount_factors,
            delivery_cumulative_discount_factors,
            delivery_total_hours,
            delivery_stage_ids,
            post_study,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{PostStudyResolved, TimeValue};

    fn fixture() -> Vec<f64> {
        vec![1.0, 0.9, 0.81, 0.729]
    }

    fn time_value_from_cumulative(cumulative: Vec<f64>) -> TimeValue {
        TimeValue::from_parts(
            vec![],
            cumulative,
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0, 1, 2, 3],
            PostStudyResolved::default(),
        )
    }

    #[test]
    fn relative_delivery_discount_from_stage_zero_is_bit_exact_to_the_absolute_factor() {
        let d = fixture();
        let tv = time_value_from_cumulative(d.clone());
        for (m, &expected) in d.iter().enumerate() {
            assert_eq!(tv.relative_delivery_discount(0, m), expected);
        }
    }

    #[test]
    fn relative_delivery_discount_divides_the_two_absolute_factors() {
        let d = fixture();
        let tv = time_value_from_cumulative(d.clone());
        assert_eq!(tv.relative_delivery_discount(1, 3), d[3] / d[1]);
    }

    /// [`TimeValue::cumulative_discount_factors`] (the study-only prefix
    /// accessor) must equal [`compute_cumulative_discount_factors`] applied to
    /// [`TimeValue::discount_factors`], bit for bit — it is the same result,
    /// never a second derivation.
    #[test]
    fn cumulative_discount_factors_prefix_matches_recomputation() {
        let discount_factors = vec![1.0, 0.9, 0.81];
        let recomputed = super::compute_cumulative_discount_factors(&discount_factors);
        // A post-study tail entry appended after the study-only prefix, to prove
        // the accessor truncates to `discount_factors.len()` rather than
        // returning the whole delivery vector.
        let mut delivery_cumulative = recomputed.clone();
        delivery_cumulative.push(0.5);
        let tv = TimeValue::from_parts(
            discount_factors,
            delivery_cumulative,
            vec![10.0, 10.0, 10.0, 10.0],
            vec![0, 1, 2, 3],
            PostStudyResolved::default(),
        );

        let actual: Vec<u64> = tv
            .cumulative_discount_factors()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let expected: Vec<u64> = recomputed.iter().map(|v| v.to_bits()).collect();
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod from_system_tests {
    use super::{AnticipatedPlants, TimeValue};
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedConfig, Bus, DeficitSegment, EntityId, Stage, SystemBuilder, Thermal,
    };

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap_or_else(|| unreachable!("hardcoded date is valid"))
    }

    fn stage(id: i32, start: NaiveDate, end: NaiveDate) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: start,
            end_date: end,
            season_id: None,
            blocks: vec![],
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

    /// A two-stage system with one anticipated thermal resolves through
    /// `from_system` to the same `delivery_stage_ids` a study-only-axis
    /// (no post-study calendar) delivery calendar carries: the study stage
    /// ids verbatim.
    #[test]
    fn from_system_derives_study_stages_and_anticipated_ids_then_resolves() {
        let s0 = stage(0, ymd(2024, 1, 1), ymd(2024, 2, 1));
        let s1 = stage(1, ymd(2024, 2, 1), ymd(2024, 3, 1));
        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: ymd(2024, 1, 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let thermal = Thermal {
            id: EntityId(7),
            name: "T1".to_string(),
            operational_start_date: ymd(2024, 1, 1),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 10.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            entry_stage_id: None,
            exit_stage_id: None,
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .stages(vec![s0, s1])
            .thermals(vec![thermal])
            .build()
            .expect("minimal two-stage anticipated system must build");

        let study_total_hours = vec![744.0, 672.0];
        let anticipated_plants = AnticipatedPlants::build(system.thermals());
        let tv = TimeValue::from_system(&system, &anticipated_plants, &study_total_hours);

        assert_eq!(tv.delivery_stage_ids(), &[0, 1]);
    }
}

#[cfg(test)]
mod post_study_resolution_tests {
    use super::{
        PostStudyResolved, compute_cumulative_discount_factors, compute_per_stage_discount_factors,
        resolve_post_study_artifacts,
    };
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId, HorizonGraph, PostStudyStage, PostStudyStages, PostStudyThermalBound,
    };
    use cobre_stochastic::season_cast::post_study_calendar_stages;

    fn two_stage_post_study() -> PostStudyStages {
        PostStudyStages {
            stages: vec![
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 11, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 720.0,
                },
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 12, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 744.0,
                },
            ],
            thermal_bounds: vec![
                PostStudyThermalBound {
                    thermal_id: EntityId(1),
                    post_study_stage_index: 0,
                    cost_per_mwh: 210.0,
                    min_mw: 0.0,
                    max_mw: 350.0,
                },
                PostStudyThermalBound {
                    thermal_id: EntityId(1),
                    post_study_stage_index: 1,
                    cost_per_mwh: 220.0,
                    min_mw: 0.0,
                    max_mw: 300.0,
                },
            ],
        }
    }

    #[test]
    fn post_study_absent_returns_default() {
        let resolved = resolve_post_study_artifacts(None, &[], &HorizonGraph::default(), 1.0, 1.0);
        assert_eq!(resolved, PostStudyResolved::default());
    }

    #[test]
    fn post_study_with_no_stages_returns_default() {
        let empty = PostStudyStages {
            stages: Vec::new(),
            thermal_bounds: Vec::new(),
        };
        let resolved =
            resolve_post_study_artifacts(Some(&empty), &[], &HorizonGraph::default(), 1.0, 1.0);
        assert_eq!(resolved, PostStudyResolved::default());
    }

    #[test]
    fn total_hours_matches_declared_duration() {
        let post_study = two_stage_post_study();
        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &HorizonGraph::default(),
            1.0,
            1.0,
        );
        assert_eq!(resolved.total_hours, vec![720.0, 744.0]);
    }

    #[test]
    fn continued_cumulative_discount_is_seed_at_zero_rate() {
        let post_study = two_stage_post_study();
        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &HorizonGraph::default(),
            0.9,
            1.0,
        );
        assert_eq!(resolved.cumulative_discount_factors, vec![0.9, 0.9]);
    }

    #[test]
    fn continued_cumulative_discount_matches_extended_horizon() {
        let post_study = two_stage_post_study();
        let pg = HorizonGraph {
            annual_discount_rate: 0.08,
            ..HorizonGraph::default()
        };
        // A synthetic two-stage study whose per-stage one-step factors are
        // `[0.95, 0.93]`: its last cumulative factor is `0.95` (the product of
        // the stages strictly before the last) and its last per-stage factor is
        // `0.93`.
        let study_per_stage = [0.95_f64, 0.93_f64];
        let last_real_cumulative = study_per_stage[0];
        let last_real_per_stage = study_per_stage[1];

        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &pg,
            last_real_cumulative,
            last_real_per_stage,
        );

        // Ground truth: extend the horizon with the post-study stages, take the
        // cumulative product over the whole thing, and read off the post-study
        // tail. A resolver that bridged by `per_stage_post[0]` instead of the
        // last study factor would diverge here.
        let calendar_stages = post_study_calendar_stages(&post_study.stages);
        let calendar_stage_refs: Vec<_> = calendar_stages.iter().collect();
        let per_stage_post = compute_per_stage_discount_factors(&calendar_stage_refs, &pg);
        let mut extended_per_stage = study_per_stage.to_vec();
        extended_per_stage.extend_from_slice(&per_stage_post);
        let extended_cumulative = compute_cumulative_discount_factors(&extended_per_stage);

        assert_eq!(
            resolved.cumulative_discount_factors,
            extended_cumulative[study_per_stage.len()..].to_vec()
        );
    }

    #[test]
    fn thermal_bound_lookup_returns_declared_triple() {
        let post_study = two_stage_post_study();
        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &HorizonGraph::default(),
            1.0,
            1.0,
        );

        assert_eq!(
            resolved.thermal_bounds.lookup(EntityId(1), 0),
            Some((210.0, 0.0, 350.0))
        );
        assert_eq!(
            resolved.thermal_bounds.lookup(EntityId(1), 1),
            Some((220.0, 0.0, 300.0))
        );
        assert_eq!(resolved.thermal_bounds.lookup(EntityId(2), 0), None);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod discount_factor_tests {
    use super::{compute_cumulative_discount_factors, compute_per_stage_discount_factors};
    use chrono::NaiveDate;
    use cobre_core::HorizonGraph;
    use cobre_core::temporal::{
        BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use std::collections::BTreeMap;

    fn one_year_stage(id: i32) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            season_id: None,
            blocks: vec![],
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

    /// A `stages[].annual_discount_rate_override` (carried on
    /// `HorizonGraph::stage_discount_rate_overrides`) sets that stage's rate,
    /// overriding the global `annual_discount_rate`.
    #[test]
    fn stage_discount_override_is_read_off_the_stage() {
        let stage = one_year_stage(0);
        let days = f64::from((stage.end_date - stage.start_date).num_days() as i32);
        let mut overrides = BTreeMap::new();
        overrides.insert(0, 0.10);
        let pg = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            nodes: vec![],
            stage_discount_rate_overrides: overrides,
            season_map: None,
        };

        let factors = compute_per_stage_discount_factors(&[&stage], &pg);
        let expected = 1.0 / (1.0_f64 + 0.10).powf(days / 365.25);
        assert!(
            (factors[0] - expected).abs() < 1e-12,
            "stage override 0.10 must set the rate, got {}",
            factors[0]
        );
        let global = 1.0 / (1.0_f64 + 0.06).powf(days / 365.25);
        assert!(
            (factors[0] - global).abs() > 1e-6,
            "override must differ from the global-rate factor"
        );
    }

    /// A stage with no override falls back to the global `annual_discount_rate`.
    #[test]
    fn stage_without_override_uses_global_rate() {
        let stage = one_year_stage(0);
        let days = f64::from((stage.end_date - stage.start_date).num_days() as i32);
        let pg = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            nodes: vec![],
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        };

        let factors = compute_per_stage_discount_factors(&[&stage], &pg);
        let expected = 1.0 / (1.0_f64 + 0.06).powf(days / 365.25);
        assert!(
            (factors[0] - expected).abs() < 1e-12,
            "absent override must fall back to the global rate, got {}",
            factors[0]
        );
    }

    #[test]
    fn cumulative_discount_factors_length_matches_n_stages() {
        let n_stages = 4_usize;
        let per_stage = vec![0.95_f64; n_stages];
        let cumulative = compute_cumulative_discount_factors(&per_stage);

        assert_eq!(
            cumulative.len(),
            n_stages,
            "cumulative_discount_factors length must equal n_stages = {n_stages}"
        );

        assert_eq!(cumulative[0], 1.0, "cumulative[0] == 1.0 (present value)");
        assert_eq!(
            cumulative[1], 0.95,
            "cumulative[1] == 0.95 = 1.0 * per_stage[0]"
        );
        // Approximate: repeated multiplication may differ from powi(3) by a ULP
        // (floating-point associativity).
        assert!(
            (cumulative[n_stages - 1] - 0.95_f64.powi(3)).abs() < 1e-15,
            "cumulative[n_stages-1] must be within 1e-15 of 0.95^(n_stages-1) (got {})",
            cumulative[n_stages - 1]
        );
    }

    #[test]
    fn cumulative_discount_factors_all_ones_when_rate_zero() {
        let per_stage = vec![1.0_f64; 3];
        let cumulative = compute_cumulative_discount_factors(&per_stage);
        assert_eq!(cumulative.len(), 3);
        for (i, &v) in cumulative.iter().enumerate() {
            assert_eq!(
                v, 1.0,
                "cumulative[{i}] must be 1.0 when per-stage factor is 1.0"
            );
        }
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::{
        TimeValue, compute_cumulative_discount_factors, compute_per_stage_discount_factors,
        resolve_post_study_artifacts,
    };
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{HorizonGraph, PostStudyStage, PostStudyStages, Stage};

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap_or_else(|| unreachable!("hardcoded date is valid"))
    }

    fn stage(id: i32, start: NaiveDate, end: NaiveDate) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: start,
            end_date: end,
            season_id: None,
            blocks: vec![],
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

    /// `resolve` concatenates the study and post-study hours, cumulative
    /// factors and synthetic ids bit-exactly, for a deck with two post-study
    /// stages.
    #[test]
    fn resolve_concatenates_study_and_post_study_delivery_vectors_bit_exact() {
        let s0 = stage(0, ymd(2024, 1, 1), ymd(2024, 2, 1));
        let s1 = stage(1, ymd(2024, 2, 1), ymd(2024, 3, 1));
        let study_stages: Vec<&Stage> = vec![&s0, &s1];
        let study_total_hours = vec![744.0, 696.0];
        let pg = HorizonGraph {
            annual_discount_rate: 0.08,
            ..HorizonGraph::default()
        };
        let post_study = PostStudyStages {
            stages: vec![
                PostStudyStage {
                    start_date: ymd(2024, 3, 1),
                    duration_hours: 720.0,
                },
                PostStudyStage {
                    start_date: ymd(2024, 4, 1),
                    duration_hours: 744.0,
                },
            ],
            thermal_bounds: vec![],
        };

        let tv = TimeValue::resolve(
            &study_stages,
            &study_total_hours,
            Some(&post_study),
            &[],
            &pg,
        );

        let per_stage = compute_per_stage_discount_factors(&study_stages, &pg);
        let cumulative = compute_cumulative_discount_factors(&per_stage);
        let post_study_resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &pg,
            cumulative.last().copied().unwrap_or(1.0),
            per_stage.last().copied().unwrap_or(1.0),
        );

        let expected_hours: Vec<f64> = study_total_hours
            .iter()
            .copied()
            .chain(post_study_resolved.total_hours.iter().copied())
            .collect();
        let expected_cumulative: Vec<f64> = cumulative
            .iter()
            .copied()
            .chain(
                post_study_resolved
                    .cumulative_discount_factors
                    .iter()
                    .copied(),
            )
            .collect();
        let expected_ids: Vec<i32> = vec![0, 1, 2, 3];

        for (i, (&h, &c)) in expected_hours.iter().zip(&expected_cumulative).enumerate() {
            assert_eq!(
                tv.delivery_total_hours(i).to_bits(),
                h.to_bits(),
                "delivery_total_hours[{i}] must match the recomputed concatenation bit-for-bit"
            );
            assert_eq!(
                tv.relative_delivery_discount(0, i).to_bits(),
                c.to_bits(),
                "cumulative_discount_factors[{i}] must match the recomputed concatenation \
                 bit-for-bit"
            );
        }
        assert_eq!(tv.delivery_stage_ids(), expected_ids.as_slice());
    }

    /// With no post-study stages declared, `delivery_stage_ids()` is
    /// element-wise identical to the study stage ids.
    #[test]
    fn resolve_with_no_post_study_delivery_stage_ids_equal_study_stage_ids() {
        let s0 = stage(0, ymd(2024, 1, 1), ymd(2024, 2, 1));
        let s1 = stage(1, ymd(2024, 2, 1), ymd(2024, 3, 1));
        let study_stages: Vec<&Stage> = vec![&s0, &s1];
        let study_total_hours = vec![744.0, 696.0];
        let pg = HorizonGraph::default();

        let tv = TimeValue::resolve(&study_stages, &study_total_hours, None, &[], &pg);

        let study_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
        assert_eq!(tv.delivery_stage_ids(), study_ids.as_slice());
    }
}
