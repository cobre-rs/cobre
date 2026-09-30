//! Bucket topology: canonical column order, global bucket count, per-stage
//! reachability mask, and the three resolved arc tables (stage-clock weights,
//! chronological spread, arrival density) for water travel-time in-transit
//! buckets — the single derivation site the LP builder threads from, never
//! re-derives.
//!
//! Depths resolve on the stage clock ([`resolve_spread`] for in-study anchors,
//! [`window_period_overlaps`] for the pre-study IC anchor); `n_blks`/block mode
//! never enter dimensioning. Every arc feeding one downstream plant collapses
//! into a single aggregated block ordered by the canonical
//! `(operational_start_date, id)` index every other state block uses.

use std::collections::HashMap;

use cobre_core::{BlockMode, EntityId, Hydro, Stage, System, window_period_overlaps};

use crate::lead_time::{SpreadResolution, resolve_arrival_density_at, resolve_spread};
use crate::lp::indexer::HydroSys;
use crate::time_value::DeliveryCalendar;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TravelTimeArc {
    pub(crate) upstream: HydroSys,
    pub(crate) downstream: HydroSys,
    pub(crate) travel_time_hours: f64,
}

/// Canonical bucket ordering, global bucket count, and per-stage reachability
/// mask.
///
/// [`Self::n_buckets`] returns `0` exactly when [`Self::arcs`] is empty.
#[derive(Debug, Clone)]
pub(crate) struct TransitBucketTopology {
    /// `(plant, lag)` pairs, `lag = 1..=L_j`, plants sorted by canonical
    /// `(operational_start_date, id)` index (plant's position in
    /// [`System::hydros`]).
    pub(crate) column_order: Vec<(HydroSys, usize)>,
    /// `per_stage_mask[t]` holds the max reachable lag per declared
    /// downstream plant, in [`Self::column_order`]'s plant order, at study
    /// stage `t` (`0` when no lag is reachable at that stage).
    pub(crate) per_stage_mask: Vec<Vec<usize>>,
    /// Per-declared-arc PARALLEL-mode stage-clock weights; see
    /// [`build_arc_stage_weights`].
    pub(crate) arc_stage_weights: HashMap<usize, Vec<Vec<f64>>>,
    /// Per-declared-arc, per-chronological-stage full [`SpreadResolution`]; see
    /// [`build_arc_spread_chrono`].
    pub(crate) arc_spread_chrono: HashMap<usize, Vec<Option<SpreadResolution>>>,
    /// Per-declared-arc, per-chronological-arrival-stage delivery density; see
    /// [`build_arc_arrival_density`].
    pub(crate) arc_arrival_density: HashMap<usize, Vec<Option<Vec<f64>>>>,
    arcs: Vec<TravelTimeArc>,
}

impl TransitBucketTopology {
    /// Declared upstream→downstream travel-time arcs, in [`System::hydros`]
    /// order.
    pub(crate) fn arcs(&self) -> &[TravelTimeArc] {
        &self.arcs
    }

    /// Global bucket count, the sum of every declared plant's depth.
    pub(crate) fn n_buckets(&self) -> usize {
        self.column_order.len()
    }

    /// A no-arc topology (`n_buckets() == 0`, every table empty), for a
    /// hand-built [`TemplateBuildCtx`](crate::lp::builder::TemplateBuildCtx)
    /// fixture with no backing [`System`] to run [`build_transit_bucket_topology`]
    /// against. The private `arcs` field makes this the only way to construct
    /// one outside this module; every other field is `pub(crate)` and
    /// mutable after construction for a fixture that needs a non-empty one.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn empty() -> Self {
        Self {
            column_order: Vec::new(),
            per_stage_mask: Vec::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            arcs: Vec::new(),
        }
    }
}

/// The one site of the arc rule: a hydro declares an arc when
/// `travel_time_hours` is `Some` and `> 0.0` (`0.0` is undeclared) and
/// `downstream_id` is `Some`, in [`System::hydros`] order.
fn resolve_travel_time_arcs(hydros: &[Hydro]) -> Vec<TravelTimeArc> {
    let positions: HashMap<EntityId, usize> = hydros
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.id, idx))
        .collect();

    hydros
        .iter()
        .enumerate()
        .filter_map(|(idx, hydro)| {
            let travel_time_hours = hydro.travel_time_hours.filter(|&t| t > 0.0)?;
            let downstream_id = hydro.downstream_id?;
            let Some(&downstream_idx) = positions.get(&downstream_id) else {
                debug_assert!(
                    false,
                    "downstream_id must resolve to a declared hydro position; cobre-io's \
                     referential validation guarantees this"
                );
                return None;
            };
            Some(TravelTimeArc {
                upstream: HydroSys::new(idx),
                downstream: HydroSys::new(downstream_idx),
                travel_time_hours,
            })
        })
        .collect()
}

/// Extends the base calendar — study stages plus any declared post-study
/// stages ([`DeliveryCalendar::total_hours`]) — with copies of its trailing
/// duration so [`resolve_spread`] never sees a window it cannot absorb; its
/// conservation check panics otherwise. The pad runs only past what the base
/// calendar already covers, so a declared post-study calendar shorter than
/// the travel time still gets padded. Horizon capping is a separate, later
/// step this does not take.
fn extend_for_resolution(base_calendar: &[f64], t_v: f64) -> Vec<f64> {
    let Some(&last) = base_calendar.last() else {
        return base_calendar.to_vec();
    };
    debug_assert!(last > 0.0, "every base calendar duration must be > 0.0");

    let mut extended = base_calendar.to_vec();
    let mut padded_hours = 0.0_f64;
    while padded_hours < t_v {
        extended.push(last);
        padded_hours += last;
    }
    extended
}

/// In-study depth at one stage anchor: [`resolve_spread`]'s overlap discards
/// its index-0 share (delivered same-stage on the water row, no bucket
/// needed).
fn in_study_depth(t_v: f64, stage: usize, extended_calendar: &[f64]) -> usize {
    resolve_spread(t_v, stage, extended_calendar, None).stage_reach
}

/// Pre-study residual depth: the in-transit water arriving over the
/// study-clock window `[0, t_v)` has no same-stage share to discard, so this
/// is the raw overlap count — the one place this feature's depth arithmetic
/// diverges from [`in_study_depth`].
fn ic_only_depth(t_v: f64, study_durations: &[f64]) -> usize {
    window_period_overlaps(0.0, t_v, study_durations).len()
}

/// Caps a stage's active lag at `n_stages − stage − 1`, the deepest lag whose
/// target stage lands inside `[0, n_stages)`. Never caps
/// [`TransitBucketTopology::column_order`], which sizes from the global max
/// over every stage anchor and must retain what the earliest stages need.
fn horizon_cap_active(active: usize, stage: usize, n_stages: usize) -> usize {
    active.min(n_stages - 1 - stage)
}

/// Build the [`TransitBucketTopology`]: confluence aggregates every arc feeding
/// one downstream plant into a single block of depth `max_i L_i` (never one
/// block per arc), sized as the max over every in-study stage anchor and the
/// pre-study IC anchor, in canonical `(operational_start_date, id)` order.
///
/// `boundary_present` (the caller's `config.policy.boundary.is_some()`) gates
/// [`horizon_cap_active`]: `false` reproduces today's capped mask byte-for-byte
/// (Terminal credit deferred); `true` un-caps the terminal deep-lag slots so
/// they stay live and reach the boundary-priced cut-state projection (the
/// "Delivery-family right-boundary pricing" contract).
pub(crate) fn build_transit_bucket_topology(
    system: &System,
    calendar: &DeliveryCalendar,
    boundary_present: bool,
) -> TransitBucketTopology {
    let n_stages = calendar.n_study();
    let arcs = resolve_travel_time_arcs(system.hydros());

    let mut column_order = Vec::new();
    let mut per_stage_mask: Vec<Vec<usize>> = vec![Vec::new(); n_stages];

    for canonical_idx in 0..system.hydros().len() {
        let downstream = HydroSys::new(canonical_idx);
        let t_vs: Vec<f64> = arcs
            .iter()
            .filter(|arc| arc.downstream == downstream)
            .map(|arc| arc.travel_time_hours)
            .collect();
        if t_vs.is_empty() {
            continue;
        }

        let mut own_release_by_stage = vec![0_usize; n_stages];
        let mut ic_depth = 0_usize;
        for &t_v in &t_vs {
            let extended = extend_for_resolution(calendar.total_hours(), t_v);
            for (stage, slot) in own_release_by_stage.iter_mut().enumerate() {
                *slot = (*slot).max(in_study_depth(t_v, stage, &extended));
            }
            ic_depth = ic_depth.max(ic_only_depth(t_v, calendar.study_total_hours()));
        }

        let in_study_max = own_release_by_stage.iter().copied().max().unwrap_or(0);
        let depth = in_study_max.max(ic_depth);
        if depth == 0 {
            continue;
        }

        for lag in 1..=depth {
            column_order.push((downstream, lag));
        }
        for (stage, mask_row) in per_stage_mask.iter_mut().enumerate() {
            // Reachability, not a zero-deposit filter: a transit slot with no
            // net deposit at this stage still carries mass through the ring
            // shift and must stay in the active range.
            let active = own_release_by_stage[stage].max(ic_depth.saturating_sub(stage));
            let capped = if boundary_present {
                active
            } else {
                let capped = horizon_cap_active(active, stage, n_stages);
                debug_assert!(
                    stage + capped < n_stages,
                    "capped active lag {capped} at stage {stage} must not target n_stages={n_stages} or beyond"
                );
                capped
            };
            mask_row.push(capped);
        }
    }

    let arc_stage_weights = build_arc_stage_weights(&arcs, calendar);
    let arc_spread_chrono = build_arc_spread_chrono(system, &arcs, calendar);
    let arc_arrival_density =
        build_arc_arrival_density(system, &arcs, calendar, &arc_stage_weights);

    let topology = TransitBucketTopology {
        column_order,
        per_stage_mask,
        arc_stage_weights,
        arc_spread_chrono,
        arc_arrival_density,
        arcs,
    };
    debug_assert!(
        topology.arcs.is_empty() == (topology.n_buckets() == 0),
        "n_buckets must be zero exactly when no arc is declared"
    );
    topology
}

/// Per-declared-arc PARALLEL-mode stage-clock weights, keyed by the arc's
/// upstream hydro system index. `k_by_stage[stage_idx]` is [`resolve_spread`]'s
/// `stage_weights` anchored at that in-study stage (`stage_weights[0]` is the
/// same-stage share); a hydro absent declares no arc.
pub(crate) fn build_arc_stage_weights(
    arcs: &[TravelTimeArc],
    calendar: &DeliveryCalendar,
) -> HashMap<usize, Vec<Vec<f64>>> {
    let n_stages = calendar.n_study();
    let mut arc_stage_weights = HashMap::new();

    for arc in arcs {
        let u_idx = arc.upstream.get();
        let t_v = arc.travel_time_hours;
        let extended = extend_for_resolution(calendar.total_hours(), t_v);
        let k_by_stage: Vec<Vec<f64>> = (0..n_stages)
            .map(|stage| resolve_spread(t_v, stage, &extended, None).stage_weights)
            .collect();
        arc_stage_weights.insert(u_idx, k_by_stage);
    }

    arc_stage_weights
}

/// Per-declared-arc, per-CHRONOLOGICAL-stage full [`SpreadResolution`]
/// (`block_deposits`, `within_stage_routing`, `arrival_density`, plus the
/// `stage_weights`/`stage_reach` [`build_arc_stage_weights`] also stores),
/// resolved with the sending stage's own block partition. Keyed like
/// `build_arc_stage_weights`; `by_stage[stage_idx]` is `None` for a `Parallel`
/// stage (no block-resolved routing there).
pub(crate) fn build_arc_spread_chrono(
    system: &System,
    arcs: &[TravelTimeArc],
    calendar: &DeliveryCalendar,
) -> HashMap<usize, Vec<Option<SpreadResolution>>> {
    let n_stages = calendar.n_study();
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    debug_assert_eq!(study_stages.len(), n_stages);

    let mut arc_spread_chrono = HashMap::new();

    for arc in arcs {
        let u_idx = arc.upstream.get();
        let t_v = arc.travel_time_hours;
        let extended = extend_for_resolution(calendar.total_hours(), t_v);
        let by_stage: Vec<Option<SpreadResolution>> = (0..n_stages)
            .map(|stage_idx| {
                if study_stages[stage_idx].block_mode != BlockMode::Chronological {
                    return None;
                }
                let blocks: Vec<f64> = study_stages[stage_idx]
                    .blocks
                    .iter()
                    .map(|b| b.duration_hours)
                    .collect();
                Some(resolve_spread(t_v, stage_idx, &extended, Some(&blocks)))
            })
            .collect();
        arc_spread_chrono.insert(u_idx, by_stage);
    }

    arc_spread_chrono
}

/// Per-declared-arc, per-CHRONOLOGICAL-arrival-stage `A` blend of every
/// contributing source stage's arrival density, resolved in `A`'s own frame
/// (ρ in the methodology): `density_b = Σ_d weight_d·source_density_{d,b} /
/// Σ_d weight_d`, `weight_d` source stage `A-d`'s stage-clock weight and
/// `source_density_d` its lag-`d` density against `A`'s own blocks
/// ([`resolve_arrival_density_at`]) — a parallel source blends exactly like a
/// chronological one. Keyed like [`build_arc_stage_weights`];
/// `by_stage[stage_idx]` is `None` for a `Parallel` stage, or when no in-study
/// source stage reaches it (total weight `== 0`, e.g. the first stage).
pub(crate) fn build_arc_arrival_density(
    system: &System,
    arcs: &[TravelTimeArc],
    calendar: &DeliveryCalendar,
    arc_stage_weights: &HashMap<usize, Vec<Vec<f64>>>,
) -> HashMap<usize, Vec<Option<Vec<f64>>>> {
    let n_stages = calendar.n_study();
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    debug_assert_eq!(study_stages.len(), n_stages);

    let mut arc_arrival_density = HashMap::new();

    for arc in arcs {
        let u_idx = arc.upstream.get();
        let t_v = arc.travel_time_hours;
        let Some(k_by_stage) = arc_stage_weights.get(&u_idx) else {
            continue;
        };
        let extended = extend_for_resolution(calendar.total_hours(), t_v);

        let density_by_stage: Vec<Option<Vec<f64>>> = (0..n_stages)
            .map(|arrival_stage| {
                arrival_frame_density(
                    t_v,
                    arrival_stage,
                    &extended,
                    &study_stages,
                    k_by_stage,
                    u_idx,
                )
            })
            .collect();

        arc_arrival_density.insert(u_idx, density_by_stage);
    }

    arc_arrival_density
}

/// One arc's arrival-frame delivery density at a single arrival stage — the
/// per-stage body [`build_arc_arrival_density`] maps over every study stage.
fn arrival_frame_density(
    t_v: f64,
    arrival_stage: usize,
    extended_calendar: &[f64],
    study_stages: &[&Stage],
    k_by_stage: &[Vec<f64>],
    u_idx: usize,
) -> Option<Vec<f64>> {
    if study_stages[arrival_stage].block_mode != BlockMode::Chronological {
        return None;
    }
    let arrival_blocks: Vec<f64> = study_stages[arrival_stage]
        .blocks
        .iter()
        .map(|b| b.duration_hours)
        .collect();

    let mut weighted_density = vec![0.0_f64; arrival_blocks.len()];
    let mut total_source_weight = 0.0_f64;

    for source_stage in 0..arrival_stage {
        let lag = arrival_stage - source_stage;
        let Some(&source_weight) = k_by_stage.get(source_stage).and_then(|k| k.get(lag)) else {
            continue;
        };
        if source_weight <= 0.0 {
            continue;
        }

        let mut source_density = resolve_arrival_density_at(
            t_v,
            source_stage,
            lag,
            extended_calendar,
            Some(&arrival_blocks),
        );
        debug_assert!(
            source_density.len() <= arrival_blocks.len(),
            "arc {u_idx} arrival stage {arrival_stage}: source_density row must not exceed A's own block count"
        );
        // window_period_overlaps omits trailing zero-overlap blocks (lead_time
        // module doc); pad back to A's own block count so each source_density
        // row aligns with `weighted_density` positionally.
        source_density.resize(arrival_blocks.len(), 0.0);

        for (acc, &density_b) in weighted_density.iter_mut().zip(&source_density) {
            *acc += source_weight * density_b;
        }
        total_source_weight += source_weight;
    }

    if total_source_weight <= 0.0 {
        return None;
    }

    let arrival_density: Vec<f64> = weighted_density
        .iter()
        .map(|&w| w / total_source_weight)
        .collect();
    debug_assert!(
        (arrival_density.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "arc {u_idx} arrival stage {arrival_stage}: arrival_density must conserve to 1.0, got {arrival_density:?}"
    );
    Some(arrival_density)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, Bus, DeficitSegment, Hydro, HydroGenerationModel, HydroPenalties,
        NoiseMethod, PostStudyStage, PostStudyStages, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig, SystemBuilder,
    };

    fn zero_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
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
            inflow_nonnegativity_cost: 0.0,
        }
    }

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn hydro(id: i32, downstream_id: Option<i32>, travel_time_hours: Option<f64>) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: date(2024, 1, 1),
            downstream_id: downstream_id.map(EntityId),
            travel_time_hours,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn stage_with_durations(id: i32, block_hours: &[f64]) -> Stage {
        Stage {
            index: usize::try_from(id).unwrap_or(0),
            id,
            start_date: date(2024, 1, 1),
            end_date: date(2024, 2, 1),
            season_id: None,
            blocks: block_hours
                .iter()
                .enumerate()
                .map(|(i, &h)| Block {
                    index: i,
                    name: format!("B{i}"),
                    duration_hours: h,
                })
                .collect(),
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

    fn chronological_stage_with_durations(id: i32, block_hours: &[f64]) -> Stage {
        Stage {
            block_mode: BlockMode::Chronological,
            ..stage_with_durations(id, block_hours)
        }
    }

    fn stages_with_durations(durations: &[f64]) -> Vec<Stage> {
        durations
            .iter()
            .enumerate()
            .map(|(i, &h)| stage_with_durations(i32::try_from(i).unwrap_or(0), &[h]))
            .collect()
    }

    fn uniform_stages(n: usize, hours: f64) -> Vec<Stage> {
        stages_with_durations(&vec![hours; n])
    }

    fn test_bus() -> Bus {
        Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: date(2024, 1, 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn build_system(hydros: Vec<Hydro>, stages: Vec<Stage>) -> cobre_core::System {
        SystemBuilder::new()
            .buses(vec![test_bus()])
            .hydros(hydros)
            .stages(stages)
            .build()
            .expect("valid system")
    }

    /// [`build_system`] with a declared post-study calendar threaded through
    /// [`SystemBuilder::post_study_stages`], mirroring `policy_export.rs`'s
    /// `system_2h_1ant` fixture shape.
    fn build_system_with_post_study(
        hydros: Vec<Hydro>,
        stages: Vec<Stage>,
        post_study: PostStudyStages,
    ) -> cobre_core::System {
        SystemBuilder::new()
            .buses(vec![test_bus()])
            .hydros(hydros)
            .stages(stages)
            .post_study_stages(Some(post_study))
            .build()
            .expect("valid system")
    }

    /// A post-study calendar with one stage per `hours` entry, each starting
    /// a calendar month apart.
    fn post_study_stages_hours(hours: &[f64]) -> PostStudyStages {
        PostStudyStages {
            stages: hours
                .iter()
                .enumerate()
                .map(|(i, &duration_hours)| PostStudyStage {
                    start_date: date(2024, 4 + u32::try_from(i).unwrap_or(0), 1),
                    duration_hours,
                })
                .collect(),
            thermal_bounds: Vec::new(),
        }
    }

    #[test]
    fn test_n_buckets_zero_when_no_arc_declared() {
        let downstream = hydro(1, None, None);
        let system = build_system(vec![downstream], uniform_stages(3, 24.0));

        let calendar = DeliveryCalendar::from_system(&system);
        let topology = build_transit_bucket_topology(&system, &calendar, false);

        assert_eq!(topology.n_buckets(), 0);
        assert!(topology.column_order.is_empty());
    }

    /// The first arc has the longer travel time, so a sort by travel time
    /// fails this test.
    #[test]
    fn travel_time_arcs_skip_undeclared_and_keep_hydro_order() {
        let system = build_system(
            vec![
                hydro(1, None, None),
                hydro(2, Some(1), Some(100.0)),
                hydro(3, Some(1), Some(24.0)),
                hydro(4, Some(1), Some(0.0)),
                hydro(5, Some(1), None),
                hydro(6, None, Some(48.0)),
            ],
            uniform_stages(3, 24.0),
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let topology = build_transit_bucket_topology(&system, &calendar, false);

        assert_eq!(
            topology.arcs(),
            &[
                TravelTimeArc {
                    upstream: HydroSys::new(1),
                    downstream: HydroSys::new(0),
                    travel_time_hours: 100.0,
                },
                TravelTimeArc {
                    upstream: HydroSys::new(2),
                    downstream: HydroSys::new(0),
                    travel_time_hours: 24.0,
                },
            ][..]
        );
    }

    #[test]
    fn test_zero_travel_time_is_treated_as_undeclared() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(0.0));
        let system = build_system(vec![downstream, upstream], uniform_stages(3, 24.0));

        let calendar = DeliveryCalendar::from_system(&system);
        let topology = build_transit_bucket_topology(&system, &calendar, false);

        assert_eq!(topology.n_buckets(), 0);
    }

    #[test]
    fn test_confluence_aggregates_to_single_block_of_max_depth() {
        let downstream = hydro(1, None, None);
        let upstream_a = hydro(2, Some(1), Some(24.0));
        let upstream_b = hydro(3, Some(1), Some(100.0));
        let system = build_system(
            vec![downstream, upstream_a, upstream_b],
            uniform_stages(10, 24.0),
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let topology = build_transit_bucket_topology(&system, &calendar, false);

        assert_eq!(topology.n_buckets(), 5);
        assert_eq!(
            topology.column_order,
            vec![
                (HydroSys::new(0), 1),
                (HydroSys::new(0), 2),
                (HydroSys::new(0), 3),
                (HydroSys::new(0), 4),
                (HydroSys::new(0), 5)
            ]
        );
    }

    #[test]
    fn test_fine_first_coarse_next_ic_anchor_deepens_beyond_in_study_max() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(30.0));
        let durations = [24.0, 720.0, 720.0, 720.0];
        let system = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let extended = extend_for_resolution(&durations, 30.0);
        let in_study_max = (0..durations.len())
            .map(|t| in_study_depth(30.0, t, &extended))
            .max()
            .unwrap_or(0);
        let ic_depth = ic_only_depth(30.0, &durations);
        assert_eq!(
            in_study_max, 1,
            "every in-study anchor must give L_arc == 1"
        );
        assert_eq!(ic_depth, 2, "the IC anchor must give L_arc(IC) == 2");

        let calendar = DeliveryCalendar::from_system(&system);
        let topology = build_transit_bucket_topology(&system, &calendar, false);

        assert_eq!(topology.n_buckets(), 2);

        // The stage-0 mask reaches the IC-residual slot 2 (decaying reachability,
        // not a zero-deposit filter); it narrows to the own-release depth once the
        // residual has drained.
        assert_eq!(topology.per_stage_mask.len(), durations.len());
        assert_eq!(topology.per_stage_mask[0], vec![2]);
        assert_eq!(topology.per_stage_mask[1], vec![1]);
        assert_eq!(topology.per_stage_mask[2], vec![1]);
    }

    #[test]
    fn test_uniform_calendar_ic_anchor_does_not_deepen() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let durations = vec![24.0; 6];
        let system = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let extended = extend_for_resolution(&durations, 24.0);
        let in_study_max = (0..durations.len())
            .map(|t| in_study_depth(24.0, t, &extended))
            .max()
            .unwrap_or(0);
        let ic_depth = ic_only_depth(24.0, &durations);
        assert_eq!(ic_depth, in_study_max, "uniform calendar: no IC deepening");

        let calendar = DeliveryCalendar::from_system(&system);
        let topology = build_transit_bucket_topology(&system, &calendar, false);

        assert_eq!(topology.n_buckets(), in_study_max);
    }

    #[test]
    fn test_horizon_cap_drops_lag_targeting_past_last_stage() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let durations = [24.0, 24.0, 24.0];
        let system = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let extended = extend_for_resolution(&durations, 72.0);
        let uncapped_active_by_stage: Vec<usize> = (0..durations.len())
            .map(|t| in_study_depth(72.0, t, &extended))
            .collect();
        let ic_depth = ic_only_depth(72.0, &durations);
        assert_eq!(
            uncapped_active_by_stage,
            vec![3, 3, 3],
            "every anchor's own-release depth must reach 3 stages ahead, past the 3-stage horizon"
        );
        assert_eq!(ic_depth, 3, "the IC anchor must also reach 3 stages ahead");

        let calendar = DeliveryCalendar::from_system(&system);
        let topology = build_transit_bucket_topology(&system, &calendar, false);

        assert_eq!(
            topology.n_buckets(),
            3,
            "global depth sizing is unaffected by the per-stage horizon cap"
        );
        assert_eq!(
            topology.column_order,
            vec![
                (HydroSys::new(0), 1),
                (HydroSys::new(0), 2),
                (HydroSys::new(0), 3)
            ]
        );

        assert_eq!(topology.per_stage_mask[0], vec![2], "cap = 3 - 1 - 0 = 2");
        assert_eq!(topology.per_stage_mask[1], vec![1], "cap = 3 - 1 - 1 = 1");
        assert_eq!(
            topology.per_stage_mask[2],
            vec![0],
            "cap = 3 - 1 - 2 = 0: the last stage targets nothing past T"
        );

        for (stage, mask_row) in topology.per_stage_mask.iter().enumerate() {
            for &max_lag in mask_row {
                assert!(
                    stage + max_lag < durations.len(),
                    "stage {stage} lag {max_lag} must not target a stage at or past n_stages"
                );
            }
        }
    }

    /// `boundary_present = true` un-caps every stage's mask to the raw
    /// `uncapped_active_by_stage` value (including the terminal stage), while
    /// sizing (`column_order`/`n_buckets`) stays identical to the gated-off
    /// build — the keep-live contract touches only the mask.
    #[test]
    fn test_boundary_present_uncaps_terminal_deep_lag_mask() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let durations = [24.0, 24.0, 24.0];
        let system = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let topology_off = build_transit_bucket_topology(&system, &calendar, false);
        let topology_on = build_transit_bucket_topology(&system, &calendar, true);

        assert_eq!(
            topology_on.column_order, topology_off.column_order,
            "un-capping the mask must not change the canonical column order"
        );
        assert_eq!(
            topology_on.n_buckets(),
            topology_off.n_buckets(),
            "un-capping the mask must not change the global bucket count"
        );

        assert_eq!(
            topology_off.per_stage_mask,
            vec![vec![2], vec![1], vec![0]],
            "gated-off mask must stay exactly the horizon-capped sequence"
        );
        assert_eq!(
            topology_on.per_stage_mask,
            vec![vec![3], vec![3], vec![3]],
            "boundary-present mask must reach the raw uncapped active lag at every \
             stage, terminal included"
        );
    }

    /// Pins the confinement `entries.rs`'s two release-site `debug_assert!`s rely
    /// on: under `boundary_present`, `per_stage_mask` dominates every arc's deposit
    /// depth on both the parallel (`arc_stage_weights`) and chronological
    /// (`arc_spread_chrono`) tables at every stage, so neither release site's row
    /// lookup can miss; with the boundary gate off, the fixture still has power —
    /// the horizon cap strictly undercuts that same depth at every stage.
    #[test]
    fn transit_bucket_mask_covers_every_arc_deposit_depth_under_boundary() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let system = build_system(
            vec![downstream, upstream],
            vec![
                chronological_stage_with_durations(0, &[12.0, 12.0]),
                stage_with_durations(1, &[24.0]),
                stage_with_durations(2, &[24.0]),
            ],
        );

        let upstream_idx = 1;
        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());
        let topology_on = build_transit_bucket_topology(&system, &calendar, true);
        let topology_off = build_transit_bucket_topology(&system, &calendar, false);
        let arc_stage_weights = build_arc_stage_weights(&arcs, &calendar);
        let arc_spread_chrono = build_arc_spread_chrono(&system, &arcs, &calendar);

        let k_by_stage = arc_stage_weights
            .get(&upstream_idx)
            .expect("declared arc must have a stage-weights entry");
        let by_stage_chrono = arc_spread_chrono
            .get(&upstream_idx)
            .expect("declared arc must have a chronological-spread entry");
        assert!(
            by_stage_chrono.iter().any(Option::is_some),
            "fixture must exercise a chronological stage so arc_spread_chrono is populated"
        );

        for (stage, mask_row) in topology_on.per_stage_mask.iter().enumerate() {
            let mask = mask_row[0];
            let parallel_depth = k_by_stage[stage].len() - 1;
            assert!(
                mask >= parallel_depth,
                "stage {stage}: boundary-present mask {mask} must dominate the parallel \
                 deposit depth {parallel_depth}"
            );
            if let Some(resolution) = &by_stage_chrono[stage] {
                for (b, deposit_row) in resolution.block_deposits.iter().enumerate() {
                    let block_depth = deposit_row.len() - 1;
                    assert!(
                        mask >= block_depth,
                        "stage {stage} block {b}: boundary-present mask {mask} must dominate \
                         the chronological deposit depth {block_depth}"
                    );
                }
            }
        }

        let undercut_stage_exists = topology_off
            .per_stage_mask
            .iter()
            .enumerate()
            .any(|(stage, mask_row)| mask_row[0] < k_by_stage[stage].len() - 1);
        assert!(
            undercut_stage_exists,
            "the fixture must have power: with no boundary, at least one stage's mask must be \
             strictly less than the arc's deposit depth"
        );
    }

    #[test]
    fn test_column_order_is_declaration_order_invariant() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(48.0));

        let system_a = build_system(
            vec![downstream.clone(), upstream.clone()],
            uniform_stages(5, 24.0),
        );
        let system_b = build_system(vec![upstream, downstream], uniform_stages(5, 24.0));

        let calendar_a = DeliveryCalendar::from_system(&system_a);
        let calendar_b = DeliveryCalendar::from_system(&system_b);
        let topology_a = build_transit_bucket_topology(&system_a, &calendar_a, false);
        let topology_b = build_transit_bucket_topology(&system_b, &calendar_b, false);

        assert_eq!(topology_a.column_order, topology_b.column_order);
        assert_eq!(topology_a.n_buckets(), topology_b.n_buckets());
    }

    #[test]
    fn test_build_arc_stage_weights_empty_when_no_arc_declared() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), None);
        let system = build_system(vec![downstream, upstream], uniform_stages(3, 24.0));
        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());

        let arc_stage_weights = build_arc_stage_weights(&arcs, &calendar);

        assert!(arc_stage_weights.is_empty());
    }

    #[test]
    fn test_build_arc_stage_weights_conserves_and_matches_topology_depth() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(24.0));
        let system = build_system(vec![downstream, upstream], uniform_stages(10, 24.0));

        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());
        let topology = build_transit_bucket_topology(&system, &calendar, false);
        let arc_stage_weights = build_arc_stage_weights(&arcs, &calendar);

        let upstream_idx = 1;
        let k_by_stage = arc_stage_weights
            .get(&upstream_idx)
            .expect("declared arc must have an entry");
        assert_eq!(k_by_stage.len(), 10, "one k vector per in-study stage");
        for k in k_by_stage {
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "k must conserve to 1.0, got {k:?}"
            );
        }
        let max_depth = k_by_stage.iter().map(|k| k.len() - 1).max().unwrap_or(0);
        assert_eq!(
            max_depth,
            topology.n_buckets(),
            "the deepest in-study k vector must match the topology's per-plant depth"
        );
    }

    #[test]
    fn test_build_arc_spread_chrono_gates_on_stage_block_mode() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(250.0));
        let system = build_system(
            vec![downstream, upstream],
            vec![
                chronological_stage_with_durations(0, &[240.0, 240.0, 240.0]),
                stage_with_durations(1, &[720.0]),
            ],
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());
        let arc_spread_chrono = build_arc_spread_chrono(&system, &arcs, &calendar);
        let upstream_idx = 1;
        let by_stage = arc_spread_chrono
            .get(&upstream_idx)
            .expect("declared arc must have an entry");

        assert!(
            by_stage[0].is_some(),
            "chronological stage 0 must resolve block_deposits/within_stage_routing/arrival_density"
        );
        assert!(
            by_stage[1].is_none(),
            "parallel stage 1 has no block-resolved routing to compute"
        );

        let resolution = by_stage[0].as_ref().expect("checked above");
        assert_eq!(
            resolution.within_stage_routing.len(),
            3,
            "one within_stage_routing row per block"
        );
        assert_eq!(
            resolution.block_deposits.len(),
            3,
            "one block_deposits row per block"
        );
        for (b, deposit_b) in resolution.block_deposits.iter().enumerate() {
            let routing_sum: f64 = resolution.within_stage_routing[b].iter().sum();
            let deposit_cross: f64 = deposit_b[1..].iter().sum();
            assert!(
                (routing_sum + deposit_cross - 1.0).abs() < 1e-9,
                "block {b}: per-column conservation must hold"
            );
        }
    }

    /// Two weekly (168h) parallel sources both reach one 720h chronological
    /// arrival stage (blocks `[20, 100, 600]`): source stage 1 delivers the
    /// whole window at lag 1 (weight `1.0`, density `[0, 88/168, 80/168]`),
    /// source stage 0 delivers a residual tail at lag 2 (weight `32/168`,
    /// density `[20/32, 12/32, 0]`). The hand-derived blend
    /// `density_b = sum_d weight_d*source_density_{d,b} / sum_d weight_d` is
    /// `[0.1, 0.5, 0.4]`.
    #[test]
    fn test_build_arc_arrival_density_multi_lag_blend_matches_hand_derived() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(200.0));
        let system = build_system(
            vec![downstream, upstream],
            vec![
                stage_with_durations(0, &[168.0]),
                stage_with_durations(1, &[168.0]),
                chronological_stage_with_durations(2, &[20.0, 100.0, 600.0]),
            ],
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());
        let arc_stage_weights = build_arc_stage_weights(&arcs, &calendar);
        let arc_arrival_density =
            build_arc_arrival_density(&system, &arcs, &calendar, &arc_stage_weights);

        let upstream_idx = 1;
        let density_by_stage = arc_arrival_density
            .get(&upstream_idx)
            .expect("declared arc must have an entry");
        assert_eq!(density_by_stage.len(), 3, "one entry per in-study stage");
        assert!(
            density_by_stage[0].is_none(),
            "no in-study source stage precedes stage 0"
        );
        assert!(
            density_by_stage[1].is_none(),
            "stage 1 is itself Parallel, no density to resolve"
        );

        let arrival_density = density_by_stage[2]
            .as_ref()
            .expect("both source stages reach the chronological arrival stage");
        let expected = [0.1, 0.5, 0.4];
        for (b, (&got, &want)) in arrival_density.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "block {b}: got {got}, want {want}"
            );
        }
    }

    /// A single Parallel source stage feeds one chronological arrival stage:
    /// the stored arrival density is the arrival-frame delivery split
    /// (`[0.8, 0.2]`), not the duration-weighted uniform fallback
    /// (`[0.4, 0.6]`).
    #[test]
    fn test_build_arc_arrival_density_parallel_sender_is_not_duration_uniform() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(50.0));
        let system = build_system(
            vec![downstream, upstream],
            vec![
                stage_with_durations(0, &[100.0]),
                chronological_stage_with_durations(1, &[40.0, 60.0]),
            ],
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());
        let arc_stage_weights = build_arc_stage_weights(&arcs, &calendar);
        let arc_arrival_density =
            build_arc_arrival_density(&system, &arcs, &calendar, &arc_stage_weights);

        let upstream_idx = 1;
        let density_by_stage = arc_arrival_density
            .get(&upstream_idx)
            .expect("declared arc must have an entry");
        let arrival_density = density_by_stage[1]
            .as_ref()
            .expect("the parallel source reaches the chronological arrival stage");

        let expected = [0.8, 0.2];
        for (b, (&got, &want)) in arrival_density.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "block {b}: got {got}, want {want}"
            );
        }

        let duration_uniform: f64 = 40.0 / 100.0;
        assert!(
            (arrival_density[0] - duration_uniform).abs() > 1e-6,
            "arrival density must be the arrival-frame blend, not the duration-weighted uniform fallback"
        );
    }

    #[test]
    fn test_build_arc_arrival_density_conserves_across_every_chronological_stage() {
        let downstream = hydro(1, None, None);
        let upstream_a = hydro(2, Some(1), Some(90.0));
        let upstream_b = hydro(3, Some(1), Some(250.0));
        let system = build_system(
            vec![downstream, upstream_a, upstream_b],
            vec![
                chronological_stage_with_durations(0, &[60.0, 40.0]),
                stage_with_durations(1, &[100.0]),
                chronological_stage_with_durations(2, &[100.0, 100.0, 100.0]),
                chronological_stage_with_durations(3, &[150.0]),
            ],
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());
        let arc_stage_weights = build_arc_stage_weights(&arcs, &calendar);
        let arc_arrival_density =
            build_arc_arrival_density(&system, &arcs, &calendar, &arc_stage_weights);

        let mut n_checked = 0;
        for density_by_stage in arc_arrival_density.values() {
            for arrival_density in density_by_stage.iter().flatten() {
                let sum: f64 = arrival_density.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-9,
                    "arrival_density must conserve to 1.0, got {arrival_density:?}"
                );
                n_checked += 1;
            }
        }
        assert!(
            n_checked > 0,
            "the system must exercise at least one resolved arrival_density vector"
        );
    }

    #[test]
    fn test_build_arc_arrival_density_is_declaration_order_invariant() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(200.0));
        let stages = vec![
            stage_with_durations(0, &[168.0]),
            stage_with_durations(1, &[168.0]),
            chronological_stage_with_durations(2, &[20.0, 100.0, 600.0]),
        ];

        let system_a = build_system(vec![downstream.clone(), upstream.clone()], stages.clone());
        let system_b = build_system(vec![upstream, downstream], stages);
        let calendar_a = DeliveryCalendar::from_system(&system_a);
        let calendar_b = DeliveryCalendar::from_system(&system_b);
        let arcs_a = resolve_travel_time_arcs(system_a.hydros());
        let arcs_b = resolve_travel_time_arcs(system_b.hydros());

        let density_a = build_arc_arrival_density(
            &system_a,
            &arcs_a,
            &calendar_a,
            &build_arc_stage_weights(&arcs_a, &calendar_a),
        );
        let density_b = build_arc_arrival_density(
            &system_b,
            &arcs_b,
            &calendar_b,
            &build_arc_stage_weights(&arcs_b, &calendar_b),
        );

        assert_eq!(density_a, density_b);
    }

    /// Hand-derived against a 72h arc, 3 x 24h study stages, and a declared
    /// post-study calendar of 2 x 12h stages: at the terminal study stage
    /// (anchor 2, `h_anchor = 24`), the arrival window `[72, 96)` falls
    /// entirely past the 2 x 12h post-study calendar (ending at `+24`) and
    /// the 6 x 12h pad beyond it, landing on pad copies 3 and 4
    /// (`k_5 = k_6 = 0.5`), so `stage_reach = 6`. The pad-only calendar
    /// (`extend_for_resolution` on the study-only vector) instead replicates
    /// 24h stages, whose window `[72, 96)` lands squarely on a single 24h pad
    /// copy (`stage_reach = 3`) — the value `test_horizon_cap_drops_lag_
    /// targeting_past_last_stage` pins for the same arc with no post-study
    /// calendar declared.
    #[test]
    fn post_study_calendar_replaces_the_synthetic_pad_in_arrival_resolution() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let durations = [24.0, 24.0, 24.0];

        let system_with_post_study = build_system_with_post_study(
            vec![downstream.clone(), upstream.clone()],
            stages_with_durations(&durations),
            post_study_stages_hours(&[12.0, 12.0]),
        );
        let system_pad_only = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let calendar_with_post_study = DeliveryCalendar::from_system(&system_with_post_study);
        let calendar_pad_only = DeliveryCalendar::from_system(&system_pad_only);
        let topology =
            build_transit_bucket_topology(&system_with_post_study, &calendar_with_post_study, true);
        let topology_pad_only =
            build_transit_bucket_topology(&system_pad_only, &calendar_pad_only, true);

        assert_eq!(
            topology.n_buckets(),
            6,
            "depth must reach the hand-derived 12-hour-stage value"
        );
        assert_eq!(
            topology.per_stage_mask[2],
            vec![6],
            "the terminal-stage mask must reach the hand-derived 12-hour-stage value"
        );
        assert_ne!(
            topology.n_buckets(),
            topology_pad_only.n_buckets(),
            "the real post-study calendar must size differently than the replicated pad"
        );
        assert_ne!(
            topology.per_stage_mask[2], topology_pad_only.per_stage_mask[2],
            "the real post-study calendar must resolve a different terminal mask than the replicated pad"
        );
    }

    #[test]
    fn post_study_calendar_shorter_than_travel_time_still_pads() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let durations = [24.0, 24.0, 24.0];
        let system = build_system_with_post_study(
            vec![downstream, upstream],
            stages_with_durations(&durations),
            post_study_stages_hours(&[24.0]),
        );

        let calendar = DeliveryCalendar::from_system(&system);
        let arcs = resolve_travel_time_arcs(system.hydros());
        let arc_stage_weights = build_arc_stage_weights(&arcs, &calendar);
        let upstream_idx = 1;
        let k_by_stage = arc_stage_weights
            .get(&upstream_idx)
            .expect("declared arc must have an entry");
        assert_eq!(k_by_stage.len(), 3, "one k vector per in-study stage");
        for k in k_by_stage {
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "k must conserve to 1.0 even though the declared post-study calendar (24h) is \
                 shorter than the travel time (72h), got {k:?}"
            );
        }
    }

    #[test]
    fn extended_calendar_topology_is_declaration_order_invariant() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(48.0));
        let post_study = post_study_stages_hours(&[12.0, 12.0]);

        let system_a = build_system_with_post_study(
            vec![downstream.clone(), upstream.clone()],
            uniform_stages(5, 24.0),
            post_study.clone(),
        );
        let system_b = build_system_with_post_study(
            vec![upstream, downstream],
            uniform_stages(5, 24.0),
            post_study,
        );

        let calendar_a = DeliveryCalendar::from_system(&system_a);
        let calendar_b = DeliveryCalendar::from_system(&system_b);
        let topology_a = build_transit_bucket_topology(&system_a, &calendar_a, false);
        let topology_b = build_transit_bucket_topology(&system_b, &calendar_b, false);

        assert_eq!(topology_a.column_order, topology_b.column_order);
        assert_eq!(topology_a.n_buckets(), topology_b.n_buckets());
    }

    /// A declared post-study calendar whose stage durations exactly match the
    /// synthetic pad (the last study stage's own duration) leaves
    /// `extend_for_resolution`'s output unchanged, so every
    /// [`TransitBucketTopology`] field and all three arc tables must equal
    /// the same system built with no post-study calendar at all.
    #[test]
    fn post_study_calendar_matching_the_pad_is_topology_neutral() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let durations = [24.0, 24.0, 24.0];

        let system_with_calendar = build_system_with_post_study(
            vec![downstream.clone(), upstream.clone()],
            stages_with_durations(&durations),
            post_study_stages_hours(&[24.0, 24.0]),
        );
        let system_no_calendar = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let calendar_with_calendar = DeliveryCalendar::from_system(&system_with_calendar);
        let calendar_no_calendar = DeliveryCalendar::from_system(&system_no_calendar);
        let topology_with_calendar =
            build_transit_bucket_topology(&system_with_calendar, &calendar_with_calendar, false);
        let topology_no_calendar =
            build_transit_bucket_topology(&system_no_calendar, &calendar_no_calendar, false);

        assert!(
            topology_no_calendar.n_buckets() > 0,
            "fixture has no power unless it declares at least one travel-time bucket"
        );
        assert_eq!(
            topology_with_calendar.n_buckets(),
            topology_no_calendar.n_buckets()
        );
        assert_eq!(
            topology_with_calendar.column_order,
            topology_no_calendar.column_order
        );
        assert_eq!(
            topology_with_calendar.per_stage_mask,
            topology_no_calendar.per_stage_mask
        );
        assert_eq!(
            topology_with_calendar.arc_stage_weights,
            topology_no_calendar.arc_stage_weights
        );
        assert_eq!(
            topology_with_calendar.arc_spread_chrono,
            topology_no_calendar.arc_spread_chrono
        );
        assert_eq!(
            topology_with_calendar.arc_arrival_density,
            topology_no_calendar.arc_arrival_density
        );
    }

    /// The power check for the neutral test above: a post-study calendar
    /// whose duration DIFFERS from the pad must change the resolved
    /// topology, or the neutral comparison would pass on a no-op
    /// implementation.
    #[test]
    fn post_study_calendar_differing_from_the_pad_changes_the_topology() {
        let downstream = hydro(1, None, None);
        let upstream = hydro(2, Some(1), Some(72.0));
        let durations = [24.0, 24.0, 24.0];

        let system_with_calendar = build_system_with_post_study(
            vec![downstream.clone(), upstream.clone()],
            stages_with_durations(&durations),
            post_study_stages_hours(&[6.0, 6.0]),
        );
        let system_no_calendar = build_system(
            vec![downstream, upstream],
            stages_with_durations(&durations),
        );

        let calendar_with_calendar = DeliveryCalendar::from_system(&system_with_calendar);
        let calendar_no_calendar = DeliveryCalendar::from_system(&system_no_calendar);
        let topology_with_calendar =
            build_transit_bucket_topology(&system_with_calendar, &calendar_with_calendar, false);
        let topology_no_calendar =
            build_transit_bucket_topology(&system_no_calendar, &calendar_no_calendar, false);

        assert!(
            topology_no_calendar.n_buckets() > 0,
            "fixture has no power unless it declares at least one travel-time bucket"
        );
        assert!(
            topology_with_calendar.per_stage_mask != topology_no_calendar.per_stage_mask
                || topology_with_calendar.column_order != topology_no_calendar.column_order,
            "a post-study calendar differing from the pad must change per_stage_mask or \
             column_order, or the neutral test above has no power"
        );
    }
}
