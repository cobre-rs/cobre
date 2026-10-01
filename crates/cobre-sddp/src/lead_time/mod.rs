//! Resolution of calendar-anchored lags into per-stage and per-block
//! factors, built on `cobre_core`'s [`window_period_overlaps`]
//! interval-overlap primitive. Two entry points share the overlap engine:
//! [`resolve_spread`] resolves a spreadable quantity's stage-clock and
//! block-clock weights; [`resolve_point`] resolves a point commitment
//! (anticipated dispatch) into a single decider per delivery stage.
//!
//! [`resolve_spread`] is the sole resolver dependency for the water
//! travel-time feature. Every factor it returns — stage-clock
//! `stage_weights` (the k-factors), block-resolved
//! `block_deposits`/`within_stage_routing` (the χ/κ factors), and
//! per-arrival-stage `arrival_density` (the delivery density) — overlaps
//! `arrival_window`'s ONE shared uniform arrival density against nested
//! calendar partitions (stage clock ⊃ block clock), so the shared-density
//! aggregation identity `Σ_b w_b·χ_{b,d} == k_d` holds by construction
//! rather than by convention.

use cobre_core::window_period_overlaps;

/// Resolved spread of one arc's travel-time arrival density, anchored at a
/// single stage `t`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpreadResolution {
    /// Deepest future stage the arrival window reaches (the depth) — the max
    /// index reached, never a count (`window_period_overlaps`'s contiguity
    /// contract).
    pub stage_reach: usize,
    /// Stage-clock weight (the k-factor) `k_d` for `d = 0..=stage_reach`;
    /// `k_0` is the same-stage share and `Σ_d k_d == 1`.
    pub stage_weights: Vec<f64>,
    /// Per-source-block deposit (the χ deposits), `block_deposits[b][d]`:
    /// index `0` is block `b`'s same-stage retained share, `1..=stage_reach`
    /// are its bucket-`d` deposits. Empty when the anchor has no block
    /// partition.
    pub block_deposits: Vec<Vec<f64>>,
    /// Per-source-block within-stage routing (the κ routing),
    /// `within_stage_routing[b][j]` to target block `b + j` (`j == 0` is
    /// block `b` routing to itself). Empty when the anchor has no block
    /// partition.
    pub within_stage_routing: Vec<Vec<f64>>,
    /// Per-arrival-stage arrival density (the delivery density) for lag
    /// `d = 1..=stage_reach` at index `d - 1`; an empty row where `k_d == 0`
    /// (nothing arrives that lag).
    pub arrival_density: Vec<Vec<f64>>,
}

/// The shared arrival window: a uniform release over the anchor stage
/// `[0, h_anchor)` delayed by `travel_time_hours`.
fn arrival_window(travel_time_hours: f64, h_anchor: f64) -> (f64, f64) {
    (travel_time_hours, travel_time_hours + h_anchor)
}

/// Resolve a scalar travel time into stage-clock weights and, for a
/// chronological anchor, block-resolved deposit/routing/delivery factors.
///
/// `block_lengths_hours` is the anchor's own block partition (`None` for a
/// parallel anchor) and must sum to `stage_lengths_hours[anchor_stage]`.
/// Delivery reuses this partition at a reached arrival stage only when that
/// stage's own length matches the anchor's; a mismatch falls back to a single
/// row rather than splitting against a partition that would not conserve the
/// target stage's own mass (`block_deposits`/`within_stage_routing`/
/// `stage_weights` are unaffected).
///
/// # Panics
///
/// Debug builds panic if `anchor_stage` is out of bounds, if
/// `travel_time_hours` is not finite and positive, or if the
/// conservation / aggregation-consistency identities do not hold.
#[must_use]
pub fn resolve_spread(
    travel_time_hours: f64,
    anchor_stage: usize,
    stage_lengths_hours: &[f64],
    block_lengths_hours: Option<&[f64]>,
) -> SpreadResolution {
    debug_assert!(
        travel_time_hours.is_finite() && travel_time_hours > 0.0,
        "travel_time_hours must be finite and > 0.0"
    );
    debug_assert!(
        anchor_stage < stage_lengths_hours.len(),
        "anchor_stage must index into stage_lengths_hours"
    );

    let future_calendar = &stage_lengths_hours[anchor_stage..];
    let h_anchor = future_calendar[0];
    let (window_start, window_end) = arrival_window(travel_time_hours, h_anchor);

    let stage_weight_overlaps = window_period_overlaps(window_start, h_anchor, future_calendar);
    let stage_weights: Vec<f64> = stage_weight_overlaps
        .iter()
        .map(|&overlap| overlap / h_anchor)
        .collect();
    let stage_reach = stage_weights.len().saturating_sub(1);

    debug_assert!(
        (stage_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "stage-clock weights must sum to 1.0 (conservation)"
    );

    let blocks = block_lengths_hours.filter(|blocks| !blocks.is_empty());
    if let Some(blocks) = blocks {
        debug_assert!(
            (blocks.iter().sum::<f64>() - h_anchor).abs() < 1e-9,
            "block_lengths_hours must sum to the anchor stage length"
        );
    }

    let BlockFactors {
        deposits: block_deposits,
        routing: within_stage_routing,
    } = blocks.map_or_else(
        || BlockFactors {
            deposits: Vec::new(),
            routing: Vec::new(),
        },
        |blocks| resolve_block_factors(travel_time_hours, blocks, future_calendar, stage_reach),
    );

    if let Some(blocks) = blocks {
        for (d, &stage_weight) in stage_weights.iter().enumerate() {
            let aggregated: f64 = block_deposits
                .iter()
                .zip(blocks)
                .map(|(row, &duration_b)| (duration_b / h_anchor) * row[d])
                .sum();
            debug_assert!(
                (aggregated - stage_weight).abs() < 1e-9,
                "block deposits must aggregate to the stage-level k_d (shared-density aggregation identity)"
            );
        }
    }

    let arrival_blocks: Vec<Option<&[f64]>> = (1..=stage_reach)
        .map(|d| blocks.filter(|_| (future_calendar[d] - h_anchor).abs() < 1e-9))
        .collect();
    let arrival_density = resolve_delivery(
        window_start,
        window_end,
        future_calendar,
        &arrival_blocks,
        stage_reach,
    );

    for (d, row) in arrival_density.iter().enumerate() {
        if stage_weights[d + 1] > 0.0 {
            debug_assert!(
                (row.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "arrival density row must sum to 1.0 where k_d > 0 (delivery conservation)"
            );
        }
    }

    SpreadResolution {
        stage_reach,
        stage_weights,
        block_deposits,
        within_stage_routing,
        arrival_density,
    }
}

/// Per-source-block deposit/routing factors resolved by [`resolve_block_factors`].
///
/// Two same-typed `Vec<Vec<f64>>` fields — kept as a named struct (rather
/// than a bare tuple) so a positional swap of `deposits`/`routing` at the
/// call site is a compile error, not a silent mix-up.
#[derive(Debug, Clone)]
struct BlockFactors {
    /// Per-source-block deposit (the `chi` factor); see
    /// [`SpreadResolution::block_deposits`].
    deposits: Vec<Vec<f64>>,
    /// Per-source-block within-stage routing (the `kappa` factor);
    /// see [`SpreadResolution::within_stage_routing`].
    routing: Vec<Vec<f64>>,
}

/// Resolve `BlockFactors` (the χ deposits and κ routing): reads the same
/// arrival window as the stage-clock weights, restricted to each block's own
/// local origin, so the aggregation identity in [`resolve_spread`] holds
/// structurally rather than needing a separate density.
fn resolve_block_factors(
    travel_time_hours: f64,
    blocks: &[f64],
    future_calendar: &[f64],
    stage_reach: usize,
) -> BlockFactors {
    let n_blocks = blocks.len();
    let mut deposits = vec![vec![0.0_f64; stage_reach + 1]; n_blocks];
    let mut routing = Vec::with_capacity(n_blocks);

    for (b, &duration_b) in blocks.iter().enumerate() {
        let mut target = Vec::with_capacity(n_blocks - b + future_calendar.len() - 1);
        target.extend_from_slice(&blocks[b..]);
        target.extend_from_slice(&future_calendar[1..]);

        let combined = window_period_overlaps(travel_time_hours, duration_b, &target);
        let routing_len = (n_blocks - b).min(combined.len());

        let mut routing_b: Vec<f64> = combined[..routing_len]
            .iter()
            .map(|&overlap| overlap / duration_b)
            .collect();
        routing_b.resize(n_blocks - b, 0.0);

        deposits[b][0] = routing_b.iter().sum();
        for (offset, &overlap) in combined[routing_len..].iter().enumerate() {
            let d = offset + 1;
            if d <= stage_reach {
                deposits[b][d] = overlap / duration_b;
            }
        }

        routing.push(routing_b);
    }

    BlockFactors { deposits, routing }
}

/// Per-arrival-stage arrival density for lag `d = 1..=stage_reach`: the same
/// `[window_start, window_end)` window restricted to arrival stage `t+d`'s
/// own local clock and split across `arrival_blocks[d - 1]`, that stage's own
/// partition (`None` delivers onto a single parallel row).
fn resolve_delivery(
    window_start: f64,
    window_end: f64,
    future_calendar: &[f64],
    arrival_blocks: &[Option<&[f64]>],
    stage_reach: usize,
) -> Vec<Vec<f64>> {
    debug_assert_eq!(
        arrival_blocks.len(),
        stage_reach,
        "arrival_blocks must supply one entry per reached arrival stage"
    );

    let mut arrival_density = Vec::with_capacity(stage_reach);
    let mut stage_start = future_calendar[0];

    for (d, &stage_len) in future_calendar[1..=stage_reach].iter().enumerate() {
        let stage_end = stage_start + stage_len;
        let blocks = arrival_blocks[d];
        if let Some(partition) = blocks {
            debug_assert!(
                (partition.iter().sum::<f64>() - stage_len).abs() < 1e-9,
                "an arrival-stage block partition must sum to that stage's own length"
            );
        }

        let overlap_start = window_start.max(stage_start);
        let overlap_end = window_end.min(stage_end);
        let width = (overlap_end - overlap_start).max(0.0);

        let row = if width > 0.0 {
            let local_start = overlap_start - stage_start;
            blocks.map_or_else(
                || vec![1.0],
                |partition| {
                    window_period_overlaps(local_start, width, partition)
                        .iter()
                        .map(|&overlap| overlap / width)
                        .collect()
                },
            )
        } else {
            Vec::new()
        };
        arrival_density.push(row);

        stage_start = stage_end;
    }

    arrival_density
}

/// Lag-`d` arrival density for a single (source stage, target arrival stage)
/// pair, resolved against an EXPLICIT `target_blocks` partition rather than
/// the anchor's own — the arrival-frame counterpart of [`resolve_spread`],
/// which instead reuses the anchor's own partition at every reached lag.
/// Lets a caller blend several source stages' deliveries into one arrival
/// stage's own frame regardless of each source's own block mode
/// (`crate::bucket_topology::build_arc_arrival_density`). `target_blocks` is
/// `None` for a parallel target (the single `1.0` row).
///
/// # Panics
///
/// Debug builds panic if `lag == 0` or `source_stage + lag` is out of bounds.
#[must_use]
pub(crate) fn resolve_arrival_density_at(
    travel_time_hours: f64,
    source_stage: usize,
    lag: usize,
    stage_lengths_hours: &[f64],
    target_blocks: Option<&[f64]>,
) -> Vec<f64> {
    debug_assert!(lag >= 1, "lag must be >= 1: d=0 is same-stage, no bucket");
    debug_assert!(
        source_stage + lag < stage_lengths_hours.len(),
        "source_stage + lag must index into stage_lengths_hours"
    );

    let future_calendar = &stage_lengths_hours[source_stage..];
    let h_anchor = future_calendar[0];
    let (window_start, window_end) = arrival_window(travel_time_hours, h_anchor);

    let mut arrival_blocks: Vec<Option<&[f64]>> = vec![None; lag];
    arrival_blocks[lag - 1] = target_blocks;

    resolve_delivery(
        window_start,
        window_end,
        future_calendar,
        &arrival_blocks,
        lag,
    )
    .pop()
    .unwrap_or_default()
}

/// A calendar-anchored point-commitment lag: a physical lead time on the
/// hour clock, or a first-class stage-count shift that never reads the
/// calendar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeadTime {
    /// Physical lead time in hours, delivery-anchored.
    Time(f64),
    /// Stage-count shift; the calendar is never consulted.
    Stages(u32),
}

/// The delivery-vs-decision domain split for [`resolve_point`]: the decision
/// axis `[0, n_decision())` is a prefix of the delivery axis
/// `[0, n_delivery())`. `n_decision() <= n_delivery()` holds by construction
/// — `n_delivery()` sums both slices' lengths, never a separately supplied
/// count — mirroring [`crate::time_value::DeliveryCalendar`]. Built by struct
/// literal (never a positional `new`) so a `study`/`post_study` transposition
/// is a field-name error at the call site, not a silent argument swap.
#[derive(Debug, Clone, Copy)]
pub struct DeliveryAxis<'a> {
    /// Study (in-horizon decision) calendar's per-stage total hours.
    pub study_stage_hours: &'a [f64],
    /// Post-study continuation calendar's per-stage total hours; empty with
    /// no post-study calendar declared. [`LeadTime::Stages`] never reads
    /// either slice's values.
    pub post_study_stage_hours: &'a [f64],
}

impl DeliveryAxis<'_> {
    /// In-study (decision) stage count.
    #[must_use]
    pub fn n_decision(self) -> usize {
        self.study_stage_hours.len()
    }

    /// Delivery-stage count, `>= n_decision()`.
    #[must_use]
    pub fn n_delivery(self) -> usize {
        self.study_stage_hours.len() + self.post_study_stage_hours.len()
    }
}

/// Resolved point-commitment lag: the delivery-anchored decider, the
/// per-decision-stage outgoing commitment sets, and the per-decision-stage
/// depths.
#[derive(Debug, Clone, PartialEq)]
pub struct PointResolution {
    /// Decision stage `c(m)` for each delivery stage `m`; `None` is a
    /// pre-study (initial-conditions) decider.
    pub decider: Vec<Option<usize>>,
    /// `C(t) = { m : c(m) = t }`, indexed by decision stage `t`.
    pub decision_sets: Vec<Vec<usize>>,
    /// `K(t) = |{ m > t : c(m) <= t }|`, indexed by decision stage `t`.
    pub depth: Vec<usize>,
    /// Full in-flight occupancy `|{ m > t : m has a carrier as of t }|` — every
    /// delivery still in flight at decision stage `t` that a pre-study
    /// (`None`-decider, `m < n_decision`) or an at-or-before-`t` in-study
    /// decision (`Some(c)`, `c <= t < n_decision`) can deposit into. Indexed by
    /// decision stage `t` (length `n_decision`). This, NOT [`Self::depth`], is
    /// the ring's per-stage occupancy boundary: `depth` structurally excludes
    /// pre-study occupancy (its sweep adds a delta only for `Some(t)`), so
    /// sizing the ring from `depth` under-counts and collides a carry row with a
    /// deposit row whenever a pre-study seed outlives the in-study in-flight
    /// count. `occupancy[t] == depth[t] + max(0, n_none − 1 − t)` (`n_none` =
    /// leading `None`-run length), so the two coincide on a uniform-calendar
    /// lead and `occupancy` is strictly larger only where the ring is
    /// under-sized today.
    pub occupancy: Vec<usize>,
}

impl PointResolution {
    /// Delivery stages where `c(m) = m` — a sub-stage lead: the physical lead
    /// is shorter than the delivery stage's own duration, so the commitment
    /// would be decided inside its own delivery stage and no anticipation
    /// binds (`K = 0`). Excluded from the ring entirely; the stage
    /// LP dispatches the plant's generation as ordinary, unconstrained
    /// thermal output.
    pub fn self_delivered_stages(&self) -> impl Iterator<Item = usize> + '_ {
        self.decider
            .iter()
            .enumerate()
            .filter_map(|(m, &c)| (c == Some(m)).then_some(m))
    }

    /// Whether delivery stage `m`'s commitment is genuinely anticipated: its
    /// decider is `None` (pre-study) or strictly earlier than `m`. `false`
    /// exactly at a `K = 0` sub-stage lead ([`Self::self_delivered_stages`]).
    /// `m` beyond [`Self::decider`]'s range defaults to `true` (no data to
    /// suggest self-delivery), keeping the fishing gate open rather than
    /// suppressing an otherwise-ordinary plant.
    #[must_use]
    pub fn is_anticipated_at(&self, m: usize) -> bool {
        self.decider.get(m).copied().flatten() != Some(m)
    }

    /// `C(t)` filtered to exclude a `K = 0` self-delivery (`c(m) = m`,
    /// [`Self::self_delivered_stages`]): the genuine, ring-eligible decisions
    /// committed at decision stage `t`. Ascending, mirroring `decision_sets[t]`'s
    /// own order — the fan-out decision-column order. `t` beyond
    /// [`Self::decision_sets`]'s range safely yields empty (no decision data,
    /// so no deposit).
    pub fn genuine_decisions_at(&self, t: usize) -> impl Iterator<Item = usize> + '_ {
        self.decision_sets
            .get(t)
            .into_iter()
            .flatten()
            .copied()
            .filter(move |&m| m != t)
    }

    /// Whether delivery stage `m`'s commitment has ALREADY been decided as of
    /// decision stage `t <= m`: either pre-study (`None`) or at or before `t`
    /// (`c(m) <= t`). `decider` is nondecreasing in `m` (`resolve_decider_*`'s
    /// end-anchored/stage-count construction), so readiness is monotonic in
    /// `m`: if slot `k` (delivery `m = t+k+1`) is ready, every shallower slot
    /// `k' < k` is ready too — a contiguous prefix from slot 0, never a
    /// `depth`-derived boundary. `depth[t]` EXCLUDES pre-study (`None`) items
    /// by construction (`build_decision_sets_and_depth`'s sweep only adds a
    /// delta for `Some(t)`), so it under-counts whenever pre-study occupancy
    /// coexists with an in-study decision at the same stage — deriving a slot
    /// boundary from it (rather than checking readiness directly per slot) is
    /// the wrong-but-plausible shortcut this method exists to rule out. `m`
    /// beyond [`Self::decider`]'s range safely yields `false` (no data, so no
    /// shift row) — callers already gate on `m < n_delivery` before reaching
    /// here in the well-formed case.
    #[must_use]
    pub fn is_ready_at(&self, m: usize, t: usize) -> bool {
        self.decider
            .get(m)
            .is_some_and(|c| c.is_none_or(|c| c <= t))
    }

    /// The plant's ring depth: `max(max_t occupancy(t), n_none_in_study)`, where
    /// `n_none_in_study` is the leading pre-study (`None`) decider run inside the
    /// decision axis. Sizing from `occupancy` alone — or, worse, from `depth` —
    /// is the wrong-but-plausible shortcut: `occupancy` subtracts the seed
    /// maturing at stage 0, so it under-counts the one moment every simultaneous
    /// pre-study seed is in flight (stage 0, before the first fishing), leaving
    /// the last seed aliased onto an earlier slot. The `None` run is capped at
    /// `decision_sets.len()` so the excised fixed-post-horizon window's `None`
    /// entries ([`Self::ring_index`] returns `None` for them) never inflate it.
    /// Single owner of the depth formula: [`AnticipatedResolution::resolve`] and
    /// the setup-time per-plant `LeadTime` lead both read it.
    #[must_use]
    pub fn ring_depth(&self) -> usize {
        let occupancy_max = self.occupancy.iter().copied().max().unwrap_or(0);
        let n_none_in_study = self
            .decider
            .iter()
            .take(self.decision_sets.len())
            .take_while(|c| c.is_none())
            .count();
        occupancy_max.max(n_none_in_study)
    }

    /// The plant's fixed post-horizon window width `g`: the leading `None`
    /// run of the post-study suffix `self.decider[n_decision..]`. A plain
    /// `filter(is_none).count()` over that suffix is the wrong-but-plausible
    /// alternative — it silently accepts a non-nondecreasing `decider` and
    /// excises the wrong indices; the leading-run form makes the contiguity
    /// check below cheap and the intent explicit. Recomputed on every call
    /// rather than cached into a field, so [`Self::ring_index`] and
    /// [`Self::physical_target`] cannot disagree about `g`.
    fn fixed_post_horizon_width(&self) -> usize {
        let n_decision = self.decision_sets.len();
        let post_study = self.decider.get(n_decision..).unwrap_or(&[]);
        let g = post_study.iter().take_while(|c| c.is_none()).count();
        debug_assert!(
            post_study[g..].iter().all(Option::is_some),
            "the post-study None run must be contiguous: decider is not nondecreasing"
        );
        g
    }

    /// The ring-axis index of physical delivery target `m`: identity below
    /// `n_decision`; `None` inside the excised fixed post-horizon window
    /// `[n_decision, n_decision + g)`, which is never a ring member; shifted
    /// left by `g` above it. Keying the ring on the raw delivery axis (`m mod
    /// k_max`) is the wrong-but-plausible alternative — injective only on a
    /// contiguous run, which the excised window breaks.
    #[must_use]
    pub fn ring_index(&self, m: usize) -> Option<usize> {
        let n_decision = self.decision_sets.len();
        let g = self.fixed_post_horizon_width();
        if m < n_decision {
            Some(m)
        } else if m < n_decision + g {
            None
        } else {
            Some(m - g)
        }
    }

    /// The left inverse of [`Self::ring_index`] on the ring axis: identity
    /// below `n_decision`, shifted right by `g` above it. Re-deriving the
    /// shift at a call site instead of routing through this method risks
    /// disagreeing with [`Self::ring_index`]'s own `g`.
    #[must_use]
    pub fn physical_target(&self, r: usize) -> usize {
        let n_decision = self.decision_sets.len();
        if r < n_decision {
            r
        } else {
            r + self.fixed_post_horizon_width()
        }
    }
}

/// Resolve a point-commitment lag into the delivery-anchored decider, the
/// per-decision-stage outgoing commitment sets, and the per-decision-stage
/// depths.
///
/// # Panics
///
/// Debug builds panic if a stage length or the lead time is not finite and
/// positive, or if a delivery stage decided in-study (decider `Some(t)`,
/// `t < axis.n_decision()`) fails to appear in its own decision set.
#[must_use]
pub fn resolve_point(lag: LeadTime, axis: DeliveryAxis<'_>) -> PointResolution {
    let decider = match lag {
        LeadTime::Time(delta_hours) => resolve_decider_physical(delta_hours, axis),
        LeadTime::Stages(lead_stages) => {
            resolve_decider_stage_count(lead_stages, axis.n_delivery())
        }
    };
    let (decision_sets, depth, occupancy) =
        build_decision_sets_and_depth(&decider, axis.n_decision());

    PointResolution {
        decider,
        decision_sets,
        depth,
        occupancy,
    }
}

/// Per-anticipated-plant delivery-anchored point-commitment resolution plus the
/// derived ring depth.
///
/// Threaded from setup onto the state layout as additive data for the in-LP
/// delivery-anchored ring; the constant-lead machinery still reads the layout's
/// per-plant `anticipated_lead_stages`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnticipatedResolution {
    /// One [`PointResolution`] per anticipated plant, in
    /// [`crate::indexer::AnticipatedPlants`] order.
    pub per_plant: Vec<PointResolution>,
}

impl AnticipatedResolution {
    /// Delivery-anchored ring depth `max_i ring_depth_i` over every plant (see
    /// [`PointResolution::ring_depth`], the single owner of the depth formula);
    /// `0` with no anticipated plants.
    #[must_use]
    pub fn anchored_depth(&self) -> usize {
        self.per_plant
            .iter()
            .map(PointResolution::ring_depth)
            .max()
            .unwrap_or(0)
    }

    /// Fan-out width `max_i max_t |genuine C_i(t)|` (bounded by
    /// [`Self::anchored_depth`] — a decision set's genuine members are a
    /// subset of the plant's in-flight count at `t`): the decision-column
    /// geometry's per-plant stride,
    /// `col_anticipated_decision_start + j * n_anticipated + local_idx` for
    /// `j in 0..max_fanout`. `1` for a single-decider study (`|C(t)| <= 1`
    /// everywhere), `0` with no anticipated plants.
    #[must_use]
    pub fn max_fanout(&self) -> usize {
        self.per_plant
            .iter()
            .flat_map(|point| {
                (0..point.decision_sets.len()).map(|t| point.genuine_decisions_at(t).count())
            })
            .max()
            .unwrap_or(0)
    }

    /// Resolve every plant's point-commitment lag against the delivery axis.
    ///
    /// `leads` is in anticipated-local order; `axis` carries the delivery
    /// calendar and the decision/delivery stage counts ([`LeadTime::Stages`]
    /// never reads the calendar). This is the single [`resolve_point`]
    /// consumer — a second call site is forbidden; the resolution threads
    /// through instead.
    #[must_use]
    pub fn resolve(leads: &[LeadTime], axis: DeliveryAxis<'_>) -> Self {
        let per_plant: Vec<PointResolution> = leads
            .iter()
            .map(|&lead| resolve_point(lead, axis))
            .collect();
        Self { per_plant }
    }

    /// Widens [`Self::anchored_depth`] to cover the deepest of `lead_stages`.
    #[must_use]
    pub(crate) fn ring_size(&self, lead_stages: &[usize]) -> usize {
        self.anchored_depth()
            .max(lead_stages.iter().copied().max().unwrap_or(0))
    }
}

/// Cumulative stage-end boundaries `S_0 = 0, S_1, .., S_n` on the hour clock.
fn cumulative_stage_boundaries(axis: DeliveryAxis<'_>) -> Vec<f64> {
    let mut boundaries = Vec::with_capacity(axis.n_delivery() + 1);
    let mut cumulative = 0.0_f64;
    boundaries.push(cumulative);
    for &length in axis
        .study_stage_hours
        .iter()
        .chain(axis.post_study_stage_hours)
    {
        debug_assert!(
            length.is_finite() && length > 0.0,
            "every stage length must be finite and > 0.0"
        );
        cumulative += length;
        boundaries.push(cumulative);
    }
    boundaries
}

/// `c(m)` = the stage containing `end_m − Δ`, anchored at the delivery
/// stage's end with boundary ties resolving to the earlier stage — a
/// sub-stage lead (`Δ < h_m`) gives `c(m) = m`; a start-anchored `start_m −
/// Δ` could never reach that. `None` when the target precedes the
/// horizon start.
fn resolve_decider_physical(delta_hours: f64, axis: DeliveryAxis<'_>) -> Vec<Option<usize>> {
    debug_assert!(
        delta_hours.is_finite() && delta_hours > 0.0,
        "delta_hours must be finite and > 0.0"
    );

    let boundaries = cumulative_stage_boundaries(axis);
    let n_delivery = axis.n_delivery();

    (0..n_delivery)
        .map(|m| {
            let target = boundaries[m + 1] - delta_hours;
            let before_target = boundaries.partition_point(|&boundary| boundary < target);
            before_target.checked_sub(1)
        })
        .collect()
}

/// `c(m) = m − ℓ` (or `None` if negative); the calendar is never read —
/// enforced by construction, since this arm takes no stage-length
/// parameter.
fn resolve_decider_stage_count(lead_stages: u32, n_delivery: usize) -> Vec<Option<usize>> {
    let lead = usize::try_from(lead_stages).unwrap_or(usize::MAX);
    (0..n_delivery).map(|m| m.checked_sub(lead)).collect()
}

/// Builds `C(t)`, `K(t)` (`depth`), and the full in-flight occupancy from
/// `decider` via difference-array prefix sums in one forward pass, avoiding an
/// O(n²) rescan of every `(m, t)` pair.
///
/// Two difference arrays share the sweep: `depth` counts only in-study-decided
/// carriers (`Some(c)`, `c < n_decision`), `occupancy` additionally counts
/// pre-study carriers (`None`, `m < n_decision`). A post-study decider
/// (`Some(c)`, `c >= n_decision`) and a pre-study decider for a post-study
/// target (`None`, `m >= n_decision`) have no carrier and enter neither.
/// A carrier is in flight over `(c_or_0, m]`: it
/// contributes `+1` at its decision stage (`c`, or `0` for a pre-study seed)
/// and `-1` at its maturity `m`, the `-1` dropped when `m >= n_decision`.
fn build_decision_sets_and_depth(
    decider: &[Option<usize>],
    n_decision: usize,
) -> (Vec<Vec<usize>>, Vec<usize>, Vec<usize>) {
    let mut decision_sets: Vec<Vec<usize>> = vec![Vec::new(); n_decision];
    let mut depth_delta = vec![0_isize; n_decision];
    let mut occupancy_delta = vec![0_isize; n_decision];

    for (m, &decided_at) in decider.iter().enumerate() {
        match decided_at {
            Some(t) if t < n_decision => {
                decision_sets[t].push(m);
                depth_delta[t] += 1;
                occupancy_delta[t] += 1;
                if m < n_decision {
                    depth_delta[m] -= 1;
                    occupancy_delta[m] -= 1;
                }
            }
            None if m < n_decision => {
                occupancy_delta[0] += 1;
                occupancy_delta[m] -= 1;
            }
            Some(_) | None => {}
        }
    }

    let mut depth_running = 0_isize;
    let depth: Vec<usize> = depth_delta
        .iter()
        .map(|&delta| {
            depth_running += delta;
            debug_assert!(
                depth_running >= 0,
                "depth sweep invariant violated: every decider must satisfy c(m) <= m"
            );
            usize::try_from(depth_running).unwrap_or(0)
        })
        .collect();

    let mut occupancy_running = 0_isize;
    let occupancy: Vec<usize> = occupancy_delta
        .iter()
        .map(|&delta| {
            occupancy_running += delta;
            debug_assert!(
                occupancy_running >= 0,
                "occupancy sweep invariant violated: a carrier is in flight over (c, m]"
            );
            usize::try_from(occupancy_running).unwrap_or(0)
        })
        .collect();

    for (m, &decided_at) in decider.iter().enumerate() {
        debug_assert!(
            decided_at.is_none_or(|t| t >= n_decision || decision_sets[t].contains(&m)),
            "every delivery stage decided in-study must appear in exactly one decision_set"
        );
    }

    (decision_sets, depth, occupancy)
}

#[cfg(test)]
mod tests;
