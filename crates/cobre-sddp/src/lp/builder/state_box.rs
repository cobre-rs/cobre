//! The admissible box per outgoing state dimension: the tightest interval the LP
//! that consumes the value enforces. Storage and transit buckets copy
//! the outgoing column's own template bounds; inflow lags stay unbounded. The
//! commitment-hold family is the one hand-written special case: its carrier
//! column is free `(-∞, ∞)`, so the box comes from the delivery stage's resolved
//! generation bound instead, mirroring `fill_anticipated_columns`'s own lookup.

use std::ops::Range;

use cobre_core::ResolvedBounds;
use cobre_solver::StageTemplate;

use crate::indexer::{
    AnticipatedLocal, AnticipatedPlants, StateSpace, anticipated_resolution_for,
    is_anticipated_decision_active_for_delivery,
};
use crate::time_value::TimeValue;

use super::delivery_ring::DeliveryRing;

/// Admissible interval per outgoing state dimension, both fields length
/// [`StateSpace::n_state`]. A reachable dimension has `lower <= upper` by
/// construction; an inverted or empty box is a load-time validator concern, not
/// this builder's.
#[derive(Debug, Clone)]
pub struct StateBox {
    /// Per-dimension lower bound.
    pub lower: Vec<f64>,
    /// Per-dimension upper bound.
    pub upper: Vec<f64>,
}

#[must_use]
pub(crate) fn build_state_box(
    template: &StageTemplate,
    layout: &StateSpace,
    stage_idx: usize,
    bounds: &ResolvedBounds,
    anticipated_plants: &AnticipatedPlants,
    anticipated_windows: &[(Option<i32>, Option<i32>)],
    time_value: &TimeValue,
) -> StateBox {
    let mut lower = vec![f64::NEG_INFINITY; layout.n_state];
    let mut upper = vec![f64::INFINITY; layout.n_state];

    fill_identity_box(&mut lower, &mut upper, template, layout.storage.clone());
    fill_identity_box(
        &mut lower,
        &mut upper,
        template,
        layout.transit_buckets_out.clone(),
    );
    fill_commitment_hold_box(
        &mut lower,
        &mut upper,
        layout,
        stage_idx,
        bounds,
        anticipated_plants,
        anticipated_windows,
        time_value,
    );

    StateBox { lower, upper }
}

/// Storage and transit-bucket outgoing columns are identity-resolved (state index
/// == LP column), so the box is the template's own `col_lower`/`col_upper`
/// verbatim.
fn fill_identity_box(
    lower: &mut [f64],
    upper: &mut [f64],
    template: &StageTemplate,
    range: Range<usize>,
) {
    for j in range {
        lower[j] = template.col_lower[j];
        upper[j] = template.col_upper[j];
    }
}

/// Commitment-hold slots (`layout.commit_out`): the carrier column is free
/// `(-∞, ∞)`, so the box is the held physical delivery target's resolved
/// generation bound instead — read via `thermal_block_base`, never
/// `template`'s own column bounds, mirroring `fill_anticipated_columns`'s
/// lookup. Every REACHABLE slot resolves its own target —
/// the fresh deposit this stage AND every already-decided, not-yet-matured
/// in-flight slot alike, since the ring holds a carried value at every reachable
/// slot, not only the one just latched. Padding (a plant's own depth shorter
/// than `k_max`) and a commissioning-dormant target's slot stay `[0, 0]`.
///
/// Mirrors `build_anticipated_slot_row_pos`'s ring-axis sweep (`layout.rs`): for
/// `depth in 0..k_max`, `r = stage_idx + depth + 1` visits every residue slot
/// exactly once, and each plant's own physical target `m =
/// point.physical_target(r)` is resolved per-plant since the fixed
/// post-horizon excision is per-plant. Deriving the slot from the raw delivery
/// axis (`m % k_max` on `m` read directly) is the wrong-but-compiling
/// alternative once a plant's fixed post-horizon window excises part of the
/// ring — `physical_target` is the sole owner of that excision.
fn fill_commitment_hold_box(
    lower: &mut [f64],
    upper: &mut [f64],
    layout: &StateSpace,
    stage_idx: usize,
    bounds: &ResolvedBounds,
    anticipated_plants: &AnticipatedPlants,
    anticipated_windows: &[(Option<i32>, Option<i32>)],
    time_value: &TimeValue,
) {
    for j in layout.commit_out.clone() {
        lower[j] = 0.0;
        upper[j] = 0.0;
    }
    if layout.n_anticipated == 0 || layout.k_max == 0 {
        return;
    }
    debug_assert_eq!(
        anticipated_windows.len(),
        layout.n_anticipated,
        "anticipated_windows must have one entry per anticipated plant"
    );

    let n_stages = bounds.n_stages();
    let n_delivery = layout.n_delivery();
    let points: Vec<_> = (0..layout.n_anticipated)
        .map(|plant| anticipated_resolution_for(layout, AnticipatedLocal::new(plant)))
        .collect();
    let ring = DeliveryRing::anticipated(layout);

    for depth in 0..layout.k_max {
        let r = stage_idx + depth + 1;
        let slot = r % layout.k_max;
        for (local_idx, point) in points.iter().enumerate() {
            let m = point.physical_target(r);
            if m >= n_delivery {
                continue;
            }
            let is_deposit = point.decider.get(m).copied().flatten() == Some(stage_idx);
            if !is_deposit && !point.is_ready_at(m, stage_idx) {
                continue;
            }
            if !is_anticipated_decision_active_for_delivery(
                AnticipatedLocal::new(local_idx),
                m,
                n_delivery,
                anticipated_windows,
                time_value.delivery_stage_ids(),
            ) {
                continue;
            }

            let bound = if m < n_stages {
                let thermal_idx = anticipated_plants
                    .thermal_of(AnticipatedLocal::new(local_idx))
                    .get();
                let cap = bounds.thermal_block_base(thermal_idx, m);
                Some((cap.min_generation_mw, cap.max_generation_mw))
            } else {
                time_value
                    .post_study()
                    .anticipated_bound(AnticipatedLocal::new(local_idx), m - n_stages)
                    .map(|(_, min_mw, max_mw)| (min_mw, max_mw))
            };

            if let Some((min_mw, max_mw)) = bound {
                let j = ring.out_col(slot, local_idx);
                lower[j] = min_mw;
                upper[j] = max_mw;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::HydroSys;
    use crate::test_support::{
        state_layout_full, state_layout_with_transit_buckets, transit_bucket_only_template,
    };
    use crate::time_value::PostStudyResolved;

    fn empty_bounds_and_time_value() -> (ResolvedBounds, TimeValue) {
        (
            ResolvedBounds::empty(),
            TimeValue::from_parts(vec![], vec![], vec![], vec![], PostStudyResolved::default()),
        )
    }

    /// AC: a storage column bounded `[0.0, 50.0]` box-copies verbatim.
    #[test]
    fn state_box_storage_takes_the_outgoing_column_bounds() {
        let layout = state_layout_full(1, 0, Vec::new());
        let mut template = transit_bucket_only_template(layout.n_state, layout.n_state);
        let storage_j = layout.storage.start;
        template.col_lower[storage_j] = 0.0;
        template.col_upper[storage_j] = 50.0;
        let (bounds, time_value) = empty_bounds_and_time_value();

        let state_box = build_state_box(
            &template,
            &layout,
            0,
            &bounds,
            &AnticipatedPlants::default(),
            &[],
            &time_value,
        );

        assert_eq!(state_box.lower[storage_j], 0.0);
        assert_eq!(state_box.upper[storage_j], 50.0);
    }

    /// AC: an inflow-lag dimension stays at the unbounded default.
    #[test]
    fn state_box_inflow_lag_is_unbounded() {
        let layout = state_layout_full(1, 1, Vec::new());
        let template = transit_bucket_only_template(layout.n_state, layout.n_state);
        let (bounds, time_value) = empty_bounds_and_time_value();

        let state_box = build_state_box(
            &template,
            &layout,
            0,
            &bounds,
            &AnticipatedPlants::default(),
            &[],
            &time_value,
        );

        let lag_j = layout.inflow_lags.start;
        assert_eq!(state_box.lower[lag_j], f64::NEG_INFINITY);
        assert_eq!(state_box.upper[lag_j], f64::INFINITY);
    }

    /// AC: a transit-bucket column box-copies its own reachable `[0, ∞)` / frozen
    /// `[0, 0]` bounds verbatim, exactly like storage.
    #[test]
    fn state_box_transit_bucket_reachable_is_zero_to_inf_frozen_is_zero_zero() {
        let layout = state_layout_with_transit_buckets(
            1,
            0,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            Vec::new(),
        );
        let mut template = transit_bucket_only_template(layout.n_state, layout.n_state);
        let reachable_j = layout.transit_buckets_out.start;
        let frozen_j = layout.transit_buckets_out.start + 1;
        template.col_lower[reachable_j] = 0.0;
        template.col_upper[reachable_j] = f64::INFINITY;
        template.col_lower[frozen_j] = 0.0;
        template.col_upper[frozen_j] = 0.0;
        let (bounds, time_value) = empty_bounds_and_time_value();

        let state_box = build_state_box(
            &template,
            &layout,
            0,
            &bounds,
            &AnticipatedPlants::default(),
            &[],
            &time_value,
        );

        assert_eq!(state_box.lower[reachable_j], 0.0);
        assert_eq!(state_box.upper[reachable_j], f64::INFINITY);
        assert_eq!(state_box.lower[frozen_j], 0.0);
        assert_eq!(state_box.upper[frozen_j], 0.0);
    }
}
