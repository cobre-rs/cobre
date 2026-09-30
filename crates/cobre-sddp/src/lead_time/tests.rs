use super::*;

use std::collections::HashMap;

use proptest::prelude::*;
use proptest::test_runner::RngSeed;

const TOL: f64 = 1e-9;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: got {actual:?}, expected {expected:?}"
    );
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() < TOL,
            "value mismatch: got {actual:?}, expected {expected:?}"
        );
    }
}

// Weekly anchor (168h stages), t_v=360h: stage_weights[0]=stage_weights[1]=0,
// stage_weights[2]=6/7, stage_weights[3]=1/7, depth 3.
#[test]
fn test_s1a_matches_decomp_fig_5_5b() {
    let stage_lengths_hours = [168.0; 6];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 3);
    assert_close(&resolution.stage_weights, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

#[test]
fn test_s1d_monthly_to_weekly_counterexample() {
    let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 3);
    assert_close(
        &resolution.stage_weights,
        &[1.0 / 2.0, 7.0 / 30.0, 7.0 / 30.0, 1.0 / 30.0],
    );
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);

    // The closed-form ceiling `⌈360/720⌉ == 1` would drop the lag-2/lag-3
    // mass; only the general max-reach depth is correct here.
    assert_ne!(resolution.stage_reach, 1);
}

#[test]
fn test_example_iii_block_factors() {
    let stage_lengths_hours = [720.0, 720.0];
    let block_lengths_hours = [240.0, 240.0, 240.0];
    let resolution = resolve_spread(250.0, 0, &stage_lengths_hours, Some(&block_lengths_hours));

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[470.0 / 720.0, 250.0 / 720.0]);

    assert_close(&resolution.block_deposits[0], &[1.0, 0.0]);
    assert_close(
        &resolution.block_deposits[1],
        &[230.0 / 240.0, 10.0 / 240.0],
    );
    assert_close(&resolution.block_deposits[2], &[0.0, 1.0]);

    // within_stage_routing[b] is self-inclusive: index 0 == block b routing
    // to itself, so the two nonzero downstream entries are
    // within_stage_routing[0][1..].
    assert_close(
        &resolution.within_stage_routing[0],
        &[0.0, 230.0 / 240.0, 10.0 / 240.0],
    );
    assert_close(&resolution.within_stage_routing[1], &[0.0, 230.0 / 240.0]);
    assert_close(&resolution.within_stage_routing[2], &[0.0]);

    assert_eq!(resolution.arrival_density.len(), 1);
    assert_close(
        &resolution.arrival_density[0],
        &[240.0 / 250.0, 10.0 / 250.0],
    );

    for (d, &k_d) in resolution.stage_weights.iter().enumerate() {
        let aggregated: f64 = resolution
            .block_deposits
            .iter()
            .zip(&block_lengths_hours)
            .map(|(row, &duration_b)| (duration_b / 720.0) * row[d])
            .sum();
        assert!((aggregated - k_d).abs() < TOL);
    }
}

// Monthly anchor, t_v=360h is exactly half the anchor length, so
// stage_weights[0]=stage_weights[1]=1/2 and depth is 1.
#[test]
fn test_s1b_monthly_half_split() {
    let stage_lengths_hours = [720.0, 720.0, 720.0];
    let resolution = resolve_spread(360.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[0.5, 0.5]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Weekly-to-monthly transition: the same t_v=360h anchored at each of 4
// weekly stages then the first monthly stage gives depths (3, 3, 2, 1, 1) —
// week 3's window skips week 4 entirely and lands in the month
// (stage_weights[2] = 1, slot 1 transit-only) — and the global max over all
// anchors is depth 3, the reachability mask the per-stage state sizing needs.
#[test]
fn test_s1c_weekly_to_monthly_transition_depths() {
    let stage_lengths_hours = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];

    let week1 = resolve_spread(360.0, 0, &stage_lengths_hours, None);
    assert_eq!(week1.stage_reach, 3);
    assert_close(&week1.stage_weights, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);

    let week2 = resolve_spread(360.0, 1, &stage_lengths_hours, None);
    assert_eq!(week2.stage_reach, 3);
    assert_close(&week2.stage_weights, &[0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0]);

    let week3 = resolve_spread(360.0, 2, &stage_lengths_hours, None);
    assert_eq!(week3.stage_reach, 2);
    assert_close(&week3.stage_weights, &[0.0, 0.0, 1.0]);

    let week4 = resolve_spread(360.0, 3, &stage_lengths_hours, None);
    assert_eq!(week4.stage_reach, 1);
    assert_close(&week4.stage_weights, &[0.0, 1.0]);

    let month1 = resolve_spread(360.0, 4, &stage_lengths_hours, None);
    assert_eq!(month1.stage_reach, 1);
    assert_close(&month1.stage_weights, &[0.5, 0.5]);

    let anchors = [&week1, &week2, &week3, &week4, &month1];
    let global_max_depth = anchors
        .iter()
        .map(|resolution| resolution.stage_reach)
        .max();
    assert_eq!(global_max_depth, Some(3));

    for resolution in anchors {
        assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
    }
}

// Negligible same-stage-crossing mass: monthly anchor, t_v=6h — the
// setup-advisory case rather than a silent fold, since the bucket still
// carries the exact mass stage_weights[1] = 6/720.
#[test]
fn test_s2_monthly_negligible_mass() {
    let stage_lengths_hours = [720.0, 720.0];
    let resolution = resolve_spread(6.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[714.0 / 720.0, 6.0 / 720.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Daily chronological block partition: t_v=6h against 24 hourly blocks gives
// stage-clock stage_weights[1] = 25%, and the chronological-adds boundary —
// blocks 0-17 route in-stage to block b+6 (block_deposits[b][0] = 1), blocks
// 18-23 cross the day boundary and deposit fully into lag 1
// (block_deposits[b][1] = 1).
#[test]
fn test_s3_daily_chronological_block_partition() {
    let stage_lengths_hours = [24.0, 24.0, 24.0];
    let block_lengths_hours = [1.0; 24];
    let resolution = resolve_spread(6.0, 0, &stage_lengths_hours, Some(&block_lengths_hours));

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[18.0 / 24.0, 6.0 / 24.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);

    // Last in-stage block: routes fully to block 17+6=23 within the same day.
    assert_close(&resolution.block_deposits[17], &[1.0, 0.0]);
    assert_close(
        &resolution.within_stage_routing[17],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    );

    // First cross-boundary block: the 6-hour delay lands entirely in the
    // next day, deposited fully rather than routed in-stage.
    assert_close(&resolution.block_deposits[18], &[0.0, 1.0]);

    for (d, &k_d) in resolution.stage_weights.iter().enumerate() {
        let aggregated: f64 = resolution
            .block_deposits
            .iter()
            .zip(&block_lengths_hours)
            .map(|(row, &duration_b)| (duration_b / 24.0) * row[d])
            .sum();
        assert!((aggregated - k_d).abs() < TOL);
    }
}

// Exact-multiple boundary: monthly anchor, t_v=720h is exactly one stage
// length, so the whole release crosses the boundary: stage_weights[0] = 0,
// stage_weights[1] = 1.
#[test]
fn test_s4_monthly_exact_multiple_boundary() {
    let stage_lengths_hours = [720.0, 720.0];
    let resolution = resolve_spread(720.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[0.0, 1.0]);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Transit-only slot: monthly anchor, t_v=1800h (75 d) — slot 1 carries zero
// deposit yet the ring shift still passes mass through it to slots 2 and 3,
// which split the release evenly; depth counts a reachable slot, not a
// nonzero-factor count.
#[test]
fn test_s5_monthly_transit_only_slot() {
    let stage_lengths_hours = [720.0; 5];
    let resolution = resolve_spread(1800.0, 0, &stage_lengths_hours, None);

    assert_eq!(resolution.stage_reach, 3);
    assert_close(&resolution.stage_weights, &[0.0, 0.0, 0.5, 0.5]);
    assert_eq!(resolution.stage_weights[1], 0.0);
    assert!((resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL);
}

// Exercises the `Σ_d k_d == 1` conservation debug_assert directly, across a
// non-uniform calendar and a non-zero `anchor_stage` slice offset — distinct
// from the weekly (`test_s1a_matches_decomp_fig_5_5b`) and monthly-transition
// (`test_s1d_monthly_to_weekly_counterexample`) fixtures above.
#[test]
fn test_stage_level_conservation_debug_assert() {
    let stage_lengths_hours = [300.0, 200.0, 400.0, 150.0, 600.0, 250.0];

    for anchor_stage in 0..3 {
        for &travel_time_hours in &[10.0, 90.0, 275.0, 501.0, 725.0] {
            let resolution =
                resolve_spread(travel_time_hours, anchor_stage, &stage_lengths_hours, None);
            assert!(
                (resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < TOL,
                "conservation violated at anchor_stage={anchor_stage}, t_v={travel_time_hours}"
            );
            assert!(resolution.stage_reach >= 1);
            assert!(resolution.block_deposits.is_empty());
            assert!(resolution.within_stage_routing.is_empty());
        }
    }
}

// Exercises the `Σ_b w_b·χ_{b,d} == k_d` shared-density-consistency
// debug_assert directly (the shared-density aggregation identity), across a
// non-uniform block partition and non-uniform future stages, distinct from
// `test_example_iii_block_factors`.
#[test]
fn test_shared_density_consistency_debug_assert() {
    let stage_lengths_hours = [720.0, 500.0, 300.0, 300.0];
    let block_lengths_hours = [100.0, 200.0, 50.0, 370.0];

    for &travel_time_hours in &[50.0, 300.0, 690.0, 1000.0] {
        let resolution = resolve_spread(
            travel_time_hours,
            0,
            &stage_lengths_hours,
            Some(&block_lengths_hours),
        );

        for (d, &k_d) in resolution.stage_weights.iter().enumerate() {
            let aggregated: f64 = resolution
                .block_deposits
                .iter()
                .zip(&block_lengths_hours)
                .map(|(row, &duration_b)| (duration_b / 720.0) * row[d])
                .sum();
            assert!(
                (aggregated - k_d).abs() < TOL,
                "shared-density consistency violated at t_v={travel_time_hours}, lag={d}"
            );
        }
    }
}

// Regression: with every arrival stage sharing the anchor's own block
// partition (uniform stage lengths), arrival_density stays bit-identical to
// this captured baseline.
#[test]
fn test_uniform_resolution_arrival_density_matches_captured_baseline() {
    let stage_lengths_hours = [720.0, 720.0];
    let block_lengths_hours = [240.0, 240.0, 240.0];
    let resolution = resolve_spread(250.0, 0, &stage_lengths_hours, Some(&block_lengths_hours));

    assert_eq!(
        resolution.arrival_density,
        vec![vec![240.0 / 250.0, 10.0 / 250.0]]
    );
}

// Multi-resolution: arrival stage t+1's own partition (summing to its own
// 500h length) differs from the anchor's 720h stage. `resolve_delivery`
// splits row 0 against t+1's own blocks, not the anchor's — block0's 300h
// covers the whole local window's first 300h, block1's 200h the rest.
#[test]
fn test_resolve_delivery_splits_against_arrival_stage_own_blocks() {
    let future_calendar = [720.0, 500.0, 400.0];
    let arrival_stage1_blocks = [300.0, 200.0];
    let arrival_blocks: [Option<&[f64]>; 2] = [Some(&arrival_stage1_blocks), None];

    let arrival_density = resolve_delivery(600.0, 1320.0, &future_calendar, &arrival_blocks, 2);

    assert_eq!(arrival_density.len(), 2);
    assert_close(&arrival_density[0], &[0.6, 0.4]);
    assert_close(&arrival_density[1], &[1.0]);
}

// Exercises the Σ_b φ_{d,b} == 1 delivery-conservation debug_assert
// directly: every nonempty arrival_density row sums to 1.0, whether the row
// split against a matching block partition (uniform calendar) or fell back
// to a single row (a length mismatch, the mixed calendar).
#[test]
fn test_arrival_density_conservation_per_lag() {
    let assert_conserved = |resolution: &SpreadResolution| {
        for row in &resolution.arrival_density {
            if row.is_empty() {
                continue;
            }
            assert!(
                (row.iter().sum::<f64>() - 1.0).abs() < TOL,
                "arrival density row must sum to 1.0: {row:?}"
            );
        }
    };

    let uniform_stage_lengths = [720.0, 720.0, 720.0];
    let uniform_blocks = [240.0, 240.0, 240.0];
    for &travel_time_hours in &[50.0, 250.0, 400.0, 690.0] {
        assert_conserved(&resolve_spread(
            travel_time_hours,
            0,
            &uniform_stage_lengths,
            Some(&uniform_blocks),
        ));
    }

    let mixed_stage_lengths = [720.0, 500.0, 300.0, 300.0];
    let mixed_blocks = [100.0, 200.0, 50.0, 370.0];
    for &travel_time_hours in &[50.0, 300.0, 690.0, 1000.0] {
        assert_conserved(&resolve_spread(
            travel_time_hours,
            0,
            &mixed_stage_lengths,
            Some(&mixed_blocks),
        ));
    }
}

// PMO calendar (4x168h weekly then 720h monthly), 30-day (720h) lead. The
// end-anchored decider computes directly against the calendar rather than
// reading a decision-anchored column, avoiding the many-to-one collision a
// decision-anchored scheme produces on this same calendar: weeks 0-3 cannot
// reach month 1 (end_4 - 720h = day -2, pre-study), so month 1 is IC; week 4
// (the stage immediately preceding month 1) is the unique decider, and each
// month thereafter decides the next at lag 1 (Δ == h_month exactly).
#[test]
fn test_pmo_end_anchored_delivery_resolution() {
    let stage_lengths_hours = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];
    let resolution = resolve_point(
        LeadTime::Time(720.0),
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(
        resolution.decider,
        vec![None, None, None, None, Some(3), Some(4), Some(5)]
    );
    assert_eq!(
        resolution.decision_sets,
        vec![vec![], vec![], vec![], vec![4], vec![5], vec![6], vec![]]
    );
    assert_eq!(resolution.depth, vec![0, 0, 0, 1, 1, 1, 0]);
}

// Sub-stage lead: a 700h stage then a 744h (31-day) month, 720h (30-day)
// lead. Δ < h_1, so end_1 - Δ falls inside stage 1's own window and c(1) ==
// 1 — the K=0 degeneracy, represented (not underflowed) with depth[1] == 0.
#[test]
fn test_sub_stage_lead_k0_degeneracy() {
    let stage_lengths_hours = [700.0, 744.0];
    let resolution = resolve_point(
        LeadTime::Time(720.0),
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(resolution.decider, vec![None, Some(1)]);
    assert_eq!(resolution.decision_sets, vec![vec![], vec![1]]);
    assert_eq!(resolution.depth, vec![0, 0]);
}

// Stage-count mode on a monthly calendar with unequal stage hours (672-744h):
// the hour clock is never consulted, so `LeadTime::Stages(2)` reproduces the
// shipped index shift identically regardless of the calendar values. Depth
// and fan-out are checked away from the array boundary, where K(t) == ℓ and
// |C(t)| == 1 hold without the natural edge truncation that bounds K(t) =
// |{m>t : c(m)<=t}| to the delivery stages that actually exist.
#[test]
fn test_stage_count_mode_unequal_monthly_hours() {
    let stage_lengths_hours = [672.0, 700.0, 744.0, 720.0, 672.0, 744.0, 700.0, 744.0];
    let resolution = resolve_point(
        LeadTime::Stages(2),
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );

    for m in 2..8 {
        assert_eq!(
            resolution.decider[m],
            Some(m - 2),
            "decider mismatch at m={m}"
        );
    }
    for t in 1..=5 {
        assert_eq!(resolution.depth[t], 2, "depth mismatch at t={t}");
        assert_eq!(
            resolution.decision_sets[t].len(),
            1,
            "decision_sets length mismatch at t={t}"
        );
    }
    assert_eq!(resolution.decision_sets[0].len(), 1);
}

#[test]
fn test_ic_boundary_decider_is_none() {
    let stage_lengths_hours = [100.0];
    let resolution = resolve_point(
        LeadTime::Time(1000.0),
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(resolution.decider, vec![None]);
    assert_eq!(resolution.decision_sets, vec![Vec::<usize>::new()]);
    assert_eq!(resolution.depth, vec![0]);
}

// Fan-out: a coarse decision stage before a fine zone commits several
// delivery stages, |C(t)| > 1. A 720h month (stage 0) anchors four 168h
// weeks (stages 1-4); at a 750h lead each week's end_m - Δ lands inside the
// month's own window, so all four weeks share decider 0. The month's own
// delivery (Δ exceeds its own 720h length) precedes the horizon and is IC.
#[test]
fn test_coarse_decision_fans_out_over_fine_delivery_stages() {
    let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0];
    let resolution = resolve_point(
        LeadTime::Time(750.0),
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(
        resolution.decider,
        vec![None, Some(0), Some(0), Some(0), Some(0)]
    );
    assert_eq!(
        resolution.decision_sets,
        vec![vec![1, 2, 3, 4], vec![], vec![], vec![], vec![]]
    );
    assert_eq!(resolution.depth, vec![4, 3, 2, 1, 0]);
    assert!(resolution.decision_sets[0].len() > 1);
}

// AnticipatedResolution::resolve batches per-plant resolutions in anticipated-
// local order and derives k_max as the global max per-stage depth. Two plants on
// a uniform monthly calendar: LeadStages(2) has depth max 2, LeadStages(4) has
// depth max 4, so the derived ring depth is 4 (the deeper plant), not the sum.
#[test]
fn test_anticipated_resolution_k_max_is_global_depth_max() {
    let stage_lengths_hours = [720.0; 8];
    let leads = [LeadTime::Stages(2), LeadTime::Stages(4)];
    let resolution = AnticipatedResolution::resolve(
        &leads,
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(resolution.per_plant.len(), 2);
    assert_eq!(resolution.per_plant[0].depth.iter().copied().max(), Some(2));
    assert_eq!(resolution.per_plant[1].depth.iter().copied().max(), Some(4));
    assert_eq!(resolution.anchored_depth(), 4);
}

#[test]
fn test_anticipated_resolution_empty_is_zero_depth() {
    let resolution = AnticipatedResolution::resolve(
        &[],
        DeliveryAxis {
            study_stage_hours: &[720.0; 4],
            post_study_stage_hours: &[],
        },
    );
    assert!(resolution.per_plant.is_empty());
    assert_eq!(resolution.anchored_depth(), 0);
}

// Fan-out `[720,168,168,168,168,168]` h `LeadTime(720)` — the exact calendar
// hand-derived to `decision_sets[0] == [1,2,3,4]`. Confirms
// `genuine_decisions_at` reproduces the same set (nothing is self-delivered
// here) and `max_fanout` reaches the fan-out width.
#[test]
fn test_genuine_decisions_at_matches_fanout_hand_derivation() {
    let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0, 168.0];
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Time(720.0)],
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );
    let point = &resolution.per_plant[0];

    assert_eq!(point.decision_sets[0], vec![1, 2, 3, 4]);
    assert_eq!(
        point.genuine_decisions_at(0).collect::<Vec<_>>(),
        vec![1, 2, 3, 4],
        "no self-delivery here, so genuine == decision_sets exactly"
    );
    assert_eq!(resolution.max_fanout(), 4);
    assert!(point.self_delivered_stages().next().is_none());
}

// `K = 0` `[744,744,744,744]` h `LeadTime(720)` — the exact calendar
// hand-derived to `depth == [0,0,0,0]`. Every delivery stage
// self-delivers (`c(m) = m`), so `genuine_decisions_at` is empty everywhere,
// `self_delivered_stages` yields every stage, and `is_anticipated_at` is
// `false` at every stage.
#[test]
fn test_k0_uniform_calendar_self_delivers_every_stage() {
    let stage_lengths_hours = [744.0, 744.0, 744.0, 744.0];
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Time(720.0)],
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );
    let point = &resolution.per_plant[0];

    assert_eq!(point.depth, vec![0, 0, 0, 0]);
    assert_eq!(resolution.anchored_depth(), 0);
    assert_eq!(resolution.max_fanout(), 0);
    assert_eq!(
        point.self_delivered_stages().collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    for t in 0..4 {
        assert!(
            point.genuine_decisions_at(t).next().is_none(),
            "stage {t}: a self-delivery must never appear as a genuine decision"
        );
        assert!(
            !point.is_anticipated_at(t),
            "stage {t}: a K=0 self-delivery must never be genuinely anticipated"
        );
    }
}

// `is_ready_at` is monotonic in `m` for a constant-lead plant: at any decision
// stage, every slot up to K-1 is ready (either IC-seeded or an earlier in-study
// decision), and slot K is not (structural padding beyond this plant's depth).
#[test]
fn test_is_ready_at_monotonic_prefix_for_constant_lead() {
    let stage_lengths_hours = [744.0; 6];
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Stages(2)],
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );
    let point = &resolution.per_plant[0];

    // At t=0: m=1 (slot 0, IC-seeded) and m=2 (slot 1, this stage's own fresh
    // decision) are both ready; m=3 (slot 2, beyond K=2) is not.
    assert!(point.is_ready_at(1, 0), "slot 0 (IC-seeded) must be ready");
    assert!(
        point.is_ready_at(2, 0),
        "slot 1 (fresh decision) must be ready"
    );
    assert!(
        !point.is_ready_at(3, 0),
        "slot 2 is beyond this plant's depth and must not be ready"
    );
}

// These pins byte-compare every field of `SpreadResolution`/`PointResolution`
// against a captured baseline: a tolerance compare would miss a projection-kernel
// extraction that subtly perturbs one field while leaving every narrower
// assertion untouched.

#[test]
fn migration_equivalence_spread_monthly_half_split() {
    let resolution = resolve_spread(360.0, 0, &[720.0, 720.0, 720.0], None);
    assert_eq!(
        resolution,
        SpreadResolution {
            stage_reach: 1,
            stage_weights: vec![0.5, 0.5],
            block_deposits: vec![],
            within_stage_routing: vec![],
            arrival_density: vec![vec![1.0]],
        }
    );
}

#[test]
fn migration_equivalence_spread_weekly_to_monthly() {
    let resolution = resolve_spread(
        360.0,
        0,
        &[168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0],
        None,
    );
    assert_eq!(
        resolution,
        SpreadResolution {
            stage_reach: 3,
            stage_weights: vec![0.0, 0.0, 6.0 / 7.0, 1.0 / 7.0],
            block_deposits: vec![],
            within_stage_routing: vec![],
            arrival_density: vec![vec![], vec![1.0], vec![1.0]],
        }
    );
}

#[test]
fn migration_equivalence_spread_daily_chronological_block() {
    let resolution = resolve_spread(6.0, 0, &[24.0, 24.0, 24.0], Some(&[1.0; 24]));

    assert_eq!(resolution.stage_reach, 1);
    assert_close(&resolution.stage_weights, &[0.75, 0.25]);

    let mut expected_deposits = vec![vec![1.0, 0.0]; 18];
    expected_deposits.extend(vec![vec![0.0, 1.0]; 6]);
    assert_eq!(resolution.block_deposits, expected_deposits);

    let expected_routing: Vec<Vec<f64>> = (0..24)
        .map(|b| {
            let width = 24 - b;
            (0..width)
                .map(|j| if b < 18 && j == 6 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    assert_eq!(resolution.within_stage_routing, expected_routing);

    assert_eq!(resolution.arrival_density, vec![vec![1.0 / 6.0; 6]]);
}

#[test]
fn migration_equivalence_point_pmo_end_anchored() {
    let resolution = resolve_point(
        LeadTime::Time(720.0),
        DeliveryAxis {
            study_stage_hours: &[168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0],
            post_study_stage_hours: &[],
        },
    );
    assert_eq!(
        resolution,
        PointResolution {
            decider: vec![None, None, None, None, Some(3), Some(4), Some(5)],
            decision_sets: vec![vec![], vec![], vec![], vec![4], vec![5], vec![6], vec![]],
            depth: vec![0, 0, 0, 1, 1, 1, 0],
            occupancy: vec![3, 2, 1, 1, 1, 1, 0],
        }
    );
}

#[test]
fn migration_equivalence_point_k0_degeneracy() {
    let resolution = resolve_point(
        LeadTime::Time(720.0),
        DeliveryAxis {
            study_stage_hours: &[700.0, 744.0],
            post_study_stage_hours: &[],
        },
    );
    assert_eq!(
        resolution,
        PointResolution {
            decider: vec![None, Some(1)],
            decision_sets: vec![vec![], vec![1]],
            depth: vec![0, 0],
            occupancy: vec![0, 0],
        }
    );
}

// resolve_point's decider/decision_sets/depth/occupancy stay byte-identical to a
// captured baseline; k_max now derives from ring_depth, so this fixture's plant-0
// LeadTime shape (leading None-run 4 > occupancy max 3, which subtracts the seed
// maturing at stage 0) grows from the old occupancy-derived 3 to 4.
#[test]
fn test_in_study_path_byte_identity_regression() {
    let stage_lengths_hours = [168.0, 168.0, 168.0, 168.0, 720.0, 720.0, 720.0];
    let resolution = resolve_point(
        LeadTime::Time(720.0),
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );
    assert_eq!(
        resolution,
        PointResolution {
            decider: vec![None, None, None, None, Some(3), Some(4), Some(5)],
            decision_sets: vec![vec![], vec![], vec![], vec![4], vec![5], vec![6], vec![]],
            depth: vec![0, 0, 0, 1, 1, 1, 0],
            occupancy: vec![3, 2, 1, 1, 1, 1, 0],
        },
        "resolve_point's decider/decision_sets/depth must stay byte-identical"
    );

    let anticipated = AnticipatedResolution::resolve(
        &[LeadTime::Time(720.0), LeadTime::Stages(2)],
        DeliveryAxis {
            study_stage_hours: &stage_lengths_hours,
            post_study_stage_hours: &[],
        },
    );
    assert_eq!(
        anticipated.anchored_depth(),
        4,
        "k_max derives from ring_depth: plant 0's leading None-run (4) exceeds its \
         occupancy max (3), so the ring grows to hold every simultaneous seed"
    );
    assert_eq!(
        anticipated.max_fanout(),
        1,
        "max_fanout must stay byte-identical"
    );
    assert_eq!(
        anticipated.per_plant[0].decider,
        vec![None, None, None, None, Some(3), Some(4), Some(5)],
        "per-plant decider must stay byte-identical"
    );
}

#[test]
fn migration_equivalence_point_stage_count() {
    let resolution = resolve_point(
        LeadTime::Stages(2),
        DeliveryAxis {
            study_stage_hours: &[672.0, 700.0, 744.0, 720.0, 672.0, 744.0, 700.0, 744.0],
            post_study_stage_hours: &[],
        },
    );
    assert_eq!(
        resolution,
        PointResolution {
            decider: vec![
                None,
                None,
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5)
            ],
            decision_sets: vec![
                vec![2],
                vec![3],
                vec![4],
                vec![5],
                vec![6],
                vec![7],
                vec![],
                vec![]
            ],
            depth: vec![1, 2, 2, 2, 2, 2, 1, 0],
            occupancy: vec![2, 2, 2, 2, 2, 2, 1, 0],
        }
    );
}

// Equal-domain axis (n_decision == n_delivery): resolve_point reproduces the
// pre-split result exactly. Hand-written expected value (not a recomputation)
// so a projection-kernel perturbation of one field is caught.
#[test]
fn test_delivery_axis_equal_domains_matches_pre_split() {
    let axis = DeliveryAxis {
        study_stage_hours: &[100.0; 4],
        post_study_stage_hours: &[],
    };
    let resolution = resolve_point(LeadTime::Time(350.0), axis);

    assert_eq!(
        resolution,
        PointResolution {
            decider: vec![None, None, None, Some(0)],
            decision_sets: vec![vec![3], vec![], vec![], vec![]],
            depth: vec![1, 1, 1, 0],
            occupancy: vec![3, 2, 1, 0],
        }
    );
    assert_eq!(resolution.decision_sets.len(), 4);
    assert_eq!(resolution.depth.len(), 4);
}

// Delivery axis wider than the decision axis (Stages mode, empty calendar): the
// decider spans all n_delivery stages, decision_sets/depth stay decision-sized.
#[test]
fn test_delivery_axis_wider_than_decision_stages_mode() {
    let axis = DeliveryAxis {
        study_stage_hours: &[720.0; 3],
        post_study_stage_hours: &[720.0; 3],
    };
    let resolution = resolve_point(LeadTime::Stages(3), axis);

    assert_eq!(
        resolution.decider,
        vec![None, None, None, Some(0), Some(1), Some(2)]
    );
    assert_eq!(resolution.decision_sets.len(), 3);
    assert_eq!(resolution.depth.len(), 3);
    assert_eq!(resolution.decision_sets[0], vec![3]);
}

// A delivery decided at a post-study stage contributes to neither decision_sets
// nor depth: m == 3 (decider Some(2), in study) appears in decision_sets[2],
// while m == 4 (decider Some(3), a post-study decision stage) appears in no
// decision set and leaves depth untouched.
#[test]
fn test_post_study_decided_delivery_absent_from_decision_domain() {
    let axis = DeliveryAxis {
        study_stage_hours: &[720.0; 3],
        post_study_stage_hours: &[720.0; 3],
    };
    let resolution = resolve_point(LeadTime::Stages(1), axis);

    assert_eq!(
        resolution.decider,
        vec![None, Some(0), Some(1), Some(2), Some(3), Some(4)]
    );
    assert_eq!(resolution.decider[3], Some(2));
    assert!(resolution.decision_sets[2].contains(&3));
    assert_eq!(resolution.decider[4], Some(3));
    assert!(resolution.decision_sets.iter().all(|set| !set.contains(&4)));
    assert_eq!(resolution.decision_sets, vec![vec![1], vec![2], vec![3]]);
    assert_eq!(resolution.depth, vec![1, 1, 1]);
    // The post-study deciders (m == 4 at Some(3), m == 5 at Some(4), both
    // c >= n_decision) have no carrier and inflate neither depth nor occupancy.
    assert_eq!(resolution.occupancy, vec![1, 1, 1]);
}

// The four-stage `LeadTime(350)` shape whose leading None-run (3) exceeds
// its max depth (1). Occupancy counts the pre-study seeds `depth` drops, so
// k_max is the true stage-0 in-flight count 3, not the depth-derived 1.
#[test]
fn test_occupancy_full_in_flight_pre_study_prefix() {
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Time(350.0)],
        DeliveryAxis {
            study_stage_hours: &[100.0; 4],
            post_study_stage_hours: &[],
        },
    );
    let point = &resolution.per_plant[0];

    assert_eq!(point.occupancy, vec![3, 2, 1, 0]);
    assert_eq!(point.depth, vec![1, 1, 1, 0]);
    assert_eq!(resolution.anchored_depth(), 3);
}

// The DECOMP shape — `LeadStages(6)` over an empty calendar with a delivery
// axis (12) twice the decision axis (6). Every in-study stage carries a full
// pre-study-plus-in-study ring of six deliveries, so occupancy is uniformly 6.
#[test]
fn test_occupancy_decomp_shape_full_ring_every_stage() {
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Stages(6)],
        DeliveryAxis {
            study_stage_hours: &[720.0; 6],
            post_study_stage_hours: &[720.0; 6],
        },
    );
    let point = &resolution.per_plant[0];

    assert_eq!(point.occupancy, vec![6, 6, 6, 6, 6, 6]);
    assert_eq!(resolution.anchored_depth(), 6);
    assert_eq!(resolution.max_fanout(), 1);
}

// Byte-identity floor: on a well-behaved uniform calendar whose leading None-run
// is <= 1 (a one-stage physical lead, n_none == max_t depth == 1), occupancy and
// depth coincide element-for-element, so sizing from occupancy leaves k_max
// unchanged from the depth-derived value.
#[test]
fn test_occupancy_equals_depth_on_short_uniform_lead() {
    let resolution = resolve_point(
        LeadTime::Time(720.0),
        DeliveryAxis {
            study_stage_hours: &[720.0; 5],
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(resolution.depth, vec![1, 1, 1, 1, 0]);
    assert_eq!(resolution.occupancy, resolution.depth);
    assert_eq!(
        resolution.occupancy.iter().max(),
        resolution.depth.iter().max()
    );
}

// The byte-identity identity `occupancy[t] == depth[t] + max(0, n_none−1−t)`
// hand-computed on the four-stage `LeadTime(350)` fixture: n_none == 3, depth == [1,1,1,0], so the
// pre-study correction is [2,1,0,0] and occupancy is [3,2,1,0].
#[test]
fn test_occupancy_depth_identity_hand_computed() {
    let resolution = resolve_point(
        LeadTime::Time(350.0),
        DeliveryAxis {
            study_stage_hours: &[100.0; 4],
            post_study_stage_hours: &[],
        },
    );

    let n_none = resolution
        .decider
        .iter()
        .take_while(|d| d.is_none())
        .count();
    assert_eq!(n_none, 3);

    let expected: Vec<usize> = resolution
        .depth
        .iter()
        .enumerate()
        .map(|(t, &d)| d + n_none.saturating_sub(1).saturating_sub(t))
        .collect();
    assert_eq!(resolution.occupancy, expected);
    assert_eq!(resolution.occupancy, vec![3, 2, 1, 0]);
}

// Edge E2: a `None`-decider whose delivery target is post-study
// (`decider[m] == None`, `m >= n_decision`) has no carrier and is excluded from
// occupancy. `LeadStages(4)` over `n_decision == 3`, `n_delivery == 6` puts
// delivery 3 (None, m == 3 == n_decision) in the post-study span; counting it
// would raise occupancy[0..3] by one.
#[test]
fn test_occupancy_excludes_none_decider_post_study_target() {
    let resolution = resolve_point(
        LeadTime::Stages(4),
        DeliveryAxis {
            study_stage_hours: &[720.0; 3],
            post_study_stage_hours: &[720.0; 3],
        },
    );

    assert_eq!(
        resolution.decider,
        vec![None, None, None, None, Some(0), Some(1)]
    );
    assert_eq!(resolution.decider[3], None);
    assert_eq!(resolution.depth, vec![1, 2, 2]);
    assert_eq!(resolution.occupancy, vec![3, 3, 2]);
}

// The bug doc's reproduction shape — LeadTime(1160) over three weeks then a
// month, plus one post-study month. The 1160 h lead resolves all four study
// deliveries pre-study, so at stage 0 four seeds are simultaneously in flight;
// occupancy subtracts the seed maturing at stage 0 and tops out at 3, while
// ring_depth covers the fourth.
#[test]
fn ring_depth_covers_every_simultaneous_pre_study_seed() {
    let resolution = resolve_point(
        LeadTime::Time(1160.0),
        DeliveryAxis {
            study_stage_hours: &[168.0, 168.0, 168.0, 648.0, 648.0][..4],
            post_study_stage_hours: &[168.0, 168.0, 168.0, 648.0, 648.0][4..],
        },
    );

    assert_eq!(resolution.decider, vec![None, None, None, None, Some(3)]);
    assert_eq!(resolution.occupancy, vec![3, 2, 1, 1]);
    assert_eq!(resolution.occupancy.iter().copied().max(), Some(3));
    assert_eq!(resolution.ring_depth(), 4);
}

// A uniform stage-count lead whose leading None-run never exceeds the in-flight
// occupancy: ring_depth is inert, agreeing bit-for-bit with the occupancy max
// the pre-fix sizing already used.
#[test]
fn ring_depth_equals_the_occupancy_max_when_no_seed_overflows() {
    let resolution = resolve_point(
        LeadTime::Stages(2),
        DeliveryAxis {
            study_stage_hours: &[720.0; 5],
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(resolution.occupancy, vec![2, 2, 2, 1, 0]);
    assert_eq!(
        resolution.ring_depth(),
        resolution.occupancy.iter().copied().max().unwrap_or(0)
    );
    assert_eq!(resolution.ring_depth(), 2);
}

// A plant with a fixed post-horizon window (g > 0): LeadStages(6) over three
// decision stages and eight deliveries leaves decider[3..6] a None run — the
// excised class-4 window, not ring occupancy. Counting the whole decider's
// leading None run would inflate ring_depth to 6; capping at the decision axis
// keeps it at the true depth 3.
#[test]
fn ring_depth_ignores_post_study_none_deciders() {
    let resolution = resolve_point(
        LeadTime::Stages(6),
        DeliveryAxis {
            study_stage_hours: &[720.0; 3],
            post_study_stage_hours: &[720.0; 5],
        },
    );

    assert_eq!(
        resolution.decider,
        vec![None, None, None, None, None, None, Some(0), Some(1)]
    );
    assert_eq!(resolution.occupancy, vec![3, 3, 2]);
    let whole_decider_leading_none = resolution
        .decider
        .iter()
        .take_while(|c| c.is_none())
        .count();
    assert_eq!(whole_decider_leading_none, 6);
    assert_eq!(resolution.ring_depth(), 3);
}

// AnticipatedResolution::resolve sizes k_max as the max ring_depth over plants:
// a lead-past-horizon plant (ring_depth 4, occupancy max 3) shares the axis with
// a shallower plant (ring_depth 2), and k_max is the deeper 4.
#[test]
fn resolve_sizes_k_max_from_the_deepest_plant_ring_depth() {
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Time(1160.0), LeadTime::Stages(2)],
        DeliveryAxis {
            study_stage_hours: &[168.0, 168.0, 168.0, 648.0, 648.0][..4],
            post_study_stage_hours: &[168.0, 168.0, 168.0, 648.0, 648.0][4..],
        },
    );

    assert_eq!(resolution.per_plant[0].ring_depth(), 4);
    assert_eq!(resolution.per_plant[1].ring_depth(), 2);
    assert_eq!(resolution.anchored_depth(), 4);
}

#[test]
fn ring_size_widens_resolution_depth_to_the_deepest_lead_stage() {
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Stages(3)],
        DeliveryAxis {
            study_stage_hours: &[720.0; 3],
            post_study_stage_hours: &[],
        },
    );

    assert_eq!(
        resolution.ring_size(&[1, 2]),
        3,
        "resolution depth dominant"
    );
    assert_eq!(resolution.ring_size(&[1, 5]), 5, "lead dominant");
    assert_eq!(
        resolution.ring_size(&[]),
        3,
        "empty lead list returns k_max"
    );
}

/// The carried in-flight set at decision stage `t` as the ring actually sweeps
/// it (`build_anticipated_slot_row_pos`): the window `m in {t+1, ..., t+k_max}`
/// capped at `n_delivery`, restricted to deliveries that are READY
/// (`PointResolution::is_ready_at` — a fresh deposit this stage or an
/// already-decided interior carry).
///
/// The modular slot key `m mod k_max` is a bijection on THIS window set, never
/// on [`carrier_predicate_targets`]'s occupancy set over `(t, n_delivery)`. The
/// two diverge when a plant's lead exceeds the horizon: a pre-study-decided
/// post-study delivery reads ready yet carries nothing, punching a hole that
/// makes the occupancy set's residues collide — while the ring only ever hosts
/// this contiguous window, so a delivery outside it occupies no slot and the
/// collision never reaches the LP. The two sets coincide whenever the lead
/// stays within the horizon.
fn ring_window_carried(
    point: &PointResolution,
    t: usize,
    k_max: usize,
    n_delivery: usize,
) -> Vec<usize> {
    ((t + 1)..(t + 1 + k_max).min(n_delivery))
        .filter(|&m| point.is_ready_at(m, t))
        .collect()
}

/// The occupancy set at decision stage `t`: delivery targets `m in (t,
/// n_delivery)` that lift `occupancy`/`depth` — a pre-study seed
/// (`decider[m] == None`, `m < n_decision`) or an at-or-before-`t` in-study
/// decision (`decider[m] == Some(c)`, `c <= t`). A post-study decider, or a
/// pre-study decider for a post-study target, has no carrier and never appears.
/// Its cardinality equals `occupancy[t]`; it is NOT the ring window
/// ([`ring_window_carried`]) when the lead exceeds the horizon.
fn carrier_predicate_targets(
    point: &PointResolution,
    t: usize,
    n_decision: usize,
    n_delivery: usize,
) -> Vec<usize> {
    ((t + 1)..n_delivery)
        .filter(|&m| match point.decider[m] {
            None => m < n_decision,
            Some(c) => c <= t,
        })
        .collect()
}

fn lead_time_strategy() -> impl Strategy<Value = LeadTime> {
    prop_oneof![
        (1u32..=12).prop_map(LeadTime::Stages),
        (24.0f64..=744.0).prop_map(LeadTime::Time),
    ]
}

prop_compose! {
    /// A resolver case over the full declared generated space: `n_decision in
    /// 1..=8`, `n_post in 0..=8`, a per-stage duration vector whose length is
    /// always `n_decision + n_post` (a shorter vector trips
    /// `resolve_decider_physical`'s own debug_assert and masks the property),
    /// and a `LeadStages(1..=12)` or `LeadTime(24.0..=744.0)` lead.
    fn resolver_case()(n_decision in 1usize..=8, n_post in 0usize..=8)(
        durations in prop::collection::vec(24.0f64..=744.0, n_decision + n_post),
        lead in lead_time_strategy(),
        n_decision in Just(n_decision),
        n_post in Just(n_post),
    ) -> (LeadTime, Vec<f64>, usize, usize) {
        (lead, durations, n_decision, n_decision + n_post)
    }
}

/// Fixed cases/seed so a failing shrink is reproducible run-to-run (the
/// determinism discipline applied to test generation).
fn resolver_property_config() -> ProptestConfig {
    ProptestConfig {
        cases: 256,
        rng_seed: RngSeed::Fixed(42),
        ..ProptestConfig::default()
    }
}

/// The mirror shape (lead equal to the decision-stage count over a post-study
/// calendar the same width) resolves a full-depth ring: every in-study stage
/// carries a complete occupancy, the ring width equals the decision-stage
/// count, and no stage fans out. Here the ring window and the occupancy set
/// coincide (nothing delivers post-study without a carrier), so the carried run
/// is exactly `{t+1, ..., n_decision + t}`.
#[test]
fn decomp_mirror_shape_resolves_full_depth_ring() {
    let anticipated = AnticipatedResolution::resolve(
        &[LeadTime::Stages(6)],
        DeliveryAxis {
            study_stage_hours: &[720.0; 6],
            post_study_stage_hours: &[720.0; 6],
        },
    );

    assert_eq!(anticipated.anchored_depth(), 6);
    assert_eq!(anticipated.max_fanout(), 1);

    let point = &anticipated.per_plant[0];
    assert_eq!(point.occupancy, vec![6; 6]);
    for t in 0..6 {
        assert_eq!(point.decider[t], None, "decider[{t}] must be pre-study");
        assert_eq!(
            point.decider[6 + t],
            Some(t),
            "decider[6 + {t}] must decide at stage {t}"
        );
    }
    assert_eq!(
        ring_window_carried(point, 0, anticipated.anchored_depth(), 12),
        vec![1, 2, 3, 4, 5, 6],
        "the full-depth ring at t=0 carries the contiguous run 1..=6"
    );
}

proptest! {
    #![proptest_config(resolver_property_config())]

    /// The modular slot key `m mod k_max` assigns a distinct residue to every
    /// delivery target the ring carries at each decision stage, over the whole
    /// generated space. Skips `k_max == 0`, where the ring is empty and the
    /// slot key is undefined.
    #[test]
    fn modular_slot_key_is_injective_on_the_carried_in_flight_set(case in resolver_case()) {
        let (lead, durations, n_decision, n_delivery) = case;
        let anticipated = AnticipatedResolution::resolve(
            &[lead],
            DeliveryAxis {
                study_stage_hours: &durations[..n_decision],
                post_study_stage_hours: &durations[n_decision..],
            },
        );
        let k_max = anticipated.anchored_depth();
        if k_max == 0 {
            return Ok(());
        }
        let point = &anticipated.per_plant[0];

        for t in 0..n_decision {
            let carried = ring_window_carried(point, t, k_max, n_delivery);
            let mut slot_owner: HashMap<usize, usize> = HashMap::new();
            for &m in &carried {
                let slot = m % k_max;
                let collided_with = slot_owner.insert(slot, m);
                prop_assert!(
                    collided_with.is_none(),
                    "modular slot collision at t={t}: targets {prev} and {m} both map to \
                     residue {slot} mod k_max={k_max}; carried set = {carried:?}",
                    prev = collided_with.unwrap_or(m),
                );
            }
        }
    }

    /// The ring's carried in-flight set at each decision stage is a contiguous
    /// run of delivery targets starting at `t + 1` (or empty when nothing is
    /// ready) — the reason the modular slot key stays injective. Holds over the
    /// whole generated space.
    #[test]
    fn carried_in_flight_set_is_contiguous(case in resolver_case()) {
        let (lead, durations, n_decision, n_delivery) = case;
        let anticipated = AnticipatedResolution::resolve(
            &[lead],
            DeliveryAxis {
                study_stage_hours: &durations[..n_decision],
                post_study_stage_hours: &durations[n_decision..],
            },
        );
        let k_max = anticipated.anchored_depth();
        if k_max == 0 {
            return Ok(());
        }
        let point = &anticipated.per_plant[0];

        for t in 0..n_decision {
            let carried = ring_window_carried(point, t, k_max, n_delivery);
            if carried.is_empty() {
                continue;
            }
            prop_assert_eq!(
                carried[0],
                t + 1,
                "carried run at t={} must start at t+1; got {:?}",
                t,
                carried
            );
            for pair in carried.windows(2) {
                prop_assert_eq!(
                    pair[1],
                    pair[0] + 1,
                    "carried run at t={} must be contiguous; gap in {:?}",
                    t,
                    carried
                );
            }
        }
    }
}

/// A pre-study decider whose delivery target is post-study
/// (`decider[m] == None`, `m >= n_decision`) has no carrier: it appears in no
/// per-stage carrier set and lifts neither `occupancy` nor `depth`.
#[test]
fn pre_study_decider_with_post_study_target_has_no_carrier() {
    let n_decision = 3;
    let n_delivery = 6;
    let study_stage_hours = vec![720.0; n_decision];
    let post_study_stage_hours = vec![720.0; n_delivery - n_decision];
    let resolution = resolve_point(
        LeadTime::Stages(4),
        DeliveryAxis {
            study_stage_hours: &study_stage_hours,
            post_study_stage_hours: &post_study_stage_hours,
        },
    );

    let m = 3;
    assert_eq!(resolution.decider[m], None);
    assert!(m >= n_decision, "target {m} must be post-study");
    assert!(
        resolution.decision_sets.iter().all(|set| !set.contains(&m)),
        "a pre-study decider must not appear in any decision set"
    );
    for t in 0..n_decision {
        let carried = carrier_predicate_targets(&resolution, t, n_decision, n_delivery);
        assert!(
            !carried.contains(&m),
            "target {m} must have no carrier, yet appears at t={t}: {carried:?}"
        );
        assert_eq!(
            carried.len(),
            resolution.occupancy[t],
            "the carrier set is the occupancy set at t={t}"
        );
    }
}

// ring_index/physical_target build PointResolution literals directly rather
// than routing through resolve_point — they are pure functions of decider
// and decision_sets.len(), so depth/occupancy are irrelevant filler here.

#[test]
fn ring_index_is_the_identity_without_a_post_study_none_run() {
    let point = PointResolution {
        decider: vec![None, None, Some(0), Some(1), Some(2)],
        decision_sets: vec![Vec::new(); 5],
        depth: vec![0; 5],
        occupancy: vec![0; 5],
    };

    for m in 0..5 {
        assert_eq!(point.ring_index(m), Some(m), "m={m}");
    }
    for r in 0..5 {
        assert_eq!(point.physical_target(r), r, "r={r}");
    }
}

#[test]
fn ring_index_excises_the_fixed_post_horizon_window() {
    let point = PointResolution {
        decider: vec![
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
        ],
        decision_sets: vec![Vec::new(); 4],
        depth: vec![0; 4],
        occupancy: vec![0; 4],
    };

    assert_eq!(point.ring_index(3), Some(3));
    assert_eq!(point.ring_index(4), None);
    assert_eq!(point.ring_index(5), None);
    assert_eq!(point.ring_index(6), None);
    assert_eq!(point.ring_index(7), Some(4));
    assert_eq!(point.ring_index(10), Some(7));
}

#[test]
fn physical_target_is_the_left_inverse_of_ring_index() {
    let point = PointResolution {
        decider: vec![
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
        ],
        decision_sets: vec![Vec::new(); 4],
        depth: vec![0; 4],
        occupancy: vec![0; 4],
    };

    for r in 0..8 {
        assert_eq!(
            point.ring_index(point.physical_target(r)),
            Some(r),
            "round-trip failed for r={r}"
        );
    }
}

#[test]
fn ring_index_degrades_to_the_identity_on_a_short_decider() {
    let empty = PointResolution {
        decider: Vec::new(),
        decision_sets: vec![Vec::new(); 5],
        depth: vec![0; 5],
        occupancy: vec![0; 5],
    };
    let short = PointResolution {
        decider: vec![None, Some(0)],
        decision_sets: vec![Vec::new(); 5],
        depth: vec![0; 5],
        occupancy: vec![0; 5],
    };

    for point in [&empty, &short] {
        for m in 0..5 {
            assert_eq!(point.ring_index(m), Some(m), "m={m}");
        }
        for r in 0..5 {
            assert_eq!(point.physical_target(r), r, "r={r}");
        }
    }
}

#[test]
#[should_panic(expected = "contiguous")]
fn fixed_post_horizon_width_rejects_a_non_contiguous_none_run() {
    let point = PointResolution {
        decider: vec![None, None, None, None, None, Some(0), None, Some(1)],
        decision_sets: vec![Vec::new(); 4],
        depth: vec![0; 4],
        occupancy: vec![0; 4],
    };

    let _ = point.fixed_post_horizon_width();
}

/// A post-study decision (`decider[m] == Some(c)`, `c >= n_decision`) is not an
/// in-study decision: it appears in no decision set and has no carrier, so it
/// lifts neither `occupancy` nor `depth`.
#[test]
fn post_study_decider_is_not_a_decision() {
    let n_decision = 3;
    let n_delivery = 6;
    let study_stage_hours = vec![720.0; n_decision];
    let post_study_stage_hours = vec![720.0; n_delivery - n_decision];
    let resolution = resolve_point(
        LeadTime::Stages(1),
        DeliveryAxis {
            study_stage_hours: &study_stage_hours,
            post_study_stage_hours: &post_study_stage_hours,
        },
    );

    let m = 4;
    let c = 3;
    assert_eq!(resolution.decider[m], Some(c));
    assert!(
        m >= n_decision && c >= n_decision,
        "decider {c} and target {m} must both be post-study"
    );
    assert!(
        resolution.decision_sets.iter().all(|set| !set.contains(&m)),
        "a post-study decision must not appear in any decision set"
    );
    for t in 0..n_decision {
        let carried = carrier_predicate_targets(&resolution, t, n_decision, n_delivery);
        assert!(
            !carried.contains(&m),
            "target {m} must have no carrier, yet appears at t={t}: {carried:?}"
        );
        assert_eq!(
            carried.len(),
            resolution.occupancy[t],
            "the carrier set is the occupancy set at t={t}"
        );
    }
}
