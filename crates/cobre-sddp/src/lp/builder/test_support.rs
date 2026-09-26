//! Shared `#[cfg(test)]` fixtures for the split builder representation modules.

use std::collections::HashMap;

use chrono::NaiveDate;
use cobre_core::{
    Block, BlockMode, HydroPenalties, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
    StageStateConfig, System,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::indexer::{AnticipatedPlants, StateSpace};
use crate::lead_time::{AnticipatedResolution, SpreadResolution};
use crate::setup::bucket_topology::build_transit_bucket_topology;
use crate::setup::resolve_anticipated_commitments_core;

use super::layout::TemplateBuildCtx;

/// Resolve the anticipated-resolution / lead-stages / per-stage-mask / arc-table /
/// `max_par_order` inputs through the same setup entry points production uses
/// (`resolve_state_layout`'s formula for `max_par_order`,
/// `build_transit_bucket_topology`'s single arc-table derivation), so a
/// builder-module test's `TemplateBuildCtx` matches what setup builds for the same
/// system.
// Rationale: the tuple threads distinct single-owner fixture outputs to one
// call site; a named struct would add ceremony without reducing the shape.
#[allow(clippy::type_complexity)]
pub(super) fn ctx_anticipated_and_mask_inputs(
    system: &System,
    par_lp: &PrecomputedPar,
) -> (
    AnticipatedResolution,
    Vec<usize>,
    Vec<Vec<usize>>,
    HashMap<usize, Vec<Vec<f64>>>,
    HashMap<usize, Vec<Option<SpreadResolution>>>,
    HashMap<usize, Vec<Option<Vec<f64>>>>,
    usize,
) {
    let (resolution, lead_stages) =
        resolve_anticipated_commitments_core(system, &AnticipatedPlants::build(system.thermals()));
    let topology = build_transit_bucket_topology(system, false);
    let max_par_order = system
        .inflow_models()
        .iter()
        .filter(|m| m.stage_id >= 0)
        .map(|m| m.ar_coefficients.len())
        .max()
        .unwrap_or(0)
        .max(par_lp.max_order());
    (
        resolution,
        lead_stages,
        topology.per_stage_mask,
        topology.arc_stage_weights,
        topology.arc_spread_chrono,
        topology.arc_arrival_density,
        max_par_order,
    )
}

/// Build the role-(a) [`StateSpace`] a `StageLayout` borrows, from a test
/// [`TemplateBuildCtx`]. Mirrors `crate::setup::resolve_state_layout` (same state
/// dimensions and PAR-derived lag counts), so the handle is byte-identical to
/// production's.
pub(super) fn state_layout_for(ctx: &TemplateBuildCtx<'_>) -> StateSpace {
    let effective_lag_counts: Vec<usize> = if ctx.max_par_order > 0 {
        (0..ctx.n_hydros)
            .map(|h| {
                if h < ctx.par_lp.n_hydros() {
                    ctx.par_lp.effective_lag_count(h)
                } else {
                    ctx.max_par_order
                }
            })
            .collect()
    } else {
        vec![0; ctx.n_hydros]
    };
    StateSpace::new(
        ctx.n_hydros,
        ctx.max_par_order,
        0,
        Vec::new(),
        ctx.n_anticipated,
        ctx.anticipated_resolution
            .ring_size(&ctx.anticipated_lead_stages),
        ctx.anticipated_lead_stages.clone(),
        &effective_lag_counts,
    )
}

/// [`state_layout_for`] plus the ctx's attached `AnticipatedResolution`. Tests
/// asserting `anticipated_resolution_for` byte-identity with production must build
/// through this — [`state_layout_for`] leaves the constant-lead fallback active.
pub(super) fn state_layout_with_resolution(ctx: &TemplateBuildCtx<'_>) -> StateSpace {
    let mut state = state_layout_for(ctx);
    state.set_anticipated_resolution(ctx.anticipated_resolution.clone());
    state
}

/// All-zero `HydroPenalties` so no fixture-side penalty cost contaminates the
/// column/objective assertions in the builder tests.
pub(super) fn zero_hydro_penalties() -> HydroPenalties {
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

/// Block-hour durations shared by every [`three_block_stage`] caller.
pub(super) const BLOCK_HOURS: [f64; 3] = [200.0, 300.0, 244.0];

/// Block count matching [`BLOCK_HOURS`].
pub(super) const N_BLKS: usize = 3;

/// Three-block `Stage` at `index`, durations [`BLOCK_HOURS`].
pub(super) fn three_block_stage(index: usize) -> Stage {
    let mut stage = two_block_stage(index, [BLOCK_HOURS[0], BLOCK_HOURS[1]]);
    stage.blocks = BLOCK_HOURS
        .iter()
        .enumerate()
        .map(|(i, &h)| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: h,
        })
        .collect();
    stage
}

/// Two-block `Stage` at `index` with per-block `durations`.
///
/// The two durations are passed by the caller rather than fixed because several
/// callers deliberately use unequal blocks (e.g. `[300.0, 444.0]`): equal
/// durations would mask a per-block divisor confusion in the code under test.
pub(super) fn two_block_stage(index: usize, durations: [f64; 2]) -> Stage {
    Stage {
        index,
        id: index as i32,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![
            Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: durations[0],
            },
            Block {
                index: 1,
                name: "BLK1".to_string(),
                duration_hours: durations[1],
            },
        ],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }
}
