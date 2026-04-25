//! Layer 5b — stage-structure semantic validation.
//!
//! Validates policy-graph transitions (`source_id`/`target_id`
//! existence), outgoing-transition probability-sum invariant,
//! cyclic-graph discount rate, block durations, and `CVaR`
//! parameters.

use std::collections::{HashMap, HashSet};

use cobre_core::temporal::{PolicyGraphType, StageRiskConfig};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};
use super::PROB_TOLERANCE;

/// Validates policy graph transitions, block durations, and `CVaR` parameters.
pub(super) fn check_stage_structure(data: &ParsedData, ctx: &mut ValidationContext) {
    let graph = &data.stages.policy_graph;
    let stages = &data.stages.stages;

    // Build a set of all valid stage IDs for fast membership tests.
    let stage_ids: HashSet<i32> = stages.iter().map(|s| s.id).collect();

    // Rule 1: Every source_id and target_id in transitions must be a valid stage ID.
    for transition in &graph.transitions {
        if !stage_ids.contains(&transition.source_id) {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "transition source_id {} does not refer to a valid stage ID",
                    transition.source_id
                ),
            );
        }
        if !stage_ids.contains(&transition.target_id) {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "transition target_id {} does not refer to a valid stage ID",
                    transition.target_id
                ),
            );
        }
    }

    // Rule 2: For each unique source_id, outgoing probability sum must be ≈ 1.0.
    // Group transitions by source_id and sum probabilities.
    let mut prob_sums: HashMap<i32, f64> = HashMap::new();
    for transition in &graph.transitions {
        *prob_sums.entry(transition.source_id).or_insert(0.0) += transition.probability;
    }
    let mut sorted_sources: Vec<i32> = prob_sums.keys().copied().collect();
    sorted_sources.sort_unstable();
    for source_id in sorted_sources {
        let total = prob_sums[&source_id];
        if (total - 1.0).abs() > PROB_TOLERANCE {
            ctx.add_error(
                ErrorKind::InvalidValue,
                "stages.json",
                None::<&str>,
                format!(
                    "outgoing transition probabilities from stage {source_id} sum to {total:.8} \
                     (expected 1.0 ±{PROB_TOLERANCE}); probability must sum to 1.0"
                ),
            );
        }
    }

    // Rule 3: Cyclic graphs require annual_discount_rate > 0.0.
    if graph.graph_type == PolicyGraphType::Cyclic && graph.annual_discount_rate <= 0.0 {
        ctx.add_error(
            ErrorKind::InvalidValue,
            "stages.json",
            None::<&str>,
            format!(
                "cyclic policy graph requires annual_discount_rate > 0.0 for convergence, \
                 got {}",
                graph.annual_discount_rate
            ),
        );
    }

    // Rule 4: Every Block.duration_hours must be > 0.0.
    for stage in stages {
        for block in &stage.blocks {
            if block.duration_hours <= 0.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: block has duration_hours {} which is not > 0.0; \
                         block duration must be positive",
                        stage.id, block.duration_hours
                    ),
                );
            }
        }
    }

    // Rule 5: CVaR alpha must be in (0, 1] and lambda must be in [0, 1].
    for stage in stages {
        if let StageRiskConfig::CVaR { alpha, lambda } = stage.risk_config {
            if alpha <= 0.0 || alpha > 1.0 {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: CVaR alpha ({alpha}) must be in (0, 1]; \
                         alpha must be a valid tail probability",
                        stage.id
                    ),
                );
            }
            if !(0.0..=1.0).contains(&lambda) {
                ctx.add_error(
                    ErrorKind::InvalidValue,
                    "stages.json",
                    Some(format!("Stage {}", stage.id)),
                    format!(
                        "Stage {}: CVaR lambda ({lambda}) must be in [0, 1]; \
                         lambda is the CVaR mixing weight",
                        stage.id
                    ),
                );
            }
        }
    }
}
