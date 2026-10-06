//! Layer 5b — stage-structure semantic validation.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use cobre_core::scenario::{SamplingScheme, ScenarioSource};
use cobre_core::temporal::{Node, PolicyGraphType, Transition};

use crate::StageIdResolver;
use crate::config::{ForwardPassesResolution, Openings};

use super::super::{ValidationContext, rules, schema::ParsedData};
use super::PROB_TOLERANCE;

/// Rules 1-3: validates policy graph transitions, outgoing transition
/// probabilities, and the cyclic-graph discount rate.
pub(super) fn check_stage_structure(data: &ParsedData, ctx: &mut ValidationContext) {
    let graph = &data.stages.policy_graph;
    let stages = &data.stages.stages;

    let stage_ids: HashSet<i32> = stages.iter().map(|s| s.id).collect();

    // Endpoints are node ids when nodes[] is declared, stage ids otherwise; a chain
    // declared with node ids satisfies both readings and any fan has an endpoint that
    // is not a valid stage id, so the node-mode endpoint check (check_node_graph)
    // catches it rather than this stage-id check silently rejecting valid node ids.
    if graph.nodes.is_empty() {
        for transition in &graph.transitions {
            if !stage_ids.contains(&transition.source_id) {
                ctx.emit(
                    &rules::SEMANTIC_TRANSITION_ENDPOINT_NOT_A_STAGE,
                    "stages.json",
                    None::<&str>,
                    format!(
                        "transition source_id {} does not refer to a valid stage ID",
                        transition.source_id
                    ),
                );
            }
            if !stage_ids.contains(&transition.target_id) {
                ctx.emit(
                    &rules::SEMANTIC_TRANSITION_ENDPOINT_NOT_A_STAGE,
                    "stages.json",
                    None::<&str>,
                    format!(
                        "transition target_id {} does not refer to a valid stage ID",
                        transition.target_id
                    ),
                );
            }
        }
    }

    let mut prob_sums: HashMap<i32, f64> = HashMap::new();
    for transition in &graph.transitions {
        *prob_sums.entry(transition.source_id).or_insert(0.0) += transition.probability;
    }
    let mut sorted_sources: Vec<i32> = prob_sums.keys().copied().collect();
    sorted_sources.sort_unstable();
    for source_id in sorted_sources {
        let total = prob_sums[&source_id];
        if (total - 1.0).abs() > PROB_TOLERANCE {
            ctx.emit(
                &rules::SEMANTIC_TRANSITION_PROBABILITY_SUM,
                "stages.json",
                None::<&str>,
                format!(
                    "outgoing transition probabilities from stage {source_id} sum to {total:.8} \
                     (expected 1.0 ±{PROB_TOLERANCE}); probability must sum to 1.0"
                ),
            );
        }
    }

    if graph.graph_type == PolicyGraphType::Cyclic && graph.annual_discount_rate <= 0.0 {
        ctx.emit(
            &rules::SEMANTIC_CYCLIC_GRAPH_DISCOUNT_RATE,
            "stages.json",
            None::<&str>,
            format!(
                "cyclic policy graph requires annual_discount_rate > 0.0 for convergence, \
                 got {}",
                graph.annual_discount_rate
            ),
        );
    }
}

/// Rule 34. Warns once when an autoregressive inflow model (PAR order `p > 0`) coexists
/// with every study stage having `state_config.inflow_lags == false`.
///
/// AR order is read as the maximum 1-based `lag` over `inflow_ar_coefficients`;
/// an empty table (no AR model, or white-noise order 0) yields order 0 and is
/// silent. Only study stages (`id >= 0`) are considered, so pre-study seed
/// stages do not satisfy the all-disabled condition on their own.
pub(super) fn check_inflow_lags_vs_par_order(data: &ParsedData, ctx: &mut ValidationContext) {
    let max_order: i32 = data
        .inflow_ar_coefficients
        .iter()
        .map(|c| c.lag)
        .max()
        .unwrap_or(0);
    if max_order == 0 {
        return;
    }

    let mut study_stages = data.stages.stages.iter().filter(|s| s.id >= 0).peekable();
    if study_stages.peek().is_none() {
        return;
    }
    if !study_stages.all(|s| !s.state_config.inflow_lags) {
        return;
    }

    ctx.emit(
        &rules::SEMANTIC_INFLOW_LAGS_DISABLED_UNDER_AR_MODEL,
        "stages.json",
        None::<&str>,
        format!(
            "inflow lags are disabled on all study stages (state_variables.inflow_lags = false) \
             despite a PAR(p>0) inflow model (AR order {max_order}), so the inflow-lag dimensions \
             are omitted from the per-stage state. This is a valid configuration for external-solver \
             interoperability; otherwise it is likely a misconfiguration"
        ),
    );
}

/// Validates a declared policy-graph `nodes[]` (node-native dialect): the
/// `scenario_id` conditional requirement (rule 36) and pointer bound
/// (rule 37) per slot-occupying external class, structural well-formedness
/// (rule 38: unknown/duplicate node id, unresolved stage, empty stage,
/// unreachable node, cycle, mid-horizon leaf), the every-edge `t → t+1` rule
/// (rule 39), and the recombinable-signature warning (rule 40). A no-op when
/// `nodes` is empty — the chain dialect is unchanged (its endpoints are stage
/// ids, validated by [`check_stage_structure`]).
pub(super) fn check_node_graph(data: &ParsedData, ctx: &mut ValidationContext) {
    let graph = &data.stages.policy_graph;
    if graph.nodes.is_empty() {
        return;
    }
    let nodes = &graph.nodes;
    let transitions = &graph.transitions;

    let (study_ids, resolver) = study_stage_ids_and_resolver(data);
    let n_stages = study_ids.len();

    let stage_index: Vec<Option<usize>> = nodes
        .iter()
        .map(|n| {
            let idx = resolver.resolve(n.stage_id);
            if idx.is_none() {
                ctx.emit(
                    &rules::SEMANTIC_NODE_GRAPH_MALFORMED,
                    "stages.json",
                    Some(format!("node {}", n.id)),
                    format!(
                        "node {} references stage_id {} which is not a declared study stage; \
                         declared study stage ids: {study_ids:?}",
                        n.id, n.stage_id
                    ),
                );
            }
            idx
        })
        .collect();

    let mut id_to_pos: HashMap<i32, usize> = HashMap::new();
    let mut dup = false;
    for (pos, n) in nodes.iter().enumerate() {
        if id_to_pos.insert(n.id, pos).is_some() {
            dup = true;
            ctx.emit(
                &rules::SEMANTIC_NODE_ID_DUPLICATE,
                "stages.json",
                Some(format!("node {}", n.id)),
                format!("duplicate policy-graph node id {}", n.id),
            );
        }
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut endpoints_ok = true;
    for tr in transitions {
        match (id_to_pos.get(&tr.source_id), id_to_pos.get(&tr.target_id)) {
            (Some(&sp), Some(&tp)) => children[sp].push(tp),
            (s, t) => {
                endpoints_ok = false;
                if s.is_none() {
                    ctx.emit(
                        &rules::SEMANTIC_NODE_GRAPH_MALFORMED,
                        "stages.json",
                        None::<&str>,
                        format!(
                            "transition source_id {} does not refer to a declared node",
                            tr.source_id
                        ),
                    );
                }
                if t.is_none() {
                    ctx.emit(
                        &rules::SEMANTIC_NODE_GRAPH_MALFORMED,
                        "stages.json",
                        None::<&str>,
                        format!(
                            "transition target_id {} does not refer to a declared node",
                            tr.target_id
                        ),
                    );
                }
            }
        }
    }

    check_realization_rules(data, nodes, &resolver, ctx);
    check_empty_stages(&stage_index, &study_ids, ctx);
    let edges_ok = check_edges_t_plus_1(nodes, transitions, &stage_index, &id_to_pos, ctx);

    let walkable = !dup && endpoints_ok && stage_index.iter().all(Option::is_some);
    if walkable {
        let acyclic = check_no_cycle(nodes, &children, ctx);
        check_reachable_nodes(nodes, &children, &stage_index, ctx);
        check_no_mid_horizon_leaf(nodes, &children, &stage_index, n_stages, ctx);
        if acyclic && edges_ok {
            check_recombinable_signature(nodes, &children, &stage_index, &study_ids, ctx);
        }
    }
}

fn study_stage_ids_and_resolver(data: &ParsedData) -> (Vec<i32>, StageIdResolver) {
    let study_ids: Vec<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let resolver = StageIdResolver::from_study_stage_ids(&study_ids);
    (study_ids, resolver)
}

/// Rules 36 and 37, applied
/// only under enumerated forward selection: a node's `scenario_id` is required
/// exactly when its stage carries a slot-occupying external class and, when
/// present, must index a valid column of every such class. Inert under any other
/// selection — a `scenario_id` under sampled forwards carries no meaning here and
/// is a setup rejection, not a check in this layer.
fn check_realization_rules(
    data: &ParsedData,
    nodes: &[Node],
    resolver: &StageIdResolver,
    ctx: &mut ValidationContext,
) {
    let Some(ForwardPassesResolution::Enumerated) = data.config.resolve_forward_passes() else {
        return;
    };
    let Ok(source) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return;
    };
    let occupancy = stage_slot_occupancy(data, &source, resolver);

    for node in nodes {
        let Some(idx) = resolver.resolve(node.stage_id) else {
            continue;
        };
        let classes = &occupancy[idx];

        match (classes.is_empty(), node.scenario_id) {
            (false, None) => ctx.emit(
                &rules::SEMANTIC_NODE_SCENARIO_ID_DECLARATION,
                "stages.json",
                Some(format!("node {}", node.id)),
                format!(
                    "node {} at stage {} carries a slot-occupying external class but has no \
                     scenario_id; scenario_id is required there",
                    node.id, node.stage_id
                ),
            ),
            (true, Some(k)) => ctx.emit(
                &rules::SEMANTIC_NODE_SCENARIO_ID_DECLARATION,
                "stages.json",
                Some(format!("node {}", node.id)),
                format!(
                    "node {} at stage {} declares scenario_id {k} but its stage carries no \
                     slot-occupying external class; scenario_id is meaningless there",
                    node.id, node.stage_id
                ),
            ),
            _ => {}
        }

        if let Some(k) = node.scenario_id {
            for (class_name, raw_c) in classes {
                if usize::try_from(k).map_or(true, |ku| ku >= *raw_c) {
                    ctx.emit(
                        &rules::SEMANTIC_NODE_SCENARIO_ID_RANGE,
                        "stages.json",
                        Some(format!("node {}", node.id)),
                        format!(
                            "node {} scenario_id {k} is out of range for external class \
                             '{class_name}' at stage {}: must be in [0, {raw_c})",
                            node.id, node.stage_id
                        ),
                    );
                }
            }
        }
    }
}

/// The external classes that occupy a realization slot at each resolved study
/// stage: those set to the external scheme in the training scenario source (the
/// phase that governs slot occupancy) that also carry at least one row there,
/// paired with their per-stage raw column count. Indexed by resolved
/// study-stage index (`resolver.resolve(stage_id)`); one pass over each
/// external table fills every stage's entry.
fn stage_slot_occupancy(
    data: &ParsedData,
    source: &ScenarioSource,
    resolver: &StageIdResolver,
) -> Vec<Vec<(&'static str, usize)>> {
    let mut occupancy = vec![Vec::new(); resolver.study_stage_ids().len()];
    if source.inflow_scheme == SamplingScheme::External {
        accumulate_slot_occupancy(
            &mut occupancy,
            resolver,
            "inflow",
            data.external_scenarios
                .iter()
                .map(|r| (r.stage_id, r.scenario_id)),
        );
    }
    if source.load_scheme == SamplingScheme::External {
        accumulate_slot_occupancy(
            &mut occupancy,
            resolver,
            "load",
            data.external_load_scenarios
                .iter()
                .map(|r| (r.stage_id, r.scenario_id)),
        );
    }
    if source.ncs_scheme == SamplingScheme::External {
        accumulate_slot_occupancy(
            &mut occupancy,
            resolver,
            "ncs",
            data.external_ncs_scenarios
                .iter()
                .map(|r| (r.stage_id, r.scenario_id)),
        );
    }
    occupancy
}

fn accumulate_slot_occupancy(
    occupancy: &mut [Vec<(&'static str, usize)>],
    resolver: &StageIdResolver,
    class_name: &'static str,
    rows: impl Iterator<Item = (i32, i32)>,
) {
    // raw_c is the distinct scenario_id count among a class's rows at a stage —
    // a per-class count, deliberately NOT rows / n_entities; the cross-class
    // agreement and the exact {0..raw_c-1} set are enforced by the
    // library-consistency validation layer, not here.
    let mut seen = vec![HashSet::new(); occupancy.len()];
    for (stage_id, scenario_id) in rows {
        if let Some(idx) = resolver.resolve(stage_id) {
            seen[idx].insert(scenario_id);
        }
    }
    for (idx, ids) in seen.into_iter().enumerate() {
        let raw_c = ids.len();
        if raw_c > 0 {
            occupancy[idx].push((class_name, raw_c));
        }
    }
}

/// Rule 38 (empty stage): every study stage must carry at least one node.
fn check_empty_stages(
    stage_index: &[Option<usize>],
    study_ids: &[i32],
    ctx: &mut ValidationContext,
) {
    let mut occupied = vec![false; study_ids.len()];
    for idx in stage_index.iter().flatten() {
        occupied[*idx] = true;
    }
    for (t, &occ) in occupied.iter().enumerate() {
        if !occ {
            let sid = study_ids[t];
            ctx.emit(
                &rules::SEMANTIC_NODE_GRAPH_MALFORMED,
                "stages.json",
                Some(format!("stage {sid}")),
                format!(
                    "study stage {sid} has no policy-graph node; every stage must carry at least \
                     one node"
                ),
            );
        }
    }
}

/// Rule 39: every graph edge advances exactly one stage (`t → t+1`).
fn check_edges_t_plus_1(
    nodes: &[Node],
    transitions: &[Transition],
    stage_index: &[Option<usize>],
    id_to_pos: &HashMap<i32, usize>,
    ctx: &mut ValidationContext,
) -> bool {
    let mut ok = true;
    for tr in transitions {
        let (Some(&sp), Some(&tp)) = (id_to_pos.get(&tr.source_id), id_to_pos.get(&tr.target_id))
        else {
            continue;
        };
        let (Some(si), Some(ti)) = (stage_index[sp], stage_index[tp]) else {
            continue;
        };
        if ti != si + 1 {
            ok = false;
            ctx.emit(
                &rules::SEMANTIC_NODE_EDGE_SKIPS_STAGE,
                "stages.json",
                None::<&str>,
                format!(
                    "transition from node {} (stage {}) to node {} (stage {}) does not advance \
                     exactly one stage; every edge must go t -> t+1",
                    tr.source_id, nodes[sp].stage_id, tr.target_id, nodes[tp].stage_id
                ),
            );
        }
    }
    ok
}

/// Rule 38 (cycle): the node graph must be acyclic. Returns `true` when acyclic.
fn check_no_cycle(nodes: &[Node], children: &[Vec<usize>], ctx: &mut ValidationContext) -> bool {
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Gray,
        Black,
    }
    fn visit(node: usize, children: &[Vec<usize>], color: &mut [Color]) -> bool {
        color[node] = Color::Gray;
        for &c in &children[node] {
            match color[c] {
                Color::White => {
                    if visit(c, children, color) {
                        return true;
                    }
                }
                Color::Gray => return true,
                Color::Black => {}
            }
        }
        color[node] = Color::Black;
        false
    }
    let mut color = vec![Color::White; nodes.len()];
    let mut found = false;
    for start in 0..nodes.len() {
        if color[start] == Color::White && visit(start, children, &mut color) {
            found = true;
            break;
        }
    }
    if found {
        ctx.emit(
            &rules::SEMANTIC_NODE_GRAPH_CYCLE,
            "stages.json",
            None::<&str>,
            "policy-graph nodes contain a cycle; the node graph must be acyclic".to_string(),
        );
    }
    !found
}

/// Rule 38 (unreachable node): every node must be reachable from a stage-0 node.
fn check_reachable_nodes(
    nodes: &[Node],
    children: &[Vec<usize>],
    stage_index: &[Option<usize>],
    ctx: &mut ValidationContext,
) {
    let roots: Vec<usize> = (0..nodes.len())
        .filter(|&p| stage_index[p] == Some(0))
        .collect();
    if roots.is_empty() {
        return;
    }
    let mut reachable = vec![false; nodes.len()];
    let mut stack = roots;
    for &r in &stack {
        reachable[r] = true;
    }
    while let Some(p) = stack.pop() {
        for &c in &children[p] {
            if !reachable[c] {
                reachable[c] = true;
                stack.push(c);
            }
        }
    }
    for (pos, node) in nodes.iter().enumerate() {
        if !reachable[pos] {
            ctx.emit(
                &rules::SEMANTIC_NODE_GRAPH_MALFORMED,
                "stages.json",
                Some(format!("node {}", node.id)),
                format!(
                    "node {} (stage {}) is unreachable from the root stage",
                    node.id, node.stage_id
                ),
            );
        }
    }
}

/// Rule 38 (mid-horizon leaf): a node with no successors must sit at the final
/// stage; a leaf before the horizon end is rejected.
fn check_no_mid_horizon_leaf(
    nodes: &[Node],
    children: &[Vec<usize>],
    stage_index: &[Option<usize>],
    n_stages: usize,
    ctx: &mut ValidationContext,
) {
    for (pos, node) in nodes.iter().enumerate() {
        if children[pos].is_empty()
            && let Some(idx) = stage_index[pos]
            && idx + 1 < n_stages
        {
            ctx.emit(
                &rules::SEMANTIC_NODE_GRAPH_MALFORMED,
                "stages.json",
                Some(format!("node {}", node.id)),
                format!(
                    "node {} at stage {} has no successors but is not at the final stage \
                     (mid-horizon leaf); a node without successors must be terminal",
                    node.id, node.stage_id
                ),
            );
        }
    }
}

/// Rule 40 (recombinable-signature warning): warns once per stage carrying two
/// or more nodes with structurally identical subtrees (same shape and
/// realization pointers) — the signature of a graph the author probably meant to
/// recombine. Runs only on an acyclic `t → t+1` graph.
fn check_recombinable_signature(
    nodes: &[Node],
    children: &[Vec<usize>],
    stage_index: &[Option<usize>],
    study_ids: &[i32],
    ctx: &mut ValidationContext,
) {
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by_key(|&p| Reverse(stage_index[p].unwrap_or_default()));

    let mut sig: Vec<String> = vec![String::new(); nodes.len()];
    for &p in &order {
        let mut child_positions = children[p].clone();
        child_positions.sort_by_key(|&c| nodes[c].id);
        let joined = child_positions
            .iter()
            .map(|&c| sig[c].clone())
            .collect::<Vec<_>>()
            .join(",");
        sig[p] = format!("r{:?};[{joined}]", nodes[p].scenario_id);
    }

    for (t, &sid) in study_ids.iter().enumerate() {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        let mut warned = false;
        for (pos, s) in sig.iter().enumerate() {
            if stage_index[pos] != Some(t) {
                continue;
            }
            let count = seen.entry(s.as_str()).or_insert(0);
            *count += 1;
            if *count == 2 && !warned {
                warned = true;
                ctx.emit(
                    &rules::SEMANTIC_NODE_RECOMBINABLE_SUBTREES,
                    "stages.json",
                    Some(format!("stage {sid}")),
                    format!(
                        "stage {sid} carries multiple nodes with structurally identical \
                         subtrees (same shape and realization pointers); without recombination \
                         this trains independent chains with fewer cuts each"
                    ),
                );
            }
        }
    }
}

/// Rule 41: `num_openings` is required at a stage carrying generated openings
/// (no slot-occupying external class) and rejected as meaningless where a stage
/// carries only external openings. Declared `nodes[]` only — the chain dialect's
/// requiredness is a parse-layer check ([`crate::stages::parse_stages`]). Generated
/// openings are determined by [`stage_slot_occupancy`], the same authoritative
/// external-class machinery rule 36 uses.
pub(super) fn check_num_openings_declaration(data: &ParsedData, ctx: &mut ValidationContext) {
    let graph = &data.stages.policy_graph;
    if graph.nodes.is_empty() {
        return;
    }
    let Ok(source) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return;
    };
    let (_, resolver) = study_stage_ids_and_resolver(data);
    let occupancy = stage_slot_occupancy(data, &source, &resolver);

    let mut staged: Vec<i32> = graph
        .nodes
        .iter()
        .filter(|n| resolver.resolve(n.stage_id).is_some())
        .map(|n| n.stage_id)
        .collect::<HashSet<i32>>()
        .into_iter()
        .collect();
    staged.sort_unstable();

    for stage_id in staged {
        let Some(idx) = resolver.resolve(stage_id) else {
            continue;
        };
        let generated = occupancy[idx].is_empty();
        let declared = data.stages.openings_declared.contains(&stage_id);
        match (generated, declared) {
            (true, false) => ctx.emit(
                &rules::SEMANTIC_NUM_OPENINGS_DECLARATION,
                "stages.json",
                Some(format!("stage {stage_id}")),
                format!(
                    "stage {stage_id} carries generated openings but declares no num_openings; \
                     num_openings is required there"
                ),
            ),
            (false, true) => ctx.emit(
                &rules::SEMANTIC_NUM_OPENINGS_DECLARATION,
                "stages.json",
                Some(format!("stage {stage_id}")),
                format!(
                    "stage {stage_id} carries only external openings but declares num_openings; \
                     num_openings is meaningless there"
                ),
            ),
            _ => {}
        }
    }
}

/// Rule 42: a per-edge `annual_discount_rate_override` is rejected under
/// `nodes[]` — the override is a per-stage quantity declared on
/// `stages[].annual_discount_rate_override`. The edge spelling stays legal in the
/// chain dialect. It is rejected rather than folded into the stage because the
/// cumulative discount factor is baked into each stage's template objective
/// coefficients, so a per-edge — hence potentially per-node — rate would force
/// per-node templates.
pub(super) fn check_edge_discount_override_under_nodes(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let graph = &data.stages.policy_graph;
    if graph.nodes.is_empty() {
        return;
    }
    for tr in &graph.transitions {
        if tr.annual_discount_rate_override.is_some() {
            ctx.emit(
                &rules::SEMANTIC_EDGE_DISCOUNT_OVERRIDE_UNDER_NODES,
                "stages.json",
                None::<&str>,
                format!(
                    "transition {} -> {} declares annual_discount_rate_override under a node graph; \
                     declare it on stages[].annual_discount_rate_override instead",
                    tr.source_id, tr.target_id
                ),
            );
        }
    }
}

/// Rule 43: `openings = {source: file}` supplies a user-declared backward
/// opening tree — a sampled, chain-dialect feature. It is rejected together with
/// a declared `nodes[]` policy graph (nodes[] declares the opening set) and under
/// enumerated forward selection (which consumes external columns, not a
/// generated/file tree). Keyed on the declaration, not file existence — an
/// undeclared `noise_openings.parquet` is ignored, so its presence is not a check
/// here.
pub(super) fn check_nodes_and_noise_openings(data: &ParsedData, ctx: &mut ValidationContext) {
    if !matches!(data.config.training_openings(), Some(Openings::File {})) {
        return;
    }

    if !data.stages.policy_graph.nodes.is_empty() {
        ctx.emit(
            &rules::SEMANTIC_FILE_OPENINGS_CONFLICT,
            "config.json",
            None::<&str>,
            "openings = {source: file} supplies a user opening tree, which conflicts with a \
             declared nodes[] policy graph (nodes[] declares the opening set); declare one or the \
             other"
                .to_string(),
        );
    }

    if matches!(
        data.config.resolve_forward_passes(),
        Some(ForwardPassesResolution::Enumerated)
    ) {
        ctx.emit(
            &rules::SEMANTIC_FILE_OPENINGS_CONFLICT,
            "config.json",
            None::<&str>,
            "openings = {source: file} supplies a generated backward opening tree, which is not \
             consumed under enumerated forward selection; use sampled selection"
                .to_string(),
        );
    }
}

/// Rule 44: `sampling_method` is inert under external openings and ill-defined
/// at a multi-node stage; a warning names such a stage and the load succeeds.
/// Declared `nodes[]` only — the chain dialect's `sampling_method` is unchanged.
pub(super) fn check_sampling_method_meaningfulness(data: &ParsedData, ctx: &mut ValidationContext) {
    let graph = &data.stages.policy_graph;
    if graph.nodes.is_empty() {
        return;
    }
    let source = data
        .config
        .training_scenario_source(Path::new("config.json"))
        .ok();
    let (_, resolver) = study_stage_ids_and_resolver(data);
    let occupancy = source
        .as_ref()
        .map(|s| stage_slot_occupancy(data, s, &resolver));

    let mut node_count: HashMap<i32, usize> = HashMap::new();
    for n in &graph.nodes {
        if resolver.resolve(n.stage_id).is_some() {
            *node_count.entry(n.stage_id).or_insert(0) += 1;
        }
    }
    let mut staged: Vec<i32> = node_count.keys().copied().collect();
    staged.sort_unstable();

    for stage_id in staged {
        let multi_node = node_count[&stage_id] >= 2;
        let external = occupancy.as_ref().is_some_and(|occ| {
            resolver
                .resolve(stage_id)
                .is_some_and(|idx| !occ[idx].is_empty())
        });
        if multi_node || external {
            ctx.emit(
                &rules::SEMANTIC_SAMPLING_METHOD_INERT,
                "stages.json",
                Some(format!("stage {stage_id}")),
                format!(
                    "stage {stage_id}: sampling_method is ignored here (inert under external \
                     openings, ill-defined at a multi-node stage)"
                ),
            );
        }
    }
}

/// Rule 52: every study stage declares at least one block, and every block's
/// `duration_hours` is finite and positive — the stage-average stored-energy
/// columns divide by a stage's summed block hours with no guard.
pub(super) fn check_study_stage_blocks(data: &ParsedData, ctx: &mut ValidationContext) {
    for stage in data.stages.stages.iter().filter(|s| s.id >= 0) {
        if stage.blocks.is_empty() {
            ctx.emit(
                &rules::SEMANTIC_STAGE_BLOCKS,
                "stages.json",
                None::<&str>,
                format!(
                    "stage {} declares no blocks; every study stage must declare at least one \
                     block",
                    stage.id
                ),
            );
            continue;
        }
        for block in &stage.blocks {
            if !block.duration_hours.is_finite() || block.duration_hours <= 0.0 {
                ctx.emit(
                    &rules::SEMANTIC_STAGE_BLOCKS,
                    "stages.json",
                    None::<&str>,
                    format!(
                        "stage {} block {} has duration_hours {}, which must be finite and > 0.0",
                        stage.id, block.index, block.duration_hours
                    ),
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::super::validate_semantic_stages_penalties_scenarios;
    use crate::test_support::*;
    use crate::validation::schema::ParsedData;
    use cobre_core::EntityId;
    use cobre_core::scenario::ExternalScenarioRow;
    use cobre_core::temporal::{
        Block, Node, PolicyGraphType, SeasonCycleType, SeasonDefinition, SeasonMap, Stage,
        Transition,
    };

    use crate::scenarios::{InflowArCoefficientRow, InflowSeasonalStatsRow};
    use crate::validation::{ErrorKind, ValidationContext};

    /// One AR coefficient row at the given 1-based lag (the PAR order is the max
    /// lag across rows).
    fn ar_row(lag: i32) -> InflowArCoefficientRow {
        InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag,
            coefficient: 0.5,
        }
    }

    // ── Rule 1: Transition stage validity ─────────────────────────────────────

    /// Transition referencing a non-existent source_id produces InvalidValue error.
    #[test]
    fn test_5b_transition_invalid_source_id() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 99, // does not exist
            target_id: 1,
            probability: 1.0,
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors.iter().any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue for invalid source_id"
        );
    }

    /// Transition referencing a non-existent target_id produces InvalidValue error.
    #[test]
    fn test_5b_transition_invalid_target_id() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 0,
            target_id: 99, // does not exist
            probability: 1.0,
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "should have InvalidValue for invalid target_id"
        );
    }

    // ── Rule 2: Transition probability sums ───────────────────────────────────

    /// Transitions from stage 0 with probability sum 0.5 produce one InvalidValue
    /// error with "probability" and "stage 0" in the message.
    #[test]
    fn test_5b_transition_probability_sum_wrong() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.policy_graph.transitions = vec![Transition {
            source_id: 0,
            target_id: 1,
            probability: 0.5, // should sum to 1.0
            annual_discount_rate_override: None,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(relevant.len(), 1, "exactly 1 InvalidValue error expected");
        let msg = &relevant[0].message;
        assert!(
            msg.contains("probability"),
            "message should contain 'probability', got: {msg}"
        );
        assert!(
            msg.contains("stage 0"),
            "message should contain 'stage 0', got: {msg}"
        );
    }

    /// Transitions from stage 0 summing exactly 1.0 produce no probability error.
    #[test]
    fn test_5b_transition_probability_sum_valid() {
        let mut stages = make_stages_5b(vec![0, 1, 2]);
        stages.policy_graph.transitions = vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.6,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.4,
                annual_discount_rate_override: None,
            },
        ];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let prob_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            prob_errors.is_empty(),
            "valid probability sum should produce no InvalidValue errors, got: {prob_errors:?}"
        );
    }

    // ── Rule 3: Cyclic discount rate ──────────────────────────────────────────

    /// Cyclic graph with annual_discount_rate = 0.0 produces InvalidValue error.
    #[test]
    fn test_5b_cyclic_zero_discount_rate() {
        let mut stages = make_stages_5b(vec![0]);
        stages.policy_graph.graph_type = PolicyGraphType::Cyclic;
        stages.policy_graph.annual_discount_rate = 0.0;
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidValue),
            "cyclic with 0 discount rate should produce InvalidValue"
        );
    }

    /// Cyclic graph with annual_discount_rate > 0.0 produces no discount rate error.
    #[test]
    fn test_5b_cyclic_positive_discount_rate_valid() {
        let mut stages = make_stages_5b(vec![0]);
        stages.policy_graph.graph_type = PolicyGraphType::Cyclic;
        stages.policy_graph.annual_discount_rate = 0.06;
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let discount_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert!(
            discount_errors.is_empty(),
            "cyclic with positive discount rate should produce no error, got: {discount_errors:?}"
        );
    }

    // ── Rule 34: inflow_lags disabled on all stages under PAR(p>0) ────────────

    /// PAR order 6 with every study stage `inflow_lags == false` produces exactly
    /// one `ModelQuality` warning and no error.
    #[test]
    fn test_5b_all_stages_inflow_lags_disabled_under_par_warns_once() {
        let mut stages = make_stages_5b(vec![0, 1, 2]); // make_stage defaults inflow_lags false
        // Order-bearing inflow_ar_coefficients trigger the PAR stationarity
        // gate, which hard-errors without a resolvable season map.
        for (i, stage) in stages.stages.iter_mut().enumerate() {
            stage.season_id = Some(i);
        }
        stages.policy_graph.season_map = Some(SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons: (0..3)
                .map(|i| SeasonDefinition {
                    id: i,
                    label: format!("Season{i}"),
                    month_start: (i % 12 + 1) as u32,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                })
                .collect(),
        });
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![ar_row(1), ar_row(6)], // max lag 6 => PAR order 6
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(!ctx.has_errors(), "warning-only check must not error");
        let model_quality: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("inflow lags"))
            .collect();
        assert_eq!(
            model_quality.len(),
            1,
            "expected exactly one inflow-lags ModelQuality warning, got: {model_quality:?}"
        );
        let msg = &model_quality[0].message;
        assert!(
            msg.contains("PAR"),
            "warning should mention PAR(p>0), got: {msg}"
        );
    }

    /// PAR order 6 but one study stage with `inflow_lags == true` produces no
    /// inflow-lags warning.
    #[test]
    fn test_5b_one_stage_inflow_lags_enabled_under_par_no_warning() {
        let mut stages = make_stages_5b(vec![0, 1, 2]);
        stages.stages[1].state_config.inflow_lags = true;
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![ar_row(1), ar_row(6)],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("inflow lags")),
            "no inflow-lags warning when any stage enables inflow_lags"
        );
    }

    /// Every study stage `inflow_lags == false` but the inflow model is order 0
    /// (white noise, no AR rows) produces no inflow-lags warning.
    #[test]
    fn test_5b_all_stages_inflow_lags_disabled_white_noise_no_warning() {
        let stages = make_stages_5b(vec![0, 1, 2]);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![], // order 0: no AR coefficients
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("inflow lags")),
            "no inflow-lags warning for white-noise (order 0) models"
        );
    }

    // ── Node graph (rules 36-40) ──────────────────────────────────────────────

    fn node(id: i32, stage_id: i32, scenario_id: Option<i32>) -> Node {
        Node {
            id,
            stage_id,
            scenario_id,
            label: None,
        }
    }

    fn edge(source_id: i32, target_id: i32, probability: f64) -> Transition {
        Transition {
            source_id,
            target_id,
            probability,
            annual_discount_rate_override: None,
        }
    }

    fn ext_inflow(stage_id: i32, scenario_id: i32) -> ExternalScenarioRow {
        ExternalScenarioRow {
            stage_id,
            scenario_id,
            hydro_id: EntityId::from(1),
            value_m3s: 1.0,
        }
    }

    /// Build Layer-5b `ParsedData` over `n_stages` study stages carrying the given
    /// node graph; other scenario inputs are minimal.
    fn node_graph_data(
        n_stages: i32,
        nodes: Vec<Node>,
        transitions: Vec<Transition>,
    ) -> ParsedData {
        let stages = make_stages_5b((0..n_stages).collect());
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        data.stages.policy_graph.nodes = nodes;
        data.stages.policy_graph.transitions = transitions;
        // Default: every study stage declares num_openings (the all-generated,
        // in-sample case). External-opening tests override this to exclude the
        // stages whose openings come from an external library.
        data.stages.openings_declared = (0..n_stages).collect();
        data
    }

    fn run(data: &ParsedData) -> ValidationContext {
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(data, &mut ctx);
        ctx
    }

    fn errs_contain(ctx: &ValidationContext, needle: &str) -> bool {
        ctx.errors().iter().any(|e| e.message.contains(needle))
    }

    /// The once-computed `stage_slot_occupancy` vector must agree, for a stage
    /// carrying multiple sibling nodes, with an independent per-node recompute
    /// over the raw external rows — and every sibling must read the same answer
    /// from the shared vector, not a per-node-drifted one.
    #[test]
    fn test_stage_slot_occupancy_matches_independent_per_node_recompute() {
        let mut data = node_graph_data(
            2,
            vec![
                node(0, 0, None),
                node(1, 1, Some(0)),
                node(2, 1, Some(1)),
                node(3, 1, Some(2)),
            ],
            vec![
                edge(0, 1, 1.0 / 3.0),
                edge(0, 2, 1.0 / 3.0),
                edge(0, 3, 1.0 / 3.0),
            ],
        );
        data.config = config_enumerated_external_inflow();
        // Two hydros duplicate the same scenario_id set at stage 1: raw_c counts
        // distinct scenario ids, not rows, so a row-count bug would read 6 here.
        data.external_scenarios = vec![
            ext_inflow(1, 0),
            ext_inflow(1, 1),
            ext_inflow(1, 2),
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id: EntityId::from(2),
                value_m3s: 1.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 1,
                hydro_id: EntityId::from(2),
                value_m3s: 1.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 2,
                hydro_id: EntityId::from(2),
                value_m3s: 1.0,
            },
        ];
        data.stages.openings_declared = [0].into_iter().collect();
        data.inflow_seasonal_stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 1,
            mean_m3s: 100.0,
            std_m3s: 5.0,
        }];

        let resolver = crate::StageIdResolver::from_study_stage_ids(&[0, 1]);
        let source = data
            .config
            .training_scenario_source(std::path::Path::new("config.json"))
            .unwrap();
        let occupancy = super::stage_slot_occupancy(&data, &source, &resolver);
        let idx = resolver.resolve(1).unwrap();

        let per_node_raw_c = data
            .external_scenarios
            .iter()
            .filter(|r| r.stage_id == 1)
            .map(|r| r.scenario_id)
            .collect::<std::collections::HashSet<i32>>()
            .len();
        assert_eq!(
            occupancy[idx],
            vec![("inflow", per_node_raw_c)],
            "once-computed occupancy must match the independent per-node recompute: {:?}",
            occupancy[idx]
        );
        assert_eq!(
            per_node_raw_c, 3,
            "raw_c counts distinct ids, not the 6 rows"
        );

        // All three siblings share stage 1 and read the same vector entry; with
        // scenario_id 0, 1, 2 all below raw_c 3, none should be rejected.
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "sibling nodes sharing a stage must all validate against the same occupancy: {:?}",
            ctx.errors()
        );
    }

    /// A K-fan (root → K distinct-realization leaves) and a 3-stage binary tree
    /// with distinct realizations load with no node-graph rule firing.
    #[test]
    fn test_node_graph_valid_k_fan_and_binary_tree() {
        // K-fan: stage 0 root (no external there ⇒ scenario_id absent), stage 1
        // carries three external realizations.
        let mut kfan = node_graph_data(
            2,
            vec![
                node(0, 0, None),
                node(1, 1, Some(0)),
                node(2, 1, Some(1)),
                node(3, 1, Some(2)),
            ],
            vec![
                edge(0, 1, 1.0 / 3.0),
                edge(0, 2, 1.0 / 3.0),
                edge(0, 3, 1.0 / 3.0),
            ],
        );
        kfan.config = config_enumerated_external_inflow();
        kfan.external_scenarios = vec![ext_inflow(1, 0), ext_inflow(1, 1), ext_inflow(1, 2)];
        // Stage 0 is generated (num_openings declared); stage 1 is external (none).
        kfan.stages.openings_declared = [0].into_iter().collect();
        kfan.inflow_seasonal_stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 1,
            mean_m3s: 100.0,
            std_m3s: 5.0,
        }];
        let ctx = run(&kfan);
        assert!(!ctx.has_errors(), "valid K-fan errors: {:?}", ctx.errors());
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("identical")),
            "distinct-realization fan must not warn recombinable"
        );

        // Binary tree branching at stages 0 and 1, with distinct realizations at
        // every stage so no two subtrees coincide.
        let mut tree = node_graph_data(
            3,
            vec![
                node(0, 0, None),
                node(1, 1, Some(0)),
                node(2, 1, Some(1)),
                node(3, 2, Some(0)),
                node(4, 2, Some(1)),
                node(5, 2, Some(2)),
                node(6, 2, Some(3)),
            ],
            vec![
                edge(0, 1, 0.5),
                edge(0, 2, 0.5),
                edge(1, 3, 0.5),
                edge(1, 4, 0.5),
                edge(2, 5, 0.5),
                edge(2, 6, 0.5),
            ],
        );
        tree.config = config_enumerated_external_inflow();
        tree.external_scenarios = vec![
            ext_inflow(1, 0),
            ext_inflow(1, 1),
            ext_inflow(2, 0),
            ext_inflow(2, 1),
            ext_inflow(2, 2),
            ext_inflow(2, 3),
        ];
        // Stage 0 generated; stages 1 and 2 external.
        tree.stages.openings_declared = [0].into_iter().collect();
        tree.inflow_seasonal_stats = vec![
            InflowSeasonalStatsRow {
                hydro_id: EntityId::from(1),
                stage_id: 1,
                mean_m3s: 100.0,
                std_m3s: 5.0,
            },
            InflowSeasonalStatsRow {
                hydro_id: EntityId::from(1),
                stage_id: 2,
                mean_m3s: 100.0,
                std_m3s: 5.0,
            },
        ];
        let ctx = run(&tree);
        assert!(!ctx.has_errors(), "valid tree errors: {:?}", ctx.errors());
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("identical")),
            "distinct-realization tree must not warn recombinable"
        );
    }

    /// A genuine fan whose two sibling successors carry the same realization
    /// pointer — the duplicate-pointer / degenerate-branching shape a retired
    /// admissibility predicate would have rejected — now loads with no error. A
    /// node's realization is defined by its own `scenario_id` (the per-class
    /// pointer bound is the only surviving pointer rule), so duplicate sibling
    /// pointers are admissible; the structurally identical siblings draw only the
    /// recombinable-signature warning, never a rejection. Asserts an OLD rule does
    /// NOT fire — the counterpart to the sibling tests that assert a surviving rule
    /// does.
    #[test]
    fn test_node_graph_duplicate_sibling_pointer_branch_admitted() {
        // Root (stage 0, generated) fans to two stage-1 leaves both pointing at
        // the single external column 0 (raw_c = 1): a real branch with degenerate
        // sibling pointers.
        let mut data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, Some(0)), node(2, 1, Some(0))],
            vec![edge(0, 1, 0.5), edge(0, 2, 0.5)],
        );
        data.config = config_enumerated_external_inflow();
        data.external_scenarios = vec![ext_inflow(1, 0)];
        data.stages.openings_declared = [0].into_iter().collect();
        data.inflow_seasonal_stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 1,
            mean_m3s: 100.0,
            std_m3s: 5.0,
        }];
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "duplicate-pointer branch must load with no error: {:?}",
            ctx.errors()
        );
    }

    /// Rule 37: a `scenario_id` equal to (and beyond) `raw_c(t)` is rejected
    /// naming the node, value, stage and bound; a value below `raw_c` is accepted.
    #[test]
    fn test_node_graph_realization_bound_at_and_beyond_raw_c() {
        // Single stage carrying two external realizations ⇒ raw_c = 2.
        let build = |rid: i32| {
            let mut data = node_graph_data(1, vec![node(0, 0, Some(rid))], vec![]);
            data.config = config_enumerated_external_inflow();
            data.external_scenarios = vec![ext_inflow(0, 0), ext_inflow(0, 1)];
            // Stage 0 is external ⇒ num_openings meaningless there; not declared.
            data.stages.openings_declared.clear();
            data
        };

        let ctx = run(&build(2));
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.message.contains("scenario_id 2 is out of range")
                    && e.message.contains("[0, 2)")
                    && e.message.contains("inflow")),
            "scenario_id == raw_c must be rejected naming value, class and bound: {:?}",
            ctx.errors()
        );

        let ctx = run(&build(5));
        assert!(
            errs_contain(&ctx, "scenario_id 5 is out of range"),
            "scenario_id beyond raw_c must be rejected"
        );

        let ctx = run(&build(1));
        assert!(
            !errs_contain(&ctx, "out of range"),
            "scenario_id < raw_c must be accepted: {:?}",
            ctx.errors()
        );
    }

    /// B3 arm 1: `scenario_id` absent at a stage carrying a slot-occupying
    /// external class is rejected.
    #[test]
    fn test_node_graph_b3_realization_required_but_absent() {
        let mut data = node_graph_data(1, vec![node(0, 0, None)], vec![]);
        data.config = config_enumerated_external_inflow();
        data.external_scenarios = vec![ext_inflow(0, 0)];
        data.stages.openings_declared.clear();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "scenario_id is required there"),
            "absent scenario_id at an external stage must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 36 is inert under sampled forward selection: a `sampled + external`
    /// stage whose node carries `scenario_id: None` raises no "required" error —
    /// nodes are Generated there, and a present pointer is a setup rejection (D2),
    /// not a check in this layer.
    #[test]
    fn test_node_graph_realization_inert_under_sampled_selection() {
        let mut data = node_graph_data(1, vec![node(0, 0, None)], vec![]);
        data.config = config_with_training_external_inflow();
        data.external_scenarios = vec![ext_inflow(0, 0)];
        data.stages.openings_declared.clear();
        let ctx = run(&data);
        assert!(
            !errs_contain(&ctx, "scenario_id"),
            "rule 36 must not fire under sampled selection: {:?}",
            ctx.errors()
        );
    }

    /// B3 arm 2: `scenario_id` present at a stage carrying no slot-occupying
    /// external class is rejected as meaningless.
    #[test]
    fn test_node_graph_b3_realization_present_but_meaningless() {
        // Enumerated, in-sample ⇒ rule 36 runs, no slot-occupying external class anywhere.
        let mut data = node_graph_data(1, vec![node(0, 0, Some(0))], vec![]);
        data.config = config_enumerated();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "scenario_id is meaningless there"),
            "present scenario_id where no external class exists must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 38: an empty study stage is rejected naming the stage.
    #[test]
    fn test_node_graph_empty_stage_rejected() {
        let data = node_graph_data(2, vec![node(0, 0, None)], vec![]);
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "study stage 1 has no policy-graph node"),
            "empty stage must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 38: an unreachable node is rejected naming the node.
    #[test]
    fn test_node_graph_unreachable_node_rejected() {
        let data = node_graph_data(
            3,
            vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 2, None),
                node(99, 1, None),
            ],
            vec![edge(0, 1, 1.0), edge(1, 2, 1.0), edge(99, 2, 1.0)],
        );
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "node 99") && errs_contain(&ctx, "unreachable"),
            "unreachable node must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 38: a cycle in the node graph is rejected.
    #[test]
    fn test_node_graph_cycle_rejected() {
        let data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, None)],
            vec![edge(0, 1, 1.0), edge(1, 0, 1.0)],
        );
        let ctx = run(&data);
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::CycleDetected),
            "cycle must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 38: a leaf before the final stage (mid-horizon leaf) is rejected.
    #[test]
    fn test_node_graph_mid_horizon_leaf_rejected() {
        let data = node_graph_data(
            3,
            vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
            ],
            vec![edge(0, 1, 0.5), edge(0, 2, 0.5), edge(2, 3, 1.0)],
        );
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "node 1") && errs_contain(&ctx, "mid-horizon leaf"),
            "mid-horizon leaf must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 39: an edge that skips a stage is rejected naming both endpoints.
    #[test]
    fn test_node_graph_stage_skipping_edge_rejected() {
        let data = node_graph_data(
            3,
            vec![node(0, 0, None), node(1, 1, None), node(2, 2, None)],
            vec![edge(0, 1, 0.5), edge(0, 2, 0.5), edge(1, 2, 1.0)],
        );
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "does not advance") && errs_contain(&ctx, "t -> t+1"),
            "stage-skipping edge must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 40: a stage with structurally identical sibling subtrees emits a
    /// recombinable-signature warning and the load succeeds.
    #[test]
    fn test_node_graph_recombinable_signature_warns() {
        let data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            vec![edge(0, 1, 0.5), edge(0, 2, 0.5)],
        );
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "recombinable graph must load: {:?}",
            ctx.errors()
        );
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality
                    && w.message.contains("structurally identical")),
            "identical sibling subtrees must warn: {:?}",
            ctx.warnings()
        );
    }

    /// A duplicate node id is rejected.
    #[test]
    fn test_node_graph_duplicate_node_id_rejected() {
        let data = node_graph_data(2, vec![node(0, 0, None), node(0, 1, None)], vec![]);
        let ctx = run(&data);
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::DuplicateId),
            "duplicate node id must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// A transition endpoint that is not a declared node is rejected.
    #[test]
    fn test_node_graph_unknown_endpoint_rejected() {
        let data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, None)],
            vec![edge(0, 99, 1.0)],
        );
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "target_id 99 does not refer to a declared node"),
            "unknown endpoint must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// A transition source that is not a declared node is rejected.
    #[test]
    fn test_node_graph_unknown_source_endpoint_rejected() {
        let data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, None)],
            vec![edge(99, 1, 1.0)],
        );
        let ctx = run(&data);
        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.message
                    .contains("source_id 99 does not refer to a declared node")
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "unknown source endpoint must be rejected once: {:?}",
            ctx.errors()
        );
        assert_eq!(matching[0].kind, ErrorKind::InvalidValue);
    }

    /// A node whose `stage_id` names no declared study stage is rejected.
    #[test]
    fn test_node_graph_unresolved_stage_rejected() {
        let data = node_graph_data(2, vec![node(0, 0, None), node(1, 5, None)], vec![]);
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "stage_id 5 which is not a declared study stage"),
            "unresolved node stage must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Absent `nodes[]` leaves the node-graph rules inert (chain dialect).
    #[test]
    fn test_node_graph_absent_nodes_is_inert() {
        let mut data = node_graph_data(2, vec![], vec![]);
        data.stages.policy_graph.transitions = vec![edge(0, 1, 1.0)];
        let ctx = run(&data);
        assert!(
            !errs_contain(&ctx, "policy-graph node")
                && !errs_contain(&ctx, "scenario_id")
                && !errs_contain(&ctx, "does not advance"),
            "chain dialect must not trigger node-graph rules: {:?}",
            ctx.errors()
        );
    }

    /// Rule 41 arm 1: `num_openings` absent at a stage carrying generated openings
    /// is rejected naming the stage.
    #[test]
    fn test_num_openings_required_at_generated_stage() {
        let mut data = node_graph_data(1, vec![node(0, 0, None)], vec![]);
        data.stages.openings_declared.clear();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "stage 0") && errs_contain(&ctx, "num_openings is required there"),
            "generated stage without num_openings must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 41 arm 2: `num_openings` present at a stage carrying only external
    /// openings is rejected as meaningless naming the stage.
    #[test]
    fn test_num_openings_meaningless_at_external_stage() {
        let mut data = node_graph_data(1, vec![node(0, 0, Some(0))], vec![]);
        data.config = config_with_training_external_inflow();
        data.external_scenarios = vec![ext_inflow(0, 0)];
        data.stages.openings_declared = [0].into_iter().collect();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "stage 0")
                && errs_contain(&ctx, "num_openings is meaningless there"),
            "external stage declaring num_openings must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 42: a per-edge `annual_discount_rate_override` under `nodes[]` is
    /// rejected naming the edge.
    #[test]
    fn test_edge_discount_override_rejected_under_nodes() {
        let mut data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, None)],
            vec![Transition {
                source_id: 0,
                target_id: 1,
                probability: 1.0,
                annual_discount_rate_override: Some(0.08),
            }],
        );
        data.stages.openings_declared = [0, 1].into_iter().collect();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "transition 0 -> 1")
                && errs_contain(&ctx, "annual_discount_rate_override")
                && errs_contain(&ctx, "stages[].annual_discount_rate_override"),
            "per-edge discount override under nodes[] must be rejected naming the edge: {:?}",
            ctx.errors()
        );
    }

    /// Rule 42 on a fan: a source node with two out-edges, one carrying
    /// `annual_discount_rate_override`, is rejected naming that edge — the branching
    /// shape that motivated the rule.
    #[test]
    fn test_fan_edge_discount_override_rejected_under_nodes() {
        let mut data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            vec![
                Transition {
                    source_id: 0,
                    target_id: 1,
                    probability: 0.5,
                    annual_discount_rate_override: Some(0.08),
                },
                edge(0, 2, 0.5),
            ],
        );
        data.stages.openings_declared = [0, 1].into_iter().collect();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "transition 0 -> 1")
                && errs_contain(&ctx, "annual_discount_rate_override")
                && errs_contain(&ctx, "stages[].annual_discount_rate_override"),
            "per-edge discount override on a fan out-edge under nodes[] must be rejected \
             naming the edge: {:?}",
            ctx.errors()
        );
    }

    /// Rule 43: `openings = {source: file}` under enumerated forward selection is
    /// rejected, stating the generated tree is not consumed there.
    #[test]
    fn test_file_openings_rejected_under_enumerated() {
        let mut data = node_graph_data(1, vec![], vec![]);
        data.config = config_enumerated_file_openings();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "not consumed under enumerated forward selection"),
            "file openings under enumerated must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 43: `openings = {source: file}` declared alongside a `nodes[]` policy
    /// graph is rejected — nodes[] declares the opening set.
    #[test]
    fn test_file_openings_rejected_with_nodes() {
        let mut data = node_graph_data(1, vec![node(0, 0, None)], vec![]);
        data.config = config_sampled_file_openings();
        let ctx = run(&data);
        assert!(
            errs_contain(&ctx, "conflicts with a declared nodes[] policy graph"),
            "file openings with a nodes[] graph must be rejected: {:?}",
            ctx.errors()
        );
    }

    /// Rule 43: `openings = {source: file}` under sampled selection with no
    /// `nodes[]` is admitted — the chain-dialect user opening-tree file arm.
    #[test]
    fn test_file_openings_admitted_sampled_chain() {
        let mut data = node_graph_data(1, vec![], vec![]);
        data.config = config_sampled_file_openings();
        let ctx = run(&data);
        assert!(
            !errs_contain(&ctx, "{source: file}"),
            "file openings under sampled + chain dialect must not be rejected by B6: {:?}",
            ctx.errors()
        );
    }

    /// Rule 44: a multi-node stage carrying `sampling_method` warns naming the stage
    /// and the load succeeds.
    #[test]
    fn test_sampling_method_warns_at_multi_node_stage() {
        let data = node_graph_data(
            2,
            vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            vec![edge(0, 1, 0.5), edge(0, 2, 0.5)],
        );
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "sampling_method meaninglessness is a warning, not an error: {:?}",
            ctx.errors()
        );
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality
                    && w.message.contains("stage 1")
                    && w.message.contains("sampling_method is ignored")),
            "multi-node stage must warn naming the stage: {:?}",
            ctx.warnings()
        );
    }

    // ── Study stage blocks (rule 52) ──────────────────────────────────────────

    fn stage_with_block_hours(id: i32, hours: &[f64]) -> Stage {
        let mut stage = make_stage(id);
        stage.blocks = hours
            .iter()
            .enumerate()
            .map(|(index, &duration_hours)| Block {
                index,
                name: format!("B{index}"),
                duration_hours,
            })
            .collect();
        stage
    }

    fn blocks_data(stages: Vec<Stage>) -> ParsedData {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        data.stages.stages = stages;
        data
    }

    /// An empty-blocks study stage produces exactly one `InvalidValue` error
    /// naming the stage, and the context fails.
    #[test]
    fn test_study_stage_blocks_empty_rejected() {
        let data = blocks_data(vec![stage_with_block_hours(3, &[])]);
        let mut ctx = ValidationContext::new();
        super::check_study_stage_blocks(&data, &mut ctx);
        let errors = ctx.errors();
        assert_eq!(errors.len(), 1, "expected exactly one error: {errors:?}");
        assert_eq!(errors[0].kind, ErrorKind::InvalidValue);
        assert!(
            errors[0].message.contains("stage 3"),
            "message must name stage 3: {}",
            errors[0].message
        );
        assert!(ctx.into_result().is_err());
    }

    /// A stage with blocks `[100.0, 0.0, -5.0, NaN]` produces exactly three
    /// errors — one per non-positive or non-finite block, naming the stage,
    /// the block index, and the offending value; the `100.0` block is clean.
    #[test]
    fn test_study_stage_blocks_mixed_hours_rejects_each_bad_block() {
        let data = blocks_data(vec![stage_with_block_hours(
            0,
            &[100.0, 0.0, -5.0, f64::NAN],
        )]);
        let mut ctx = ValidationContext::new();
        super::check_study_stage_blocks(&data, &mut ctx);
        let errors = ctx.errors();
        assert_eq!(errors.len(), 3, "expected exactly three errors: {errors:?}");
        for (index, value) in [(1, "0"), (2, "-5"), (3, "NaN")] {
            assert!(
                errors.iter().any(|e| e.kind == ErrorKind::InvalidValue
                    && e.message.contains("stage 0")
                    && e.message.contains(&format!("block {index}"))
                    && e.message.contains(&format!("duration_hours {value},"))),
                "expected an error naming stage 0, block {index}, duration_hours {value}: \
                 {errors:?}"
            );
        }
    }

    /// A well-formed multi-stage deck (every stage carries positive-duration
    /// blocks) produces no error.
    #[test]
    fn test_study_stage_blocks_well_formed_deck_no_error() {
        let data = blocks_data(vec![
            make_stage_with_blocks(0, 3),
            make_stage_with_blocks(1, 1),
            make_stage_with_blocks(2, 2),
        ]);
        let mut ctx = ValidationContext::new();
        super::check_study_stage_blocks(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "well-formed deck must not error: {:?}",
            ctx.errors()
        );
    }
}
