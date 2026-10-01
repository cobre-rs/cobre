//! Runtime node graph: node identity, canonical successor order, the
//! `node → pool` map, and per-node opening/weight views — the node-native
//! engine's single traversal structure, recorded as a
//! [`crate::context::TrainingContext`] field.
//!
//! Absent `nodes[]` this degenerates byte-exactly to one node per stage, 1:1
//! pools, and uniform `q`.

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use cobre_core::HorizonGraph;
use cobre_io::StageIdResolver;
use cobre_stochastic::{StochasticContext, select_transition_child};

use crate::error::SddpError;
use crate::simulation::SimulationWeighting;
use crate::workspace::CapturedBasis;

/// A declared JSON node id — the value stored in [`NodeGraph::node_ids`], never
/// a position into the graph's parallel arrays.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct NodeId(pub i32);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// A 0-based positional stage index — the study-stage array position
/// (`stage_data.stages[stage]`, `for t in 0..n_stages`), never [`cobre_core`]'s
/// declared `StageId`. Runtime-only: an infra crate or a wire payload takes
/// the bare `usize`/`i32`/`u32`, converted at that boundary exactly as
/// [`NodeId`] converts at the cut/policy codecs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageIdx(pub usize);

impl fmt::Display for StageIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl StageIdx {
    /// The next stage position, `self + 1` — the newtype carries no
    /// arithmetic operator, so every `t -> t+1` advance in the crate routes
    /// through this one method rather than an ad hoc `.0`-rewrap.
    #[must_use]
    pub fn next(self) -> StageIdx {
        StageIdx(self.0 + 1)
    }
}

/// A 0-based canonical position into the node graph's own parallel arrays
/// (`NodeGraph::nodes`, `successors`, `node_ids`) — never a study-stage index
/// and never a declared JSON node id. On a chain the two coincide
/// numerically (`NodePos(t)` and `StageIdx(t)` carry the same raw value at
/// every position); that numeric coincidence is exactly what let a stage
/// index stand in for a node position at a call site, silently, across this
/// crate — `NodePos` makes the substitution a compile error. Runtime-only:
/// converted to a bare `usize`/`i32` at an infra-crate or wire boundary
/// (`policy/codec.rs`'s `stage_id` field, the `cobre-cli` broadcast payload),
/// exactly as [`StageIdx`] converts at its own boundaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodePos(pub usize);

impl fmt::Display for NodePos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// A [`TypedVec`] index space: converts to and from the raw storage
/// position. Adding a new index space wires exactly one impl of this trait;
/// `pub` only so `TypedVec<Idx, T>`'s own `pub` API (`Index`, `get`) stays
/// callable from outside this crate, never implemented outside it in
/// practice.
pub trait TypedIndex: Copy {
    /// The raw storage position this index resolves to.
    fn as_usize(self) -> usize;

    /// Wraps a raw storage position as this index — the inverse of
    /// [`TypedIndex::as_usize`], letting [`TypedVec::iter_indexed`] hand back
    /// a typed position without a manual `.enumerate().map(Idx)` at every
    /// call site.
    fn from_usize(pos: usize) -> Self;
}

impl TypedIndex for StageIdx {
    fn as_usize(self) -> usize {
        self.0
    }

    fn from_usize(pos: usize) -> Self {
        StageIdx(pos)
    }
}

impl TypedIndex for NodePos {
    fn as_usize(self) -> usize {
        self.0
    }

    fn from_usize(pos: usize) -> Self {
        NodePos(pos)
    }
}

/// A `Vec<T>` indexable ONLY by `Idx` (`StageIdx` or `NodePos`) — never by a
/// bare `usize`, so a stage position and a node position cannot be
/// substituted for each other at a read. No `Deref`: a
/// `Deref<Target = Vec<T>>` would leak `usize` indexing back in via
/// coercion, defeating the whole point of the wrapper.
#[derive(Debug, Clone, PartialEq)]
#[repr(transparent)]
pub struct TypedVec<Idx, T>(Vec<T>, PhantomData<Idx>);

impl<Idx, T> TypedVec<Idx, T> {
    /// An empty `TypedVec`.
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new(), PhantomData)
    }

    /// An empty `TypedVec` pre-allocated for `capacity` elements.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity), PhantomData)
    }

    /// Iterator over `&T`, in storage order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    /// Iterator over `&mut T`, in storage order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.0.iter_mut()
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the vector holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Appends `value` at the end.
    pub fn push(&mut self, value: T) {
        self.0.push(value);
    }

    /// Borrows the backing storage as a plain `usize`-indexable slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<Idx: TypedIndex, T> TypedVec<Idx, T> {
    /// `&T` at `idx`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, idx: Idx) -> Option<&T> {
        self.0.get(idx.as_usize())
    }

    /// Iterator over `(Idx, &T)` pairs, in storage order — the typed
    /// counterpart of `Vec::iter().enumerate()`; without it, every read site
    /// that needs both a value and its own position would fall back to
    /// enumerating as a bare `usize` and re-wrapping by hand.
    pub fn iter_indexed(&self) -> impl Iterator<Item = (Idx, &T)> + '_ {
        self.0
            .iter()
            .enumerate()
            .map(|(i, t)| (Idx::from_usize(i), t))
    }
}

impl<Idx, T> Default for TypedVec<Idx, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Idx: TypedIndex, T> Index<Idx> for TypedVec<Idx, T> {
    type Output = T;

    fn index(&self, idx: Idx) -> &T {
        &self.0[idx.as_usize()]
    }
}

impl<Idx: TypedIndex, T> IndexMut<Idx> for TypedVec<Idx, T> {
    fn index_mut(&mut self, idx: Idx) -> &mut T {
        &mut self.0[idx.as_usize()]
    }
}

impl<Idx, T> FromIterator<T> for TypedVec<Idx, T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(Vec::from_iter(iter), PhantomData)
    }
}

impl<Idx, T> IntoIterator for TypedVec<Idx, T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, Idx, T> IntoIterator for &'a TypedVec<Idx, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, Idx, T> IntoIterator for &'a mut TypedVec<Idx, T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<Idx, T> From<Vec<T>> for TypedVec<Idx, T> {
    fn from(items: Vec<T>) -> Self {
        Self(items, PhantomData)
    }
}

/// Which substrate a node's [`NodeOpenings`] view addresses. Both are
/// single-storage, read-only realization stores — `Generated` reads
/// [`StochasticContext::opening_tree`], `External` reads the standardized
/// external library for the node's stage (`cobre_stochastic::ExternalScenarioLibrary`,
/// via `ScenarioLibraries::training`) — never a copy of realization values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpeningSource {
    /// `offset..offset+len` are opening indices into this node's stage's
    /// `OpeningTree` block (`StochasticContext::opening_tree`).
    Generated,
    /// `offset` is a raw scenario column (`Node::scenario_id`) in the
    /// standardized external library at this node's stage; `len` is always
    /// `1` — within-node weighted openings are deferred.
    External,
}

/// A node's Ω (opening set) view: `offset`/`len` address `source`'s own
/// per-stage ordinal space, never a copy of the realization values.
///
/// `q` is the uniform per-opening weight, computed as `1.0 / (len as f64)` —
/// the exact bit pattern
/// `chain_degeneracy_one_node_per_stage_1to1_pools_uniform_q_bit_pattern` pins
/// for a generated node's `len` openings, and (by the same formula,
/// `len == 1`) exactly `1.0` for a degenerate `External` view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeOpenings {
    /// Which substrate `offset`/`len` address.
    pub source: OpeningSource,
    /// Starting opening/scenario index within the node's stage, in `source`'s
    /// own per-stage ordinal space.
    pub offset: usize,
    /// Number of consecutive openings/scenarios in this node's Ω, starting at
    /// `offset`.
    pub len: usize,
    /// Uniform per-opening weight.
    pub q: f64,
}

impl NodeOpenings {
    #[expect(
        clippy::cast_precision_loss,
        reason = "opening counts are far inside f64's exact-integer range"
    )]
    fn new(source: OpeningSource, offset: usize, len: usize) -> Self {
        Self {
            source,
            offset,
            len,
            q: 1.0 / (len as f64),
        }
    }
}

/// One out-edge from a node to a canonical-order successor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeSuccessor {
    /// Dense canonical position of the child in [`NodeGraph::nodes`] — not
    /// its declared JSON node id.
    pub child: NodePos,
    /// Normalized transition probability `P(n -> m)` — cobre-io's load-time,
    /// once-only Neumaier normalization (`normalize_out_edge_probabilities`);
    /// never re-normalized here.
    pub probability: f64,
}

/// One runtime node: its stage, pool, and Ω view. Discount is deliberately
/// NOT carried here — it stays per-stage, reached through `stage` (`t(n)`) on
/// `StageContext::cumulative_discount_factors`; a per-node array would store
/// `|nodes|` copies of `T` distinct values.
#[derive(Debug, Clone, Copy)]
pub struct NodeRuntime {
    /// Study-stage array index (`stage_data.stages[stage]`), not the JSON
    /// stage id.
    pub stage: StageIdx,
    /// Pool id into `FutureCostFunction::pools` (pool *contents* — capacity,
    /// archives, basis tagging — are out of scope here). Leaves
    /// share one id; a node with successors always owns its own.
    pub pool_id: usize,
    /// This node's Ω view.
    pub openings: NodeOpenings,
}

/// The runtime node graph: every node in canonical (ascending declared
/// id) order, its pool assignment, its Ω view, and its canonically-ordered
/// successor list.
///
/// Absent `nodes[]`, `build_node_graph` synthesizes the byte-exact chain
/// degeneracy: one node per stage, `node_ids[t] = NodeId(t as i32)` (a
/// synthetic id — the source declared no `nodes[]`), pools 1:1.
#[derive(Debug, Clone)]
pub struct NodeGraph {
    /// Declared JSON node id at each canonical position (`node_ids.len() ==
    /// nodes.len()`).
    pub node_ids: TypedVec<NodePos, NodeId>,
    /// Runtime nodes, canonical (ascending `node_ids`) order.
    pub nodes: TypedVec<NodePos, NodeRuntime>,
    /// `successors[i]` is node `i`'s out-edge list, ascending child node id —
    /// the load-bearing canonical order (`CVaR`'s tie-break is index-order-sensitive).
    pub successors: TypedVec<NodePos, Vec<NodeSuccessor>>,
    /// `max(pool_id) + 1`.
    pub n_pools: usize,
    /// `pool_stage[p]` is the study-stage index of the node(s) owning pool `p`
    /// — indexed by POOL id, never by [`NodePos`] (the pool-id space is
    /// distinct from both node position and stage index, and is not typed by
    /// this newtype sweep). The base LP structure a per-pool frozen overlay
    /// composes onto (`freeze(templates[pool_stage[p]], active_cuts(p))`).
    /// Well-defined because every node sharing a pool shares one stage (a
    /// non-leaf pool has exactly one owner; the shared leaf pool's leaves are
    /// all terminal). The frozen overlay MUST index its base by this, never
    /// by a pool ordinal used as a stage: on a branching graph
    /// `pool_id != stage`. On a chain `pool_stage[t] == StageIdx(t)`.
    pub pool_stage: Vec<StageIdx>,
}

impl NodeGraph {
    /// Node → pool map, one entry per canonical node position
    /// (`nodes[i].pool_id`). The load-path basis reconstruction
    /// ([`build_basis_cache_from_checkpoint`](crate::build_basis_cache_from_checkpoint))
    /// keys each node's cut records through this.
    #[must_use]
    pub fn node_pool_ids(&self) -> TypedVec<NodePos, usize> {
        self.nodes.iter().map(|n| n.pool_id).collect()
    }
}

/// Warm every same-pool `None` sibling of a node-indexed simulation basis cache
/// from the pool's representative capture (the lowest [`NodePos`] carrying a
/// `Some`), re-tagged with the target node's own [`NodeId`].
///
/// A branching graph's terminal leaves share one pool and load the identical
/// frozen template, so a sibling's captured basis is structurally valid at any
/// same-pool leaf. Re-tagging with the target's own `node_id` is load-bearing:
/// the apply-site node-tag filter (`run_stage_solve`) reuses a stored basis only
/// when `captured.node_id == inputs.node_id`, so a filled slot MUST carry the
/// target's id — matching by pool instead warms from the wrong node's LP, which
/// CLP accepts silently. A pool with no `Some` capture is left `None` (cold); a
/// chain is a no-op (one node per pool). Deterministic and declaration-order
/// invariant, so every rank fills identically.
pub(crate) fn pool_fill_basis_cache(
    cache: &mut [Option<CapturedBasis>],
    node_pool_ids: &TypedVec<NodePos, usize>,
    node_ids: &TypedVec<NodePos, NodeId>,
) {
    let n = cache.len();
    if node_pool_ids.len() != n || node_ids.len() != n {
        return;
    }
    let n_pools = node_pool_ids.iter().copied().max().map_or(0, |m| m + 1);
    let mut representative: Vec<Option<usize>> = vec![None; n_pools];
    for (node, slot) in cache.iter().enumerate() {
        if slot.is_some() {
            let pool = node_pool_ids[NodePos(node)];
            if representative[pool].is_none() {
                representative[pool] = Some(node);
            }
        }
    }
    let fills: Vec<(usize, CapturedBasis)> = (0..n)
        .filter_map(|node| {
            if cache[node].is_some() {
                return None;
            }
            let src = representative[node_pool_ids[NodePos(node)]]?;
            let mut basis = cache[src].as_ref()?.clone();
            basis.node_id = node_ids[NodePos(node)];
            Some((node, basis))
        })
        .collect();
    for (node, basis) in fills {
        cache[node] = Some(basis);
    }
}

/// Build the runtime node graph from the validated, normalized `HorizonGraph`.
///
/// MUST run after `build_scenario_libraries`: an `External`-bound node's
/// Ω addresses the standardized library's raw scenario axis, so binding
/// earlier would race the library's own standardization.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if a transition or node names an id that
/// resolves to no declared node/stage — unreachable given upstream
/// structural validation, asserted rather than silently ignored.
pub(crate) fn build_node_graph(
    graph: &HorizonGraph,
    n_stages: usize,
    resolver: &StageIdResolver,
    stochastic: &StochasticContext,
) -> Result<NodeGraph, SddpError> {
    if graph.nodes.is_empty() {
        return Ok(build_chain_node_graph(
            graph, n_stages, resolver, stochastic,
        ));
    }
    build_declared_node_graph(graph, resolver, stochastic)
}

/// Chain degeneracy: one node per stage, pools 1:1, `q` the
/// `1.0 / (n as f64)` bit pattern (never a normalized accumulation), the
/// per-stage cumulative discount arrays untouched (this function never reads
/// or writes them).
fn build_chain_node_graph(
    graph: &HorizonGraph,
    n_stages: usize,
    resolver: &StageIdResolver,
    stochastic: &StochasticContext,
) -> NodeGraph {
    let tree = stochastic.opening_tree();
    let mut nodes = TypedVec::with_capacity(n_stages);
    let mut successors = TypedVec::with_capacity(n_stages);
    // On a chain, stage position, pool id, and node position all coincide
    // numerically with the loop counter `t` — but each is constructed into
    // its own typed space (`StageIdx`, the untyped pool-id space, `NodePos`)
    // at its own use, rather than one raw value standing in for all three.
    for t in 0..n_stages {
        let stage = StageIdx(t);
        let pool_id = t;
        let len = tree.n_openings(t);
        nodes.push(NodeRuntime {
            stage,
            pool_id,
            openings: NodeOpenings::new(OpeningSource::Generated, 0, len),
        });
        if t + 1 < n_stages {
            successors.push(vec![NodeSuccessor {
                child: NodePos(t + 1),
                probability: chain_transition_probability(graph, resolver, stage),
            }]);
        } else {
            successors.push(Vec::new());
        }
    }
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "stage counts are far below i32::MAX"
    )]
    let node_ids: TypedVec<NodePos, NodeId> = (0..n_stages as i32).map(NodeId).collect();
    NodeGraph {
        node_ids,
        nodes,
        successors,
        n_pools: n_stages,
        pool_stage: (0..n_stages).map(StageIdx).collect(),
    }
}

/// The chain dialect's departing-edge probability for stage index `t`, read
/// off `graph.transitions` (already normalized to 1.0 within 1 ULP by
/// `normalize_out_edge_probabilities` at load time, when declared); `1.0`
/// when no transition departs the stage — the fully-implicit chain, where a
/// single deterministic out-edge is structurally required.
fn chain_transition_probability(
    graph: &HorizonGraph,
    resolver: &StageIdResolver,
    t: StageIdx,
) -> f64 {
    let Some(stage_id) = resolver.id_at(t.0) else {
        return 1.0;
    };
    graph
        .transitions
        .iter()
        .find(|tr| tr.source_id == stage_id)
        .map_or(1.0, |tr| tr.probability)
}

/// The declared (`nodes[]` non-empty) node graph: canonical id order is
/// derived locally (never trusted from the caller's iteration order), the
/// `node → pool` map applies leaf sharing unconditionally under leaf-ness —
/// no boundary-source discriminator — and each node's Ω binds through
/// its own `scenario_id` (a single external-library column, degenerate
/// `|Ω| = 1`) or the stage's generated set (`scenario_id: None`).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] on a duplicate node id, or on a transition
/// / node naming an id that resolves to no declared node/stage.
fn build_declared_node_graph(
    graph: &HorizonGraph,
    resolver: &StageIdResolver,
    stochastic: &StochasticContext,
) -> Result<NodeGraph, SddpError> {
    let tree = stochastic.opening_tree();

    // `order[k]` is the original `graph.nodes` index (a `cobre_core`-owned
    // array, never this crate's own node position) of the node that becomes
    // canonical position `k`; the two loops below over `order` therefore
    // build `node_ids`/`nodes` in ascending canonical `NodePos` order.
    let mut order: Vec<usize> = (0..graph.nodes.len()).collect();
    order.sort_by_key(|&i| graph.nodes[i].id);
    let node_ids: TypedVec<NodePos, NodeId> =
        order.iter().map(|&i| NodeId(graph.nodes[i].id)).collect();
    // Reject duplicate node ids: the forward/backward warm-start apply guard
    // matches a stored basis to the current LP by `CapturedBasis.node_id`, so a
    // duplicate id would false-match one node's basis onto another's LP (the CLP
    // backend accepts a shape-mismatched basis silently). Ascending order makes
    // duplicates adjacent.
    if let Some(dup) = node_ids
        .iter()
        .zip(node_ids.iter().skip(1))
        .find_map(|(a, b)| (a == b).then_some(*a))
    {
        return Err(SddpError::Validation(format!(
            "node graph: duplicate node id {dup} declared more than once"
        )));
    }
    let id_to_position: HashMap<NodeId, NodePos> = node_ids
        .iter_indexed()
        .map(|(pos, &id)| (id, pos))
        .collect();
    let n = order.len();

    // Out-edges per source, canonical (ascending child node id) order — the
    // load-bearing order CVaR's tie-break depends on. Indexed by raw node
    // position (never `NodePos` itself): purely local bookkeeping over
    // `graph.transitions` indices, an infra array with its own index space.
    let mut out_edge_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, tr) in graph.transitions.iter().enumerate() {
        let src_pos = *id_to_position.get(&NodeId(tr.source_id)).ok_or_else(|| {
            SddpError::Validation(format!(
                "node graph: transition source id {} names no declared node",
                tr.source_id
            ))
        })?;
        out_edge_idx[src_pos.0].push(i);
    }

    let mut successors: TypedVec<NodePos, Vec<NodeSuccessor>> = TypedVec::with_capacity(n);
    let mut is_leaf = vec![true; n];
    for (pos, edges) in out_edge_idx.iter_mut().enumerate() {
        edges.sort_by_key(|&i| graph.transitions[i].target_id);
        let mut list = Vec::with_capacity(edges.len());
        for &i in edges.iter() {
            let tr = &graph.transitions[i];
            let child_pos = *id_to_position.get(&NodeId(tr.target_id)).ok_or_else(|| {
                SddpError::Validation(format!(
                    "node graph: transition target id {} names no declared node",
                    tr.target_id
                ))
            })?;
            list.push(NodeSuccessor {
                child: child_pos,
                probability: tr.probability,
            });
        }
        is_leaf[pos] = list.is_empty();
        successors.push(list);
    }

    // node -> pool: non-leaves own a pool each, in canonical node order; all
    // leaves then share ONE trailing pool id — leaf-ness is the whole
    // condition, unconditional, no boundary-source discriminator. Pool id is
    // its own untyped index space (not part of this newtype sweep), so this
    // stays a raw-usize-indexed array.
    let mut pool_id = vec![0usize; n];
    let mut next_pool = 0usize;
    for pos in 0..n {
        if !is_leaf[pos] {
            pool_id[pos] = next_pool;
            next_pool += 1;
        }
    }
    let has_leaf = is_leaf.iter().any(|&leaf| leaf);
    let shared_leaf_pool = next_pool;
    if has_leaf {
        for pos in 0..n {
            if is_leaf[pos] {
                pool_id[pos] = shared_leaf_pool;
            }
        }
    }
    let n_pools = if has_leaf { next_pool + 1 } else { next_pool };

    let mut nodes: TypedVec<NodePos, NodeRuntime> = TypedVec::with_capacity(n);
    for &pos in &order {
        let node = &graph.nodes[pos];
        let canonical_pos = id_to_position[&NodeId(node.id)];
        let stage = StageIdx(resolver.resolve(node.stage_id).ok_or_else(|| {
            SddpError::Validation(format!(
                "node graph: node {} stage id {} names no declared study stage",
                node.id, node.stage_id
            ))
        })?);
        let openings = match node.scenario_id {
            Some(k) => {
                let offset = usize::try_from(k).map_err(|_| {
                    SddpError::Validation(format!(
                        "node graph: node {} scenario_id {k} is negative",
                        node.id
                    ))
                })?;
                NodeOpenings::new(OpeningSource::External, offset, 1)
            }
            None => NodeOpenings::new(OpeningSource::Generated, 0, tree.n_openings(stage.0)),
        };
        nodes.push(NodeRuntime {
            stage,
            pool_id: pool_id[canonical_pos.0],
            openings,
        });
    }

    // pool -> stage: every node sharing a pool shares one stage (a non-leaf
    // pool has one owner; leaves sharing the terminal pool are all terminal),
    // so the last writer per pool is also the only stage value written. The
    // debug-only conflict check guards a future leaf-terminality relaxation or a
    // programmatic construction path that bypasses cobre-io validation.
    let mut pool_stage = vec![StageIdx(0); n_pools];
    #[cfg(debug_assertions)]
    let mut pool_stage_seen = vec![false; n_pools];
    for node in &nodes {
        #[cfg(debug_assertions)]
        {
            if pool_stage_seen[node.pool_id] {
                debug_assert_eq!(
                    pool_stage[node.pool_id], node.stage,
                    "node graph: pool {} owned by nodes at differing stages; every node \
                     sharing a pool must share one stage",
                    node.pool_id,
                );
            }
            pool_stage_seen[node.pool_id] = true;
        }
        pool_stage[node.pool_id] = node.stage;
    }

    // Every edge goes t -> t+1 (upstream structural validation); every
    // successor of a node therefore sits exactly one stage downstream —
    // asserted, not re-derived, since this is what makes pool dimension
    // well-defined by construction with no heterogeneity rule.
    for (pos, node) in nodes.iter_indexed() {
        for succ in &successors[pos] {
            debug_assert_eq!(
                nodes[succ.child].stage,
                node.stage.next(),
                "node graph: successor stage must be exactly one downstream of its parent \
                 (t -> t+1) — parent stage {}, child stage {}",
                node.stage,
                nodes[succ.child].stage
            );
        }
    }

    Ok(NodeGraph {
        node_ids,
        nodes,
        successors,
        n_pools,
        pool_stage,
    })
}

/// Flatten `successors` into `out`, canonical (ascending child node id) order,
/// each entry the un-re-normalized product `P(n→child)·q_{child,ω}` — both
/// factors are already load-time-normalized on `node_graph`. `CVaR`'s tail
/// weighting is index-order-sensitive (sddp.md "Backward opening order is
/// warm-start-only"), so the caller-supplied canonical order is load-bearing.
///
/// Single owner for the backward pass
/// (`training::backward_pass_state::assemble_successor_outcome_weights`) and
/// the lower-bound root evaluation
/// (`training::lower_bound::assemble_root_outcome_weights`) — both delegate here so
/// the length precompute, weight formula, and fill order can never diverge
/// between the two call sites.
pub(crate) fn assemble_outcome_weights(
    node_graph: &NodeGraph,
    successors: &[NodeSuccessor],
    out: &mut Vec<f64>,
) {
    let expected_len = successor_outcome_count(node_graph, successors);
    out.clear();
    out.reserve(expected_len);
    for succ in successors {
        let child = &node_graph.nodes[succ.child];
        let weight = succ.probability * child.openings.q;
        out.extend(std::iter::repeat_n(weight, child.openings.len));
    }
    debug_assert_eq!(
        out.len(),
        expected_len,
        "assemble_outcome_weights: assembled outcome set must have length Σ_(m∈successors)|Ω_m|"
    );
}

/// `Σ_(m∈successors)|Ω_m|` — a node's flattened outcome-set length, the exact
/// quantity [`assemble_outcome_weights`] fills `out` to. Standalone so a
/// workspace-sizing caller (`max_successor_outcome_count`) shares this
/// flattening instead of re-deriving it.
fn successor_outcome_count(node_graph: &NodeGraph, successors: &[NodeSuccessor]) -> usize {
    successors
        .iter()
        .map(|s| node_graph.nodes[s.child].openings.len)
        .sum()
}

impl NodeGraph {
    /// Maximum, over every node, of its own [`successor_outcome_count`] — the
    /// backward pass's true per-node `n_openings` upper bound
    /// (`compute_one_backward_node`'s `n_openings` is exactly this quantity for
    /// whichever node it processes). A leaf's empty successor list contributes
    /// `0`. Absent `nodes[]`, node `t`'s successor is stage `t+1`, so this is
    /// `max` over stages `1..num_stages` of `tree.n_openings` — it excludes stage
    /// 0's own opening count (no node's successor is ever the first stage), a
    /// DIFFERENT quantity the lower-bound evaluation needs separately: a caller
    /// sizing a buffer shared with that evaluation takes the max of both, rather
    /// than assuming this function alone covers it.
    pub(crate) fn max_successor_outcome_count(&self) -> usize {
        self.successors
            .iter()
            .map(|succs| successor_outcome_count(self, succs))
            .max()
            .unwrap_or(0)
    }

    /// Reverse-topological cut-sharing levels: cut-generating nodes (those with at
    /// least one successor — a leaf produces no cut) grouped by stage, the outer
    /// list descending by stage and each inner list ascending by canonical node
    /// position. Sibling subtrees at one level are independent (nested per-node
    /// risk), so a level is one cut-sharing barrier. Absent `nodes[]` every stage
    /// carries one such node, so each level holds exactly one node (== that stage)
    /// and the sweep reduces to the reversed stage loop.
    pub(crate) fn backward_cut_levels(&self) -> Vec<Vec<NodePos>> {
        let cut_generating = |pos: NodePos| !self.successors[pos].is_empty();
        let Some(max_stage) = self
            .nodes
            .iter_indexed()
            .filter(|&(pos, _)| cut_generating(pos))
            .map(|(_, n)| n.stage)
            .max()
        else {
            return Vec::new();
        };

        let mut levels: Vec<Vec<NodePos>> = Vec::new();
        for stage in (0..=max_stage.0).rev().map(StageIdx) {
            let level: Vec<NodePos> = self
                .nodes
                .iter_indexed()
                .filter(|&(pos, n)| n.stage == stage && cut_generating(pos))
                .map(|(pos, _)| pos)
                .collect();
            if !level.is_empty() {
                levels.push(level);
            }
        }
        levels
    }
}

/// Per-node predecessor flag: `false` at position `i` iff no edge names node
/// `i` as its child — a predecessor-free node is a traversal root. Shared by
/// every enumerated/pool walk below that needs to find the graph's roots.
fn build_has_predecessor(graph: &NodeGraph) -> TypedVec<NodePos, bool> {
    let mut has_predecessor: TypedVec<NodePos, bool> = vec![false; graph.nodes.len()].into();
    for succs in &graph.successors {
        for succ in succs {
            has_predecessor[succ.child] = true;
        }
    }
    has_predecessor
}

/// Number of fully-enumerated scenarios the node graph encodes: the sum over
/// every root→leaf path of the product of `openings.len` along that path
/// (`f(n) = |Ω_n| · Σ_child f(child)`, `f(leaf) = |Ω_leaf|`, then summed over
/// the predecessor-free roots). A leaf has no successors; a root has no
/// predecessor.
///
/// Overflow-safe: a `K^T` fan overflows `u64` for even a modest branching
/// factor and horizon, so every product and sum is checked and an overflow is
/// an `Err`, never a wrapped count. This owns the count only — the derived-count
/// reconciliation that rejects an over-large enumeration consumes it elsewhere.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when the path-product-sum exceeds `u64`.
pub(crate) fn enumerated_scenario_count(graph: &NodeGraph) -> Result<u64, SddpError> {
    fn overflow_err() -> SddpError {
        SddpError::Validation(
            "enumerated scenario count exceeds u64: the policy graph's root→leaf \
             path-product-sum (Π |Ω| per path, summed over paths) overflows"
                .to_string(),
        )
    }

    let n = graph.nodes.len();
    let has_predecessor = build_has_predecessor(graph);

    // Resolve `f` in descending-stage order: every successor sits exactly one
    // stage downstream (t -> t+1), so a node's children are already resolved
    // when it is visited.
    let mut order: Vec<NodePos> = (0..n).map(NodePos).collect();
    order.sort_by(|&a, &b| graph.nodes[b].stage.cmp(&graph.nodes[a].stage));

    let mut subtree: TypedVec<NodePos, u64> = vec![0_u64; n].into();
    for &pos in &order {
        let len = u64::try_from(graph.nodes[pos].openings.len).map_err(|_| overflow_err())?;
        let children = if graph.successors[pos].is_empty() {
            1_u64
        } else {
            let mut acc = 0_u64;
            for succ in &graph.successors[pos] {
                acc = acc
                    .checked_add(subtree[succ.child])
                    .ok_or_else(overflow_err)?;
            }
            acc
        };
        subtree[pos] = len.checked_mul(children).ok_or_else(overflow_err)?;
    }

    let mut total = 0_u64;
    for (pos, &has_pred) in has_predecessor.iter_indexed() {
        if !has_pred {
            total = total.checked_add(subtree[pos]).ok_or_else(overflow_err)?;
        }
    }
    Ok(total)
}

/// Standard-deviation multiplier for [`NodeGraph::pool_cut_stride`]'s per-node
/// margin (mean + this many σ under the binomial approximation for a node's
/// realized visit count). A chain node's probability is exactly `1.0`, so its
/// variance term is exactly `0.0` regardless of this value — the chain
/// identity does not depend on the multiplier chosen here.
const VISIT_BOUND_SIGMA_MULTIPLIER: f64 = 3.0;

impl NodeGraph {
    /// Per-pool construction-time visit floor under sampled forward-pass
    /// selection: for each node `n`, `⌈F·P(n) + σ·√(F·P(n)·(1−P(n)))⌉` where
    /// `P(n)` is the root→`n` product of (already load-time-normalized) edge
    /// probabilities (`P(root) = 1`, `P(m) = Σ_{n: m∈n's successors} P(n)·prob(n→m)`),
    /// summed over the nodes sharing a pool id and capped at `F` (only `F` passes
    /// exist, so no pool can realize more visits than that). This is the
    /// top-down dual of [`enumerated_scenario_count`]'s bottom-up opening-count
    /// walk, propagated in the same ascending/descending-stage order.
    ///
    /// A cut-receipt stride — how many times a pool's cuts get a fresh slot per
    /// iteration — distinct from [`Self::forward_solve_counts`]'s per-node LP-solve
    /// count; the two coincide only where the graph makes visits statically known
    /// (a derived-count-one enumerated graph).
    ///
    /// This floor is statistical, not a guarantee: a rare iteration's realized
    /// routed count can exceed it whenever `0 < P(n) < 1`. The training backward
    /// pass rejects that excess rather than silently addressing past the pool's
    /// slot block — see `check_visit_bound` in `training/backward_pass_state.rs`.
    ///
    /// A chain node's single successor normalizes to exactly `1.0`
    /// (`chain_transition_probability`), so `P(n) = 1.0`, the variance term is
    /// exactly `0.0`, and the floor is `F` bit-for-bit for every pool — the
    /// sacred-parity chain identity falls out of this one formula with no shape
    /// branch.
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the ceiled probability-weighted pass count is non-negative and far below u64::MAX"
    )]
    pub(crate) fn pool_cut_stride(&self, forward_passes: u32) -> Vec<u64> {
        let n = self.nodes.len();
        let has_predecessor = build_has_predecessor(self);

        // Ascending-stage order: every successor sits exactly one stage
        // downstream (t -> t+1), so a node's own probability is fully
        // accumulated from its predecessors before it propagates to its children.
        let mut order: Vec<NodePos> = (0..n).map(NodePos).collect();
        order.sort_by_key(|&i| self.nodes[i].stage);

        let mut prob: TypedVec<NodePos, f64> = vec![0.0f64; n].into();
        for &i in &order {
            if !has_predecessor[i] {
                prob[i] = 1.0;
            }
            for succ in &self.successors[i] {
                prob[succ.child] += prob[i] * succ.probability;
            }
        }

        let f = f64::from(forward_passes);
        let mut pool_sum = vec![0u64; self.n_pools];
        for (i, node) in self.nodes.iter_indexed() {
            let p = prob[i];
            let mean = f * p;
            let variance = (f * p * (1.0 - p)).max(0.0);
            let margin = mean + VISIT_BOUND_SIGMA_MULTIPLIER * variance.sqrt();
            pool_sum[node.pool_id] += margin.ceil() as u64;
        }

        let cap = u64::from(forward_passes);
        pool_sum.into_iter().map(|s| s.min(cap)).collect()
    }
}

/// Per-pool cut-receipt stride under exhaustive (`enumerated`) node-native
/// traversal: the distinct-incoming-state count per pool = **1** for every
/// non-leaf pool and **0** for the shared leaf pool. A non-leaf pool's owner
/// node is in-degree 1 (a recombination join is rejected upstream), so every
/// trajectory reaching it saw the one persisted incoming state and the backward
/// appends exactly one cut per node per iteration; a leaf generates no cut, so
/// its pool receives none.
///
/// The enumerated counterpart of [`NodeGraph::pool_cut_stride`]'s sampled
/// statistical margin (mean + σ, capped at `forward_passes`): under node-native
/// enumeration the true stride is known exactly, so sizing at the sampled bound
/// would keep the per-pool capacity/basis/broadcast/checkpoint reservation the
/// enumerated engine never fills. The [`Traversal::Enumerated`] arm routes here;
/// the [`Traversal::Sampled`] arm keeps [`NodeGraph::pool_cut_stride`] unchanged.
pub(crate) fn enumerated_pool_cut_stride(graph: &NodeGraph) -> Vec<u64> {
    let mut stride = vec![0u64; graph.n_pools];
    for (pos, node) in graph.nodes.iter_indexed() {
        if !graph.successors[pos].is_empty() {
            stride[node.pool_id] = 1;
        }
    }
    stride
}

/// Per-node exact prefix count under exhaustive (`enumerated`) forward-pass
/// selection: `π(n)`, the number of distinct root→node prefixes,
/// `π(n) = Σ_{(parent, n) edges} π(parent)·|Ω_parent|`, `π(root) = 1`, in
/// canonical node-position order. The per-node dual of
/// [`enumerated_scenario_count`]'s bottom-up opening walk;
/// [`NodeGraph::forward_solve_counts`] sums it per pool. The enumerated forward driver
/// solves node `n` exactly `π(n)` times per iteration. Overflow-safe
/// (`checked_mul`/`checked_add`), mirroring [`enumerated_scenario_count`].
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a path-product-sum exceeds `u64`.
pub(crate) fn enumerated_node_visit_counts(
    graph: &NodeGraph,
) -> Result<TypedVec<NodePos, u64>, SddpError> {
    fn overflow_err() -> SddpError {
        SddpError::Validation(
            "enumerated visit bound exceeds u64: the policy graph's root→node \
             path-product-sum overflows"
                .to_string(),
        )
    }

    let n = graph.nodes.len();
    let has_predecessor = build_has_predecessor(graph);

    let mut order: Vec<NodePos> = (0..n).map(NodePos).collect();
    order.sort_by_key(|&i| graph.nodes[i].stage);

    let mut prefix_count: TypedVec<NodePos, u64> = vec![0_u64; n].into();
    for &i in &order {
        if !has_predecessor[i] {
            prefix_count[i] = 1;
        }
        let len = u64::try_from(graph.nodes[i].openings.len).map_err(|_| overflow_err())?;
        let contribution = prefix_count[i].checked_mul(len).ok_or_else(overflow_err)?;
        for succ in &graph.successors[i] {
            prefix_count[succ.child] = prefix_count[succ.child]
                .checked_add(contribution)
                .ok_or_else(overflow_err)?;
        }
    }
    Ok(prefix_count)
}

impl NodeGraph {
    /// Per-pool exact visit count under exhaustive (`enumerated`) forward-pass
    /// selection: [`enumerated_node_visit_counts`] summed over the nodes sharing
    /// each pool id (leaf-sharing aggregates several nodes into one pool). The
    /// top-down dual of [`enumerated_scenario_count`]'s bottom-up walk:
    /// `Σ_leaf π(leaf)·|Ω_leaf| == enumerated_scenario_count(graph)`, and the
    /// enumerated forward driver's single-rank per-iteration solve total is
    /// `Σ` of this vector. Overflow-safe, mirroring that function's guard.
    ///
    /// A per-node forward-solve count — how many times the enumerated engine
    /// actually solves each pool's LP — distinct from [`Self::pool_cut_stride`]'s
    /// cut-receipt stride; the two coincide only where the graph makes visits
    /// statically known (a derived-count-one enumerated graph).
    ///
    /// # Errors
    ///
    /// Returns [`SddpError::Validation`] when a path-product-sum exceeds `u64`.
    pub(crate) fn forward_solve_counts(&self) -> Result<Vec<u64>, SddpError> {
        let prefix_count = enumerated_node_visit_counts(self)?;
        let mut pool_sum = vec![0_u64; self.n_pools];
        for (i, node) in self.nodes.iter_indexed() {
            pool_sum[node.pool_id] = pool_sum[node.pool_id]
                .checked_add(prefix_count[i])
                .ok_or_else(|| {
                    SddpError::Validation(
                        "enumerated visit bound exceeds u64: the policy graph's root→node \
                         path-product-sum overflows"
                            .to_string(),
                    )
                })?;
        }
        Ok(pool_sum)
    }
}

/// Whether the enumerated forward's cross-rank state elision at world >= 2 is
/// sound: sound iff every non-leaf (cut-generating) node lies on every
/// root→leaf path — a deterministic trunk that branches only at its own
/// terminal fan. Returns the first (ascending canonical position) interior
/// branch node — one whose own subtree does not span every leaf of the
/// graph, so some root→leaf path bypasses it entirely — or `None` when the
/// graph has no interior branching (a chain, or any trunk+terminal-fan
/// shape).
///
/// [`crate::training::forward::run_enumerated_forward`] populates
/// a node's persisted outgoing state only on the ranks whose assigned paths
/// visit it, leaving every other rank's slot zero-filled
/// (`EnumeratedForwardScratch::ensure_sized`'s zero-fill); a trunk node sits
/// on every path and so is replicated on every rank, but an interior branch
/// node is not.
/// [`crate::training::backward::run_backward_node_replicated`]
/// partitions a cut-generating node's successor openings across ALL ranks
/// regardless of whether a rank actually holds that node's true state — sound
/// only when this predicate returns `None`; the caller hard-rejects otherwise.
pub(crate) fn enumerated_requires_state_exchange(graph: &NodeGraph) -> Option<NodeId> {
    let n = graph.nodes.len();
    let mut is_leaf: TypedVec<NodePos, bool> = vec![false; n].into();
    for (pos, succs) in graph.successors.iter_indexed() {
        is_leaf[pos] = succs.is_empty();
    }
    let total_leaves = is_leaf.iter().filter(|&&leaf| leaf).count();

    // Descending-stage order: every successor sits exactly one stage
    // downstream (t -> t+1), so a node's children are already resolved when
    // it is visited — mirrors `enumerated_scenario_count`'s subtree walk.
    let mut order: Vec<NodePos> = (0..n).map(NodePos).collect();
    order.sort_by(|&a, &b| graph.nodes[b].stage.cmp(&graph.nodes[a].stage));

    let mut leaf_count: TypedVec<NodePos, usize> = vec![0usize; n].into();
    for &pos in &order {
        leaf_count[pos] = if is_leaf[pos] {
            1
        } else {
            graph.successors[pos]
                .iter()
                .map(|s| leaf_count[s.child])
                .sum()
        };
    }

    (0..n)
        .map(NodePos)
        .find(|&pos| !is_leaf[pos] && leaf_count[pos] != total_leaves)
        .map(|pos| graph.node_ids[pos])
}

// ── Driver-shared traversal helpers ─────────────────────────────────────────
//
// Structural resolution helpers shared by the forward training driver
// (`training::forward_pass_state`) and the simulation driver
// (`simulation::pipeline`/`simulation::state`) — both walk this graph the
// same way, so the resolution logic has one owner.

impl NodeGraph {
    /// Ascending node-graph positions with `nodes[pos].stage == stage` — the
    /// stage's alive frontier. Structural only (reads [`NodeGraph::nodes`]; no
    /// sampling or enumeration selection): a synthesized or declared chain's
    /// frontier is always a singleton, and below a recombination join a declared
    /// graph's frontier carries more than one node.
    pub(crate) fn stage_frontier(&self, stage: StageIdx) -> impl Iterator<Item = NodePos> + '_ {
        self.nodes
            .iter_indexed()
            .filter(move |(_, n)| n.stage == stage)
            .map(|(pos, _)| pos)
    }

    /// The study's sole root: stage `stage`'s one alive node, or `None` if the
    /// stage carries no alive node. Used only for stage-0 root resolution — every
    /// sampled trajectory starts here (a multi-root graph is out of scope for the
    /// sampled walk) — so a stage carrying more than one node here is a
    /// `debug_assert`-checked construction invariant, never silently resolved to
    /// the first candidate.
    pub(crate) fn frontier_node(&self, stage: StageIdx) -> Option<NodePos> {
        let mut frontier = self.stage_frontier(stage);
        let node = frontier.next();
        debug_assert!(
            frontier.next().is_none(),
            "frontier_node: stage {stage} carries more than one alive node"
        );
        node
    }

    /// Any one node at `stage` — the first in ascending canonical order, or
    /// `None` if the stage carries no alive node. Unlike [`Self::frontier_node`],
    /// this tolerates a stage carrying several nodes: every node at the terminal
    /// stage is a leaf (no edge crosses the horizon), and leaves sharing a stage
    /// share ONE pool (this module's leaf-sharing rule), so any representative
    /// resolves the same pool.
    pub(crate) fn any_stage_node(&self, stage: StageIdx) -> Option<NodePos> {
        self.stage_frontier(stage).next()
    }

    /// The pool every terminal-stage node shares (this module's leaf-sharing
    /// rule): `nodes[any_stage_node(num_stages - 1)].pool_id`. `None` for a
    /// zero-stage horizon.
    #[inline]
    #[must_use]
    pub fn terminal_pool(&self, num_stages: usize) -> Option<usize> {
        let last = num_stages.checked_sub(1)?;
        Some(self.nodes[self.any_stage_node(StageIdx(last))?].pool_id)
    }
}

/// Advance a sampled trajectory from `node` to the node it visits at the next
/// stage: a single out-edge is taken with probability 1 WITHOUT deriving a
/// seed, so a chain never perturbs the within-node noise stream
/// (chain-parity). Single owner for the training forward pass
/// (`training::forward_pass_state::run_forward_worker`) and the simulation
/// pass (`simulation::pipeline::advance_simulation_node`) — both delegate here
/// so the chain-parity contract is stated once.
///
/// # Panics (debug builds only)
///
/// Panics if `node` has no out-edge — a graph-construction invariant for any
/// call site that is not the terminal stage.
pub(crate) fn advance_sampled_node(
    node_graph: &NodeGraph,
    node: NodePos,
    iteration: u32,
    scenario: u32,
    stage: u32,
) -> NodePos {
    let successors = &node_graph.successors[node];
    debug_assert!(
        !successors.is_empty(),
        "advance_sampled_node: node {node} at stage {stage} has no out-edge before the \
         terminal stage (a graph-construction invariant)"
    );
    if successors.len() == 1 {
        successors[0].child
    } else {
        let idx = select_transition_child(
            iteration,
            scenario,
            stage,
            successors.iter().map(|s| s.probability),
        );
        successors[idx].child
    }
}

impl NodeGraph {
    /// Single-node test accessor for a node's canonical predecessor (`None` at a
    /// root), mirroring [`NodeGraph::build_parent_map`] — production reads that
    /// O(1) map, not this O(n) scan. A node reached by more than one edge is a
    /// recombination join — `debug_assert`-checked, never silently resolved.
    #[cfg(test)]
    pub(crate) fn node_parent(&self, node: NodePos) -> Option<NodePos> {
        let mut candidates = self
            .successors
            .iter_indexed()
            .filter(|(_, succs)| succs.iter().any(|s| s.child == node))
            .map(|(pos, _)| pos);
        let parent = candidates.next();
        debug_assert!(
            candidates.next().is_none(),
            "node_parent: node {node} has more than one predecessor (a recombination join)"
        );
        parent
    }

    /// A visit's node-Ω sub-range for the within-node opening draw: `node`'s own
    /// `(offset, len)`. A `Generated` view addresses its stage's generated tree (a
    /// chain-degenerate node's spans the whole stage); an `External` view addresses
    /// its own scenario column `k` as the degenerate `(k, 1)` — within-node
    /// weighted openings are deferred, so an External view's `len` is always `1`.
    pub(crate) fn node_opening_range(&self, node: NodePos) -> (usize, usize) {
        let openings = self.nodes[node].openings;
        match openings.source {
            OpeningSource::Generated => (openings.offset, openings.len),
            OpeningSource::External => (openings.offset, 1),
        }
    }

    /// The scenario column to pin for a node's within-node draw, or `None` for a
    /// `Generated` node — the value `SampleRequest`/`ClassSampleRequest.pinned_scenario`
    /// takes. Single owner for the forward training driver and the simulation walk so
    /// the two cannot drift: an `External` node pins its declared column `k`
    /// (`openings.offset`), so forward, backward, and simulation all read the same
    /// `eta_slice(stage, k)`; a `Generated` node yields `None`, so the sampler
    /// hash-selects exactly as it does today (the sampled/generated byte-parity path).
    pub(crate) fn node_pinned_scenario(&self, node: NodePos) -> Option<usize> {
        let openings = self.nodes[node].openings;
        match openings.source {
            OpeningSource::External => Some(openings.offset),
            OpeningSource::Generated => None,
        }
    }

    /// `true` iff `node` is a terminal, External, single-opening leaf — the
    /// only case where the forward capture and the backward's own solve read
    /// the byte-identical LP, licensing reuse of a captured forward leaf
    /// slice on the backward. Single source of fusion eligibility for the
    /// forward capture and the backward consume, so the two cannot drift. A
    /// Generated terminal leaf (forward samples one opening, backward
    /// integrates all of them) or an interior node returns `false` — reusing
    /// a single forward opening there understates the cut and drives
    /// `final_lb` above `final_ub`.
    pub(crate) fn is_external_terminal_leaf(&self, node: NodePos, num_stages: usize) -> bool {
        let Some(terminal_stage) = num_stages.checked_sub(1) else {
            return false;
        };
        self.nodes[node].stage.0 >= terminal_stage
            && self.node_pinned_scenario(node).is_some()
            && self.node_opening_range(node).1 == 1
    }

    /// Single canonical predecessor of every node (`None` at a root), indexed by
    /// canonical node position. The enumerated forward driver reads it to pin each
    /// visited node's incoming state to its parent visit's outgoing state (every
    /// edge is `t -> t+1`, so a node's parent sits exactly one stage upstream). A
    /// node reached by more than one edge is a recombination join, which the graph
    /// does not yet admit — `debug_assert`-checked, never silently resolved to an
    /// arbitrary predecessor.
    pub(crate) fn build_parent_map(&self) -> TypedVec<NodePos, Option<NodePos>> {
        let mut parent: TypedVec<NodePos, Option<NodePos>> = vec![None; self.nodes.len()].into();
        for (pos, succs) in self.successors.iter_indexed() {
            for succ in succs {
                debug_assert!(
                    parent[succ.child].is_none(),
                    "build_parent_map: node {} has more than one predecessor (a recombination join)",
                    succ.child
                );
                parent[succ.child] = Some(pos);
            }
        }
        parent
    }
}

/// Canonical root→leaf path enumeration for the exhaustive (`enumerated`)
/// forward: one entry per fully-enumerated scenario path, in depth-first order
/// with roots and successors both taken in canonical (ascending node id)
/// order — the order `enumerated_scenario_count` counts and the exact upper
/// bound reduces over.
///
/// Every admitted enumerated graph has `|Ω_n| == 1` at every node (within-node
/// opening enumeration is deferred and rejected upstream), so a path is a
/// distinct root→leaf node sequence and the path count is
/// [`enumerated_scenario_count`].
#[derive(Debug)]
pub(crate) struct EnumeratedForwardPaths {
    /// Leaf node position of each path, canonical order. Length = path count.
    pub leaf: Vec<NodePos>,
    /// `P(ℓ)` of each path, canonical order: the fixed-order root→leaf product
    /// of edge transition probabilities and within-node opening weights `q`
    /// (both already load-time-normalized). Length = path count.
    pub weight: Vec<f64>,
}

/// Enumerate every root→leaf path in canonical order (see
/// [`EnumeratedForwardPaths`]). Roots are the predecessor-free nodes in
/// ascending canonical position; each root's subtree is walked depth-first in
/// canonical successor order.
pub(crate) fn enumerate_forward_paths(node_graph: &NodeGraph) -> EnumeratedForwardPaths {
    fn walk(
        node_graph: &NodeGraph,
        node: NodePos,
        weight_before_node: f64,
        leaf: &mut Vec<NodePos>,
        weight: &mut Vec<f64>,
    ) {
        let w = weight_before_node * node_graph.nodes[node].openings.q;
        let succs = &node_graph.successors[node];
        if succs.is_empty() {
            leaf.push(node);
            weight.push(w);
            return;
        }
        for succ in succs {
            walk(node_graph, succ.child, w * succ.probability, leaf, weight);
        }
    }

    let has_predecessor = build_has_predecessor(node_graph);
    let mut paths = EnumeratedForwardPaths {
        leaf: Vec::new(),
        weight: Vec::new(),
    };
    for (root, &has_pred) in has_predecessor.iter_indexed() {
        if !has_pred {
            walk(node_graph, root, 1.0, &mut paths.leaf, &mut paths.weight);
        }
    }
    paths
}

// ── Traversal: the resolved forward-traversal axis ─────────────────────────

/// The resolved forward-traversal axis — sampled or exhaustive — for one
/// selection declaration (training's `training_enumerated`, or simulation's
/// `simulation_enumerated`). Resolved once, from the graph and an
/// already-admissibility-checked selection flag, via [`Traversal::resolve`];
/// never re-derived per iteration. Enum dispatch (never `Box<dyn Trait>`) so
/// "selected exhaustive but carries no plan" is unrepresentable — the plan
/// lives inline in the `Enumerated` variant, not behind a separate flag and
/// `Option`.
///
/// Distinct from the config wire forms (`training_enumerated: bool`,
/// [`crate::setup::SimulationEnumeratedRequest`]), which stay on the MPI
/// broadcast unchanged; `Traversal` is the runtime resolution of them and is
/// never itself broadcast.
#[derive(Debug)]
pub enum Traversal {
    /// Monte-Carlo forward-pass selection; `forward_passes` is the resolved
    /// (not placeholder) per-iteration trajectory count.
    Sampled {
        /// Resolved forward-pass count for this axis (training's
        /// `forward_passes` or simulation's `n_scenarios`).
        forward_passes: u32,
    },
    /// Exhaustive root→leaf path selection over the resolved node graph.
    Enumerated(EnumeratedPlan),
}

impl Default for Traversal {
    /// `Sampled { forward_passes: 0 }` — the placeholder a `ForwardPassState`
    /// carries until `set_traversal` installs the resolved axis, and the
    /// value `run()` swaps back in around the `Enumerated` arm's borrow
    /// (never observed by a real caller).
    fn default() -> Self {
        Traversal::Sampled { forward_passes: 0 }
    }
}

impl Traversal {
    /// Resolve the traversal axis from an already-admissibility-checked
    /// selection: `is_enumerated` and `forward_passes` must already reflect
    /// the caller's own guard-checked resolution (training routes through
    /// `resolve_enumerated_training_count`, simulation through
    /// `resolve_enumerated_simulation_count` — both share one admissibility
    /// guard) — this constructor re-derives no guard and cannot fail.
    #[must_use]
    pub fn resolve(node_graph: &NodeGraph, is_enumerated: bool, forward_passes: u32) -> Self {
        if is_enumerated {
            Traversal::Enumerated(EnumeratedPlan::new(node_graph))
        } else {
            Traversal::Sampled { forward_passes }
        }
    }

    /// Per-path probabilities `P(ℓ)` in canonical path order, for the exact
    /// `Σ P(ℓ)·C(ℓ)` forward bound — `None` under `Sampled`, where the bound
    /// is the statistical Welford estimate instead.
    #[must_use]
    pub fn path_weights(&self) -> Option<&[f64]> {
        match self {
            Traversal::Sampled { .. } => None,
            Traversal::Enumerated(plan) => Some(&plan.paths.weight),
        }
    }

    /// The simulation cost-aggregation weighting this axis licenses: an exact
    /// census under `Enumerated` (weighted by the same per-path probabilities
    /// [`Self::path_weights`] returns), the uniform Monte-Carlo weighting
    /// under `Sampled`. The single derivation point for
    /// [`crate::simulation::SimulationWeighting::Census`] — an estimator
    /// carrying no evidence of its own that the walk it weights was
    /// exhaustive, so every caller reaches it through here rather than
    /// constructing it beside a selection flag.
    #[must_use]
    pub fn simulation_weighting(&self) -> SimulationWeighting<'_> {
        match self.path_weights() {
            Some(weights) => SimulationWeighting::Census { weights },
            None => SimulationWeighting::Uniform,
        }
    }
}

/// Graph-invariant exhaustive-traversal plan: the canonical root→leaf path
/// enumeration and the single-predecessor parent map, both pure functions of
/// the resolved [`NodeGraph`]. Carries no per-run mutable scratch — the
/// enumerated forward engine's reused arenas are
/// `training::forward::enumerated::EnumeratedForwardScratch`, owned
/// separately by `ForwardPassState` and grown lazily against this plan.
#[derive(Debug)]
pub struct EnumeratedPlan {
    pub(crate) paths: EnumeratedForwardPaths,
    pub(crate) parent: TypedVec<NodePos, Option<NodePos>>,
    /// Precomputed tree structure the nested upper-bound recursion reduces over
    /// (children / roots / valuation order / node marginals) — pure functions of
    /// `parent` and `paths`, resolved once here instead of every training
    /// iteration. Boxed to keep [`Traversal::Enumerated`] lean.
    pub(crate) nested_ub_topology: Box<NestedUbTopology>,
}

impl EnumeratedPlan {
    fn new(node_graph: &NodeGraph) -> Self {
        Self::from_parts(
            enumerate_forward_paths(node_graph),
            node_graph.build_parent_map(),
        )
    }

    /// Assemble a plan from its enumerated paths and parent map, precomputing the
    /// nested-UB reduction topology once.
    pub(crate) fn from_parts(
        paths: EnumeratedForwardPaths,
        parent: TypedVec<NodePos, Option<NodePos>>,
    ) -> Self {
        let nested_ub_topology =
            Box::new(NestedUbTopology::new(&parent, &paths.leaf, &paths.weight));
        Self {
            paths,
            parent,
            nested_ub_topology,
        }
    }

    /// Walk path `p`'s leaf up to its root, writing the canonical root→leaf node
    /// sequence (ascending stage) into `out`.
    pub(crate) fn walk_path(&self, p: usize, num_stages: usize, out: &mut Vec<NodePos>) {
        walk_leaf_to_root(&self.parent, self.paths.leaf[p], out);
        debug_assert_eq!(
            out.len(),
            num_stages,
            "root→leaf path must visit exactly one node per stage"
        );
    }
}

/// Write the canonical root→leaf node sequence (ascending stage) for the path
/// ending at `leaf_node` into `out`, walking the single-predecessor parent map.
/// The single owner of this walk — [`EnumeratedPlan::walk_path`] and
/// [`NestedUbTopology::new`] both drive off it.
pub(crate) fn walk_leaf_to_root(
    parent: &TypedVec<NodePos, Option<NodePos>>,
    leaf_node: NodePos,
    out: &mut Vec<NodePos>,
) {
    out.clear();
    let mut cur = Some(leaf_node);
    while let Some(node) = cur {
        out.push(node);
        cur = parent[node];
    }
    out.reverse();
}

/// Precomputed reduction structure over an enumerated scenario tree, consumed by
/// the nested upper-bound recursion. Every field is a pure function of the
/// `parent` map and per-path `weight`s, so it is resolved once at plan
/// construction rather than every training iteration; only the per-node realized
/// costs change between iterations.
#[derive(Debug)]
pub struct NestedUbTopology {
    /// Stage (root-distance) of each node.
    pub(crate) node_stage: Vec<usize>,
    /// A representative path index reaching each node — any path works, since a
    /// node's immediate cost is identical on every path through it.
    pub(crate) node_path: Vec<usize>,
    /// Marginal probability of each node (Σ path weights through it, in canonical
    /// path order for rank-invariance).
    pub(crate) node_prob: Vec<f64>,
    /// Children of each node in canonical node-position order — the tie-break the
    /// risk measure's tail selection depends on.
    pub(crate) children: Vec<Vec<NodePos>>,
    /// Predecessor-free roots, canonical order.
    pub(crate) roots: Vec<NodePos>,
    /// Nodes deepest-stage first, so a node's children are valued before it.
    pub(crate) valuation_order: Vec<NodePos>,
}

impl NestedUbTopology {
    pub(crate) fn new(
        parent: &TypedVec<NodePos, Option<NodePos>>,
        leaf: &[NodePos],
        weight: &[f64],
    ) -> Self {
        let n_nodes = parent.len();
        let mut node_stage = vec![0_usize; n_nodes];
        let mut node_path = vec![0_usize; n_nodes];
        let mut node_prob = vec![0.0_f64; n_nodes];
        let mut seq: Vec<NodePos> = Vec::new();
        for (p, &leaf_node) in leaf.iter().enumerate() {
            walk_leaf_to_root(parent, leaf_node, &mut seq);
            let w = weight[p];
            for (t, &node) in seq.iter().enumerate() {
                node_stage[node.0] = t;
                node_path[node.0] = p;
                node_prob[node.0] += w;
            }
        }

        let mut children: Vec<Vec<NodePos>> = vec![Vec::new(); n_nodes];
        let mut roots: Vec<NodePos> = Vec::new();
        for node in (0..n_nodes).map(NodePos) {
            match parent[node] {
                Some(p) => children[p.0].push(node),
                None => roots.push(node),
            }
        }

        let mut valuation_order: Vec<NodePos> = (0..n_nodes).map(NodePos).collect();
        valuation_order.sort_by(|&a, &b| node_stage[b.0].cmp(&node_stage[a.0]));

        Self {
            node_stage,
            node_path,
            node_prob,
            children,
            roots,
            valuation_order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cut::FutureCostFunction;
    use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, Transition};
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    fn cb(node_id: i32, marker: u32) -> CapturedBasis {
        let mut c = CapturedBasis::new(0, 0, 0, 1, 0, NodeId(node_id));
        c.cut_row_slots.push(marker);
        c
    }

    fn fingerprint(cache: &[Option<CapturedBasis>]) -> Vec<Option<(i32, Vec<u32>)>> {
        cache
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .map(|c| (c.node_id.0, c.cut_row_slots.clone()))
            })
            .collect()
    }

    #[test]
    fn pool_fill_warms_same_pool_none_siblings_retagged_with_own_node_id() {
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2, 3, 3, 3, 3].into();
        let node_ids: TypedVec<NodePos, NodeId> = vec![10, 11, 12, 13, 14, 15, 16]
            .into_iter()
            .map(NodeId)
            .collect();
        let mut cache: Vec<Option<CapturedBasis>> = vec![
            Some(cb(10, 100)),
            Some(cb(11, 101)),
            Some(cb(12, 102)),
            Some(cb(13, 103)),
            None,
            None,
            None,
        ];

        pool_fill_basis_cache(&mut cache, &node_pools, &node_ids);

        for (node, own_id) in [(4usize, 14i32), (5, 15), (6, 16)] {
            let filled = cache[node].as_ref().expect("sibling leaf must be filled");
            assert_eq!(
                filled.node_id,
                NodeId(own_id),
                "filled leaf carries its OWN node_id, never the source's"
            );
            assert_eq!(
                filled.cut_row_slots,
                vec![103_u32],
                "filled leaf copies pool 3 representative (leaf 3) content"
            );
        }
        assert_eq!(cache[3].as_ref().expect("leaf 3").node_id, NodeId(13));
        assert_eq!(cache[0].as_ref().expect("node 0").node_id, NodeId(10));
    }

    #[test]
    fn pool_fill_leaves_no_capture_pool_unfilled() {
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 1].into();
        let node_ids: TypedVec<NodePos, NodeId> = vec![0, 1, 2].into_iter().map(NodeId).collect();
        let mut cache: Vec<Option<CapturedBasis>> = vec![Some(cb(0, 100)), None, None];

        pool_fill_basis_cache(&mut cache, &node_pools, &node_ids);

        assert!(cache[1].is_none(), "no capture in pool 1 → left cold");
        assert!(cache[2].is_none(), "no capture in pool 1 → left cold");
    }

    #[test]
    fn pool_fill_chain_is_a_no_op() {
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2].into();
        let node_ids: TypedVec<NodePos, NodeId> = vec![0, 1, 2].into_iter().map(NodeId).collect();
        let mut cache: Vec<Option<CapturedBasis>> = vec![Some(cb(0, 100)), None, Some(cb(2, 102))];

        pool_fill_basis_cache(&mut cache, &node_pools, &node_ids);

        assert!(
            cache[1].is_none(),
            "chain pool has no same-pool sibling to fill from"
        );
        assert_eq!(
            cache[0].as_ref().expect("node 0").cut_row_slots,
            vec![100_u32]
        );
        assert_eq!(
            cache[2].as_ref().expect("node 2").cut_row_slots,
            vec![102_u32]
        );
    }

    #[test]
    fn pool_fill_representative_is_lowest_nodepos_with_capture() {
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2, 3, 3, 3, 3].into();
        let node_ids: TypedVec<NodePos, NodeId> = vec![10, 11, 12, 13, 14, 15, 16]
            .into_iter()
            .map(NodeId)
            .collect();
        let mut cache: Vec<Option<CapturedBasis>> = vec![
            Some(cb(10, 100)),
            Some(cb(11, 101)),
            Some(cb(12, 102)),
            None,
            Some(cb(14, 140)),
            None,
            Some(cb(16, 160)),
        ];

        pool_fill_basis_cache(&mut cache, &node_pools, &node_ids);

        for (node, own_id) in [(3usize, 13i32), (5, 15)] {
            let filled = cache[node].as_ref().expect("gap leaf must be filled");
            assert_eq!(filled.node_id, NodeId(own_id));
            assert_eq!(
                filled.cut_row_slots,
                vec![140_u32],
                "filled from node 4 (lowest NodePos with a capture), never node 6"
            );
        }
        assert_eq!(cache[6].as_ref().expect("leaf 6").node_id, NodeId(16));
    }

    #[test]
    fn pool_fill_is_deterministic_across_repeat_calls() {
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 1, 1].into();
        let node_ids: TypedVec<NodePos, NodeId> =
            vec![0, 1, 2, 3].into_iter().map(NodeId).collect();
        let build = || vec![Some(cb(0, 100)), Some(cb(1, 200)), None, None];
        let mut a = build();
        let mut b = build();

        pool_fill_basis_cache(&mut a, &node_pools, &node_ids);
        pool_fill_basis_cache(&mut b, &node_pools, &node_ids);

        assert_eq!(
            fingerprint(&a),
            fingerprint(&b),
            "identical sparse input → identical fill (single shared helper on both sim paths)"
        );
    }

    fn transition(source_id: i32, target_id: i32, probability: f64) -> Transition {
        Transition {
            source_id,
            target_id,
            probability,
            annual_discount_rate_override: None,
        }
    }

    fn node(id: i32, stage_id: i32, scenario_id: Option<i32>) -> PolicyNode {
        PolicyNode {
            id,
            stage_id,
            scenario_id,
            label: None,
        }
    }

    fn empty_graph() -> HorizonGraph {
        HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: Vec::new(),
            nodes: Vec::new(),
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            season_map: None,
        }
    }

    /// A minimal, generated-only `StochasticContext` over `n_stages` stages
    /// with `n_hydros` hydro entities carrying independent noise and
    /// `branching_factor` openings per stage — enough to exercise
    /// `stochastic.opening_tree()` without any external library.
    #[expect(
        clippy::too_many_lines,
        reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
    )]
    fn stochastic_context(
        n_stages: usize,
        n_hydros: usize,
        branching_factor: usize,
    ) -> StochasticContext {
        use chrono::NaiveDate;
        use cobre_core::entities::bus::{Bus, DeficitSegment};
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{
            BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
            HydroBlockBounds, HydroPenalties, HydroStageBounds, InflowModel, LineBlockBounds,
            LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
            PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
            ThermalBlockBounds, ThermalStageBounds,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let hydros: Vec<Hydro> = (0..n_hydros)
            .map(|h| {
                let mut hydro = Hydro {
                    unit_groups: Vec::new(),
                    id: EntityId(10 + h as i32),
                    name: format!("H{h}"),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    downstream_id: None,
                    travel_time_hours: None,
                    entry_stage_id: None,
                    exit_stage_id: None,
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    specific_productivity_mw_per_m3s_per_m: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    tailrace: None,
                    hydraulic_losses: None,
                    efficiency: None,
                    evaporation_coefficients_mm: None,
                    evaporation_reference_volumes_hm3: None,
                    diversion: None,
                    filling: None,
                    penalties: HydroPenalties {
                        spillage_cost: 0.01,
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
                        inflow_nonnegativity_cost: 1000.0,
                    },
                };
                hydro.declare_mirror_unit_group(EntityId(1));
                hydro
            })
            .collect();
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();
        let inflow_models: Vec<InflowModel> = (0..n_hydros)
            .flat_map(|h| {
                (0..n_stages).map(move |i| InflowModel {
                    hydro_id: EntityId(10 + h as i32),
                    stage_id: i as i32,
                    mean_m3s: 80.0,
                    std_m3s: 20.0,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                })
            })
            .collect();
        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("stochastic_context fixture: valid system");

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: None,
                load: None,
                ncs: None,
            },
        )
        .expect("stochastic_context fixture: build_stochastic_context")
    }

    // ── Chain degeneracy ────────────────────────────────────────────────────

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "test counts are small, far inside f64's exact-integer range"
    )]
    fn chain_degeneracy_one_node_per_stage_1to1_pools_uniform_q_bit_pattern() {
        let n_stages = 4;
        let branching = 5;
        let stochastic = stochastic_context(n_stages, 1, branching);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();

        assert_eq!(ng.nodes.len(), n_stages, "one node per stage");
        assert_eq!(ng.n_pools, n_stages, "pools 1:1");
        for (t, n) in ng.nodes.iter().enumerate() {
            assert_eq!(n.stage, StageIdx(t));
            assert_eq!(n.pool_id, t, "chain pools are 1:1 (identity)");
            assert_eq!(n.openings.source, OpeningSource::Generated);
            assert_eq!(n.openings.len, branching);
            let expected_q = 1.0 / (branching as f64);
            assert_eq!(
                n.openings.q.to_bits(),
                expected_q.to_bits(),
                "q must be the exact 1.0/(n as f64) bit pattern, not a normalized accumulation"
            );
        }
        // Last stage is the terminal leaf: no successors.
        assert!(ng.successors[NodePos(n_stages - 1)].is_empty());
        for t in 0..n_stages - 1 {
            assert_eq!(ng.successors[NodePos(t)].len(), 1);
            assert_eq!(ng.successors[NodePos(t)][0].child, NodePos(t + 1));
            assert_eq!(ng.successors[NodePos(t)][0].probability, 1.0);
        }
    }

    #[test]
    fn chain_degeneracy_does_not_touch_discount_arrays() {
        // NodeGraph carries no discount-related field at all — the type
        // itself is the proof; this test pins that `build_node_graph`'s
        // output is silent on discount by construction (no field to inspect).
        let n_stages = 3;
        let stochastic = stochastic_context(n_stages, 1, 2);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();
        assert_eq!(ng.nodes.len(), n_stages);
        // Compile-time: NodeRuntime/NodeGraph have no discount field to read.
    }

    // ── Leaf pool sharing ───────────────────────────────────────────────────

    #[test]
    fn k_fan_leaves_share_one_pool_id_non_leaf_owns_its_own() {
        // Root at stage 0 (id 0), K=4 leaves at stage 1 (ids 1..4).
        let stochastic = stochastic_context(2, 1, 4);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 1, None),
                node(4, 1, None),
            ],
            transitions: vec![
                transition(0, 1, 0.25),
                transition(0, 2, 0.25),
                transition(0, 3, 0.25),
                transition(0, 4, 0.25),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        // Root (position 0) owns its own pool.
        let root_pool = ng.nodes[NodePos(0)].pool_id;
        // The four leaves (positions 1..4) share exactly one pool id, distinct
        // from the root's.
        let leaf_pools: Vec<usize> = (1..=4).map(|i| ng.nodes[NodePos(i)].pool_id).collect();
        assert!(
            leaf_pools.iter().all(|&p| p == leaf_pools[0]),
            "all leaves must share one pool id: {leaf_pools:?}"
        );
        assert_ne!(
            root_pool, leaf_pools[0],
            "a node with successors never shares the leaf pool"
        );
        assert_eq!(ng.n_pools, 2, "one pool for the root, one shared leaf pool");
    }

    #[test]
    fn terminal_pool_resolves_the_chain_and_the_shared_fan_leaf_pool() {
        let chain = crate::test_support::oracle_chain_setup(1);
        let chain_num_stages = chain.training_ctx().horizon.num_stages();
        assert_eq!(
            chain.inputs.node_graph.terminal_pool(chain_num_stages),
            Some(chain.fcf.pools.len() - 1),
            "a chain's terminal pool is its last stage's identity pool"
        );

        let fan = crate::test_support::terminal_generated_fan_setup(2, 1);
        let fan_num_stages = fan.training_ctx().horizon.num_stages();
        let terminal = fan
            .inputs
            .node_graph
            .terminal_pool(fan_num_stages)
            .expect("the fan's terminal stage carries alive nodes");
        assert_eq!(
            terminal,
            fan.fcf.pools.len() - 1,
            "the shared leaf pool owns the highest id"
        );
        let last = StageIdx(fan_num_stages - 1);
        assert!(
            fan.inputs
                .node_graph
                .nodes
                .iter()
                .filter(|n| n.stage == last)
                .all(|n| n.pool_id == terminal),
            "every terminal-stage node must resolve to the same pool"
        );
    }

    #[test]
    fn duplicate_node_id_is_rejected_at_build() {
        // Two nodes declared with the same id 1; the basis apply-guard keys on
        // node id, so a duplicate must be rejected before any solve.
        let stochastic = stochastic_context(2, 1, 2);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(1, 1, None)],
            transitions: vec![transition(0, 1, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let err = build_node_graph(&graph, 2, &resolver, &stochastic)
            .expect_err("a duplicate node id must be rejected");
        match err {
            SddpError::Validation(msg) => {
                assert!(
                    msg.contains("duplicate node id 1"),
                    "the rejection must name the duplicate id: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    #[test]
    fn pool_stage_maps_each_pool_to_its_owning_stage_over_a_k_fan() {
        // Root (stage 0, own pool) fans into two interior nodes at stage 1 (each
        // its own pool) with two leaves each at stage 2 (shared leaf pool).
        let stochastic = stochastic_context(3, 1, 2);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
                node(4, 2, None),
                node(5, 2, None),
                node(6, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        assert_eq!(ng.pool_stage.len(), ng.n_pools);
        // Every pool's recorded stage must equal the stage of every node owning
        // it (leaf pool: all owners terminal → one shared stage).
        for (pos, n) in ng.nodes.iter().enumerate() {
            assert_eq!(
                ng.pool_stage[n.pool_id], n.stage,
                "pool_stage[pool of node {pos}] must equal that node's stage"
            );
        }
        // The shared leaf pool maps to the terminal stage 2.
        let leaf_pool = ng.nodes[NodePos(3)].pool_id;
        assert_eq!(ng.pool_stage[leaf_pool], StageIdx(2));
    }

    #[test]
    fn pool_stage_chain_is_identity() {
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 3);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic).unwrap();
        assert_eq!(
            ng.pool_stage,
            vec![StageIdx(0), StageIdx(1), StageIdx(2), StageIdx(3)]
        );
    }

    #[test]
    fn leaf_sharing_guard_is_never_applied_to_a_node_with_successors() {
        // A 2-stage chain-like fan where stage 1 also has ITS OWN successor
        // (stage 2) — stage-1 node must own its own pool, not share.
        let stochastic = stochastic_context(3, 1, 2);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 2, None)],
            transitions: vec![transition(0, 1, 1.0), transition(1, 2, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        // Only node 2 (position 2) is a leaf; nodes 0 and 1 have successors
        // and must own distinct pools from each other and from the leaf pool.
        let pools: Vec<usize> = ng.nodes.iter().map(|n| n.pool_id).collect();
        assert_ne!(pools[0], pools[1], "non-leaf nodes never share a pool");
        assert_ne!(
            pools[1], pools[2],
            "the leaf pool is distinct from its parent's"
        );
        assert_eq!(ng.n_pools, 3);
    }

    // ── Canonical successor order ───────────────────────────────────────────

    #[test]
    fn successor_order_is_ascending_child_node_id_not_declaration_order() {
        // Declare the root's out-edges and the nodes themselves in reverse /
        // shuffled order; the runtime successor list must still come out
        // ascending by child node id.
        let stochastic = stochastic_context(2, 1, 3);
        let graph = HorizonGraph {
            // Nodes declared out of id order (5, 0, 3, 1).
            nodes: vec![
                node(5, 1, None),
                node(0, 0, None),
                node(3, 1, None),
                node(1, 1, None),
            ],
            // Transitions declared out of target-id order.
            transitions: vec![
                transition(0, 5, 0.2),
                transition(0, 1, 0.3),
                transition(0, 3, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        // Canonical node order is ascending declared id: 0, 1, 3, 5.
        assert_eq!(
            ng.node_ids.as_slice(),
            [NodeId(0), NodeId(1), NodeId(3), NodeId(5)]
        );
        let root_pos = NodePos(ng.node_ids.iter().position(|&id| id == NodeId(0)).unwrap());
        let child_ids: Vec<NodeId> = ng.successors[root_pos]
            .iter()
            .map(|s| ng.node_ids[s.child])
            .collect();
        assert_eq!(
            child_ids,
            vec![NodeId(1), NodeId(3), NodeId(5)],
            "successors must be ascending child node id regardless of declaration order"
        );
    }

    // ── Ω views: no realization copy ────────────────────────────────────────

    #[test]
    fn generated_node_omega_is_a_view_offset_len_q_into_the_opening_tree() {
        let n_stages = 2;
        let branching = 6;
        let stochastic = stochastic_context(n_stages, 1, branching);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();

        for n in &ng.nodes {
            assert_eq!(n.openings.source, OpeningSource::Generated);
            assert_eq!(n.openings.offset, 0);
            assert_eq!(
                n.openings.len,
                stochastic.opening_tree().n_openings(n.stage.0)
            );
        }
        // NodeOpenings carries no realization buffer field — inspection of
        // the type itself (offset/len/q, no `Vec<f64>`) is the "no second
        // copy" proof; std::mem::size_of pins it stays a small POD view.
        assert!(std::mem::size_of::<NodeOpenings>() <= 32);
    }

    #[test]
    fn external_node_omega_is_a_degenerate_view_at_scenario_id() {
        let stochastic = stochastic_context(1, 1, 1);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, Some(7))],
            transitions: vec![],
            ..empty_graph()
        };
        let study_stage_ids = [0];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng = build_node_graph(&graph, 1, &resolver, &stochastic).unwrap();

        assert_eq!(ng.nodes.len(), 1);
        let openings = ng.nodes[NodePos(0)].openings;
        assert_eq!(openings.source, OpeningSource::External);
        assert_eq!(openings.offset, 7);
        assert_eq!(openings.len, 1);
        assert_eq!(openings.q.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn node_opening_range_external_is_scenario_column_generated_is_offset_len() {
        let stochastic = stochastic_context(2, 1, 6);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, Some(4)), node(1, 1, None)],
            transitions: vec![transition(0, 1, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        assert_eq!(
            ng.nodes[NodePos(0)].openings.source,
            OpeningSource::External
        );
        assert_eq!(ng.node_opening_range(NodePos(0)), (4, 1));

        assert_eq!(
            ng.nodes[NodePos(1)].openings.source,
            OpeningSource::Generated
        );
        let generated = ng.nodes[NodePos(1)].openings;
        assert_eq!(
            ng.node_opening_range(NodePos(1)),
            (generated.offset, generated.len)
        );
    }

    #[test]
    fn node_pinned_scenario_external_pins_declared_column_generated_none() {
        let stochastic = stochastic_context(2, 1, 6);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, Some(4)), node(1, 1, None)],
            transitions: vec![transition(0, 1, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        // External node pins its declared column `k`, and it is exactly the
        // `node_opening_range` offset the forward driver reads — the single-owner
        // agreement the training and simulation sites depend on.
        assert_eq!(ng.node_pinned_scenario(NodePos(0)), Some(4));
        assert_eq!(
            ng.node_pinned_scenario(NodePos(0)),
            Some(ng.node_opening_range(NodePos(0)).0)
        );
        // Generated node yields `None` → the sampler hash-selects unchanged.
        assert_eq!(ng.node_pinned_scenario(NodePos(1)), None);
    }

    // ── Fusion eligibility: terminal + External + single-opening ───────────

    #[test]
    fn is_external_terminal_leaf_true_for_terminal_external_single_opening() {
        let stochastic = stochastic_context(2, 1, 6);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, Some(4))],
            transitions: vec![transition(0, 1, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        assert!(ng.is_external_terminal_leaf(NodePos(1), 2));
    }

    #[test]
    fn is_external_terminal_leaf_false_for_generated_terminal_leaf() {
        let stochastic = stochastic_context(2, 1, 6);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None)],
            transitions: vec![transition(0, 1, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        assert!(!ng.is_external_terminal_leaf(NodePos(1), 2));
    }

    #[test]
    fn is_external_terminal_leaf_false_for_interior_external_node() {
        let stochastic = stochastic_context(2, 1, 6);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, Some(4)), node(1, 1, None)],
            transitions: vec![transition(0, 1, 1.0)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        assert!(!ng.is_external_terminal_leaf(NodePos(0), 2));
    }

    // ── Discount: per-stage, never per-node ─────────────────────────────────

    #[test]
    fn nodes_at_the_same_stage_resolve_through_the_same_stage_index() {
        // Two sibling nodes at stage 1 (a K=2 fan) both carry `stage == 1`;
        // a discount consumer reading `cumulative_discount_factors[node.stage]`
        // necessarily reads the SAME array slot for both — no per-edge rate
        // is consulted here and no agreement check is performed (there is
        // nothing to agree over: one shared index).
        let stochastic = stochastic_context(2, 1, 2);
        let graph = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            transitions: vec![transition(0, 1, 0.5), transition(0, 2, 0.5)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();

        let sibling_stages: Vec<StageIdx> = (1..=2).map(|i| ng.nodes[NodePos(i)].stage).collect();
        assert_eq!(sibling_stages, vec![StageIdx(1), StageIdx(1)]);
    }

    // ── Rank-invariance (mpiexec -n1 vs -n2 substitute) ─────────────────────

    #[test]
    fn node_graph_construction_is_deterministic_across_independent_builds() {
        // NodeGraph construction reads nothing rank-count- or thread-count-
        // dependent (no RNG, no partition split); building it twice from
        // identical inputs — the same guarantee a real `mpiexec -n1` vs
        // `-n2` run relies on, since every rank receives the identical
        // already-broadcast `System`/config and independently reconstructs
        // an identical `StochasticContext` — must produce bitwise-identical
        // results.
        let stochastic_a = stochastic_context(3, 2, 3);
        let stochastic_b = stochastic_context(3, 2, 3);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
                node(4, 2, None),
                node(5, 2, None),
                node(6, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let a = build_node_graph(&graph, 3, &resolver, &stochastic_a).unwrap();
        let b = build_node_graph(&graph, 3, &resolver, &stochastic_b).unwrap();

        assert_eq!(a.node_ids, b.node_ids);
        assert_eq!(a.n_pools, b.n_pools);
        for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(na.stage, nb.stage);
            assert_eq!(na.pool_id, nb.pool_id);
            assert_eq!(na.openings.source, nb.openings.source);
            assert_eq!(na.openings.offset, nb.openings.offset);
            assert_eq!(na.openings.len, nb.openings.len);
            assert_eq!(na.openings.q.to_bits(), nb.openings.q.to_bits());
        }
        for (sa, sb) in a.successors.iter().zip(b.successors.iter()) {
            assert_eq!(sa.len(), sb.len());
            for (ea, eb) in sa.iter().zip(sb.iter()) {
                assert_eq!(ea.child, eb.child);
                assert_eq!(ea.probability.to_bits(), eb.probability.to_bits());
            }
        }
    }

    // ── pool_cut_stride: chain identity ─────────────────────────────────

    #[test]
    fn pool_cut_stride_chain_matches_forward_passes_over_a_sweep() {
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 5);
        let graph = empty_graph();
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, n_stages, &resolver, &stochastic).unwrap();

        for &(warm_start, max_iterations, forward_passes) in &[
            (0u32, 1u64, 1u32),
            (5, 10, 100),
            (0, 100, 1),
            (1000, 1, 4096),
            (7, 37, 13),
            (12, 1, 1),
        ] {
            let visit_bound = ng.pool_cut_stride(forward_passes);
            assert_eq!(visit_bound.len(), ng.n_pools);
            for &vb in &visit_bound {
                assert_eq!(
                    vb,
                    u64::from(forward_passes),
                    "chain visit_bound must equal forward_passes exactly"
                );
            }

            let fcf = FutureCostFunction::new_per_pool(
                &vec![1; ng.n_pools],
                1,
                forward_passes,
                max_iterations,
                &vec![warm_start; ng.n_pools],
                &visit_bound,
            );
            for pool in &fcf.pools {
                assert_eq!(
                    pool.capacity,
                    warm_start as usize + max_iterations as usize * forward_passes as usize,
                    "chain capacity must equal warm_start + max_iterations * forward_passes exactly"
                );
            }
        }
    }

    #[test]
    fn pool_cut_stride_declared_chain_matches_synthesized_chain() {
        let n_stages = 3;
        let forward_passes = 17u32;
        let stochastic_a = stochastic_context(n_stages, 1, 5);
        let stochastic_b = stochastic_context(n_stages, 1, 5);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

        let ng_synthesized =
            build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic_a).unwrap();

        // The same chain, declared explicitly (one node per stage, a single
        // successor at probability 1.0 each) rather than synthesized from an
        // empty `nodes[]`.
        let declared = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 2, None)],
            transitions: vec![transition(0, 1, 1.0), transition(1, 2, 1.0)],
            ..empty_graph()
        };
        let ng_declared = build_node_graph(&declared, n_stages, &resolver, &stochastic_b).unwrap();

        assert_eq!(
            ng_synthesized.pool_cut_stride(forward_passes),
            ng_declared.pool_cut_stride(forward_passes),
            "a declared chain and the synthesized one must derive identical visit bounds"
        );
    }

    // ── enumerated_pool_cut_stride: unit stride per non-leaf pool ─────────

    #[test]
    fn enumerated_pool_cut_stride_is_one_per_nonleaf_pool_zero_for_leaf() {
        // Root (stage 0, own pool) fans into K=4 leaves (stage 1) sharing ONE
        // leaf pool: the non-leaf root pool sizes at stride 1, the shared leaf
        // pool at 0 (leaves generate no cut).
        let stochastic = stochastic_context(2, 1, 1);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 1, None),
                node(4, 1, None),
            ],
            transitions: vec![
                transition(0, 1, 0.25),
                transition(0, 2, 0.25),
                transition(0, 3, 0.25),
                transition(0, 4, 0.25),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();
        assert_eq!(ng.n_pools, 2, "one root pool, one shared leaf pool");

        let stride = enumerated_pool_cut_stride(&ng);
        assert_eq!(stride.len(), ng.n_pools);
        let root_pool = ng.nodes[NodePos(0)].pool_id;
        let leaf_pool = ng.nodes[NodePos(1)].pool_id;
        assert_eq!(stride[root_pool], 1, "the non-leaf root pool sizes at 1");
        assert_eq!(
            stride[leaf_pool], 0,
            "the shared leaf pool sizes at 0 — leaves generate no cut"
        );
    }

    #[test]
    fn enumerated_pool_cut_stride_chain_is_one_per_nonleaf_zero_terminal() {
        // A 4-stage chain: stages 0..2 are non-leaf (stride 1); the terminal
        // stage-3 node is a leaf (stride 0).
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 3);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic).unwrap();

        let stride = enumerated_pool_cut_stride(&ng);
        assert_eq!(stride, vec![1, 1, 1, 0]);
    }

    // ── forward_solve_counts: exact prefix counts ─────────────────────────

    #[test]
    fn forward_solve_counts_k_fan_matches_hand_computed_prefix_counts() {
        // Root (stage 0, own pool) fans into K=4 leaves (stage 1, sharing ONE
        // pool per the leaf-sharing rule). branching_factor=1 gives every
        // node's own opening count exactly 1, so each root->leaf path
        // contributes exactly 1.
        let stochastic = stochastic_context(2, 1, 1);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 1, None),
                node(4, 1, None),
            ],
            transitions: vec![
                transition(0, 1, 0.25),
                transition(0, 2, 0.25),
                transition(0, 3, 0.25),
                transition(0, 4, 0.25),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 2, &resolver, &stochastic).unwrap();
        assert_eq!(ng.n_pools, 2, "one pool for the root, one shared leaf pool");

        let visit_bound = ng.forward_solve_counts().unwrap();
        let root_pool = ng.nodes[NodePos(0)].pool_id;
        let leaf_pool = ng.nodes[NodePos(1)].pool_id;
        assert_eq!(
            visit_bound[root_pool], 1,
            "root: a single root prefix, π(root) = 1"
        );
        assert_eq!(
            visit_bound[leaf_pool], 4,
            "shared leaf pool: 4 leaves each contribute π(leaf_i) = 1, summed"
        );

        let warm_start = 5u32;
        let max_iterations = 10u64;
        let fcf = FutureCostFunction::new_per_pool(
            &vec![1; ng.n_pools],
            1,
            1,
            max_iterations,
            &vec![warm_start; ng.n_pools],
            &visit_bound,
        );
        assert_eq!(
            fcf.pools[root_pool].capacity,
            warm_start as usize + max_iterations as usize
        );
        assert_eq!(
            fcf.pools[leaf_pool].capacity,
            warm_start as usize + max_iterations as usize * 4
        );
    }

    #[test]
    fn forward_solve_counts_duality_with_enumerated_scenario_count() {
        // K-fan (branching_factor=1): duality holds with every leaf's own
        // contribution at 1.
        let stochastic_kfan = stochastic_context(2, 1, 1);
        let graph_kfan = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            transitions: vec![transition(0, 1, 0.5), transition(0, 2, 0.5)],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng_kfan = build_node_graph(&graph_kfan, 2, &resolver, &stochastic_kfan).unwrap();
        assert_prefix_duality(&ng_kfan);

        // A non-DECOMP shape: the root carries its OWN within-node branching
        // (|Ω_root| = 3), so a child's prefix count is not simply the node
        // count -- the case that distinguishes the prefix-product formula
        // from a bare per-node count.
        let stochastic_branchy = stochastic_context(2, 1, 3);
        let graph_branchy = HorizonGraph {
            nodes: vec![node(0, 0, None), node(1, 1, None), node(2, 1, None)],
            transitions: vec![transition(0, 1, 0.5), transition(0, 2, 0.5)],
            ..empty_graph()
        };
        let ng_branchy =
            build_node_graph(&graph_branchy, 2, &resolver, &stochastic_branchy).unwrap();
        let visit_bound = ng_branchy.forward_solve_counts().unwrap();
        assert_eq!(
            visit_bound[ng_branchy.nodes[NodePos(1)].pool_id],
            6,
            "the shared leaf pool sums both children's π = |Ω_root| each, not divided between them"
        );
        assert_prefix_duality(&ng_branchy);
    }

    // ── enumerated_requires_state_exchange: world >= 2 soundness predicate ──

    #[test]
    fn enumerated_requires_state_exchange_flags_interior_branch_node() {
        // Root fans into two INTERIOR nodes at stage 1 (each with its own two
        // leaves at stage 2) — mirrors
        // `backward_cut_levels_multi_node_level_is_ascending_node_id_descending_stage`.
        // Node 1 (and node 2) sit on only one of the two root→leaf paths.
        let stochastic = stochastic_context(3, 1, 2);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
                node(4, 2, None),
                node(5, 2, None),
                node(6, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        assert_eq!(
            enumerated_requires_state_exchange(&ng),
            Some(NodeId(1)),
            "node 1 is a non-leaf node not shared by every root→leaf path — the path \
             through node 2 bypasses it"
        );
    }

    #[test]
    fn enumerated_requires_state_exchange_is_none_on_a_trunk_terminal_fan() {
        // Root → trunk → three leaves: every non-leaf node (root, trunk) lies
        // on every root→leaf path.
        let stochastic = stochastic_context(3, 1, 3);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 2, None),
                node(3, 2, None),
                node(4, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 1.0),
                transition(1, 2, 1.0 / 3.0),
                transition(1, 3, 1.0 / 3.0),
                transition(1, 4, 1.0 / 3.0),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        assert_eq!(
            enumerated_requires_state_exchange(&ng),
            None,
            "a trunk + terminal fan has no interior branching"
        );
    }

    #[test]
    fn enumerated_requires_state_exchange_is_none_on_a_chain() {
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 2);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic).unwrap();

        assert_eq!(
            enumerated_requires_state_exchange(&ng),
            None,
            "a chain has exactly one leaf; every non-leaf node's subtree spans it"
        );
    }

    // ── backward_cut_levels: reverse-topological level decomposition ────────

    #[test]
    fn backward_cut_levels_chain_is_one_node_per_level_descending_stage() {
        let n_stages = 4;
        let stochastic = stochastic_context(n_stages, 1, 3);
        let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&empty_graph(), n_stages, &resolver, &stochastic).unwrap();

        let levels = ng.backward_cut_levels();
        // The terminal leaf (stage 3) generates no cut; stages 2,1,0 each hold
        // exactly one cut-generating node, descending.
        assert_eq!(
            levels,
            vec![vec![NodePos(2)], vec![NodePos(1)], vec![NodePos(0)]]
        );
    }

    #[test]
    fn backward_cut_levels_multi_node_level_is_ascending_node_id_descending_stage() {
        // Root (stage 0) fans into two INTERIOR nodes at stage 1 (each with its
        // own two leaves at stage 2). The stage-1 level therefore carries two
        // cut-generating nodes; the leaves generate none.
        let stochastic = stochastic_context(3, 1, 2);
        let graph = HorizonGraph {
            nodes: vec![
                node(0, 0, None),
                node(1, 1, None),
                node(2, 1, None),
                node(3, 2, None),
                node(4, 2, None),
                node(5, 2, None),
                node(6, 2, None),
            ],
            transitions: vec![
                transition(0, 1, 0.5),
                transition(0, 2, 0.5),
                transition(1, 3, 0.5),
                transition(1, 4, 0.5),
                transition(2, 5, 0.5),
                transition(2, 6, 0.5),
            ],
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let ng = build_node_graph(&graph, 3, &resolver, &stochastic).unwrap();

        let levels = ng.backward_cut_levels();
        // Stage 1 (both interior nodes, ascending position) before stage 0.
        assert_eq!(levels, vec![vec![NodePos(1), NodePos(2)], vec![NodePos(0)]]);
        for level in &levels {
            assert!(
                level.windows(2).all(|w| w[0] < w[1]),
                "each level's node positions must be strictly ascending"
            );
            let stages: Vec<StageIdx> = level.iter().map(|&p| ng.nodes[p].stage).collect();
            assert!(
                stages.windows(2).all(|w| w[0] == w[1]),
                "every node in a level shares one stage"
            );
        }
    }

    /// `Σ_leaf π(leaf)·|Ω_leaf| == enumerated_scenario_count(graph)` — the
    /// prefix-count/scenario-count duality [`NodeGraph::forward_solve_counts`] cites.
    /// Assumes every leaf sharing a pool carries the same `openings.len`
    /// (true of every fixture this helper is called on).
    fn assert_prefix_duality(graph: &NodeGraph) {
        let visit_bound = graph.forward_solve_counts().unwrap();
        let expected = enumerated_scenario_count(graph).unwrap();

        let mut leaf_len_by_pool: HashMap<usize, usize> = HashMap::new();
        for (i, n) in graph.nodes.iter_indexed() {
            if graph.successors[i].is_empty() {
                leaf_len_by_pool.insert(n.pool_id, n.openings.len);
            }
        }
        let total: u64 = leaf_len_by_pool
            .iter()
            .map(|(&pool, &len)| visit_bound[pool] * len as u64)
            .sum();
        assert_eq!(
            total, expected,
            "Σ_leaf π(leaf)*|Ω_leaf| must equal enumerated_scenario_count"
        );
    }

    // ── enumerated forward traversal: per-node counts, paths, parents ───────

    /// A 3-stage K-fan (root -> fan -> leaf), `branching_factor` 1, non-uniform
    /// root out-edges `i / Σj` — the shape the enumerated forward driver walks.
    #[expect(
        clippy::cast_precision_loss,
        reason = "test counts are small, far inside f64's exact-integer range"
    )]
    fn enumerated_k_fan(k: usize) -> NodeGraph {
        let stochastic = stochastic_context(3, 1, 1);
        let mut nodes = vec![node(0, 0, None)];
        let mut transitions = Vec::new();
        let total: f64 = (1..=k).map(|i| i as f64).sum();
        for i in 1..=k {
            let fan = i as i32;
            let leaf = fan + k as i32;
            nodes.push(node(fan, 1, None));
            nodes.push(node(leaf, 2, None));
            transitions.push(transition(0, fan, (i as f64) / total));
            transitions.push(transition(fan, leaf, 1.0));
        }
        let graph = HorizonGraph {
            nodes,
            transitions,
            ..empty_graph()
        };
        let study_stage_ids = [0, 1, 2];
        let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        build_node_graph(&graph, 3, &resolver, &stochastic).unwrap()
    }

    #[test]
    fn enumerated_node_visit_counts_are_one_per_node_and_total_below_naive() {
        let k = 5usize;
        let ng = enumerated_k_fan(k);
        let counts = enumerated_node_visit_counts(&ng).unwrap();
        assert_eq!(counts.len(), ng.nodes.len());
        assert!(
            counts.iter().all(|&c| c == 1),
            "every |Ω|=1 tree node has exactly one root->node prefix: {counts:?}"
        );

        let total: u64 = ng.forward_solve_counts().unwrap().into_iter().sum();
        assert_eq!(
            total,
            ng.nodes.len() as u64,
            "Σ forward_solve_counts sums π(n)=1 over every node = the node count"
        );
        // The dedup total beats naive per-path re-solving.
        let paths = enumerated_scenario_count(&ng).unwrap();
        let path_length = 3u64;
        assert_eq!(paths, k as u64, "one path per leaf on a |Ω|=1 K-fan");
        assert!(
            total < paths * path_length,
            "Σ forward_solve_counts ({total}) must be strictly below \
             enumerated_scenario_count * path_length ({})",
            paths * path_length
        );
    }

    #[test]
    fn enumerate_forward_paths_weights_are_canonical_edge_probabilities() {
        let k = 4usize;
        let ng = enumerated_k_fan(k);
        let paths = enumerate_forward_paths(&ng);
        assert_eq!(paths.leaf.len(), k, "one path per leaf");
        assert_eq!(paths.weight.len(), k);

        // Canonical DFS order visits the root's successors ascending; each
        // successor's edge probability IS that path's P(ℓ) (q = 1, single leaf
        // edge at prob 1).
        let root = (0..ng.nodes.len())
            .map(NodePos)
            .find(|&p| ng.nodes[p].stage == StageIdx(0))
            .unwrap();
        let expected: Vec<f64> = ng.successors[root].iter().map(|s| s.probability).collect();
        for (got, want) in paths.weight.iter().zip(expected.iter()) {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "path weight must be the canonical edge probability, bit-for-bit"
            );
        }
        let sum: f64 = paths.weight.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "enumerated path probabilities must sum to 1 (got {sum})"
        );
    }

    #[test]
    fn build_parent_map_resolves_the_single_predecessor() {
        let ng = enumerated_k_fan(3);
        let parent = ng.build_parent_map();
        let root = (0..ng.nodes.len())
            .map(NodePos)
            .find(|&p| ng.nodes[p].stage == StageIdx(0))
            .unwrap();
        assert_eq!(parent[root], None, "the root has no predecessor");
        for (pos, node) in ng.nodes.iter_indexed() {
            if node.stage == StageIdx(0) {
                continue;
            }
            let p = parent[pos].expect("a non-root node has exactly one parent");
            assert_eq!(
                ng.nodes[p].stage,
                StageIdx(node.stage.0 - 1),
                "a node's parent sits exactly one stage upstream"
            );
        }
    }

    #[test]
    fn walk_path_returns_the_ascending_stage_sequence_spanning_every_stage() {
        let k = 3usize;
        let num_stages = 3usize;
        let ng = enumerated_k_fan(k);
        let plan = EnumeratedPlan::new(&ng);

        let mut out = Vec::new();
        for p in 0..plan.paths.leaf.len() {
            plan.walk_path(p, num_stages, &mut out);
            assert_eq!(out.len(), num_stages, "path must span every stage");
            let stages: Vec<StageIdx> = out.iter().map(|&pos| ng.nodes[pos].stage).collect();
            assert_eq!(
                stages,
                vec![StageIdx(0), StageIdx(1), StageIdx(2)],
                "walk_path must return nodes in ascending stage order"
            );
            assert_eq!(
                out[num_stages - 1],
                plan.paths.leaf[p],
                "the last entry must be the path's own leaf"
            );
        }
    }

    // ── Traversal ────────────────────────────────────────────────────────

    #[test]
    fn traversal_resolve_sampled_carries_the_resolved_count() {
        let ng = enumerated_k_fan(3);
        let traversal = Traversal::resolve(&ng, false, 7);
        assert!(matches!(
            traversal,
            Traversal::Sampled { forward_passes: 7 }
        ));
        assert_eq!(traversal.path_weights(), None);
    }

    #[test]
    fn traversal_resolve_enumerated_builds_the_plan_once() {
        let ng = enumerated_k_fan(3);
        let traversal = Traversal::resolve(&ng, true, 0);
        let expected = enumerate_forward_paths(&ng);
        let weights = traversal
            .path_weights()
            .expect("enumerated traversal must carry path weights");
        assert_eq!(weights, expected.weight.as_slice());
    }

    /// `SimulationWeighting::Census` is derivable ONLY from an enumerated
    /// traversal: a sampled traversal must resolve to `Uniform`, never
    /// `Census`, regardless of what weights a caller might otherwise be
    /// tempted to supply beside it.
    #[test]
    fn simulation_weighting_census_underivable_from_sampled_traversal() {
        use crate::simulation::SimulationWeighting;

        let ng = enumerated_k_fan(3);
        let sampled = Traversal::resolve(&ng, false, 3);
        assert!(matches!(
            sampled.simulation_weighting(),
            SimulationWeighting::Uniform
        ));

        let enumerated = Traversal::resolve(&ng, true, 0);
        assert!(matches!(
            enumerated.simulation_weighting(),
            SimulationWeighting::Census { .. }
        ));
    }
}
