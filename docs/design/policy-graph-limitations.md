# Policy Graph — Known Limitations and SDDP.jl Compatibility

> **Status:** Live spec — reflects shipped behavior. Verify the cited symbols against the tree before acting.

The node-native engine models a policy graph — nodes, transition-weighted edges,
and a per-node value function — in the same family as
[SDDP.jl](https://github.com/odow/SDDP.jl). This document records where the current
implementation is a deliberate **subset of**, or **differs in convention from**, a
general Markovian policy graph, so the boundaries are explicit for anyone porting a
model from SDDP.jl or extending the engine. It also reconciles the two independent
probability axes the engine carries with its separate treatment of discounting, and
records the standing engine behaviors that replaced the structural validation rules
retired when the graph became node-native.

## Compatible core

For a **finite, acyclic, leaf-terminated** graph the engine maps directly onto the
SDDP.jl policy-graph model:

- A node is a `(stage, state)` subproblem, and there may be more than one node per
  stage (the Markovian case: several discrete states at the same stage).
- Each non-leaf node owns its own value function — one future-cost pool per node,
  carrying that node's own Benders cuts.
- Edges carry conditional transition probabilities; the backward pass builds a
  node's cut by aggregating its successors' value functions weighted by those
  probabilities.
- The forward pass samples a root-to-leaf trajectory by following the transition
  probabilities.

The scenario-tree case — a single known initial state, interior branching, and
terminal leaves — is fully supported and is the primary target of the node engine.

## Two probability axes

A declared `nodes[]` policy graph carries randomness on **two independent axes
that compose but never mix**, and treats discounting as a quantity separate from
either.

- **Between-node axis — graph transition edges.** Each out-edge of a node carries
  the conditional probability of moving to that successor; these node-edge weights
  are first-class inputs of the node-native dialect. A source node's out-edge
  weights are **normalized to sum to 1** at load by
  `normalize_out_edge_probabilities`, subject to a per-source `Σ ≈ 1` agreement
  check within `PROB_TOLERANCE`.
- **Within-node axis — per-node openings.** A node also owns an opening set `Ω_n`
  (its inflow realizations). See "How a node's openings are bound (both modes)"
  below for the two binding paths, and "Within-node openings must be singleton"
  for the enumeration restriction.

The two axes compose multiplicatively: the path probability of a root→leaf
trajectory is the product of the normalized out-edge weights along its edges,
times one opening weight per node. The exact upper bound `Σ_ℓ P(ℓ)·C(ℓ)` is the
probability-weighted walk over every root→leaf path (`enumerate_forward_paths`).

## Known limitations (recorded for a future release)

### 1. Single initial node — no multi-state first stage

The engine requires exactly one node at the first stage. `find_root_position`
(`crates/cobre-sddp/src/training/lower_bound.rs`) rejects a first stage that holds
more than one node. A general Markovian graph can begin from an initial probability
distribution over several first-stage states (an uncertain initial regime); the
engine cannot represent that today.

Closing it needs a dedicated root source that carries the initial distribution
`P(root → n)` across the first-stage siblings, wired through the lower-bound,
forward, and backward passes. It is not required when the initial state is known,
which is the usual case (current storage and a known starting condition).

### 2. One shared boundary future-cost function, evaluated at each leaf's own state

All terminal (leaf) nodes share one value-function pool — see `NodeRuntime` in
`crates/cobre-sddp/src/setup/node_graph.rs`: leaves share one pool id, while a node
with successors owns its own. In ordinary training terminal nodes accumulate no
cost-to-go, so the shared pool holds nothing.

Under a boundary policy that shared pool holds one future-cost function
`F(x) = maxₖ(αₖ + βₖ·x)` — the injected boundary cuts. Each terminal leaf prices its
post-horizon value-to-go by evaluating that **same** `F` at its **own** ending state
`x_leaf` (storage plus inflow-lag state): the leaf's LP carries the terminal epigraph
row `θ ≥ αₖ + βₖ·x_leaf` for every cut in the pool, so `θ` realizes `F(x_leaf)` at the
state that leaf actually reaches. A wetter ending and a drier ending therefore already
receive different terminal values from the one shared pool — it holds a single
water-value _function_, not a single water-value _number_. For the supported
single-boundary-policy input this is the correct representation, not a limitation: one
shared static terminal template, baked once and evaluated at each leaf's own state,
with the terminal boundary future cost booked in the reported cost
(`CutPool::has_warm_start_cuts` on the terminal stage's pool; see the "Terminal boundary
FCF" contract in `.claude/rules/sddp.md`).

A distinct future-cost _function_ per leaf — different `α/β`, say one water-value
function for a wet-basin ending and another for a dry-basin ending — is a different
model. It requires more than one boundary policy on input, one per terminal regime,
which is a reserved future feature (per-leaf boundary input), not a fix to the
single-policy path.

The shared stage template across leaves is correct while a state affects only the
inflow realization, not the LP structure — which holds for the hydrothermal model.

## Deliberate differences from SDDP.jl (by design, not limitations)

### Discount is a per-stage rate, not edge mass

Out-edge probabilities are normalized to sum to 1 at load time
(`normalize_out_edge_probabilities`), and discounting is carried as a **separate
per-stage factor**, never folded into the edge weights. The discount override is a
**per-stage** quantity, declared on `stages[].annual_discount_rate_override` and
compounded into the per-stage `cumulative_discount_factors` the stage templates
consume. Probability mass and the discount factor are validated and applied
independently.

A **per-edge** — hence potentially per-node — rate is **rejected under `nodes[]`**
by `check_edge_discount_override_under_nodes` (rule 42). The cumulative discount
factor is baked into each stage's template objective coefficients, so a per-edge
rate would force per-node templates. The edge spelling stays legal in the chain
dialect, where every edge from a stage shares one factor.

### SDDP.jl mapping: explicit rate vs. discount-in-edge-mass

SDDP.jl and cobre make the same modeling distinction differently, and the
difference is exactly where discounting lives.

- **SDDP.jl folds discounting into edge mass.** A node's out-arc weights are
  **not required to sum to 1**; the deficit `1 − Σ_j w_{ij}` is the probability of
  _terminating_ (leaving the graph). Discounting is expressed through that same
  deficit: in a cyclic (infinite-horizon) graph, the **probability on the arc that
  completes a cycle equals the discount factor** — the cycle-closing arc carries
  weight `ρ`, so continuation costs enter the Bellman update multiplied by `ρ` per
  cycle. Probability and discount share one channel.
- **cobre keeps the two channels separate.** Out-edge weights are normalized to
  `Σ ≈ 1` per source (`normalize_out_edge_probabilities`), discounting is carried
  as an **explicit per-stage rate** (`cumulative_discount_factors`), and a
  trajectory _terminates at a leaf_ — a node with no out-edges at the final stage,
  with a mid-horizon leaf rejected by rule 38 — never through a probability
  deficit.

The two coincide for a finite horizon. An SDDP.jl graph that encodes a discount as
an out-arc probability deficit must therefore be **translated** — normalize the
edges to sum to 1 and set the per-stage discount rate — not transcribed
edge-for-edge.

### Finite horizon only

The engine supports finite, acyclic, leaf-terminated graphs. It has no cyclic,
infinite-horizon (discounted-cycle) construction.

### Autocorrelation model

Inter-stage inflow dependence is modeled with a continuous periodic-autoregressive
lag state (its depth fitted from the PAR model and widened to a loaded boundary
policy), not by discretizing the process into a Markov lattice as SDDP.jl's
"Markov chain approach" does. The
node graph is for **explicit discrete branching** (scenario trees, regime states);
the two compose — a branching graph whose nodes each carry the continuous lag state.

## Enumerated forward selection requires a structural tree of single-realization nodes

`training.selection = enumerated` computes the exact upper bound `Σ_ℓ P(ℓ)·C(ℓ)` by
walking every root→leaf path. It enumerates only the **structural** axis of the graph
— the node/edge branching — solving each node once per distinct incoming prefix, and
multiplying a single opening weight per node into the path probability
(`enumerate_forward_paths`, `crates/cobre-sddp/src/setup/node_graph.rs`). It does
**not** enumerate the other two ways a graph can carry randomness, and rejects them at
setup rather than compute a biased bound.

### Within-node openings must be singleton (`|Ω_n| = 1`)

A node also owns an opening set `Ω_n` (its inflow realizations). Because the engine
solves each node once and its probability walk multiplies one opening weight per node,
a node with `|Ω_n| > 1` would cover only a subset of scenarios, with weights that no
longer sum to one and a single-sample cost — an invalid bound. `setup` rejects
`|Ω_n| > 1` under enumerated training with a named error
(`reject_within_node_opening_enumeration`, `crates/cobre-sddp/src/setup/mod.rs`).
Express the stochasticity structurally (one realization per node) or use sampled
selection.

### The graph must be a tree (single predecessor per node)

A node reached from two different parents — a recombination join, i.e. the
dense-transition-matrix Markov case — has a different incoming continuous state per
incoming prefix. The engine reconstructs that state from a single-predecessor map
(`build_parent_map`), so a recombining node would be solved once under one arbitrarily
chosen parent's state while paths arriving via the other parent silently read the
wrong state — an invalid bound in a release build. `setup` rejects a node with
in-degree ≥ 2 under enumerated training with a named error
(`reject_recombining_node_enumeration`). Full recombination support — per-prefix state
reconstruction for a multi-parent node — is a reserved seam, and is what a genuine
Markovian graph with a dense transition matrix would need under enumerated selection.

### Sampled selection has neither restriction

`training.selection = sampled` Monte-Carlo draws one opening per visited node and
carries per-trajectory state, so it handles both `|Ω_n| > 1` and recombining graphs
natively. It is the general mode; enumerated is the exact-bound specialization for
small structural trees (scenario trees, the extensive-form oracle fixtures). A
multi-opening chain has `Π|Ω|` enumerated scenarios and is intractable to enumerate
regardless.

### How a node's openings are bound (both modes)

A declared node's opening set is either a single external-library column
(node `scenario_id` present → `|Ω_n| = 1`) or the **stage's** generated opening set
(`scenario_id` absent → `|Ω_n| = n_openings(stage)`), the latter shared by every
generated node at that stage. So a node with its own distinct multi-opening
distribution — different from its siblings at the same stage — is not yet expressible:
siblings either share the stage's generated set or are pinned to single external
columns (`build_declared_node_graph`, `crates/cobre-sddp/src/setup/node_graph.rs`).
That per-node opening-weight axis is a **deferred future feature** — it would return
with a per-node Markov configuration — not an input a study supplies today.

## Standing behaviors that replaced retired validation rules

The move to a node-native graph removed a family of collapse-era structural rules.
Their intent survives as standing engine behavior, not as re-added validation:

- **A node's realization is defined by its own pointer.** A node's `scenario_id`
  fixes its realization directly; there is no descendant-agreement or trunk
  constancy to enforce across a shared trunk. The only surviving pointer rule is
  the per-class bound (rule 37).
- **Output identity re-keys to the leaf node id.** Per-node outputs and the
  simulation summary key on the node id, so injectivity across realizations is
  automatic rather than validated.
- **Probability totals come from the normalized transitions.** The census
  enumerates the graph's leaves and `scenario_summary` keys by leaf node id, so
  library exhaustion is no longer a soundness condition — the totals are the
  normalized out-edge weights. Coverage is not a validation rule.
- **Genuine branching is admissible.** A node with multiple successors — including
  sibling successors that share a realization pointer — loads without error. The
  admissibility predicate and the duplicate-pointer / degenerate-branching
  rejections are gone; a structurally recombinable signature draws a warning
  (rule 40), not a rejection. `check_node_graph`'s surviving structural rules
  (well-formedness, every edge `t → t+1`, the pointer bound) are the whole gate.

Retired rule ids in the cobre-io validation rule table
(`crates/cobre-io/src/validation/rules.rs`) are **never reused** — a retired
id is listed as never assignable, so a rule id always denotes the same
behavior across the project's life.

## References

- SDDP.jl policy graphs — <https://sddp.dev/stable/tutorial/first_steps/>
- SDDP.jl general policy graphs (arc weights, termination, discounting) —
  <https://sddp.dev/stable/guides/create_a_general_policy_graph/>
- SDDP.jl Markov chain approach — <https://sddp.dev/stable/tutorial/arma/>
- SDDP.jl source — <https://github.com/odow/SDDP.jl>
</content>

</invoke>
