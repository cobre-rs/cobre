# Deferred Features

## Purpose

This spec documents features that are planned but not yet implemented in the Cobre SDDP solver. Each feature includes a description, rationale for deferral, prerequisites, and estimated effort. The data model is designed to accommodate these extensions without breaking changes.

## C.1 GNL Thermal Plants

**Status**: DEFERRED

**Description**: Gas Natural Liquefeito (GNL) thermal plants have complex operational constraints including:

- Minimum take-or-pay contracts
- Variable fuel costs based on LNG spot market
- Start-up and shutdown constraints
- Fuel inventory management

**Planned Formulation**:

- Binary variables for unit commitment (requires MIP solver integration)
- Fuel inventory balance constraints
- Contract fulfillment constraints
- Piecewise-linear fuel cost functions

**Why Deferred**: Requires MIP solver integration (SDDiP or Lagrangian relaxation) which is a significant architectural change. Unit commitment is not in the immediate roadmap for medium/long-term planning.

**Prerequisites**:

- Duality handler infrastructure (Lagrangian relaxation for MIP subproblems)
- MIP solver integration (Gurobi, CPLEX, or Cbc)
- GNL-specific data model extensions

**Estimated Effort**: Large (3-4 weeks). Requires SDDiP infrastructure before GNL-specific modeling.

**Reference**: CEPEL NEWAVE/DECOMP GNL modeling documentation.

## C.2 Battery Energy Storage Systems

**Status**: DEFERRED

**Description**: Grid-scale batteries with:

- State-of-charge management
- Charge/discharge efficiency losses
- Degradation modeling (cycle counting)
- Capacity fade over time

**Planned Formulation**:

**Variables**:

| Variable        | Domain           | Units | Description       |
| --------------- | ---------------- | ----- | ----------------- |
| $e_{b,k}$       | $[0, \bar{E}_b]$ | MWh   | State of charge   |
| $p^{ch}_{b,k}$  | $\geq 0$         | MW    | Charging power    |
| $p^{dis}_{b,k}$ | $\geq 0$         | MW    | Discharging power |

**Energy Balance**:

$$
e_{b,k} = e_{b,k-1} + \eta^{ch} \cdot p^{ch}_{b,k} \cdot \Delta t - \frac{p^{dis}_{b,k} \cdot \Delta t}{\eta^{dis}}
$$

**Load Balance Contribution**:

$$
\sum_{b \in \mathcal{B}} (p^{dis}_{b,k} - p^{ch}_{b,k}) \text{ added to generation}
$$

**Data Model**: The `system/batteries.json` schema is fully specified with fields for capacity (energy/charge/discharge), efficiency, initial SOC, and SOC limits. LP variables include `battery_soc` (state), `battery_charge`, and `battery_discharge` (controls). Output schema in `simulation/batteries/` is also defined. See [Input System Entities](data-model/input-system-entities.md) for the entity registry pattern and [Internal Structures](data-model/internal-structures.md) for how entities are resolved at runtime.

**Why Deferred**: Batteries are linear storage devices (no integer variables needed), but require:

- New state variable dimension (SOC per battery)
- Integration with bus load balance
- Testing with realistic battery degradation scenarios

**Prerequisites**:

- State variable infrastructure supports dynamic dimension
- Bus balance constraint generation handles battery charge/discharge
- Output schema for battery simulation results

**Estimated Effort**: Medium (2-3 weeks). LP formulation is straightforward; main effort is data pipeline and testing.

## C.3 Multi-Cut Formulation

**Status**: DEFERRED

**Description**: Alternative to single-cut aggregation that creates one cut per scenario.

**Formulation**:

Instead of a single aggregated future cost variable $\theta$, introduce per-scenario variables $\theta_\omega$:

$$
\theta = \sum_{\omega \in \Omega_t} p(\omega) \cdot \theta_\omega
$$

With per-scenario cuts:

$$
\theta_\omega \geq \alpha_k(\omega) + \pi_k(\omega)^\top x \quad \forall k, \omega
$$

**Trade-offs**:

| Aspect                  | Single-Cut                         | Multi-Cut                                      |
| ----------------------- | ---------------------------------- | ---------------------------------------------- |
| **Cuts per iteration**  | 1                                  | $\lvert\Omega\rvert$ (one per scenario)        |
| **LP size**             | Smaller (1 future cost variable)   | Larger ($\lvert\Omega\rvert$ future cost vars) |
| **Convergence rate**    | Slower (more iterations)           | Faster (fewer iterations)                      |
| **Time per iteration**  | Faster                             | Slower                                         |
| **Memory**              | Lower                              | Higher                                         |
| **Numerical stability** | More stable                        | Can have issues with risk measures             |
| **Best for**            | Large scenario counts, risk-averse | Small scenario counts, risk-neutral            |

**Why Deferred**: Multi-cut requires significant changes:

- LP construction must handle multiple future cost variables
- Cut storage and selection becomes more complex
- Interaction with CVaR risk measures needs careful implementation
- Performance tuning (when to use which formulation) is problem-dependent

**Prerequisites**:

- LP builder supports variable number of future cost variables
- Cut pool indexed by scenario
- Cut selection adapted for multi-cut pools

**Estimated Effort**: Medium (2-3 weeks). Core algorithm change with wide-reaching effects on cut management.

**Reference**: Birge, J.R. (1985). "Decomposition and partitioning methods for multistage stochastic linear programs." _Operations Research_, 33(5), 989-1007.

## C.4 Markovian Policy Graphs

**Status**: DEFERRED

**Description**: Extension to handle scenario-dependent transitions (e.g., different inflow regimes).

**Current Limitation**: Cobre assumes stage-wise independent scenarios. The same scenario tree structure is used regardless of which scenario was realized in the previous stage.

**Planned Extension**:

- Markov chain over "regimes" (e.g., wet/dry/normal)
- Transition probabilities between regimes
- Regime-dependent inflow distributions
- Cut sharing across nodes in same regime

**Formulation**:

Let $M$ be a Markov chain with states $\mathcal{M} = \{1, \ldots, m\}$ and transition matrix $P$.

The policy graph becomes:

- Nodes: $(t, r)$ for stage $t$ and regime $r$
- Edges: $(t, r) \to (t+1, r')$ with probability $P_{r,r'}$

Value function approximation:

$$
V_{t,r}(x) \approx \max_{k \in \mathcal{K}_{t,r}} \{\alpha_k + \pi_k^\top x\}
$$

Cuts are regime-specific and only shared within the same regime.

**Data Model**: The `stages.json` schema supports optional `markov_states` fields. Transitions include `source_markov` and `target_markov` fields. Cut files are indexed by `(stage_id, markov_state)` as `stage_XXX_markov_YYY.bin`. See [Input Scenarios](data-model/input-scenarios.md) for the stages.json schema and [Binary Formats](data-model/binary-formats.md) for cut file persistence.

**Why Deferred**: Markovian policy graphs substantially increase algorithm complexity:

- Forward passes must track Markov state transitions
- Backward passes generate cuts for each `(stage, markov_state)` node
- State space grows by factor of `|markov_states|`
- Requires careful handling of stagewise-independent noise within each Markov state

**Prerequisites**:

- Policy graph supports multi-node-per-stage structure
- Cut pool indexed by `(stage, markov_state)` tuples
- Forward/backward pass logic handles Markov transitions
- Scenario generation respects regime-dependent distributions

**Estimated Effort**: Large (3-4 weeks). Fundamental change to policy graph and forward/backward pass logic.

**Reference**: Philpott, A.B., & de Matos, V.L. (2012). "Dynamic sampling algorithms for multi-stage stochastic programs with risk aversion." _European Journal of Operational Research_, 218(2), 470-483.

## C.5 Non-Controllable Sources (Wind/Solar)

**Status**: IMPLEMENTED

> **Note**: Non-controllable sources are no longer deferred. The LP formulation (generation bounds, curtailment variables, bus balance participation) is specified in [Equipment Formulations §6](./math/equipment-formulations.md). The data model (entity registry, scenario pipeline, penalty overrides) is specified in [Input System Entities §7](./data-model/input-system-entities.md). The penalty system includes curtailment cost as a Category 3 regularization penalty in [Penalty System §2](./data-model/penalty-system.md). The formulation and data model descriptions below are retained for reference.

**Variables**:

| Variable       | Domain                                  | Units | Description                 |
| -------------- | --------------------------------------- | ----- | --------------------------- |
| $g^{nc}_{r,k}$ | $[0, \bar{G}_r \cdot \alpha_r(\omega)]$ | MW    | Non-controllable generation |
| $\kappa_{r,k}$ | $\geq 0$                                | MW    | Curtailment                 |

**Generation Constraint**:

$$
g^{nc}_{r,k} + \kappa_{r,k} = \bar{G}_r \cdot \alpha_r(\omega)
$$

where $\alpha_r(\omega) \in [0, 1]$ is the stochastic availability factor.

**Curtailment Penalty**:

$$
+ \sum_{r \in \mathcal{R}} c^{curt} \cdot \kappa_{r,k} \cdot \Delta t_k
$$

**Data Model**: The `system/non_controllable_sources.json` schema defines source type, bus assignment, capacity, and curtailment settings. Generation models in `scenarios/non_controllable_models.parquet` provide mean and standard deviation per source per stage. Correlation with inflows is supported via `correlation.json` blocks. Output schema in `simulation/non_controllables/` is defined.

## C.6 FPHA Enhancements

**Status**: DEFERRED (Partial -- core FPHA implemented)

**Description**: Advanced extensions to the FPHA (Four-Point Head Approximation) model for improved accuracy in hydroelectric production function modeling.

### C.6.1 Variable Efficiency Curves

**Current**: Constant turbine-generator efficiency $\eta_{ref}$.

**Enhancement**: Flow-dependent efficiency using characteristic curves:

$$
\eta(q) = \eta_{max} \times f\left(\frac{q}{q_{nom}}\right)
$$

where $f$ is a hill chart approximation, typically:

$$
f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 \quad \text{for } x \in [x_{min}, 1]
$$

**Data Requirements**:

- Efficiency curve coefficients in `hydro_production_data.parquet`
- `efficiency_type = "flow_dependent"`
- `efficiency_coeffs = [a_0, a_1, a_2, a_3]`

**Impact on FPHA**: Hyperplane fitting must use $\eta(q_j)$ at each grid point. Increases nonlinearity captured by the approximation. More planes may be needed for same accuracy.

### C.6.2 Pumped Hydro Production Function

**Current**: Pumping modeled separately from generation.

**Enhancement**: Unified production function for reversible hydro plants:

**Generation Mode** (standard FPHA):

$$
g_{h,k}^{gen} \leq \phi(v, q^{gen}, s)
$$

**Pumping Mode** (reversed inequality -- pumping power increases with head):

$$
p_{h,k}^{pump} \geq \phi_{pump}(v, q^{pump}, s)
$$

**Pumping Production Function**:

$$
p_{pump} = \frac{\rho \times q^{pump} \times h_{net,pump}}{\eta_{pump}}
$$

where $h_{net,pump}$ = pumping head (downstream to upstream) and $\eta_{pump}$ = pumping efficiency (typically 0.85-0.90).

**Hyperplane Form**:

$$
p_{h,k}^{pump} \geq \gamma_{0,pump}^m + \gamma_{v,pump}^m \cdot v_h^{avg} + \gamma_{q,pump}^m \cdot q_{h,k}^{pump}
$$

**Operational Constraints**:

- Mutual exclusion: $q_{h,k}^{gen} \times q_{h,k}^{pump} = 0$ (nonlinear)
- Alternative: Big-M or SOS1 constraints (introduces integer variables)
- Cobre approach: Allow simultaneous gen/pump with high penalty (relaxation)

### C.6.3 Dynamic FPHA Recomputation

**Current**: FPHA hyperplanes fixed per stage configuration.

**Enhancement**: Recompute hyperplanes based on expected operating region.

**Approach**:

1. Track volume distribution per stage across forward passes
2. Adjust $[v_{min}, v_{max}]$ to cover observed operation
3. Generate new planes for updated windows
4. Manage cut validity after FPHA updates

**Configuration** (future):

```json
{
  "fpha_config": {
    "dynamic_recomputation": {
      "enabled": true,
      "recompute_every_n_iterations": 50,
      "window_adaptation": "narrow_only",
      "percentile_margin": 5
    }
  }
}
```

> **Warning**: Dynamic recomputation changes the LP structure across iterations. This may affect SDDP convergence guarantees.

**Why Deferred**: Core FPHA is implemented; these are accuracy improvements requiring:

- Variable efficiency: Hill chart data and fitting infrastructure
- Pumped hydro: Unified gen/pump production function with mutual exclusion
- Dynamic recomputation: Online hyperplane refitting with cut validity management

**Prerequisites**:

- Core FPHA operational and validated
- Hyperplane fitting infrastructure supports refitting
- Performance baseline established to measure improvements

**Estimated Effort**: Medium-Large (2-4 weeks total across all three sub-features).

## C.7 Temporal Scope Decoupling

**Status**: DEFERRED

**Description**: Advanced temporal decomposition inspired by SPARHTACUS that decouples the physical time resolution (decision dynamics) from the SDDP stage decomposition (Benders cut generation points). Enables flexible multi-resolution modeling with controlled cut growth.

**Motivation**: In conventional SDDP, three temporal scopes are tightly coupled: stage (SDDP decomposition unit), decision period (physical time resolution), and stochastic process (uncertainty realization base). This forces a trade-off between fine temporal resolution (accurate physics, exponential cut growth) and coarse resolution (manageable cuts, poor short-term dynamics).

**Three Independent Temporal Scopes** (SPARHTACUS nomenclature):

| Scope                         | Portuguese Term                 | Cobre Current                             | Purpose                                                |
| ----------------------------- | ------------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| **Optimization Period**       | Periodo de otimizacao           | `stages[t]`                               | SDDP decomposition unit, Benders cut generation        |
| **Study Period**              | Periodo de estudo               | _(coupled to stage)_                      | Physical time resolution for constraints and decisions |
| **Stochastic Process Period** | Periodo do processo estocastico | `inflow_seasonal_stats.parquet` per stage | Base for uncertainty realization                       |

**Extended Multi-Period Stage Subproblem**:

Let stage $t$ contain $K_t$ decision periods indexed by $k = 1, \ldots, K_t$.

$$
V_t(x_t, \{\omega_{t,k}\}_{k=1}^{K_t}) = \min_{\{y_{t,k}\}_{k=1}^{K_t}} \left\{ \sum_{k=1}^{K_t} c_{t,k}^\top y_{t,k} + \mathbb{E} [V_{t+1}(x_{t+1}, \cdot)] \right\}
$$

Subject to:

- **Period 1 constraints**: $A_{t,1} y_{t,1} = b_{t,1}(\omega_{t,1}) - B_t x_t$
- **Period k constraints**: $A_{t,k} y_{t,k} + D_{t,k} y_{t,k-1} = b_{t,k}(\omega_{t,k})$ for $k = 2, \ldots, K_t$
- **State transition**: $x_{t+1} = E_t y_{t,K_t} + F_t x_t$

**Key property**: State variable dimension is unchanged -- cuts reference only $x_t$ at stage boundaries, not intermediate periods. LP size increases by factor $K_t$ but cut count remains constant.

**LP Size Impact**:

| Configuration                    | Stages | Avg LP Size | Cut Pool             |
| -------------------------------- | ------ | ----------- | -------------------- |
| Standard monthly                 | 60     | 1500 vars   | ~1200 cuts @ 20 iter |
| Hybrid (4 weekly + rest monthly) | 60     | ~1600 vars  | ~1200 cuts           |
| All weekly                       | 260    | 1500 vars   | ~5200 cuts @ 20 iter |

**Why Deferred**: Requires significant architectural changes:

- Multi-period LP construction within single stage
- Modified forward/backward pass logic
- Data model changes (periods array within stages)
- Interaction with chronological blocks (nested: Stage → Periods → Blocks)

**Prerequisites**:

- Core SDDP training loop validated and stable
- Chronological block formulation operational
- Performance profiling shows cut growth is the bottleneck

**Estimated Effort**: Large (4-6 weeks). Fundamental change to LP construction, data model, and algorithm.

**References**:

- SPARHTACUS/SPTcpp: [Escopo Temporal](https://github.com/SPARHTACUS/SPTcpp/wiki/Escopo-Temporal)
- Pereira, M.V.F., & Pinto, L.M.V.G. (1991): Original SDDP paper with monthly stages

## C.8 CEPEL PAR(p)-A Variant

**Status**: DEFERRED (standard PAR(p) now available in v0.1.1)

> **Note**: Standard PAR(p) fitting from historical inflow data is now implemented as of v0.1.1 (Levinson-Durbin recursion, AIC-based order selection, spectral spatial correlation). The CEPEL-specific PAR(p)-A extensions listed below remain deferred.

**Description**: CEPEL's PAR(p)-A model (referenced in Rel-1941_2021) extends the standard PAR(p) with:

- **Order constraint**: Maximum AR order often fixed at 12 (annual cycle)
- **Stationarity enforcement**: Coefficients adjusted to ensure $\sum_\ell \psi_{m,\ell} < 1$
- **Lognormal transformation**: Working with $\ln(a_{h,t})$ for strictly positive inflows
- **Regional correlation**: Cross-correlation between hydros in the same river basin

**Why Deferred**: The standard PAR(p) model covers most practical use cases. The lognormal variant is primarily relevant for basins with highly skewed inflow distributions where negative synthetic inflows become problematic. The inflow non-negativity handling strategies (see [Inflow Non-Negativity](./math/inflow-nonnegativity.md)) provide adequate mitigation for the standard model.

**Prerequisites**:

- Standard PAR(p) fitting and validation operational (met as of v0.1.1)
- Lognormal transformation infrastructure (log-space fitting, back-transformation)
- Validation suite comparing PAR(p) vs. PAR(p)-A on representative basins

**Data Model Compatibility**: The current input format (`inflow_seasonal_stats.parquet` + `inflow_ar_coefficients.parquet`) supports PAR(p)-A — the lognormal transformation is applied to history before computing seasonal stats, and the resulting μ, s, ψ values are stored in the same schema.

**Estimated Effort**: Small-Medium (1-2 weeks). Mathematical extensions are straightforward; main effort is validation and testing.

**Reference**: CEPEL Rel-1941_2021.

## C.9 Policy Compatibility Validation

**Status**: DEFERRED (requires dedicated spec)

**Description**: When the solver runs in simulation-only, warm-start, or checkpoint resume mode, it must validate that the current input data is compatible with the previously trained policy (cuts). Incompatible inputs produce silently incorrect results — cut coefficients encode LP structural information that becomes invalid if the system changes.

**Properties to validate** (non-exhaustive):

| Property                             | Why It Matters                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| Block mode per stage                 | Cut duals come from different water balance structures (parallel vs. chronological) |
| Block count and durations per stage  | LP column/row dimensions change                                                     |
| Number of hydro plants               | State variable dimension changes                                                    |
| AR orders per hydro                  | State variable dimension changes                                                    |
| Cascade topology                     | Water balance constraint structure changes                                          |
| Number of buses, lines               | Load balance constraint structure changes                                           |
| Number of thermals, contracts, NCS   | LP column count changes                                                             |
| Production model per hydro per stage | FPHA vs. constant affects generation constraint and dual structure                  |
| Penalty configuration                | Affects objective coefficients and thus cut intercepts                              |

**Design considerations**:

- The training phase must persist a **policy metadata record** alongside the cut data, capturing all input properties that affect LP structure
- The validation phase compares the current input against this metadata and produces a **hard error** on any mismatch
- The metadata format should be versioned for forward compatibility
- Some properties may allow controlled relaxation (e.g., adding a new thermal plant at the end might be compatible if the policy's state variables are unchanged) — these exceptions require careful analysis

**Why Deferred**: This is a complex cross-cutting concern that touches training, simulation, checkpointing, and input loading. It requires a dedicated specification that defines:

1. The complete list of validated properties
2. The metadata format persisted by training
3. The validation algorithm (exact match vs. compatible subset)
4. Error reporting format
5. Interaction with checkpoint versioning

**Prerequisites** (all now met — approved specs):

- Training loop architecture finalized — [Training Loop](./architecture/training-loop.md) approved
- Checkpoint format defined — [Checkpointing](./hpc/checkpointing.md) approved, [Binary Formats §3-§4](./data-model/binary-formats.md) approved
- Simulation architecture finalized — [Simulation Architecture](./architecture/simulation-architecture.md) approved
- Input loading pipeline defined — [Input Loading Pipeline](./architecture/input-loading-pipeline.md) approved

C.9 is unblocked and can proceed to its own dedicated specification.

**Estimated Effort**: Medium (2-3 weeks). Primarily specification and testing; implementation is straightforward once the property list is defined.

**Cross-references**:

- [Block Formulations §4](./math/block-formulations.md) — block mode validation requirement (first identified property)
- [Training Loop](./architecture/training-loop.md) — must persist policy metadata
- [Simulation Architecture](./architecture/simulation-architecture.md) — must validate on entry
- [Checkpointing](./hpc/checkpointing.md) — checkpoint format must include metadata
- [Binary Formats](./data-model/binary-formats.md) — policy file format that carries metadata

## C.10 Fine-Grained Temporal Resolution (Typical Days)

**Status**: DEFERRED (requires research)

**Description**: Decompose each stage into D representative typical day types (e.g., weekday, weekend, peak-season day), each containing N chronological blocks (e.g., 24 hourly periods). This enables accurate modeling of daily cycling patterns within monthly or weekly stages.

**Motivation**: The current block formulation uses a single level of temporal decomposition — a few load blocks (e.g., peak/shoulder/off-peak) with duration τ_k representing the total monthly hours for that load level. This is adequate for long-term planning but cannot capture:

- Daily storage cycling (pumped hydro, batteries)
- Hourly renewable generation profiles (solar peak, wind patterns)
- Time-of-use pricing dynamics
- Intra-day ramp constraints

**Key design questions** (require research):

1. **Day-type chaining**: Should typical days chain sequentially within a stage (day 1 end-storage → day 2 initial storage), or operate independently with weighted-average end-of-stage storage?
2. **Objective weighting**: Block durations within a day type are the actual hourly duration (e.g., 1 hour), but the objective contribution must be scaled by the day-type weight (number of days it represents). This requires separating the current τ_k into two components: block duration (for water balance) and effective duration (for objective).
3. **Two-phase architecture**: Train with aggregated blocks, simulate with full typical-day resolution. Cut compatibility conditions need formal verification.
4. **Input data requirements**: Day-type definitions, weights, hourly demand profiles per day type, hourly renewable availability profiles per day type.

**LP size impact** (estimated for 100 hydros):

| Configuration           | Blocks/stage | Water balance rows | Additional LP vars |
| ----------------------- | ------------ | ------------------ | ------------------ |
| Current (3 load blocks) | 3            | 100-300            | 0-200              |
| 3 day types × 24 hours  | 72           | 7,200              | 7,100              |
| 5 day types × 24 hours  | 120          | 12,000             | 11,900             |

**References**:

- PSR SDDP v17.3 release notes — "Typical Days" representation
- NZ Battery Project (Jacobs/PSR, 2023) — 21 chronological blocks per weekly stage
- Garcia et al. (2020), "Genesys" — 3 chronological blocks per day in weekly stages
- SPARHTACUS temporal decoupling (see §C.7 above)

**Prerequisites**:

- Core parallel and chronological block modes operational and validated
- Simulation architecture supports variable block configurations
- Research on day-type chaining and two-phase cut compatibility completed

**Estimated Effort**: Large (4-6 weeks). Significant input schema design, LP construction changes, and research required.

---

## C.11 User-Supplied Noise Openings

**Status**: Implemented (v0.1.3)

**Implemented in**: [ADR-008](../../cobre/docs/adr/008-user-supplied-opening-tree.md) (cobre repository). The design questions identified when this feature was deferred are all resolved in ADR-008.

**Description**: Allow users to directly input pre-sampled, pre-correlated noise values for the opening tree, bypassing the internal noise generation and correlation pipeline entirely.

**Motivation**: The standard pipeline generates noise internally (sample independent $N(0,1)$, apply spectral correlation). Some use cases require full user control over the noise:

- Importing noise realizations from external stochastic models that use non-Gaussian distributions
- Reproducing exact noise sequences from legacy tools for validation
- Using domain-specific spatial correlation structures not captured by the spectral decomposition approach
- Research workflows where specific noise patterns are under study

**Input File**: `scenarios/noise_openings.parquet` (optional)

See ADR-008 for the authoritative Parquet schema (four columns: `stage_id`, `opening_index`, `entity_index`, `value`), validation rules, bypass logic, and interaction with forward-pass noise. The schema used by the export path (ADR-009) is identical to the import schema, enabling a round-trip invariant.

**Cross-references**:

- [ADR-008: User-Supplied Opening Tree](../../cobre/docs/adr/008-user-supplied-opening-tree.md) — Authoritative design record (cobre repository)
- [ADR-009: Stochastic Artifact Export](../../cobre/docs/adr/009-stochastic-artifact-export.md) — Export counterpart and round-trip invariant (cobre repository)
- [Scenario Generation §2.3](./architecture/scenario-generation.md) — Opening tree architecture

## C.12 Complete Tree Solver Integration

**Status**: DEFERRED (concept documented, solver integration pending)

**Description**: Full solver integration for complete tree mode — enumerating all nodes in an explicit scenario tree and solving a subproblem per node using Benders decomposition, as in CEPEL's DECOMP model.

The concept and tree structure are documented in [Scenario Generation §7](./architecture/scenario-generation.md). This deferred item covers the solver-side integration:

**Required Components**:

1. **Tree enumerator** — Given per-stage branching counts $N_t$, enumerate all tree nodes with parent-child relationships and branching probabilities
2. **Node-to-subproblem mapping** — Each tree node maps to a stage subproblem with specific RHS values (from the node's branching realization) and a specific state variable value (from the parent node's solution)
3. **Tree-aware backward pass** — Instead of SDDP's sample-based backward pass, solve backward through the tree: at each node, aggregate cuts from its children (weighted by branching probabilities)
4. **Result aggregation** — Collect optimal values, state trajectories, and duals from all tree nodes; compute expected cost and policy metrics
5. **Exhaustive forward pass** — Replace sampling with deterministic tree traversal, visiting every path from root to leaf

**DECOMP Special Case**: When $N_t = 1$ for all $t < T$, the tree degenerates into a deterministic trunk with branching only at stage $T$. The forward pass solves a single path up to $T$, then branches. This is the standard DECOMP configuration for short-term planning.

**LP Construction**: Each node's LP is identical in structure to the SDDP stage subproblem — the complete tree mode reuses the existing LP builder. The difference is purely in the traversal logic (enumeration vs. sampling) and cut aggregation (tree-weighted vs. sample-averaged).

**Why Deferred**: Solver integration requires:

- Tree traversal infrastructure (BFS/DFS node enumeration with MPI distribution)
- Modified training loop that replaces SDDP's forward/backward sampling with tree enumeration
- Result storage for potentially millions of tree nodes
- Convergence criterion different from SDDP (the tree solution is exact — convergence means Benders gap closure on the tree, not statistical bound convergence)

**Prerequisites**:

- Core SDDP training loop operational
- Opening tree infrastructure (§2.3) supports variable per-stage branching counts
- External scenario integration validated
- Performance profiling establishes tractability bounds for target problem sizes

**Estimated Effort**: Medium-Large (3-4 weeks). LP construction is reused; main effort is tree traversal, MPI distribution of tree nodes, and result aggregation.

**References**:

- CEPEL DECOMP: [Modelo DECOMP](https://see.cepel.br/manual/libs/latest/modelos_computacionais/modelo_decomp.html) — Short-term hydrothermal dispatch with scenario trees
- [Scenario Generation §7](./architecture/scenario-generation.md) — Complete tree concept and structure

---

## C.13 Alternative Forward Pass Model

**Status**: DEFERRED

**Description**: Solve a different LP model in the forward pass — one that includes simulation-only features (linearized head, unit commitment, bottom discharge) — to generate trial points that better reflect real-world operations, while keeping the training LP (convex, no simulation-only features) for backward pass cut generation.

**Motivation**: The default forward pass solves the training LP, which excludes simulation-only features (see [Simulation Architecture](./architecture/simulation-architecture.md)). Trial points from this simplified model may not visit states that are realistic under full operational modeling. An alternative forward pass addresses this gap by solving a richer model to generate more representative trial points.

**Design** (inspired by SDDP.jl's `AlternativeForwardPass`):

1. Maintain two LP models per stage: the **training LP** (convex) and the **alternative LP** (includes simulation-only features)
2. Forward pass solves the alternative LP at each stage, producing trial states $\hat{x}_t$
3. Backward pass solves the training LP at each stage using these trial states, generating valid Benders cuts
4. After each backward pass, new cuts are added to **both** LP models (the alternative LP needs cuts too, for its $\theta$ variable)

**Key properties**:

- Cuts remain valid because they are generated from the convex training LP
- Trial points are more realistic because they come from the richer alternative LP
- Convergence may be slower (trial points from the alternative model may not be optimal for the training model)
- Memory cost approximately doubles (two LP models per stage)

**Why Deferred**: Requires maintaining two parallel LP models per stage, which doubles memory and complicates solver workspace management. The benefit depends on how different simulation-only features make trial points — this is problem-dependent and needs empirical investigation.

**Prerequisites**:

- Core SDDP training loop operational and validated
- Simulation-only LP construction (linearized head, unit commitment) implemented
- Solver workspace infrastructure supports multiple LP models per stage

**Estimated Effort**: Medium (2-3 weeks). LP construction infrastructure is reused; main effort is dual LP management, cut copying, and convergence testing.

**Reference**: SDDP.jl `AlternativeForwardPass` and `AlternativePostIterationCallback` in `src/plugins/forward_passes.jl`.

## C.14 Monte Carlo Backward Sampling

**Status**: DEFERRED

**Description**: Sample $n$ openings with replacement from the fixed opening tree instead of evaluating all $N_{\text{openings}}$ in the backward pass. This reduces backward pass cost from $O(N_{\text{openings}})$ to $O(n)$ LP solves per stage per trial point.

**Motivation**: When $N_{\text{openings}}$ is large (e.g., 500+ for high-fidelity tail representation), the backward pass dominates iteration time. Sampling a subset provides an unbiased cut estimator with reduced computation.

**Design** (inspired by SDDP.jl's `MonteCarloSampler`):

- At each backward stage, sample $n$ noise vectors uniformly with replacement from the opening tree
- Compute cut coefficients from only these $n$ solves
- The resulting cut is an unbiased estimator of the full cut (with higher variance)

**Trade-offs**:

| Aspect           | Complete (current)                 | MonteCarlo(n)                       |
| ---------------- | ---------------------------------- | ----------------------------------- |
| Solves per stage | $N_{\text{openings}}$              | $n$                                 |
| Cut quality      | Exact (for given tree)             | Unbiased estimator, higher variance |
| Convergence      | Monotonic lower bound              | Non-monotonic (stochastic cuts)     |
| Best for         | Small-medium $N_{\text{openings}}$ | Large $N_{\text{openings}}$         |

**Why Deferred**: Introduces non-monotonic lower bound behavior, which complicates convergence monitoring. Requires careful tuning of $n$ relative to $N_{\text{openings}}$. The default complete evaluation is reliable and performant for typical production sizes (50-200 openings).

**Prerequisites**:

- Core SDDP with complete backward sampling validated
- Convergence monitoring supports non-monotonic lower bounds
- Empirical study of $n$ vs. convergence rate trade-off

**Estimated Effort**: Small (1 week). Sampling infrastructure is trivial; main effort is convergence monitoring adaptation.

**Reference**: SDDP.jl `MonteCarloSampler` in `src/plugins/backward_sampling_schemes.jl`.

## C.15 Risk-Adjusted Forward Sampling

**Status**: DEFERRED (supersedes "Risk-Adjusted Forward Passes" in Additional Variants below)

**Description**: Oversample scenarios from distribution tails in the forward pass, improving exploration of worst-case outcomes for risk-averse policies. This is a forward sampling scheme variant that biases sampling toward extreme scenarios.

**Design** (inspired by SDDP.jl's `RiskAdjustedForwardPass`):

- After solving each forward pass stage, evaluate the risk measure over the candidate noise terms
- Re-weight or re-sample the next-stage noise based on the risk adjustment
- The resulting forward trajectory visits states that are more relevant for the tail of the cost distribution

**Complementary approach** — Importance Sampling (inspired by SDDP.jl's `ImportanceSamplingForwardPass`):

- Weight forward trajectories by their likelihood ratio under the risk-adjusted distribution vs. the nominal distribution
- Use these weights in upper bound estimation for tighter risk-averse bounds

**Why Deferred**: Requires integration with the CVaR risk measure configuration and careful handling of importance weights. Default uniform sampling is sufficient for most applications. The benefit is primarily for strongly risk-averse configurations ($\lambda$ close to 1.0).

**Prerequisites**:

- CVaR risk measure implemented and validated
- Forward sampling scheme abstraction supports per-stage re-weighting
- Upper bound evaluation handles importance-weighted trajectories

**Estimated Effort**: Medium (2-3 weeks). Algorithm is well-documented in literature; main effort is integration with risk measure and convergence analysis.

**References**:

- SDDP.jl `RiskAdjustedForwardPass` in `src/plugins/forward_passes.jl`
- Philpott, A.B., & de Matos, V.L. (2012). "Dynamic sampling algorithms for multi-stage stochastic programs with risk aversion."

## C.16 Revisiting Forward Pass

**Status**: DEFERRED

**Description**: Encourage state-space diversity by occasionally re-solving forward passes from previously visited states rather than always starting from the root node. This helps the algorithm explore under-represented regions of the state space.

**Design** (inspired by SDDP.jl's `RevisitingForwardPass`):

- Maintain a buffer of previously visited states $\{\hat{x}^{(k)}_t\}$ from past iterations
- With some probability, start a forward pass from a randomly selected historical state at a randomly selected stage (rather than from stage 1 with $x_0$)
- The resulting trial points provide cuts in previously unvisited regions

**Why Deferred**: Modifies the forward pass initialization, which affects upper bound estimation (partial trajectories don't contribute full-horizon costs). Requires careful interaction with convergence monitoring. The benefit is primarily for problems with large state spaces where standard forward passes repeatedly visit similar states.

**Prerequisites**:

- Core SDDP training loop validated
- State history buffer infrastructure
- Convergence monitoring handles partial-trajectory iterations

**Estimated Effort**: Small-Medium (1-2 weeks). State buffer is straightforward; main effort is convergence analysis for partial trajectories.

**Reference**: SDDP.jl `RevisitingForwardPass` in `src/plugins/forward_passes.jl`.

---

## C.17 Forward Pass State Deduplication

**Status**: DEFERRED

**Description**: When multiple forward scenarios visit identical (or near-identical) states at a given stage, the backward pass redundantly evaluates the cost-to-go from duplicate trial points. State deduplication merges these duplicates before the backward pass, reducing the number of backward LP solves without affecting cut quality (identical states produce identical cuts).

**Design Considerations**:

- **Exact deduplication** — Hash-based deduplication of state vectors with identical storage volumes and AR lag values. Straightforward but limited benefit (exact duplicates are rare with continuous state spaces)
- **Approximate deduplication** — Merge states within a tolerance $\varepsilon$ using spatial indexing (e.g., k-d tree). Greater reduction in backward solves, but introduces approximation error in cut coefficients. Requires analysis of the impact on convergence guarantees

**Why Deferred**: Exact duplicates are uncommon in continuous-state SDDP, so the benefit of exact deduplication is marginal. Approximate deduplication requires careful analysis of convergence implications. Should be implemented only when profiling shows the backward pass is bottlenecked on trial point count rather than per-solve cost.

**Prerequisites**:

- Core SDDP training loop validated and profiled
- Backward pass timing data showing trial point count as the dominant cost factor

**Estimated Effort**: Small (< 1 week) for exact deduplication; Medium (2-3 weeks) for approximate with convergence analysis.

**Cross-references**:

- [Training Loop §5.2](./architecture/training-loop.md) — State lifecycle where deduplication would be applied

---

## C.18 Pipelined Backward Pass

**Status**: DEFERRED

**Description**: The standard (sequential) backward pass uses $V_{t+1}^k$ — the freshly computed approximation from the current iteration — requiring a synchronization barrier at each stage boundary. The pipelined variant uses $V_{t+1}^{k-1}$ (previous iteration's approximation), allowing stages to be computed with overlapped communication. This eliminates per-stage barriers and enables non-blocking MPI broadcasts (`ibroadcast`) to overlap with computation at the next stage.

**Trade-offs**:

| Aspect                 | Sequential (current)            | Pipelined (deferred)                          |
| ---------------------- | ------------------------------- | --------------------------------------------- |
| Cut source             | $V_{t+1}^k$ (current iteration) | $V_{t+1}^{k-1}$ (previous iteration)          |
| Cut quality            | Tighter                         | Looser                                        |
| Iterations to converge | Fewer                           | More                                          |
| Time per iteration     | Longer (per-stage barriers)     | Shorter (overlapped communication)            |
| Implementation         | Simple                          | Complex (async state management)              |
| MPI primitives         | `MPI_Allgatherv` (blocking)     | `MPI_Iallgatherv` + `MPI_Wait` (non-blocking) |

**Mathematical validity**: Both approaches produce valid cuts — supporting hyperplanes of the true value function. The difference is in tightness: sequential cuts incorporate more recent information (downstream cuts from the same iteration), leading to faster convergence per iteration.

**When to consider**: When profiling shows backward pass per-stage barrier synchronization accounts for a significant fraction (>20%) of backward pass time. This is most likely with very large stage counts (>100 stages) and high inter-node network latency.

**Why Deferred**: The sequential mode is simpler, produces tighter cuts, and is the default in SDDP.jl and most production implementations. The benefit of pipelining depends on hardware-specific communication latency, which cannot be determined without profiling production workloads.

**Prerequisites**:

- Core SDDP training loop validated and profiled
- Backward pass timing data showing barrier synchronization as a bottleneck
- Non-blocking MPI collective support in ferrompi

**Estimated Effort**: Medium (2-3 weeks). Requires async state management, non-blocking collective wrappers, and convergence regression testing.

**Cross-references**:

- [Work Distribution §2](./hpc/work-distribution.md) — Current sequential backward pass distribution
- [Training Loop §6](./architecture/training-loop.md) — Backward pass execution with per-stage barrier
- [Communication Patterns](./hpc/communication-patterns.md) — Non-blocking ferrompi collectives

---

## C.19 Dynamic Forward-Passes Scheduler

**Status**: DEFERRED

**Description**: A scheduler that adjusts the number of forward passes per iteration (`forward_passes`) during training based on convergence metrics or wall-clock budget.

**Why Deferred**: The cut pool capacity is pre-allocated using the formula `warm_start_cuts + max_iterations × forward_passes` (see [Cut Management Implementation SS1.3](./architecture/cut-management-impl.md)). A dynamic scheduler would require either a worst-case static allocation (wasteful for most runs) or dynamic reallocation (invalidates the deterministic slot assignment in SS1.2). Both alternatives need profiling data to choose correctly.

---

## Additional Deferred Algorithm Variants

The following algorithm variants are also deferred:

### Objective States

Extends SDDP to handle exogenous random processes that affect objective coefficients (e.g., electricity spot prices following an AR process). Uses inner approximation (Lipschitz interpolation) for the value function component that depends on objective states.

**Why Deferred**: Not typically needed for hydrothermal dispatch where prices are marginal cost-based. Can often be approximated by scenario-based approaches.

### Belief States (POMDP)

Extends SDDP to partially observable Markov decision processes where the agent maintains a probability distribution (belief) over hidden states. Useful for modeling phenomena like unobservable climate regimes.

**Why Deferred**: Research-level feature with limited practical benefit for most hydrothermal applications. Can often be approximated by expanding Markov states.

### Duality Handlers (Lagrangian Relaxation)

Methods to generate valid cuts from MIP subproblems when integer variables are present (e.g., unit commitment). Includes Lagrangian relaxation and strengthened Benders cuts.

**Why Deferred**: Significant complexity in Lagrangian multiplier updates. Alternative: solve unit commitment deterministically after SDDP provides marginal values.

## Cross-References

- [Configuration Reference](./configuration/configuration-reference.md) -- Config options for implemented features
- [SDDP Algorithm](./math/sddp-algorithm.md) -- Core algorithm that deferred features extend
- [Hydro Production Models](./math/hydro-production-models.md) -- Base FPHA that C.6 enhances
- [Cut Management](./math/cut-management.md) -- Cut formulation that multi-cut (C.3) modifies
- [Risk Measures](./math/risk-measures.md) -- Risk framework that Markovian (C.4) and risk-adjusted passes extend
- [Block Formulations](./math/block-formulations.md) -- Block structure that temporal decoupling (C.7) generalizes; policy validation requirement (§4) motivates C.9
- [PAR Inflow Model](./math/par-inflow-model.md) -- Standard PAR(p) that CEPEL PAR(p)-A (C.8) extends
- [Equipment Formulations](./math/equipment-formulations.md) -- Thermal formulations that GNL (C.1) extends
- [Input System Entities](./data-model/input-system-entities.md) -- Entity schemas for batteries (C.2) and non-controllables (C.5)
- [Input Scenarios](./data-model/input-scenarios.md) -- External scenario data model (C.11, C.12)
- [Scenario Generation](./architecture/scenario-generation.md) -- Opening tree (C.11), complete tree concept (C.12), sampling scheme abstraction (C.13-C.16)
- [Training Loop](./architecture/training-loop.md) -- Must persist policy metadata for C.9 validation; forward/backward pass structure for C.13-C.16
- [Simulation Architecture](./architecture/simulation-architecture.md) -- Must validate policy compatibility (C.9) on entry
- [Checkpointing](./hpc/checkpointing.md) -- Checkpoint format that stores policy metadata (C.9)
- [Binary Formats](./data-model/binary-formats.md) -- Policy file format that carries metadata (C.9)
