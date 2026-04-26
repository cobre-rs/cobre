# Solver Abstraction Layer

## Purpose

This spec defines the multi-solver abstraction layer: the unified interface through which optimization algorithms interact with LP solvers, including the solver interface contract, LP layout convention, cut pool design, error categories, retry logic contract, dual normalization, basis storage, compile-time solver selection, stage LP template strategy, and solver-specific optimization paths.

For thread-local solver infrastructure and LP scaling, see [Solver Workspaces & LP Scaling](./solver-workspaces.md). For solver-specific implementations, see [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md).

## 1. Design Rationale

Optimization algorithms using this layer must be solver-agnostic. The solver interface abstracts LP solver details behind a unified contract, designed and validated against both HiGHS and CLP to avoid single-solver bias.

Key design decisions:

1. **Dual-solver validation** — The interface is designed against both HiGHS (C API) and CLP (C API + thin C++ wrappers) as first-class reference implementations. Every operation in the interface contract (SS3) must be naturally expressible through both solver APIs. This prevents designing an interface that maps cleanly to one solver but awkwardly to another.
2. **Compile-time solver selection** — The active solver is selected at compile time to avoid virtual dispatch overhead on the hot path (millions of LP solves)
3. **Encapsulated retry logic** — Each solver handles its own numerical difficulties internally; the calling algorithm only sees success or failure
4. **Cuts stored in physical units** — Scaling transformations are applied at solve time, not stored in the cut pool
5. **Pre-assembled stage templates** — Each stage's structural LP is assembled once at initialization in solver-ready CSC form (both HiGHS and CLP use column-major internally), minimizing per-iteration rebuild cost

## 2. LP Layout Convention

The LP row and column ordering is standardized to enable performance optimizations across the hot path: slice-based state transfer, fast RHS patching, efficient dual extraction for cut coefficients, and favorable basis persistence.

### 2.1 Column Layout (Variables)

State variables are placed contiguously at the beginning of the variable vector. All index formulas below use 0-based indexing. Let $N$ denote the number of **operating** hydros at the current stage (non-existing and decommissioned hydros per [Internal Structures SS2](../data-model/internal-structures.md) have zero LP presence) and $L$ the maximum PAR order across all operating hydros (i.e., $L = \max_{h} P_h$).

**Column layout diagram:**

```
Column index:
  0              N                   N+N       N+2N  ...  N+L·N     N+L·N+N    n_state+N+1          n_cols
  ├──────────────┼────────────────────┼─────────┼────────────┼──────────┼──────────┼──────┼───────────┤
  │ storage      │ lag 0              │ lag 1   │    ...     │ lag L-1  │   v^in   │  θ   │ decision  │
  │ v_0..v_{N-1} │ a_{0,0}..a_{N-1,0} │  ...    │            │ v^in_0.. │          │      │   vars    │ └──────────────┴────────────────────┴─────────┴────────────┴──────────┴──────────┴──────┴───────────┘
  ╰──────────────────────────── state prefix (n_state columns) ────────────────────╯╰───── aux ───────╯
```

**Exact index formulas:**

| Column Index Formula          | Contents                                                                                                   | Count       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------- |
| $h$                           | Storage volume $v_h$ for hydro $h \in [0, N)$                                                              | $N$         |
| $N + \ell \cdot N + h$        | AR inflow lag $a_{h,\ell}$ for hydro $h \in [0, N)$, lag $\ell \in [0, L)$                                 | $N \cdot L$ |
| $N(1 + L) + h$                | Incoming storage $v^{in}_h$ for hydro $h \in [0, N)$ — see [LP Formulation §4a](../math/lp-formulation.md) | $N$         |
| $N(1 + L) + N = N(2 + L)$     | Future cost variable $\theta$                                                                              | $1$         |
| $[N(2 + L) + 1, \; n_{cols})$ | Decision variables (see ordering convention below)                                                         | varies      |

**State dimension:**

$$n_{state} = N + N \cdot L = N \cdot (1 + L)$$

**Incoming storage variables**: The $N$ columns at positions $[N(1+L), N(2+L))$ hold the incoming storage variables $v^{in}_h$, introduced by the storage fixing constraint ([LP Formulation §4a](../math/lp-formulation.md)). These are **not** part of the state vector — they are auxiliary variables whose only purpose is to provide a clean dual for cut coefficient extraction. They have no bounds (their values are fully determined by the fixing constraints). Placing them after the state prefix but before $\theta$ keeps the state prefix contiguous and undisturbed.

**Uniform lag storage**: All operating hydros store $L$ lags regardless of their individual AR order $P_h$. Hydros with $P_h < L$ have zero-valued AR coefficients ($\psi_\ell = 0$ for $\ell > P_h$) in their lag positions beyond their actual order. This uniform stride enables:

- SIMD vectorization of dot products between cut coefficients and state vectors (all hydros have the same memory stride)
- A single contiguous slice for the entire state vector (no variable-length per-hydro regions)
- Simplified index arithmetic (the formula $N + \ell \cdot N + h$ works for all hydros without per-hydro offset tables)

**State transfer operation**: Transferring state from stage $t$ to stage $t+1$ is a single contiguous slice copy. The oldest lag ($\ell = L-1$) is dropped because it will not be needed in the next stage's AR dynamics (it ages out of the lag window). The transfer copies:

```
primal[0 .. N·(1 + L_transfer)]   where L_transfer = L - 1
```

This copies $N$ storage values plus $N \cdot (L-1)$ lag values (lags $\ell = 0, \ldots, L-2$). The new stage then shifts these lags by one position (lag $\ell$ becomes lag $\ell+1$), and the newly computed inflow $a_{h,t}$ is written into lag position $\ell = 0$. The transfer dimension is:

$$n_{transfer} = N \cdot (1 + (L - 1)) = N \cdot L$$

**Decision variable ordering** (columns $[N(2+L)+1, \; n_{cols})$): Decision variables are ordered by entity type, then by entity ID within type, then by block within entity. This ordering is secondary to performance (decision variables are not in the state vector) but important for debugging and output interpretation.

| Sub-region (in order) | Contents                                                                                                                                                          | Count                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Thermals              | $g_{j,k,s}$ by thermal $j$, block $k$, segment $s$                                                                                                                | $N_{th} \times N_{blk} \times \bar{N}_{seg}$       |
| Lines                 | $f^+_{\ell,k}$, $f^-_{\ell,k}$ by line $\ell$, block $k$                                                                                                          | $2 \times N_{line} \times N_{blk}$                 |
| Buses                 | $\delta_{b,k,s}$, $\epsilon_{b,k}$ by bus $b$, block $k$                                                                                                          | $N_{bus} \times N_{blk} \times (N_{def\_seg} + 1)$ |
| Hydro per-block       | $q_{h,k}$, $s_{h,k}$, $g_{h,k}$, $\tilde{a}_{h,k}$, $e_{h,k}$, $r_{h,k}$, $u_{h,k}$ by hydro $h$, block $k$                                                       | $N \times N_{blk} \times n_{h\_vars}$              |
| Hydro slacks          | $\sigma^{v-}_h$, $\sigma^{fill}_h$, $\sigma^{q-}_{h,k}$, $\sigma^{o-}_{h,k}$, $\sigma^{o+}_{h,k}$, $\sigma^{g-}_{h,k}$, $\sigma^{e\pm}_{h,k}$, $\sigma^{r}_{h,k}$ | varies                                             |
| Contracts             | $\chi_{c,k}$ by contract $c$, block $k$                                                                                                                           | $(N_{imp} + N_{exp}) \times N_{blk}$               |
| Pumping               | $p_{j,k}$, power by pump $j$, block $k$                                                                                                                           | $N_{pump} \times N_{blk} \times 2$                 |
| NCS                   | $g^{nc}_{r,k}$ by NCS $r$, block $k$                                                                                                                              | $N_{ncs} \times N_{blk}$                           |

**Key benefits**:

- The complete state vector (storage + AR lags) occupies columns $[0, n_{state})$. Transferring state from one stage to the next is a single contiguous slice copy -- no gathering from scattered column positions.
- The incoming storage variables $v^{in}_h$ occupy columns $[n_{state}, n_{state} + N)$, separated from the state prefix. They are auxiliary variables for dual extraction, not part of the state.
- The $\theta$ variable sits at a fixed, easily computed position $n_{state} + N = N(2 + L)$ immediately after the incoming storage block.
- Decision variables follow a predictable type-then-ID-then-block ordering for debuggability.

### 2.2 Row Layout (Constraints)

Constraints are ordered in three regions. All index formulas use 0-based indexing. Let $N$ and $L$ be defined as in SS2.1.

| Region                        | Row Range                          | Contents                                                                                                  | Lifecycle                                            |
| ----------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Dual-extraction region**    | `[0, n_dual_relevant)`             | All constraints whose duals contribute to subgradient/gradient computation (sub-regions defined below)    | Static structure per stage; RHS patched per scenario |
| **Static non-dual region**    | `[n_dual_relevant, n_static)`      | All other static constraints: load balance, generation/flow bounds, penalty bounds, remaining constraints | Static per stage                                     |
| **Dynamic constraint region** | `[n_static, n_static + n_dynamic)` | Dynamic constraints (added at runtime via batch `add_rows`; e.g., Benders cuts in SDDP)                   | Rebuilt each iteration via batch `add_rows`          |

**Top region — fixing constraints for cut coefficient extraction:**

The dual-extraction region contains exactly the **fixing constraints** — the equality constraints that bind each incoming state variable to its trial value. Their duals are the cut coefficients, read as a single contiguous slice from the dual vector. No other constraints (water balance, FPHA, generic) appear in this region; their contributions to the marginal value of incoming state are captured automatically by LP duality through the fixing constraint duals (see [LP Formulation §4a](../math/lp-formulation.md) and [Cut Management §2](../math/cut-management.md)).

**Row layout diagram (dual-extraction region):**

```
Row index:
  0          N         N+N·L = n_state
  ├──────────┼──────────┤
  │ storage  │ AR lag   │
  │ fixing   │ fixing   │
  │ (N rows) │(N·L rows)│
  └──────────┴──────────┘
```

**Exact row index formulas (dual-extraction region):**

| Row Index Formula      | Contents                                                                                                                                                       | Count       | Dual Symbol          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------- |
| $h$                    | Storage fixing constraint $v^{in}_h = \hat{v}_h$ for hydro $h \in [0, N)$ — see [LP Formulation §4a](../math/lp-formulation.md)                                | $N$         | $\pi^{fix}_h$        |
| $N + \ell \cdot N + h$ | AR lag fixing constraint $a_{h,\ell} = \hat{a}_{h,\ell}$ for hydro $h \in [0, N)$, lag $\ell \in [0, L)$ — see [LP Formulation §5a](../math/lp-formulation.md) | $N \cdot L$ | $\pi^{lag}_{h,\ell}$ |

**Dual-relevant row count:**

$$n_{dual\_relevant} = N + N \cdot L = n_{state}$$

The dual-relevant row count equals the state dimension. The row index formulas use the same structure as the column index formulas in SS2.1 (index $h$ and $N + \ell \cdot N + h$ respectively). This symmetry is exact: the dual of row $r$ **is** the cut coefficient for state variable at column $r$. Cut coefficient extraction is a single contiguous slice read:

```
cut_coefficients[0..n_state] = dual[0..n_state]
```

No postprocessing, no dual combination, no FPHA scaling factors, no generic constraint mapping. The LP solver resolves all contributions through the fixing constraint duals automatically.

**Middle region — static, non-dual-relevant constraints:**

The static non-dual region contains all static constraints whose duals do not contribute directly to cut coefficients. Their exact internal ordering follows the [LP Formulation](../math/lp-formulation.md) convention:

| Sub-region (in order) | Contents                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| Water balance         | Hydro water balance ([LP Formulation §4](../math/lp-formulation.md)) -- $N$ rows                    |
| Load balance          | Bus power balance ([LP Formulation §3](../math/lp-formulation.md)) -- $N_{bus} \times N_{blk}$ rows |
| Inflow AR dynamics    | PAR model dynamics equation ([LP Formulation §5](../math/lp-formulation.md)) -- $N$ rows            |
| FPHA hyperplanes      | FPHA production constraints ([LP Formulation §6](../math/lp-formulation.md)) -- $n_{fpha}$ rows     |
| Constant productivity | $g_{h,k} = \rho_h \cdot q_{h,k}$ for non-FPHA hydros -- $(N - N_{fpha}) \times N_{blk}$ rows        |
| Outflow definition    | $o_{h,k} = q_{h,k} + s_{h,k}$ -- $N \times N_{blk}$ rows                                            |
| Outflow bounds        | Min/max outflow -- $2 \times N \times N_{blk}$ rows                                                 |
| Turbined flow min     | $q_{h,k} \geq \underline{Q}_h - \sigma^{q-}_{h,k}$ -- $N \times N_{blk}$ rows                       |
| Generation min        | $g_{h,k} \geq \underline{G}_h - \sigma^{g-}_{h,k}$ -- $N \times N_{blk}$ rows                       |
| Evaporation           | Evaporation constraints -- $N \times N_{blk}$ rows                                                  |
| Water withdrawal      | Withdrawal constraints -- $N \times N_{blk}$ rows                                                   |
| Storage bounds        | $\underline{V}_h - \sigma^{v-}_h \leq v_h \leq \bar{V}_h$ -- $N$ rows                               |
| Generic constraints   | All user-defined generic constraints ([LP Formulation §10](../math/lp-formulation.md)) -- varies    |

**Dynamic constraint region:**

Dynamic constraints are appended at rows $[n_{static}, \; n_{static} + n_{dynamic})$ via batch `add_rows` (e.g., Benders cuts in SDDP). Each cut row has the form $\theta - \boldsymbol{\pi}^\top \mathbf{x} \geq \alpha$ where $\boldsymbol{\pi}$ is the cut coefficient vector and $\alpha$ is the intercept (see [Solver Abstraction SS5.3](./solver-abstraction.md)).

**Key benefits**:

- **Fixing constraints at top** — All duals needed for cut coefficient computation are in the first $n_{state}$ rows. Extracting them is a single contiguous slice read: `cut_coefficients = dual[0..n_state]` — no index gathering, no postprocessing, no FPHA/generic constraint dual mapping.
- **Exact row-column symmetry** — Storage fixing row $h$ corresponds to storage column $h$; lag fixing row $N + \ell \cdot N + h$ corresponds to lag column $N + \ell \cdot N + h$. The dual of row $r$ is the cut coefficient for state variable at column $r$, for all $r \in [0, n_{state})$.
- **Dynamic constraints at bottom** — Everything above the dynamic constraint boundary is identical across iterations within a stage. A cached basis from the previous iteration applies directly to the static portion (rows `[0, n_static)`). Only the new dynamic constraint rows need their basis status initialized.
- **Static non-dual region** — Water balance, FPHA, load balance, and all other non-dual-relevant constraints. Their ordering within this region follows the [LP Formulation](../math/lp-formulation.md) convention.

### 2.3 Interaction with Basis Persistence

When warm-starting from a cached basis after adding new dynamic constraints:

1. The basis status codes for rows `[0, n_static)` are reused directly from the cached basis
2. New dynamic constraint rows (appended at the bottom) are initialized with status **Basic** — meaning the slack variable for each new constraint is in the basis. The solver will price these rows in and pivot as needed during the solve.
3. If cuts were removed since the cached basis was saved, the basis is truncated to match the new row count

This is possible precisely because dynamic constraints are at the bottom — the static portion of the basis is position-stable across iterations.

### 2.4 Worked Example

This section applies the index formulas from SS2.1 and SS2.2 to a concrete small system, providing a verification baseline for implementers.

**System definition:**

| Entity        | Count         | Details                                                                                 |
| ------------- | ------------- | --------------------------------------------------------------------------------------- |
| Hydros        | $N = 3$       | H0 (constant productivity), H1 (FPHA with $M_1 = 4$ planes), H2 (constant productivity) |
| Max PAR order | $L = 2$       | H0 uses PAR(1), H1 uses PAR(2), H2 uses PAR(1); all store $L = 2$ lags                  |
| Buses         | 2             | B0, B1                                                                                  |
| Thermals      | 1             | T0 (single cost segment)                                                                |
| Lines         | 1             | L0 (B0 $\to$ B1)                                                                        |
| Blocks        | $N_{blk} = 1$ | Single block for simplicity                                                             |

**Column layout (SS2.1):**

State dimension: $n_{state} = N \cdot (1 + L) = 3 \cdot (1 + 2) = 9$.

| Column | Formula                 | Contents                         |
| -----: | ----------------------- | -------------------------------- |
|      0 | $h = 0$                 | $v_0$ — storage H0               |
|      1 | $h = 1$                 | $v_1$ — storage H1               |
|      2 | $h = 2$                 | $v_2$ — storage H2               |
|      3 | $N + 0 \cdot N + 0 = 3$ | $a_{0,0}$ — H0 lag 0             |
|      4 | $N + 0 \cdot N + 1 = 4$ | $a_{1,0}$ — H1 lag 0             |
|      5 | $N + 0 \cdot N + 2 = 5$ | $a_{2,0}$ — H2 lag 0             |
|      6 | $N + 1 \cdot N + 0 = 6$ | $a_{0,1}$ — H0 lag 1             |
|      7 | $N + 1 \cdot N + 1 = 7$ | $a_{1,1}$ — H1 lag 1             |
|      8 | $N + 1 \cdot N + 2 = 8$ | $a_{2,1}$ — H2 lag 1             |
|      9 | $N(1+L) + 0 = 9$        | $v^{in}_0$ — incoming storage H0 |
|     10 | $N(1+L) + 1 = 10$       | $v^{in}_1$ — incoming storage H1 |
|     11 | $N(1+L) + 2 = 11$       | $v^{in}_2$ — incoming storage H2 |
|     12 | $N(2 + L) = 12$         | $\theta$ — future cost           |
|    13+ | $\geq N(2+L)+1 = 13$    | Decision variables               |

Note: H0 uses PAR(1) and H2 uses PAR(1), so their lag-1 positions (columns 6 and 8) have zero AR coefficients ($\psi_2 = 0$) but are still allocated for uniform stride. The incoming storage variables (columns 9-11) have no bounds — their values are determined by the storage fixing constraints.

State transfer copies `primal[0..6]` — that is, $n_{transfer} = N \cdot L = 3 \cdot 2 = 6$ values: 3 storages + 3 lag-0 values. The lag-1 values (columns 6-8) are dropped as the oldest lag. The incoming storage variables (columns 9-11) are not part of the state transfer.

**Row layout (SS2.2):**

Dual-relevant row count: $n_{dual\_relevant} = N + N \cdot L = n_{state} = 9$.

| Row | Formula                       | Contents                                           | Dual              |
| --: | ----------------------------- | -------------------------------------------------- | ----------------- |
|   0 | $h = 0$                       | Storage fixing $v^{in}_0 = \hat{v}_0$ (H0)         | $\pi^{fix}_0$     |
|   1 | $h = 1$                       | Storage fixing $v^{in}_1 = \hat{v}_1$ (H1)         | $\pi^{fix}_1$     |
|   2 | $h = 2$                       | Storage fixing $v^{in}_2 = \hat{v}_2$ (H2)         | $\pi^{fix}_2$     |
|   3 | $N + 0 \cdot N + 0 = 3$       | AR lag fixing H0, $\ell=0$                         | $\pi^{lag}_{0,0}$ |
|   4 | $N + 0 \cdot N + 1 = 4$       | AR lag fixing H1, $\ell=0$                         | $\pi^{lag}_{1,0}$ |
|   5 | $N + 0 \cdot N + 2 = 5$       | AR lag fixing H2, $\ell=0$                         | $\pi^{lag}_{2,0}$ |
|   6 | $N + 1 \cdot N + 0 = 6$       | AR lag fixing H0, $\ell=1$                         | $\pi^{lag}_{0,1}$ |
|   7 | $N + 1 \cdot N + 1 = 7$       | AR lag fixing H1, $\ell=1$                         | $\pi^{lag}_{1,1}$ |
|   8 | $N + 1 \cdot N + 2 = 8$       | AR lag fixing H2, $\ell=1$                         | $\pi^{lag}_{2,1}$ |
|  9+ | $\geq n_{dual\_relevant} = 9$ | Static non-dual region (water balance, FPHA, etc.) |                   |

Note: The FPHA hyperplanes for H1 (4 planes × 1 block = 4 rows) and water balance constraints (3 rows) are now in the **static non-dual region** (rows 9+). Their duals are no longer needed for cut coefficient computation — the storage fixing constraint duals ($\pi^{fix}_h$) capture all contributions automatically.

**Summary for this example:**

| Quantity               | Value |
| ---------------------- | ----- |
| $N$ (operating hydros) | 3     |
| $L$ (max PAR order)    | 2     |
| $n_{state}$            | 9     |
| $n_{transfer}$         | 6     |
| $\theta$ column        | 12    |
| $n_{dual\_relevant}$   | 9     |

**Verification**: A developer applying the formulas to this system should obtain the exact column and row indices shown in the tables above. The row-column symmetry is exact: `dual[0..9]` gives the cut coefficient vector for state variables at columns `[0..9)`. No postprocessing is needed.

### 2.5 Performance Notes

**SIMD alignment**: State vector arrays (primal solution buffer, cut coefficient vectors) should be allocated with 64-byte alignment. This is the AVX-512 register width, accommodating 8 `f64` values per SIMD lane. The alignment requirement is stated as a specification constraint — specific Rust allocator APIs (e.g., `std::alloc::Layout::from_size_align`) are an implementation choice.

**Cache locality of the state prefix**: The state prefix occupies the first $N \cdot (1 + L) \times 8$ bytes of the primal vector. At production scale ($N = 160$, $L = 12$):

$$n_{state} = 160 \times (1 + 12) = 2{,}080$$

$$\text{State prefix size} = 2{,}080 \times 8 = 16{,}640 \text{ bytes} = 260 \text{ cache lines (64 B each)}$$

This fits comfortably in the L2 cache of modern server CPUs (typically 1-2 MB per core) and represents a small fraction of L1 data cache (32-48 KB). State transfer, cut coefficient dot products, and dual extraction all operate within this cache-hot region.

For a smaller production configuration ($N = 160$, $L = 6$):

$$n_{state} = 160 \times (1 + 6) = 1{,}120$$

$$\text{State prefix size} = 1{,}120 \times 8 = 8{,}960 \text{ bytes} = 140 \text{ cache lines}$$

This fits entirely in L1 data cache.

**Contiguous dual extraction**: The first $n_{state}$ values of the dual vector contain all duals needed for cut coefficient computation. Reading them is a single contiguous memory access:

```
cut_coefficients = dual[0 .. n_state]
```

No index indirection, gather operation, or postprocessing is required. Each dual value maps directly to the cut coefficient at the corresponding state variable position — a single `memcpy`-equivalent operation produces the complete cut coefficient vector. This is a significant simplification over designs that require combining duals from multiple constraint types (water balance, FPHA, generic constraints).

**Cut coefficient dot product**: The most frequent numerical operation on state vectors is the dot product $\alpha_k + \boldsymbol{\pi}_k^\top \mathbf{x}$ evaluated for each active cut during the forward pass (to compute $\theta$'s lower bound). With uniform stride and 64-byte aligned arrays, this reduces to a SIMD-friendly dense dot product of length $n_{state}$.

## 3. Abstraction Hierarchy

The solver abstraction separates concerns into four layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SOLVER ABSTRACTION HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Stage LP Template (Data Holder)                    │  │
│  │  - Pre-assembled structural LP in CSC form (built once per stage)     │  │
│  │  - Solver-agnostic problem representation                             │  │
│  │  - Row/column layout follows SS2 convention                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      LP Scaling (Pre-processing)                      │  │
│  │  - Row/column scaling factors for numerical stability                 │  │
│  │  - Transforms problem ↔ scaled space                                  │  │
│  │  - Persisted for cut coefficient computation                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       LP Solver (Execution)                           │  │
│  │  - Solve → success or error                                           │  │
│  │  - Encapsulates retry logic, warm-start, basis management             │  │
│  │  - Compile-time selected (HiGHS or CLP as reference impls)            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     LP Solution (Result Data)                         │  │
│  │  - Primal values, dual values (constraint & variable)                 │  │
│  │  - Objective value, solve status                                      │  │
│  │  - Basis information (optional, for warm-starting)                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Decision [DEC-008](../overview/decision-log.md#dec-008) (active):** LP scaling is delegated to the solver backend (`SolverAuto`); Cobre does not apply its own scaling in the minimal viable solver.
>
> - **HiGHS**: internal scaling is enabled by default (built-in geometric mean strategy, `kSimplexScaleStrategy = 2`).
> - **CLP**: internal scaling is enabled by default (CLP's automatic scaling).
> - **Cobre-managed scaling**: the `GeometricMean` and `Equilibration` methods defined in [Solver Workspaces SS2.2--2.4](./solver-workspaces.md) are retained as reference material but are **not active** in the minimal viable solver. Two-phase scaling (re-scale after cuts accumulate) is likewise deferred.
>
> **Rationale**: (a) HiGHS and CLP both have mature, well-tested internal scaling that handles the coefficient magnitude ranges typical of SDDP LPs; (b) applying Cobre-managed scaling on top of solver-internal scaling risks double-scaling, where the combined effect degrades rather than improves conditioning; (c) deferring Cobre-managed scaling reduces initial implementation complexity -- the `col_scale`/`row_scale` arrays, the unscaling transforms in solution extraction, and the cut coefficient re-transformation are all eliminated from the critical path; (d) this decision can be revisited if profiling with production-scale problems shows numerical difficulties (excessive retries, poor convergence) attributable to solver-internal scaling alone. See [Solver Workspaces SS2.5](./solver-workspaces.md) for the impact on the stage solve workflow.

**Stage Template vs Transient Solver State:**

| Aspect    | Stage LP Template (Static Data)                      | Solver State (Transient)      |
| --------- | ---------------------------------------------------- | ----------------------------- |
| Lifetime  | Built once at initialization, persists for algorithm | Per-solve invocation          |
| Ownership | Shared read-only across threads                      | Thread-local                  |
| Contents  | CSC matrix, bounds, objective, coefficients          | Basis factors, working arrays |
| Memory    | One per stage, pre-assembled once                    | Created/reused per solve      |

This separation enables the key optimization: template data is read-only and shared, while each thread maintains its own solver workspace for warm-starting. See [Solver Workspaces](./solver-workspaces.md) for the thread-local workspace design.

## 4. Solver Interface Contract

The solver interface defines the behavioral contract that all solver implementations must satisfy. Optimization algorithms interact with solvers exclusively through this interface.

### 4.1 Required Operations

| Operation           | Input                                     | Output                    | Description                                                                                                                      |
| ------------------- | ----------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Load model          | Stage LP template (CSC)                   | —                         | Bulk-load the pre-assembled structural LP into the solver via `passModel`/`loadProblem`. Fast memcpy-like operation.             |
| Add constraint rows | Dynamic constraints in CSR format (batch) | —                         | Batch-add dynamic constraints via a single `add_rows` call. Constraints are appended at the dynamic constraint region per SS2.2. |
| Patch row bounds    | Set of (row_index, lower, upper) triples  | —                         | Update scenario-dependent constraint RHS values (inflows, state fixing constraints) without structural LP changes.               |
| Patch col bounds    | Set of (col_index, lower, upper) triples  | —                         | Update variable bounds. Not used in minimal viable SDDP; included for completeness.                                              |
| Solve               | —                                         | Solution or error         | Solve the loaded LP. Handles retries internally, extracts solution with normalized duals.                                        |
| Solve with basis    | Basis from prior solve                    | Solution or error         | Set basis for warm-start, then solve. Reduces simplex iterations.                                                                |
| Reset               | —                                         | —                         | Clear all internal solver state (caches, basis, factorization).                                                                  |
| Get basis           | —                                         | Current basis             | Extract the current simplex basis for caching.                                                                                   |
| Statistics          | —                                         | Accumulated solve metrics | Total solves, total iterations, retries, failures, timing.                                                                       |

### 4.2 Contract Guarantees

- **Thread safety** — Each solver instance is exclusively owned by one thread. No concurrent access.
- **Retry encapsulation** — The calling algorithm never sees retry attempts. It calls solve and receives either a valid solution or a terminal error (see SS6 and SS7).
- **Dual normalization** — All returned dual values use the canonical sign convention (see SS8). Solver-specific sign differences are handled internally.
- **Solver identification** — Each implementation exposes a name string for logging and diagnostics.

### 4.3 Dual-Solver Validation

Every operation in the interface contract above must be verifiable against both reference solver APIs:

| Operation           | HiGHS C API                       | CLP C API / C++                                         |
| ------------------- | --------------------------------- | ------------------------------------------------------- |
| Load model          | `Highs_passModel`                 | `Clp_loadProblem`                                       |
| Add constraint rows | `Highs_addRows` (CSR batch)       | `Clp_addRows` (CSR batch)                               |
| Patch row bounds    | `Highs_changeRowsBoundsBySet`     | Mutable `double*` via `Clp_rowLower()`/`Clp_rowUpper()` |
| Patch col bounds    | `Highs_changeColsBoundsBySet`     | Mutable `double*` via `Clp_colLower()`/`Clp_colUpper()` |
| Solve               | `Highs_run`                       | `Clp_dual` (warm-start) or `Clp_initialSolve` (cold)    |
| Set/get basis       | `Highs_setBasis`/`Highs_getBasis` | `Clp_copyinStatus`/`Clp_statusArray`                    |
| Reset               | `Highs_clearSolver`               | Reconstruct or `Clp_initialSolve`                       |

See [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md) for solver-specific details.

## 5. Cut Pool Design

```mermaid
stateDiagram-v2
    direction TB
    [*] --> Generated: computed from duals
    Generated --> Active: add to LP (addRows)
    Active --> Deactivated: selection removes
    Deactivated --> Active: re-activate
    note right of Active
        warm-start from checkpoint
        → loaded cuts start Active
    end note
    note right of Deactivated
        retained in pool
        (never deleted)
    end note
```

**Memory layout.** Cuts live in a pre-allocated flat array sized `K_max × (1 + n_state) × 8 bytes`; each cut occupies a deterministic slot `warm_start + iter × M + fwd_idx`, which enables lock-free concurrent writes from multiple threads. A parallel activity bitmap marks which slots are currently Active; only Active cuts are added to the solver LP via `addRows`. Per-cut metadata (`iteration`, `stage`, `forward_pass`, `rhs`, `coefficients[]`) is stored alongside.

### 5.1 Scope Clarification: Cut Pool vs Solver LP

Two distinct concepts must be clearly separated:

| Concept       | What                                                                                | Where                                      | Preallocation                                                                       |
| ------------- | ----------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------- |
| **Cut pool**  | In-memory shared data structure holding all cuts (active + inactive) for all stages | Shared across threads, one per MPI rank    | Yes — deterministic slot assignment, activity bitmap, contiguous dense coefficients |
| **Solver LP** | Transient LP loaded into a thread-local solver instance for a single solve          | Thread-local, rebuilt per stage transition | No — only active cuts are added via `addRows`                                       |

Under the adopted LP rebuild strategy (SS11), the solver LP is transient — it is constructed, used, and discarded at every stage transition. Pre-allocating inactive dynamic constraint rows in the solver LP would add memory pressure and solver overhead (larger factorization) with no benefit. Only the cut pool uses preallocation.

### 5.2 Cut Pool Preallocation

The cut pool pre-allocates capacity for all cuts that could be generated during the entire training run:

**Capacity formula:**

```
capacity = warm_start_cuts + max_iterations × forward_passes_per_iteration
```

**Example** (production-scale):

| Parameter                       | Value                             |
| ------------------------------- | --------------------------------- |
| Warm-start cuts                 | 5,000                             |
| Max iterations × forward passes | 50 × 192 = 9,600                  |
| **Total capacity per stage**    | **15,000**                        |
| State dimension                 | 2,080                             |
| Memory per stage                | 15,000 × 2,080 × 8 bytes ≈ 238 MB |
| Total for 60 stages             | ~14.3 GB per rank                 |

**Deterministic slot assignment:** Cut slots are computed, not allocated at runtime. The slot for a new cut is:

```
slot = warm_start_count + iteration × forward_passes + forward_pass_index
```

This is a pure function with no side effects — the same inputs always produce the same slot. This eliminates thread-safety concerns and non-determinism.

`forward_passes_per_iteration` is immutable after initialization — see [Cut Management Implementation SS1.3](./cut-management-impl.md) for the precondition rationale.

An activity bitmap tracks which slots are currently active. The count of active slots is maintained alongside the bitmap to avoid scanning.

> Under StageLpCache (SS11.4), cut coefficients are absorbed into the per-stage CSC. The cut pool retains only metadata.

### 5.3 Mathematical Basis

A Benders cut has the form: $\theta \geq \alpha + \boldsymbol{\pi}' \mathbf{x}$, where $\theta$ is the future cost variable, $\alpha$ is the cut intercept, $\boldsymbol{\pi}$ is the vector of cut coefficients, and $\mathbf{x}$ is the state vector. Rearranging to standard LP row form: $\theta - \boldsymbol{\pi}' \mathbf{x} \geq \alpha$.

For details on cut generation and selection, see [Cut Management](../math/cut-management.md). For the in-memory layout requirements (contiguous dense coefficients, CSR-friendly, cache-aligned), see [Binary Formats SS3.4](../data-model/binary-formats.md).

### 5.4 How Active Cuts Enter the Solver LP

> **Decision [DEC-007](../overview/decision-log.md#dec-007) (active):** Selective cut addition is the baseline for cut loading: only active cuts are loaded into the solver LP; no inactive rows are parked in the LP.

Only active cuts (as determined by the cut pool activity bitmap) are added via a single `addRows` call at each stage transition. Inactive cuts are excluded from the solver LP entirely -- they are never loaded, and no bound-toggling deactivation is performed.

**Rationale:**

1. **Smaller LPs.** Selective addition produces LPs with fewer rows and a smaller basis factorization. At production scale with Level-1 cut selection, the active ratio $n_\text{active} / n_\text{total}$ can be substantially below 1.0, so excluding inactive cuts yields a meaningful row-count reduction.
2. **Natural fit for StageLpCache workflow.** Under the StageLpCache approach (SS11.2), the solver LP is rebuilt from the pre-assembled cache at every stage transition. There is no persistent LP to keep inactive rows "parked" in; selective addition aligns directly with the load-from-cache workflow (load StageLpCache, patch scenario bounds, warm-start, solve).
3. **Presolve is disabled.** Presolve is currently disabled for warm-start basis compatibility (see SS11.2 step 4 and the retry escalation in SS7.2). Bulk loading with bound deactivation relies on presolve to eliminate the redundant inactive rows before factorization. With presolve off, those rows would remain in the LP as non-binding constraints, increasing factorization cost for no benefit.
4. **O(1) per-cut activity status.** The activity bitmap in the cut pool ([Cut Management Implementation](./cut-management-impl.md) SS1.1) already provides O(1) lookup for whether a cut is active, so filtering during `CutBatch` assembly adds negligible overhead.

**CutBatch assembly implications.** The `CutBatch` struct ([Solver Interface Trait](./solver-interface-trait.md) SS4.5) contains only active cuts. Assembly iterates the activity bitmap and copies only active cut data (intercepts and dense coefficient rows) into the CSR arrays. The assembly cost is $O(n_\text{active} \times n_\text{state})$, not $O(n_\text{total} \times n_\text{state})$. At production scale ($n_\text{state} = 2{,}080$), the per-cut assembly cost is dominated by the coefficient copy (2,081 doubles per cut row including the $\theta$ column), so the active-only filtering provides a directly proportional speedup relative to the active ratio.

**Presolve interaction note.** With selective addition, presolve can remain disabled as required for warm-start basis compatibility. If bulk loading were adopted in a future optimization pass, the presolve interaction would need re-evaluation: presolve would be required to eliminate inactive rows, but enabling presolve may invalidate the warm-start basis, creating a tension that selective addition avoids entirely.

**Deferred optimization: bulk loading with bound deactivation.** An alternative approach loads all cuts (active + inactive) in a single bulk operation and "deactivates" inactive cuts by setting their lower bound to $-\infty$ (making the constraint non-binding). This simplifies the cut loading logic (one bulk call, no bitmap filtering) and may interact favorably with solver-internal optimizations, but increases the nominal LP size and depends on presolve being effective. This alternative is deferred to a later optimization pass, contingent on benchmarking with realistic cut pool sizes and resolution of the presolve/warm-start tension.

> Selective cut addition is the adopted baseline. Under StageLpCache (SS11.4), active cuts are pre-assembled into the per-stage CSC rather than added at each stage transition via `addRows`.

**Cut deactivation**: Under the rebuild strategy, cuts are not deactivated in the solver LP — they simply aren't included in the next rebuild's `addRows` call. Deactivation happens in the cut pool (the activity bitmap is updated), and the next LP rebuild naturally excludes deactivated cuts.

## 6. Error Categories

> **Decision [DEC-015](../overview/decision-log.md#dec-015) (active):** `SolverError` hard-stop vs proceed-with-partial mapping: `Infeasible`, `Unbounded`, `InternalError` are hard-stops; `NumericalDifficulty`, `TimeLimitExceeded`, `IterationLimit` permit partial results.

The solver interface returns a categorized error when a solve fails. The calling algorithm uses the error category to determine its response. Solver-internal errors (e.g., factorization failures) are resolved by the retry logic (SS7) before reaching this level.

| Error Category       | Meaning                                     | SDDP Response                                               | May Have Partial Solution |
| -------------------- | ------------------------------------------- | ----------------------------------------------------------- | :-----------------------: |
| Infeasible           | No feasible solution exists                 | Data error — check bounds, constraints. Hard stop.          |            No             |
| Unbounded            | Objective is unbounded below                | Modeling error — check objective signs. Hard stop.          |            No             |
| Numerical difficulty | Solver retries exhausted without resolution | Log warning; may proceed with partial solution if available |            Yes            |
| Time limit exceeded  | Per-solve time budget exhausted             | Log warning; may proceed with best solution found           |            Yes            |
| Iteration limit      | Solver iteration limit hit                  | Log warning; may proceed with best solution found           |            Yes            |
| Internal error       | Unrecoverable solver-internal failure       | Log and hard stop                                           |            No             |

**Infeasibility diagnostics**: When available, the solver provides an infeasibility ray (proof of infeasibility) or unbounded direction for debugging.

**Numerical recovery hints**: When a numerical difficulty error is returned, the solver may suggest recovery actions: apply scaling, try cold start, tighten tolerances, or report that the problem is inherently ill-conditioned.

## 7. Retry Logic Contract

Retry logic is **encapsulated within each solver implementation**. The calling algorithm never sees retry details — it only receives the final result.

### 7.1 Behavioral Contract

Each solver implementation defines its own retry sequence based on its failure modes. The retry logic must satisfy these behavioral requirements:

| Requirement           | Description                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| Maximum attempts      | A configurable upper bound on retry attempts (default: 5)                                         |
| Time budget           | A configurable wall-clock budget for all attempts combined                                        |
| Escalating strategies | Retries should escalate from least-disruptive (clear basis) to most-disruptive (switch algorithm) |
| Final disposition     | After exhausting all retries, return a terminal error with the best partial solution if available |
| Logging               | Each retry attempt is logged at debug level with the strategy used and outcome                    |

### 7.2 Typical Retry Escalation

The following is the general pattern; solver-specific details are in the respective implementation specs:

1. Clear warm-start basis (cold start)
2. Disable presolve
3. Switch to alternative algorithm (e.g., interior point instead of simplex)
4. Relax solver tolerances

For solver-specific retry strategies, see [HiGHS Implementation](./solver-highs-impl.md) and [CLP Implementation](./solver-clp-impl.md).

## 8. Dual Variable Normalization

Different LP solvers report dual multipliers with different sign conventions. The solver implementation **must** normalize duals to the canonical form before returning to the SDDP algorithm.

**Canonical sign convention:**

| Constraint Form                | Positive Dual Means                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| $\mathbf{a}'\mathbf{x} \leq b$ | Increasing RHS $b$ increases objective (shadow price $\partial z^*/\partial b > 0$) |
| $\mathbf{a}'\mathbf{x} \geq b$ | Sign flipped relative to $\leq$ form                                                |
| $\mathbf{a}'\mathbf{x} = b$    | Unrestricted sign                                                                   |

This normalization is critical for correct cut coefficient computation. A sign error in duals produces cuts that point in the wrong direction, leading to divergence of the algorithm.

## 9. Basis Storage for Warm-Starting

The solver interface supports saving and restoring a simplex basis for warm-starting subsequent solves. The basis captures which variables and constraints are basic (in the basis), at lower bound, at upper bound, free, or fixed.

**Basis status values:**

| Status   | Meaning                                   |
| -------- | ----------------------------------------- |
| At lower | Variable is at its lower bound            |
| Basic    | Variable is in the basis (between bounds) |
| At upper | Variable is at its upper bound            |
| Free     | Variable is free (superbasic)             |
| Fixed    | Variable is fixed (lower = upper)         |

A basis consists of one status value per column (variable) and one per row (constraint).

**Key design decisions:**

- Bases are stored in the **original problem space** (not presolved), ensuring portability across solver versions and presolve strategies
- The forward pass basis at each stage is retained for warm-starting the backward pass at the same stage (see [Training Loop SS4.4](./training-loop.md))
- Basis storage uses compact representation (one byte per variable/constraint)
- Basis structure splits naturally at the dynamic constraint boundary per SS2.3: static rows are position-stable, dynamic constraint rows are appended/truncated

## 10. Compile-Time Solver Selection

> **Decision [DEC-005](../overview/decision-log.md#dec-005) (active):** Compile-time solver selection via Cargo feature flags; exactly one solver active per binary; HiGHS and CLP are first-class reference implementations.

The active solver implementation is selected at compile time via Cargo feature flags. Only one solver is active per build. This design:

- **Eliminates virtual dispatch** — The concrete solver type is known at compile time, enabling inlining and monomorphization
- **Simplifies the hot path** — No trait object indirection on the millions-of-solves critical path
- **Supports multiple backends** — HiGHS and CLP (open-source reference implementations), CPLEX and Gurobi (commercial) can be selected without code changes

HiGHS and CLP are first-class reference implementations — the solver abstraction is designed and tested against both. Commercial solvers (CPLEX, Gurobi) are compile-time selectable but are not primary validation targets.

## 11. Stage LP Template and Rebuild Strategy

### 11.1 Pre-Assembled Stage LP Templates

**Crate ownership:** `StageTemplate` and `StageIndexer` belong to `cobre-solver`. They encode LP-specific structure (variable counts, constraint counts, coefficient offsets) derived from the `System` struct during solver initialization. The temporal structure (`Stage`, `Block`, `PolicyGraph`) lives in `cobre-core`; the LP structure derived from it lives in `cobre-solver`. Construction of `StageTemplate` is performed by `cobre-sddp`, which calls a builder function that takes a reference to the resolved `System` and produces the populated struct (see [Solver Interface Trait SS4.4](./solver-interface-trait.md) for the construction signature). `cobre-solver` receives `StageTemplate` as an opaque data holder and bulk-loads its CSC arrays into the underlying LP solver without interpreting column or row semantics. `StageIndexer` (see [Training Loop SS5.5](./training-loop.md)) is likewise defined in `cobre-solver` and shared read-only across all threads and ranks.

At initialization, each stage's structural LP (matrix, bounds, objective — everything except cuts and scenario-dependent RHS) is assembled once into a canonical CSC (column-major) representation called the **stage template**. This template:

- Is built from the resolved internal structures (see [Internal Structures](../data-model/internal-structures.md)) with full clarity and correctness — this is the initialization phase where data clarity matters more than performance
- Follows the column and row layout convention defined in SS2
- Stores the CSC arrays (`col_starts`, `row_indices`, `values`, `col_lower`, `col_upper`, `row_lower`, `row_upper`, `objective`) in solver-ready format — CSC is used because both reference solvers (HiGHS and CLP) internally store LP matrices in column-major format; passing CSC avoids per-stage-transition transposition overhead
- Is shared read-only across all threads within an MPI rank

### 11.2 LP Rebuild Strategy

**Adopted decision: StageLpCache — pre-assembled complete LPs per stage.**

The StageLpCache (SS11.4) holds a complete pre-assembled LP per stage in CSC format (structural constraints + active Benders cuts + bounds + objective). At each stage transition, each thread:

1. **Load StageLpCache** — Bulk-load `StageLpCache[t]` into the solver via `passModel`/`loadProblem`. The CSC contains the structural template plus all active cuts — no `addRows` call is needed. Since the LP is already in CSC form (the native internal format for both HiGHS and CLP), this is a fast sequential bulk read.
2. **Patch scenario values** — Update the ~2,240 scenario-dependent row bounds (incoming storage as RHS of water balance, AR lag fixing constraint RHS, current-stage noise fixing) via `patch_row_bounds` ([Solver Interface Trait SS2.3](./solver-interface-trait.md)).
3. **Warm-start** — Apply the cached basis from the previous iteration's solve at this stage.
4. **Solve** — Solve the LP.

**Between-iterations update** (off the critical path, leader-only SharedRegion write):

1. Cut selection determines active/inactive cut sets per stage — stages are distributed across ranks and threads for parallel evaluation ([Cut Selection Strategy Trait SS2.2a](./cut-selection-trait.md)), with DeactivationSets gathered via `allgatherv` ([Synchronization §1.4a](../hpc/synchronization.md))
2. For each stage: leader rank updates the StageLpCache CSC — insert new cut coefficients, adjust row bounds for deactivated cuts (set lower bound to $-\infty$ to make constraint non-binding)
3. Write updated StageLpCache to SharedRegion (leader rank, `fence()` + barrier ensures visibility)

> The prior per-thread LP rebuild approach is documented in [Binary Formats SS3](../data-model/binary-formats.md) and [Solver Workspaces SS1.10](./solver-workspaces.md).

### 11.3 Solver-Specific Optimization Paths

The StageLpCache is the **adopted baseline** for cut in-memory management, working identically across all solver backends. The following table tracks the status of solver-specific optimizations:

| Optimization                | Solver | Mechanism                                                                                                                                                                         | Status                                                       |
| --------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Pre-assembled CSC templates | All    | Stage template in solver-ready CSC → fast `passModel`/`loadProblem` with no format transposition                                                                                  | **Adopted** (SS11.1)                                         |
| StageLpCache                | All    | Complete per-stage CSC (template + active cuts) via `SharedRegion<T>`, loaded via `passModel`                                                                                     | **Adopted** (SS11.4)                                         |
| Direct memory patching      | CLP    | Mutable `double*` pointers (`Clp_rowLower()`/`Clp_rowUpper()`, `Clp_colLower()`/`Clp_colUpper()`) for zero-copy in-place bound patching via `patch_row_bounds`/`patch_col_bounds` | Anticipated — see [CLP Implementation](./solver-clp-impl.md) |

These optimizations do not change the interface contract (SS4) — they are internal to each solver implementation.

### 11.4 StageLpCache Design

The StageLpCache is a complete pre-assembled LP per stage in CSC format. It combines structural constraints (from the StageTemplate) with active Benders cuts (15K pre-allocated slots) plus column/row bounds and objective, into a single solver-ready CSC matrix.

**Structure**: The CSC is constructed once at initialization from the structural template plus empty cut slots. Cut slots are filled as cuts arrive between iterations. Each cut slot occupies dense entries in the $n_{state}$ state columns plus the $\theta$ column (2,081 entries per cut). Structural rows are interleaved in the same column arrays.

**Sizing**: Cut CSC data dominates (92–99% per stage at capacity):

| Component                           | Per Stage                                                       |
| ----------------------------------- | --------------------------------------------------------------- |
| Cut nnz (at 15K capacity)           | 15,000 × 2,081 = 31.2M entries                                  |
| CSC cut data (row_indices + values) | 31.2M × 4 (row idx) + 31.2M × 8 (value) ≈ **375 MB**            |
| Structural CSC (template portion)   | ~5 MB (from StageTemplate)                                      |
| Bounds + objective                  | ~1.3 MB (col_lower, col_upper, row_lower, row_upper, objective) |
| **Mid/late stage total**            | **~378 MB**                                                     |
| **Early stage total** (larger LP)   | ~407 MB (more structural rows from FPHA constraints)            |
| **60 stages (all at capacity)**     | **~22.3 GB**                                                    |

**Ownership**: A single shared copy across all ranks on the same node via `SharedRegion<T>` with NUMA-interleaved allocation (`mbind(MPOL_INTERLEAVE)` — see [Shared Memory Aggregation §1](../hpc/shared-memory-aggregation.md)). All ranks read the same data during passes.

**Update contract**: SharedRegion writes are performed by the leader rank between iterations; cut selection computation is distributed across all ranks ([Cut Management Implementation SS7](./cut-management-impl.md)). Per-iteration update: ~200 new cuts × 59 stages × 2,081 × 12 bytes ≈ 300 MB written → ~5 ms at DRAM bandwidth. New cuts are written into pre-allocated CSC slots; deactivated cuts have their row lower bound set to $-\infty$ (making the constraint non-binding without structural modification). Readers observe updates only after `fence()` + barrier.

**Read contract**: Read-only during forward and backward passes. `passModel(StageLpCache[t])` = sequential bulk read of ~378 MB at ~44 GB/s (NUMA-interleaved across 4 domains) → **~8.6 ms** per stage transition.

> **Decision [DEC-001](../overview/decision-log.md#dec-001) (active):** StageLpCache as LP construction baseline: one complete pre-assembled LP per stage in CSC format loaded via `passModel`/`loadProblem`, replacing the previous per-thread memory model and reducing node-wide memory from ~91.8 GB to ~22.3 GB.

With the StageLpCache (via SharedRegion):

| Component                   |       Size | Sharing         |
| --------------------------- | ---------: | --------------- |
| StageLpCache                | ~22,300 MB | 1 copy per node |
| Cut metadata                |     ~12 MB | 1 copy per node |
| Opening tree                |    ~0.8 MB | 1 copy per node |
| Input case data             |     ~20 MB | 1 copy per node |
| Thread-local workspaces     |  ~1,737 MB | Per rank        |
| Forward state + MPI buffers |     ~28 MB | Per rank        |
| **Node total**              |            | **~27.7 GB**    |

Net savings: **~64 GB** per node (91.8 → 27.7 GB).

**Sensitivity**: At 3K active cuts: ~7.6 GB StageLpCache, ~2.9 ms stage transition. At 15K: ~22.3 GB, ~8.6 ms. Cut selection remains the highest-leverage optimization for controlling StageLpCache size and stage transition time.

### 11.5 Within-Stage Incremental Updates

Within the same stage (e.g., multiple backward pass scenarios at stage $t$, or multiple forward passes at the same stage within a batch), only scenario-dependent values change — the structural LP and cuts are identical. In this case, the solver skips step 1 and only performs:

- **Patch scenario values** via `set_row_bounds` (step 2), passing separate `indices`, `lower`, and `upper` slices
- **Solve** with implicit warm-start from the previous solve's basis (step 4)

This is significantly faster than a full stage transition and is the common case in the backward pass.

## Cross-References

- Solver Architecture Decisions — The 5 architectural decisions are documented inline throughout this spec (SS2 LP layout, SS4.3 dual-solver validation, SS5 cut preallocation, SS10 compile-time solver selection, SS11 pre-assembled templates and rebuild strategy)
- [Solver Workspaces & LP Scaling](./solver-workspaces.md) — Thread-local solver infrastructure and LP scaling specification
- [HiGHS Implementation](./solver-highs-impl.md) — HiGHS-specific implementation: retry strategy, batch operations, memory footprint
- [CLP Implementation](./solver-clp-impl.md) — CLP-specific implementation: C++ wrapper strategy, mutable pointer access, cloning optimization path
- [LP Formulation](../math/lp-formulation.md) — Constraint structure that the solver operates on
- [Cut Management](../math/cut-management.md) — How cuts are generated; this spec handles how they are stored in the cut pool and loaded into the solver LP
- [Training Loop](./training-loop.md) — Forward pass (parallel solve) and backward pass (cut addition) that drive solver invocations
- [Binary Formats](../data-model/binary-formats.md) — FlatBuffers schema for cut persistence, cut pool memory layout (SS3.4)
- [Internal Structures](../data-model/internal-structures.md) — Logical in-memory data model from which stage templates are built
- [Hybrid Parallelism](../hpc/hybrid-parallelism.md) — OpenMP threading model that requires thread-local solvers
- [Memory Architecture](../hpc/memory-architecture.md) — NUMA-aware allocation for solver workspaces, StageLpCache memory budget
- [Shared Memory Aggregation](../hpc/shared-memory-aggregation.md) — SharedRegion for NUMA-interleaved StageLpCache allocation
- [Configuration Reference](../configuration/configuration-reference.md) — Solver configuration parameters
