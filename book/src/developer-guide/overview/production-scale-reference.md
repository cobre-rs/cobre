# Production Scale Reference

## Purpose

This spec defines the production-scale dimensions of the Cobre SDDP solver: system sizes, LP variable and constraint counts, state dimension formulas, and performance expectations by scale. It serves as the reference for capacity planning, memory budgeting, and performance regression detection.

## 1. Production Scale Dimensions

Based on the target production scenario:

| Dimension            | Value                      | Memory Impact                          |
| -------------------- | -------------------------- | -------------------------------------- |
| Stages               | 60                         | Graph size                             |
| Blocks per Stage     | 1-24 (varies), typically 3 | LP structure, outputs                  |
| Hydros               | 160                        | State dimension                        |
| Max AR Order         | 12                         | State dimension, variables/constraints |
| Thermals             | 130                        | Variables                              |
| Buses                | 6                          | Variables/constraints                  |
| Lines                | 10                         | Variables/constraints                  |
| Forward Passes       | 192                        | Parallelism                            |
| Openings             | 10                         | Backward pass LP solves                |
| Iterations           | 50                         | Cut pool size                          |
| Simulation Scenarios | 2000                       | Output size                            |

> **Decision [DEC-009](./decision-log.md#dec-009) (active):** 60 stages is the production-scale reference baseline for all capacity planning, memory budgets, and performance targets across the corpus. Some specs reference 120 stages as a theoretical maximum or worst-case bound (e.g., FLOP cost calculations, temporal flattening memory estimates, hypothetical sizing). Such references are explicitly labeled as "worst-case" or "hypothetical maximum" to distinguish from the production baseline.

## 2. State Dimension Estimates

### 2.1 State Variables and Dimension

The **state dimension** determines the size of Benders cuts. State variables include:

| Component     | Count                | Description                                  | Status                           |
| ------------- | -------------------- | -------------------------------------------- | -------------------------------- |
| Storage       | $N_{hydro}$          | End-of-stage reservoir volume for each hydro | Implemented                      |
| AR Lags       | $\sum_{h} P_h$       | Inflow lag values for AR($P_h$) models       | Implemented                      |
| Battery SOC   | $N_{battery}$        | Battery state of charge                      | Deferred ([C.2](../deferred.md)) |
| GNL Committed | $\sum_{gnl} L_{gnl}$ | GNL dispatch pipeline                        | Deferred ([C.1](../deferred.md)) |

**State Dimension Formula**:

$$
N_{state} = N_{hydro} + \sum_{h=1}^{N_{hydro}} P_h + N_{battery} + \sum_{gnl} L_{gnl}
$$

For the current implementation (batteries and GNL deferred, both zero):

$$
N_{state} = N_{hydro} + \sum_{h=1}^{N_{hydro}} P_h
$$

For production scale (160 hydros, AR order up to 12):

- Storage: 160
- AR lags: $160 \times 12 = 1{,}920$ (all hydros at max order)
- **Total: 2,080** (production assumption)

> **Note**: The production assumption is $D_{state} = 2{,}080$, corresponding to all hydros using AR(12). This is the worst-case ceiling and the value used for all capacity planning and performance budgeting in this spec. If most hydros use AR(6), the dimension reduces to $160 + 160 \times 6 = 1{,}120$, which is the optimistic case.

## 3. Variable and Constraint Counts

### 3.1 Variable Count per Subproblem

| Component                   | Formula                                             | Typical Count       |
| --------------------------- | --------------------------------------------------- | ------------------- |
| Future cost                 | $1$                                                 | 1                   |
| Deficit                     | $N_{bus} \times N_{block} \times N_{seg}$           | 6 × 3 × 1 = 18      |
| Excess                      | $N_{bus} \times N_{block}$                          | 6 × 3 = 18          |
| Exchange (direct + reverse) | $2 \times N_{line} \times N_{block}$                | 2 × 10 × 3 = 60     |
| Hydro storage               | $N_{hydro}$                                         | 160                 |
| Hydro incremental inflow AR | $N_{hydro} \times P$                                | 160 × 12 = 1,920    |
| Hydro turbined flow         | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro spillage              | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro generation            | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro inflow (per-block)    | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro diversion             | $N_{div} \times N_{block}$                          | ~10 × 3 = 30        |
| Hydro evaporation           | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro withdrawal            | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480       |
| Hydro slacks                | $N_{hydro} \times N_{block} \times 6$               | 160 × 3 × 6 = 2,880 |
| Thermal generation          | $N_{thermal} \times N_{block} \times \bar{N}_{seg}$ | 130 × 3 × 1 = 390   |
| Contracts                   | $(N_{imp} + N_{exp}) \times N_{block}$              | 5 × 3 = 15          |
| Pumping (flow + power)      | $N_{pump} \times N_{block} \times 2$                | 5 × 3 × 2 = 30      |
| **Total Variables**         |                                                     | **~8,400**          |

### 3.2 Constraint Count per Subproblem

| Component                        | Formula                                             | Typical Count         |
| -------------------------------- | --------------------------------------------------- | --------------------- |
| Load balance                     | $N_{bus} \times N_{block}$                          | 6 × 3 = 18            |
| Hydro water balance              | $N_{hydro}$                                         | 160                   |
| Incremental inflow AR dynamics   | $N_{hydro}$                                         | 160                   |
| Lagged incremental inflow fixing | $N_{hydro} \times P$                                | 160 × 12 = 1,920      |
| Hydro generation (constant)      | $(N_{hydro} - N_{fpha}) \times N_{block}$           | (160 − 50) × 3 = 330  |
| Hydro generation (FPHA)          | $N_{fpha} \times N_{block} \times \bar{M}_{planes}$ | 50 × 3 × 125 = 18,750 |
| Outflow definition               | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480         |
| Outflow bounds (min/max)         | $2 \times N_{hydro} \times N_{block}$               | 2 × 160 × 3 = 960     |
| Turbined min                     | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480         |
| Generation min                   | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480         |
| Evaporation                      | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480         |
| Water withdrawal                 | $N_{hydro} \times N_{block}$                        | 160 × 3 = 480         |
| Generic constraints              | $N_{generic}$                                       | ~50                   |
| **Active constraints**           |                                                     | **~24,750**           |
| **Benders cuts (pre-allocated)** | $N_{cuts}$                                          | 10,000-15,000         |
| **Total Constraints**            |                                                     | **~35,000-40,000**    |

> **Note**: Among active constraints, FPHA hyperplane constraints dominate (~18,750 of ~24,750). The total constraint count including pre-allocated Benders cut slots reaches ~35,000-40,000, but during early iterations most cut constraints are inactive (bounds set to $[-\infty, +\infty]$). See [Solver Abstraction §5](../architecture/solver-abstraction.md) for pre-allocation design.

### 3.3 Counting Formulas (Exact)

For precise sizing, use the following formulas where parameters come from the system configuration. The production-scale values in §3.1 and §3.2 are obtained by substituting the parameters from §1.

**Variables**:

```
N_VAR = 1                                                      # theta
      + N_BUS × N_BLOCK × (AVG_DEF_SEGMENTS + 1)              # deficit + excess
      + 2 × N_LINE × N_BLOCK                                   # exchange
      + N_HYDRO                                                 # storage
      + N_HYDRO × AR_ORDER                                      # incremental inflow model
      + N_HYDRO × N_BLOCK × 4                                   # q, s, g, inflow
      + N_HYDRO_DIV × N_BLOCK                                   # diversion
      + N_HYDRO_EVAP × N_BLOCK                                  # evaporation
      + N_HYDRO_WITHDRAWAL × N_BLOCK                            # withdrawal
      + N_HYDRO × N_BLOCK × N_SLACK_TYPES                       # slack vars
      + N_THERMAL × N_BLOCK × AVG_COST_SEGMENTS                 # thermal
      + (N_CONTRACT_IMP + N_CONTRACT_EXP) × N_BLOCK             # contracts
      + N_PUMP × N_BLOCK × 2                                    # pump flow + power
```

**Constraints**:

```
N_CON = N_BUS × N_BLOCK                                        # load balance
      + N_HYDRO                                                 # water balance
      + N_HYDRO                                                 # inflow AR dynamics
      + N_HYDRO × AR_ORDER                                      # lagged inflow fixing
      + (N_HYDRO - N_HYDRO_FPHA) × N_BLOCK                       # generation (constant)
      + N_HYDRO_FPHA × N_BLOCK × AVG_FPHA_PLANES                # FPHA (additional)
      + N_HYDRO × N_BLOCK                                       # outflow definition
      + N_HYDRO × N_BLOCK × 2                                   # outflow bounds
      + N_HYDRO × N_BLOCK                                       # turbined min
      + N_HYDRO × N_BLOCK                                       # generation min
      + N_HYDRO_EVAP × N_BLOCK                                  # evaporation
      + N_HYDRO_WITHDRAWAL × N_BLOCK                            # withdrawal
      + N_GENERIC                                                # generic constraints
      + N_CUT_CAPACITY                                           # Benders cuts
```

**State Dimension**:

```
N_STATE = N_HYDRO                                               # storage
        + SUM(AR_ORDER[h] for h in HYDROS)                      # AR lags
        # + N_BATTERY  (deferred C.2)
        # + SUM(GNL_LAG[t] for t in GNL_THERMALS)  (deferred C.1)
```

**Production-scale parameter values**: `AVG_DEF_SEGMENTS = 1`, `AVG_COST_SEGMENTS = 1`, `AVG_FPHA_PLANES = 125`, `N_HYDRO_EVAP = N_HYDRO = 160`, `N_HYDRO_WITHDRAWAL = N_HYDRO = 160`, `AR_ORDER = 12` (worst case). All other parameters as in §1.

## 4. Performance Expectations by Scale

> **Note**: The timing targets in §4.2 are engineering targets for each test system scale, serving as upper bounds for performance regression testing. Section §4.6 provides a separate first-principles wall-clock model for the Production configuration using LP solve time KPIs (§4.3) and parallelism parameters. The complete derivation is in the [timing model analysis](../../../plans/spec-consistency-audit/epic-04-wall-clock-time-model/timing-model-analysis.md) audit artifact. All estimates assume fully warm-started steady-state iterations and do not account for I/O, checkpointing, or convergence check overhead.

> **Purpose**: This table provides expected timing targets for different problem scales, enabling performance validation and regression detection. Timings are per-iteration unless otherwise noted.

### 4.1 Hardware Assumptions

| Component | Specification                                         |
| --------- | ----------------------------------------------------- |
| CPU       | AMD EPYC 9R14 or equivalent (192 cores, 3.7 GHz base) |
| Memory    | DDR5, 384 GB/node                                     |
| Network   | InfiniBand HDR (200 Gb/s) or equivalent               |
| Storage   | NVMe SSD for I/O operations                           |

### 4.2 Test Systems

#### System Definition

| Scale          | Stages | Hydros | Thermals | AR Order | Hydro Production | Fwd Passes | Openings |
| -------------- | -----: | -----: | -------: | -------: | ---------------- | ---------: | -------: |
| **Unit Test**  |      4 |      1 |        2 |        0 | Constant         |          2 |       10 |
| **Small**      |     12 |      2 |        4 |        0 | Constant         |          4 |       10 |
| **Medium**     |     24 |      4 |       20 |        0 | Constant         |         12 |       20 |
| **Large**      |     24 |      4 |       20 |        6 | Constant         |         16 |       20 |
| **XLarge**     |     36 |     12 |      130 |       12 | Constant         |         32 |       20 |
| **2XLarge**    |     36 |     12 |      130 |       12 | FPHA (125)       |         32 |       20 |
| **Production** |     60 |    160 |      130 |       12 | FPHA (125)       |        192 |       10 |

#### Hardware Configuration and Performance Targets

| Scale          | Ranks | Threads/Rank | Forward Time | Backward Time | Memory/Rank |
| -------------- | ----: | -----------: | ------------ | ------------- | ----------- |
| **Unit Test**  |     1 |            2 | <0.2s        | <1s           | —           |
| **Small**      |     2 |            2 | <0.3s        | <1.5s         | —           |
| **Medium**     |     2 |            6 | <1s          | <10s          | —           |
| **Large**      |     2 |            8 | <1.2s        | <12s          | —           |
| **XLarge**     |     2 |            8 | <2s          | <20s          | —           |
| **2XLarge**    |     2 |            8 | <3s          | <30s          | —           |
| **Production** |     4 |           48 | <10s         | <90s          | —           |

> **Note**: All timing values are engineering targets based on domain experience. The Production row model-derived estimates from §4.6 are substantially lower than the engineering targets, indicating headroom for I/O, cold-starts, and solver variability. Memory/Rank values are to be calculated using the sizing tool; approximate sizing can be derived from the formulas in §3 and [Memory Architecture §2.1](../hpc/memory-architecture.md).

### 4.3 Key Performance Indicators

| Metric                        | Target  | Measurement Point                    |
| ----------------------------- | ------- | ------------------------------------ |
| LP solve (warm-start)         | ≤25 ms  | Hot-path, ~8,400-variable subproblem |
| LP solve (cold-start)         | ≤250 ms | First solve or basis invalid         |
| RHS batch update              | <100 us | ~25,000 constraint updates           |
| Solution extraction           | <50 us  | Primal + basis to buffers            |
| Cut exchange (MPI_Allgatherv) | <5 ms   | Per-stage, all ranks exchange cuts   |
| Parallel efficiency           | >80%    | At 4 ranks vs. 1 rank                |
| Warm-start hit rate           | >70%    | Forward pass consecutive stages      |

### 4.4 Scaling Expectations

| Dimension       | Scaling Behavior                                                                                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Forward pass    | Time $\propto$ (stages $\times$ LP solve time) / (ranks $\times$ threads). Near-linear speedup.                                                                                           |
| Backward pass   | Time $\propto$ (stages $- 1$) $\times$ openings $\times$ LP solve time. Sequential stage and opening dependency (warm-start). Each rank redundantly solves all openings for cut locality. |
| Communication   | With 4 ranks on InfiniBand, pure data transfer and synchronization are negligible (<0.1% of iteration time). See §4.6 time budget.                                                        |
| Memory per rank | $\approx$ solver workspaces (~36 MB $\times$ threads) + StageLpCache via SharedRegion (~22.3 GB node-wide). See [Memory Architecture §2.1](../hpc/memory-architecture.md)                 |
| Cut pool growth | Logical growth only (pre-allocated slots). Memory stable after initialization.                                                                                                            |

> **Thread utilization at 4 ranks**: The forward and backward passes have different utilization characteristics. **Forward pass**: With $M = 192$ trajectories distributed across $R = 4$ ranks, each rank receives $192 / 4 = 48$ trajectories. With $T_h = 48$ threads per rank, each thread handles exactly 1 trajectory — **100% thread utilization**. **Backward pass**: The $M = 192$ trial states (gathered from the forward pass) are distributed across 4 ranks (48 per rank) and then across 48 threads (1 trial state per thread). Each thread evaluates all $N_{open} = 10$ openings sequentially for its trial state, maintaining warm-start benefit. Per stage: $48 \times 10 = 480$ LP solves per rank, 10 sequential solves per thread — **100% thread utilization**, with sequential depth $N_{open} \times \tau_{LP}$. All 4 ranks are active (each rank independently generates cuts from its assigned trial states). The per-iteration compute time is determined by the sequential depth per stage: $(T-1) \times N_{open} \times \tau_{LP}$ for the backward pass. See the [timing model analysis §8](../../../plans/spec-consistency-audit/epic-04-wall-clock-time-model/timing-model-analysis.md) for the complete utilization comparison.

### 4.5 Convergence Reference

| Problem Type                       | Typical Iterations | Optimality Gap  |
| ---------------------------------- | ------------------ | --------------- |
| Simple (few hydros, short horizon) | 10-20              | <0.1%           |
| Medium (regional system)           | 30-50              | <0.5%           |
| Complex (full national grid)       | 50-100             | <1.0%           |
| With CVaR risk measure             | +20-50% iterations | Same gap target |

> **Note**: Iteration counts assume reasonable initial policy (warm-start from previous study). Cold-start may require 2-3x more iterations. The Production system (60 stages, 160 hydros, AR(12)) falls in the "Complex" category.

### 4.6 Wall-Clock Time Budget

The following per-iteration time budget is derived from a first-principles model at $\tau_{LP} = 25$ ms, $R = 4$ ranks, $T_h = 48$ threads/rank, $D_{state} = 2{,}080$ (worst-case AR(12)), InfiniBand HDR. The complete derivation with all intermediate steps is in the [timing model analysis](../../../plans/spec-consistency-audit/epic-04-wall-clock-time-model/timing-model-analysis.md).

| Component                    | Per-Iteration | Fraction | Category        |
| ---------------------------- | ------------: | -------: | --------------- |
| Forward pass compute         |       1.500 s |    9.07% | Compute         |
| Backward pass compute        |      14.750 s |   89.15% | Compute         |
| Trial point `Allgatherv`     |      <0.001 s |   <0.01% | Communication   |
| Cut exchange (59 stages)     |      <0.001 s |   <0.01% | Communication   |
| Convergence `Allreduce`      |      <0.001 s |   <0.01% | Communication   |
| Barrier overhead (59 stages) |       0.295 s |    1.78% | Synchronization |
| **Per-iteration total**      |  **16.545 s** | **100%** |                 |

**Forward pass**: $T \times \lceil M / (R \times T_h) \rceil \times \tau_{LP} = 60 \times 1 \times 25 = 1{,}500$ ms. Each rank receives $192/4 = 48$ trajectories, fully utilizing 48 threads.

**Backward pass**: $(T - 1) \times N_{open} \times \tau_{LP} = 59 \times 10 \times 25 = 14{,}750$ ms. Each rank sequentially solves all 10 openings per stage for warm-start benefit.

**50-iteration projection**:

| Metric                 |     Value | Notes                                           |
| ---------------------- | --------: | ----------------------------------------------- |
| 50-iteration total     |   827.3 s | $50 \times 16.545$                              |
| Wall-clock budget      | 7,200.0 s | 2-hour operational requirement                  |
| Budget fraction        |     11.5% | Headroom: 6,373 s (88.5%)                       |
| Headroom available for |           | I/O, checkpointing, cold-starts, cut management |

#### Sensitivity: Critical LP Solve Time

The model is linear in $\tau_{LP}$. LP solves per iteration: $60 + 59 \times 10 = 650$. Including barrier overhead (2% of backward compute), the LP-equivalent coefficient is $650 + 11.8 = 661.8$. The total 50-iteration compute time is:

$$
T_{total}^{full} \approx 50 \times 661.8 \times \tau_{LP} = 33{,}090 \times \tau_{LP} \text{ seconds}
$$

Setting $T_{total}^{full} = 7{,}200$ s and solving for the critical LP solve time:

$$
\tau_{LP}^{crit} = \frac{7{,}200}{33{,}090} \approx 217.6 \text{ ms}
$$

If the effective LP solve time exceeds approximately **218 ms**, the 50-iteration budget is violated. With 60 stages, 10 openings, and 4 ranks, the solver has substantial headroom — the critical threshold is nearly 9x the baseline assumption.

| Scenario                | $\tau_{LP}$ | 50-iter total |  Budget % | Verdict             |
| ----------------------- | ----------: | ------------: | --------: | ------------------- |
| Optimistic warm-start   |       10 ms |        ~331 s |      4.6% | Well under budget   |
| **Baseline (spec KPI)** |   **25 ms** |    **~827 s** | **11.5%** | **Under budget**    |
| Moderate degradation    |       50 ms |      ~1,655 s |     23.0% | Under budget        |
| Elevated solve time     |      100 ms |      ~3,309 s |     46.0% | Under budget        |
| Heavy degradation       |      150 ms |      ~4,964 s |     68.9% | Under budget        |
| **Critical threshold**  | **~218 ms** |  **~7,200 s** |  **100%** | **At budget limit** |

#### Sensitivity: Backward Pass Warm-Start Rate

The backward pass constitutes ~89% of compute. If the backward pass warm-start hit rate $\eta_{ws}^{bwd}$ drops below 100% (the design target from [Work Distribution §2.3](../hpc/work-distribution.md)), LP solve times blend between warm-start ($\tau_{ws} = 25$ ms) and cold-start ($\tau_{cs} = 250$ ms):

| $\eta_{ws}^{bwd}$ | Effective $\tau_{LP}^{bwd}$ | 50-iter total | Budget % | Verdict      |
| ----------------: | --------------------------: | ------------: | -------: | ------------ |
|              100% |                       25 ms |        ~827 s |    11.5% | Under budget |
|               90% |                     47.5 ms |      ~1,504 s |    20.9% | Under budget |
|               80% |                       70 ms |      ~2,181 s |    30.3% | Under budget |
|               70% |                     92.5 ms |      ~2,858 s |    39.7% | Under budget |

With 10 openings and 60 stages, warm-start remains important for performance but is not an existential threat to the 2-hour budget. Even at 70% hit rate, the solver uses ~40% of the budget, leaving 60% headroom for other overhead. The sequential opening evaluation design ([Work Distribution §2.3](../hpc/work-distribution.md)) still targets near-100% warm-start for optimal LP throughput.

### 4.7 Model Assumptions

Every timing model estimate in §4.6 depends on the assumptions below. The table distinguishes spec-defined KPIs (which are targets to be met by the implementation), spec-defined architecture parameters (which are design decisions), and engineering estimates (which need validation).

| Assumption                               | Value      | Source                                                                      | Category             | Sensitivity |
| ---------------------------------------- | ---------- | --------------------------------------------------------------------------- | -------------------- | ----------- |
| LP solve time (warm-start)               | 25 ms      | §4.3 KPI                                                                    | Spec KPI             | **High**    |
| LP solve time (cold-start)               | 250 ms     | §4.3 KPI                                                                    | Spec KPI             | Medium      |
| Forward pass warm-start hit rate         | 70%        | §4.3 KPI                                                                    | Spec KPI             | Low         |
| Backward pass warm-start hit rate        | ~100%      | [Work Distribution §2.3](../hpc/work-distribution.md) (sequential openings) | Spec Architecture    | Low         |
| MPI ranks                                | 4          | §4.2                                                                        | Spec Architecture    | Low         |
| Threads per rank                         | 48         | §4.2                                                                        | Spec Architecture    | Low         |
| Stages                                   | 60         | §1                                                                          | Spec Architecture    | Low         |
| Forward passes                           | 192        | §1                                                                          | Spec Architecture    | Low         |
| Openings                                 | 10         | §1                                                                          | Spec Architecture    | Low         |
| Iterations                               | 50         | §1                                                                          | Spec Architecture    | Low         |
| Backward stages = $T - 1$                | 59         | [Training Loop §6.1](../architecture/training-loop.md)                      | Spec Architecture    | Low         |
| One LP solve per stage per trajectory    | 1          | §3 (blocks within LP, not separate solves)                                  | Spec Architecture    | Low         |
| Sequential opening evaluation per thread | Sequential | [Work Distribution §2.3](../hpc/work-distribution.md)                       | Spec Architecture    | Low         |
| InfiniBand HDR bandwidth                 | 25 GB/s    | §4.1                                                                        | Spec Architecture    | Low         |
| IB protocol overhead factor              | 50%        | [Communication Patterns §3.2](../hpc/communication-patterns.md)             | Engineering Estimate | Low         |
| MPI base latency (InfiniBand)            | 2 us       | Conservative estimate for modern IB HCA                                     | Engineering Estimate | Low         |
| LP solve time standard deviation         | 15%        | [Work Distribution §4.1](../hpc/work-distribution.md) (upper end of 5-15%)  | Engineering Estimate | Low         |
| Load imbalance barrier overhead          | 2%         | Derived from statistical model of per-stage max-of-ranks                    | Engineering Estimate | Low         |

**High-sensitivity parameters**: LP solve time ($\tau_{ws} = 25$ ms) is the primary assumption with material impact on the time budget, though the critical threshold (~218 ms) provides nearly 9x headroom above the baseline. With 10 openings, backward pass warm-start hit rate ($\eta_{ws}^{bwd}$) is a moderate concern — at 70% hit rate, the budget fraction reaches ~40%, still well under the limit but with less headroom than the baseline. Early solver benchmarking should prioritize measuring LP solve time under warm-start conditions with production-scale subproblems (~8,400 variables, ~25,000 active constraints).

## Cross-References

- [Design Principles](./design-principles.md) — Format selection criteria and design goals
- [Notation Conventions](./notation-conventions.md) — Mathematical symbols used in formulas above
- [LP Formulation](../math/lp-formulation.md) — Complete LP subproblem that these dimensions describe
- [SDDP Algorithm](../math/sddp-algorithm.md) — Forward/backward pass structure driving performance targets
- [Solver Abstraction §5](../architecture/solver-abstraction.md) — Cut pool pre-allocation and capacity management
- [Solver Workspaces §1.2](../architecture/solver-workspaces.md) — Per-thread workspace sizing (~36 MB HiGHS, ~21 MB CLP)
- [Solver Abstraction SS11.4](../architecture/solver-abstraction.md) — StageLpCache design and sizing (~22.3 GB SharedRegion)
- [Memory Architecture §2](../hpc/memory-architecture.md) — Node memory budget (~27.7 GB), two-tier model
- [Hybrid Parallelism](../hpc/hybrid-parallelism.md) — MPI+OpenMP architecture for achieving these scaling targets
- [Communication Patterns §3](../hpc/communication-patterns.md) — Communication volume analysis, pure transfer < 0.1%
- [Work Distribution §2](../hpc/work-distribution.md) — Forward/backward pass distribution, thread-trajectory affinity
- [Training Loop §6](../architecture/training-loop.md) — Backward pass stage loop, cut synchronization per stage
- [SLURM Deployment](../hpc/slurm-deployment.md) — Job scripts for the test system scales above
