# Communication Patterns

## Purpose

This spec defines the communication patterns used by Cobre during SDDP training through the `Communicator` trait ([Communicator Trait §1](./communicator-trait.md)): the collective operations, their data payloads, wire formats, communication volume analysis, and optimization opportunities. Communication is performed through a pluggable backend selected at compile time (see [Backend Selection](./backend-selection.md)) -- the SDDP training loop is generic over `C: Communicator` and never calls backend-specific APIs directly. This spec details the communication mechanics that [Synchronization](./synchronization.md) defines at the protocol level and [Cut Management Implementation §4](../architecture/cut-management-impl.md) defines for cut wire format.

## 1. Collective Operations

### 1.1 Operations Summary

Cobre uses two distinct collective operations through the `Communicator` trait during SDDP training: `allgatherv` (three call sites) and `broadcast` (one call site). All operations are invoked through `comm: &C` where `C: Communicator` (see [Communicator Trait §3](./communicator-trait.md) for generic parameterization). The communicator is `Send + Sync` to support hybrid communication+Rayon execution.

| Operation    | Communicator Trait Method                             | When                    | Data                          | Frequency                |
| ------------ | ----------------------------------------------------- | ----------------------- | ----------------------------- | ------------------------ |
| `allgatherv` | `comm.allgatherv(&send, &mut recv, &counts, &displs)` | Forward → backward      | Visited states (trial points) | Once per iteration       |
| `allgatherv` | `comm.allgatherv(&send, &mut recv, &counts, &displs)` | Backward stage boundary | New cuts at stage $t$         | Once per stage ($T - 1$) |
| `allgatherv` | `comm.allgatherv(&send, &mut recv, &counts, &displs)` | Post-forward            | Per-scenario cost vectors     | Once per iteration       |
| `broadcast`  | `comm.broadcast(&mut buf, 0)`                         | Post-backward           | Lower bound (1 scalar)        | Once per iteration       |

Additionally, initialization uses standard (non-iterative) collectives:

| Operation   | Communicator Trait Method        | When       | Data                     |
| ----------- | -------------------------------- | ---------- | ------------------------ |
| `broadcast` | `comm.broadcast(&mut buf, root)` | Startup    | Configuration, case data |
| `barrier`   | `comm.barrier()`                 | Checkpoint | Synchronization only     |

For method contracts (preconditions, postconditions, error semantics), see [Communicator Trait §2](./communicator-trait.md).

### 1.2 No Point-to-Point Messaging

The approved architecture uses only collective operations -- no point-to-point (`Send`/`Recv`) communication. The symmetric `allgatherv` pattern ensures all ranks have identical data after each synchronization point, eliminating the need for a master/worker protocol.

## 2. Data Payloads

### 2.1 Trial Point Payload (Forward → Backward)

After the forward pass, each rank contributes its visited states to `allgatherv`. The payload per trial point consists of the state vector:

| Component       | Type    | Size per trial point              |
| --------------- | ------- | --------------------------------- |
| Storage volumes | `[f64]` | $N_{\text{hydro}} \times 8$ bytes |
| AR inflow lags  | `[f64]` | $\sum_h P_h \times 8$ bytes       |
| Stage index     | `u32`   | 4 bytes                           |

At worst-case scale ($N_{\text{hydro}} = 160$, average $P_h = 6$ lags, $M = 192$ trajectories, $T = 120$ stages; note: the production baseline per DEC-009 is $T = 60$ stages — these figures represent the hypothetical maximum):

- State dimension: $160 + 160 \times 6 = 1{,}120$ doubles = 8,960 bytes per trial point
- Trial points per stage: 192
- Total payload: $192 \times 8{,}964 \approx 1.72$ MB per stage, or $1.72 \times 120 \approx 206$ MB for all stages

The `allgatherv` counts and displacements are computed from the contiguous block assignment (see [Work Distribution §3.1](./work-distribution.md)).

### 2.2 Cut Payload (Backward Stage Boundary)

After generating cuts at each backward stage, ranks exchange cuts via `allgatherv`. The wire format is a compact binary representation optimized for bandwidth -- see [Cut Management Implementation §4.2](../architecture/cut-management-impl.md) for the complete specification.

| Field              | Type    | Size (production scale)              |
| ------------------ | ------- | ------------------------------------ |
| Slot index         | `u32`   | 4 bytes                              |
| Iteration          | `u32`   | 4 bytes                              |
| Forward pass index | `u32`   | 4 bytes                              |
| Intercept          | `f64`   | 8 bytes                              |
| Coefficients       | `[f64]` | $D_{\text{state}} \times 8$ bytes    |
| **Total per cut**  |         | **~16,660 bytes** (at $D = 2{,}080$) |

At production scale with $M = 192$ forward passes and $R = 16$ ranks, each rank generates $\lfloor 192/16 \rfloor = 12$ cuts per stage. The `allgatherv` payload is $192 \times 16{,}660 \approx 3.2$ MB per stage.

### 2.3 Convergence Statistics Payload (Post-Forward)

The post-forward synchronization uses `allgatherv` to collect the full per-scenario cost vector from all ranks. Each rank contributes its local `scenario_costs: Vec<f64>` (one `f64` per forward-pass scenario solved by that rank). After `allgatherv`, every rank holds the complete global cost vector in canonical rank order.

| Component          | Type    | Size per rank                     | Purpose                                           |
| ------------------ | ------- | --------------------------------- | ------------------------------------------------- |
| Per-scenario costs | `[f64]` | $N_{\text{local}} \times 8$ bytes | Individual scenario costs for canonical summation |

Total payload: $N_{\text{scenarios}} \times 8$ bytes (e.g., $192 \times 8 = 1{,}536$ bytes at production scale with $M = 192$ forward passes).

**Why `allgatherv` instead of `allreduce` of 3 scalar statistics:** Floating-point addition is non-associative. An `allreduce(Sum)` of partial sums would produce results that vary with the number of ranks and the MPI implementation's reduction tree shape, making the upper bound non-deterministic across rank counts. By gathering the full cost vector and performing canonical-order sequential summation on every rank (iterating `global_costs[0], global_costs[1], ..., global_costs[N-1]` in rank order), all ranks produce bit-identical mean, standard deviation, and 95% confidence interval statistics regardless of rank count. This eliminates floating-point non-associativity as a source of non-determinism in convergence checking.

The statistics are computed locally from the gathered vector:

| Statistic               | Formula                                                  |
| ----------------------- | -------------------------------------------------------- |
| Mean (UB)               | $\bar{c} = \frac{1}{N} \sum_{i=0}^{N-1} c_i$             |
| Standard deviation      | $\sigma = \sqrt{\frac{\sum c_i^2 - N \bar{c}^2}{N - 1}}$ |
| 95% confidence interval | $1.96 \cdot \sigma / \sqrt{N}$                           |

See [Work Distribution §1.4](./work-distribution.md) and [Convergence Monitoring §3](../architecture/convergence-monitoring.md).

The lower bound is evaluated separately after the backward pass. Rank 0 solves the stage-0 LP for all openings and computes the risk-adjusted lower bound; the scalar result is then broadcast to all ranks via `comm.broadcast(&mut [lb], 0)` (8 bytes). See [Training Loop SS4.3b](../architecture/training-loop.md) and [Convergence Monitoring SS3.2](../architecture/convergence-monitoring.md).

## 3. Communication Volume Analysis

### 3.1 Per-Iteration Budget

Worst-case reference configuration: $R = 16$ ranks, $T = 120$ stages (hypothetical maximum; DEC-009 baseline is $T = 60$), $M = 192$ forward passes, $D_{\text{state}} = 2{,}080$.

| Operation                | Per-stage | Per-iteration        | Notes                                                  |
| ------------------------ | --------- | -------------------- | ------------------------------------------------------ |
| Trial point `allgatherv` | --        | ~206 MB (once)       | All stages' visited states at once                     |
| Cut `allgatherv`         | ~3.2 MB   | ~381 MB (119 stages) | Per cut-management-impl.md §4.2                        |
| UB `allgatherv`          | --        | ~1.5 KB (once)       | $M \times 8$ bytes (192 scenario costs, canonical sum) |
| LB `broadcast`           | --        | 8 bytes (once)       | 1 scalar (lower bound, rank 0 to all)                  |
| **Total per iteration**  |           | **~587 MB**          |                                                        |

### 3.2 Bandwidth Requirements

On InfiniBand HDR (200 Gb/s = 25 GB/s):

- 587 MB takes ~23 ms at wire speed
- With protocol overhead (~50%), ~46 ms per iteration
- At 200 iterations total: ~9.2 seconds of communication
- Pure data transfer communication fraction: **< 1%**

On 100 Gbps Ethernet (12.5 GB/s):

- 587 MB takes ~47 ms at wire speed
- With TCP/RDMA overhead (~100%), ~94 ms per iteration
- At 200 iterations: ~18.8 seconds → communication fraction: **~1-2%**

SDDP's communication-to-computation ratio is low. The LP solve time dominates.

> **Pure communication vs. total synchronization**: The fractions above measure pure data transfer time (wire time plus protocol overhead). They do not include load imbalance barrier overhead at per-stage `allgatherv` synchronization points in the backward pass. A first-principles timing model at production scale ($R = 4$, $\tau_{LP} = 25$ ms) confirms pure communication remains well below 1% of iteration time. See [Production Scale Reference §4.6](../overview/production-scale-reference.md) for the complete time budget.

> **Backend note**: These bandwidth estimates assume direct network transfer (MPI or TCP). The shm backend uses shared memory buffers instead of network I/O, with effectively zero transfer latency for intra-node communication. See [Shm Backend §3](./backend-shm.md).

## 4. Persistent Collectives (Ferrompi Backend)

This section describes an optimization specific to the **ferrompi backend** (`FerrompiBackend`). MPI 4.0 persistent collectives allow pre-negotiating communication patterns at initialization and reusing them across iterations. For the complete ferrompi backend specification, see [Ferrompi Backend](./backend-ferrompi.md). Other backends use their own optimization strategies: the TCP backend uses persistent socket connections ([TCP Backend §2](./backend-tcp.md)); the shm backend uses persistent shared memory segments ([Shm Backend §1](./backend-shm.md)).

### 4.1 Optimization Opportunity

MPI 4.0 persistent collectives (`MPI_Allgatherv_init`, `MPI_Allreduce_init`) allow pre-negotiating communication patterns at initialization and reusing them across iterations. This amortizes setup cost over the ~100-200 iterations of an SDDP training run.

| Aspect                  | Standard Collective  | Persistent Collective    |
| ----------------------- | -------------------- | ------------------------ |
| Setup cost per call     | Protocol negotiation | None (pre-negotiated)    |
| Subsequent call latency | Full negotiation     | Reduced                  |
| Buffer requirements     | Any buffer per call  | Fixed buffers at init    |
| ferrompi API            | `comm.allgatherv()`  | `comm.allgatherv_init()` |

### 4.2 Applicability to SDDP

The three collective operations in §1.1 are candidates for persistent collectives:

| Operation                   | Persistent candidate? | Notes                                                                       |
| --------------------------- | --------------------- | --------------------------------------------------------------------------- |
| Cut `allgatherv`            | Yes                   | Same pattern every stage, buffer sizes vary per iteration (cut count grows) |
| UB cost vector `allgatherv` | Yes                   | Fixed $M \times 8$-byte payload, identical every iteration                  |
| Trial point `allgatherv`    | Conditional           | Only if $M$ is fixed across iterations; if adaptive, buffer sizes change    |

> **Implementation note**: Persistent collectives require fixed buffer addresses at initialization. If the cut count per rank varies across iterations (which it may, due to cut selection), the send buffer must be pre-allocated at the maximum expected size. This is consistent with the cut pool preallocation strategy in [Solver Abstraction §5](../architecture/solver-abstraction.md).

### 4.3 Design Decision

Whether to use persistent or standard collectives is an **implementation choice**, not an architectural requirement. The approved synchronization model ([Synchronization §1.1](./synchronization.md)) specifies the collective operations and their semantics but does not mandate persistence. The decision should be based on profiling: if communication accounts for less than 5% of total training time (§3.2), the 5-10x speedup from persistent collectives yields marginal absolute improvement.

## 5. Intra-Node Shared Memory

### 5.1 SharedRegion\<T\>

The `SharedMemoryProvider` trait ([Communicator Trait §4](./communicator-trait.md)) enables ranks on the same physical node to share memory regions without replication. The concrete region type (`SharedRegion<T>`) is backend-specific: `FerrompiRegion<T>` for MPI windows, `ShmRegion<T>` for POSIX shared memory, or `HeapRegion<T>` for the heap fallback (local and TCP backends). See [Communicator Trait §4.2](./communicator-trait.md) for the `SharedRegion<T>` trait definition.

| Capability            | Trait Method                           | Use Case                                          |
| --------------------- | -------------------------------------- | ------------------------------------------------- |
| Region creation       | `provider.create_shared_region(count)` | Allocate shared region on intra-node communicator |
| Intra-node grouping   | `provider.split_local()`               | Identify co-located ranks                         |
| Read access           | `region.as_slice()`                    | Zero-copy reads from shared region                |
| Write synchronization | `region.fence()`                       | Ensure visibility of writes across ranks          |

### 5.2 Shared Data Candidates

| Data Structure         | Per-Rank Size (production) | Shareable? | Rationale                                                      |
| ---------------------- | -------------------------- | ---------- | -------------------------------------------------------------- |
| Scenario noise vectors | Large (opening tree)       | Yes        | Read-only during training, identical across ranks on same node |
| Input case data        | Moderate                   | Yes        | Read-only after initialization                                 |
| Cut pool               | Large (grows each iter)    | Partial    | Read-heavy in forward pass, written at stage boundaries only   |
| Solver workspace       | Per-thread                 | No         | Thread-local mutable state, must not be shared                 |

The memory savings from `SharedRegion<T>` are quantified in [Memory Architecture](./memory-architecture.md).

> **Design point**: The extent to which `SharedRegion<T>` is used for the cut pool depends on the access pattern analysis in [Shared Memory Aggregation](./shared-memory-aggregation.md). The baseline approach (each rank maintains its own cut pool, synchronized via `allgatherv`) is simple and correct; shared memory is an optimization to reduce memory footprint on memory-constrained nodes.

## 6. Deterministic Communication

### 6.1 Reproducibility Invariant

All collective operations through the `Communicator` trait in the SDDP training loop are deterministic: given the same inputs and rank count, every rank produces identical results after synchronization. This is critical for the SDDP correctness requirement that all ranks have identical FCFs (see [Cut Management Implementation §4.3](../architecture/cut-management-impl.md)).

The rank-ordered receive semantics are a formal postcondition of `allgatherv` (see [Communicator Trait §2.1](./communicator-trait.md)) that all backends must satisfy.

Determinism sources:

- **Cut slot assignment** -- Computed from `(iteration, forward_pass_index)`, deterministic across all ranks
- **Contiguous block distribution** -- Forward pass scenarios assigned by rank index, reproducible
- **`allgatherv` ordering** -- Receives data in rank order (rank 0, rank 1, ..., rank $R-1$)

### 6.2 Floating-Point Determinism

The post-forward UB synchronization uses `allgatherv` (not `allreduce`) to collect the full per-scenario cost vector, followed by canonical-order sequential summation on every rank. This design eliminates floating-point non-associativity: all ranks iterate the same global cost vector in the same order, producing bit-identical statistics regardless of rank count or MPI implementation. See §2.3 for details.

The lower bound uses `broadcast` from rank 0, which is exact -- rank 0 computes the single authoritative value and distributes it to all ranks.

`allreduce` with `ReduceOp::Sum` is used only for simulation-mode global min/max aggregation, where small floating-point variations are acceptable. Non-MPI backends (TCP, shm) produce deterministic reduction results because they use a fixed coordinator/rank-0 reduction order -- see [TCP Backend §3.2](./backend-tcp.md) and [Shm Backend §3.2](./backend-shm.md).

## Cross-References

- [Communicator Trait §1-§3](./communicator-trait.md) -- Communicator trait definition, method contracts, generic parameterization
- [Communicator Trait §4](./communicator-trait.md) -- SharedMemoryProvider trait, SharedRegion\<T\> type
- [Backend Selection](./backend-selection.md) -- Feature flags, runtime selection, factory pattern
- [Ferrompi Backend](./backend-ferrompi.md) -- FerrompiBackend: MPI delegation, persistent collectives, FerrompiRegion\<T\>
- [TCP Backend](./backend-tcp.md) -- TcpBackend: coordinator pattern, message framing, deterministic reductions
- [Shm Backend](./backend-shm.md) -- ShmBackend: shared buffer protocols, atomic barriers, ShmRegion\<T\>
- [Local Backend](./backend-local.md) -- LocalBackend: identity/no-op operations, HeapRegion\<T\>
- [Synchronization §1.1](./synchronization.md) -- Three collective operations per iteration, their timing and semantics
- [Synchronization §1.4](./synchronization.md) -- Per-stage barrier via `allgatherv` implicit synchronization
- [Work Distribution §1.4](./work-distribution.md) -- Post-forward `allgatherv` with per-scenario cost vector for canonical-order UB computation
- [Work Distribution §2.2](./work-distribution.md) -- Per-stage backward pass execution, `allgatherv` for cuts
- [Work Distribution §3](./work-distribution.md) -- Contiguous block assignment arithmetic, `allgatherv` parameters
- [Cut Management Implementation §4](../architecture/cut-management-impl.md) -- Wire format, deterministic slot assignment, synchronization protocol
- [Hybrid Parallelism §1.2](./hybrid-parallelism.md) -- ferrompi capabilities table, `SharedWindow<T>`, `split_shared()`
- [Convergence Monitoring §3](../architecture/convergence-monitoring.md) -- Cross-rank bound aggregation
- [Training Loop §5.2](../architecture/training-loop.md) -- `allgatherv` for trial point collection
- [Training Loop §6.3](../architecture/training-loop.md) -- `allgatherv` for cut distribution
- [Shared Memory Aggregation](./shared-memory-aggregation.md) -- Intra-node shared memory patterns, hierarchical cut aggregation
- [Memory Architecture](./memory-architecture.md) -- Memory budget, shared memory savings quantification
- [Production Scale Reference §4.6](../overview/production-scale-reference.md) -- Wall-clock time budget, sync overhead in context of total iteration time
