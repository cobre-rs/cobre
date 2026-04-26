# Backend Testing and Conformance

## Purpose

This spec defines the conformance test suite, interchangeability tests, and performance regression tests for the four `Communicator` backend implementations (local, ferrompi, TCP, shm). The conformance suite verifies that each backend satisfies the method contracts from [Communicator Trait §2](./communicator-trait.md) and the `SharedMemoryProvider` lifecycle from [Communicator Trait §4](./communicator-trait.md). The interchangeability tests verify that swapping backends produces equivalent SDDP results -- the central claim of the backend abstraction architecture. The performance regression tests ensure that the trait abstraction layer introduces no measurable overhead relative to direct API calls.

## 1. Conformance Test Suite

Tests are parameterized by backend and rank count. Each test specifies its name, applicable backends, rank configurations, input scenario, and expected observable behavior (postcondition assertion).

**Test naming convention:** `test_{backend}_{method}_{scenario}` where `{backend}` is one of `local`, `ferrompi`, `tcp`, `shm`, and `{method}` is the `Communicator` or `SharedMemoryProvider` method under test.

**Rank count configurations:** Tests are run with 1, 2, and 4 ranks unless otherwise noted. The local backend always runs with size=1. The ferrompi, TCP, and shm backends run with size=2 and size=4. Size=1 tests on multi-rank backends verify degenerate-case correctness.

### 1.1 allgatherv Conformance

`allgatherv` is the most performance-critical method ([Communicator Trait §2.1](./communicator-trait.md)). The conformance tests verify rank-ordered receive, data integrity, and correct handling of heterogeneous send sizes and boundary cases.

| Test Name                                  | Ranks | Input Scenario                                                                                                                                                                                 | Expected Observable Behavior                                                                                                                                                                   |
| ------------------------------------------ | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_{backend}_allgatherv_heterogeneous`  | 2, 4  | Rank 0 sends 3 elements `[1.0, 2.0, 3.0]`; rank 1 sends 5 elements `[4.0, 5.0, 6.0, 7.0, 8.0]`; remaining ranks send rank-specific element counts. `counts` and `displs` computed accordingly. | `recv` buffer on all ranks contains rank 0's data at `displs[0]`, rank 1's data at `displs[1]`, ..., rank $R-1$'s data at `displs[R-1]`. All ranks have bit-for-bit identical `recv` contents. |
| `test_{backend}_allgatherv_identity_size1` | 1     | Single rank sends `[10.0, 20.0, 30.0]`; `counts=[3]`, `displs=[0]`.                                                                                                                            | `recv[0..3] == send[0..3]`. Identity copy semantics per [Local Backend §2.1](./backend-local.md).                                                                                              |
| `test_{backend}_allgatherv_empty_send`     | 2, 4  | One rank sends 0 elements (`counts[r]=0` for that rank); other ranks send non-empty data.                                                                                                      | `recv` buffer is populated correctly for all ranks. The zero-count rank's region in `recv` is untouched. All ranks have identical `recv` contents.                                             |
| `test_{backend}_allgatherv_single_element` | 2, 4  | Each rank sends exactly 1 element (its rank index as `f64`). `counts` is all 1s.                                                                                                               | `recv` contains `[0.0, 1.0, ..., (R-1) as f64]` on all ranks.                                                                                                                                  |
| `test_{backend}_allgatherv_large_payload`  | 2     | Each rank sends 100,000 `f64` elements initialized to `rank as f64`.                                                                                                                           | `recv` contains correct data for both ranks. Validates that shared buffer sizing and framing handle large payloads.                                                                            |

### 1.2 allreduce Conformance

`allreduce` aggregates convergence statistics ([Communicator Trait §2.2](./communicator-trait.md)). The conformance tests verify both `ReduceOp::Sum` and `ReduceOp::Min` correctness and identity semantics for size=1.

| Test Name                                 | Ranks | Input Scenario                                                                             | Expected Observable Behavior                                                                   |
| ----------------------------------------- | ----- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `test_{backend}_allreduce_sum`            | 2, 4  | Each rank sends `[1.0, 2.0, 3.0, 4.0]` (the SDDP convergence statistics shape: 4 x `f64`). | `recv == [R * 1.0, R * 2.0, R * 3.0, R * 4.0]` on all ranks, where $R$ is the number of ranks. |
| `test_{backend}_allreduce_min`            | 2, 4  | Rank $r$ sends `[r as f64, (R - r) as f64]`.                                               | `recv == [0.0, 1.0]` on all ranks (element-wise minimum).                                      |
| `test_{backend}_allreduce_identity_size1` | 1     | Single rank sends `[42.0, 99.0]` with `ReduceOp::Sum`.                                     | `recv == [42.0, 99.0]`. Reduction of a single operand is the identity.                         |
| `test_{backend}_allreduce_single_element` | 2, 4  | Each rank sends `[rank as f64]` (single-element buffer) with `ReduceOp::Sum`.              | `recv == [sum(0..R) as f64]` on all ranks.                                                     |
| `test_{backend}_allreduce_max`            | 2, 4  | Rank $r$ sends `[r as f64, (R - 1 - r) as f64]`. `ReduceOp::Max`.                          | `recv == [(R-1) as f64, (R-1) as f64]` on all ranks.                                           |

### 1.3 broadcast Conformance

`broadcast` distributes initialization data ([Communicator Trait §2.3](./communicator-trait.md)). The conformance tests verify data integrity with different root ranks.

| Test Name                                 | Ranks | Input Scenario                                                                                   | Expected Observable Behavior                                                                                   |
| ----------------------------------------- | ----- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `test_{backend}_broadcast_root0`          | 2, 4  | Root=0. Rank 0's `buf` contains `[3.14, 2.72, 1.41]`. Other ranks' `buf` is zero-initialized.    | After broadcast, all ranks have `buf == [3.14, 2.72, 1.41]`.                                                   |
| `test_{backend}_broadcast_root_last`      | 2, 4  | Root=$R-1$. Root rank's `buf` contains `[100.0, 200.0]`. Other ranks' `buf` is zero-initialized. | After broadcast, all ranks have `buf == [100.0, 200.0]`.                                                       |
| `test_{backend}_broadcast_data_integrity` | 4     | Root=0. `buf` contains 10,000 `f64` elements with a known pattern (`buf[i] = i as f64`).         | All ranks have identical `buf` contents after broadcast. Element-by-element comparison confirms no corruption. |

### 1.4 barrier Conformance

`barrier` provides global synchronization ([Communicator Trait §2.4](./communicator-trait.md)). The conformance test verifies the all-ranks-must-enter semantics.

| Test Name                                        | Ranks | Input Scenario                                                                                                                                      | Expected Observable Behavior                                                                                                                                                                                   |
| ------------------------------------------------ | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_{backend}_barrier_write_before_read_after` | 2, 4  | Each rank writes its rank index to a shared location (shared memory buffer or file), then calls `barrier()`, then reads all other ranks' locations. | After barrier returns, all ranks can observe writes from all other ranks. No rank reads a value that was not yet written. This is the canonical write-before-barrier, read-after-barrier verification pattern. |
| `test_{backend}_barrier_repeated`                | 2, 4  | Ranks execute three consecutive barriers.                                                                                                           | All three barriers complete without deadlock. Verifies correct barrier reset between consecutive calls (generation counter correctness per [Shared Memory Backend §1.4](./backend-shm.md)).                    |

### 1.5 rank and size Conformance

`rank()` and `size()` are infallible accessors ([Communicator Trait §2.5](./communicator-trait.md)). The conformance tests verify the consistency properties across all ranks.

| Test Name                        | Ranks   | Input Scenario                                               | Expected Observable Behavior                                                    |
| -------------------------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `test_{backend}_rank_in_range`   | 1, 2, 4 | Each rank calls `rank()` and `size()`.                       | `rank()` returns a value in `0..size()` on every rank.                          |
| `test_{backend}_size_consistent` | 2, 4    | All ranks call `size()` and broadcast their result.          | All ranks report the same `size()` value.                                       |
| `test_{backend}_rank_unique`     | 2, 4    | All ranks call `rank()` and gather results via `allgatherv`. | The collected rank values form a permutation of `0..size()` with no duplicates. |

### 1.6 Compound Conformance: Collective Sequencing

This test verifies that consecutive collective operations do not interfere with each other -- a critical property for the SDDP training loop which interleaves `allgatherv`, `allreduce`, and `barrier` calls across iterations.

| Test Name                            | Ranks | Input Scenario                                                                                                                                                          | Expected Observable Behavior                                                                                                                                                                                                                 |
| ------------------------------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_{backend}_collective_sequence` | 2, 4  | Execute in order: `allgatherv` (each rank sends rank-specific data), `allreduce` with `ReduceOp::Sum`, `barrier`, second `allgatherv` (each rank sends different data). | Each operation returns correct results independently. The second `allgatherv` does not contain stale data from the first. Verifies collective sequencing (shm: `collective_seq` counter per [Shared Memory Backend §3.5](./backend-shm.md)). |

### 1.7 SharedMemoryProvider Lifecycle Tests

These tests verify the `SharedMemoryProvider` trait methods and the `SharedRegion<T>` lifecycle defined in [Communicator Trait §4](./communicator-trait.md). All four backends implement `SharedMemoryProvider` (ferrompi and shm with true shared memory; local and TCP with `HeapFallback` per [Communicator Trait §4.4](./communicator-trait.md)).

| Test Name                                | Ranks | Input Scenario                                                                                                                                                                | Expected Observable Behavior                                                                                                                                                                                                                                                         |
| ---------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `test_{backend}_shared_region_lifecycle` | 2, 4  | Leader calls `create_shared_region::<f64>(100)`, writes `[0.0, 1.0, ..., 99.0]` via `as_mut_slice()`, calls `fence()`. Follower calls `fence()`, then reads via `as_slice()`. | Follower reads exactly `[0.0, 1.0, ..., 99.0]`. Verifies the allocation -> population -> fence -> read lifecycle from [Communicator Trait §4.2](./communicator-trait.md).                                                                                                            |
| `test_{backend}_shared_region_is_leader` | 2, 4  | All ranks call `is_leader()` and gather results.                                                                                                                              | For ferrompi and shm: exactly one rank (local rank 0) returns `true`; all others return `false`. For local and TCP (`HeapFallback`): all ranks return `true` per [Communicator Trait §4.4](./communicator-trait.md).                                                                 |
| `test_{backend}_split_local_rank_size`   | 2, 4  | All ranks call `split_local()` and query `rank()` and `size()` on the returned communicator.                                                                                  | For ferrompi and shm: the returned communicator has `size()` equal to the number of co-located ranks and `rank()` in `0..size()`. For local and TCP: the returned communicator has `size() == 1` and `rank() == 0` (per [Communicator Trait §4.4](./communicator-trait.md)).         |
| `test_{backend}_shared_region_drop`      | 2     | Create a `SharedRegion<f64>`, populate it, read it, then drop it.                                                                                                             | No resource leaks. For ferrompi: `MPI_Win_free` called on drop. For shm: `shm_unlink` + `munmap` called on drop. For `HeapFallback`: `Vec<T>` dropped. Verified by running under a resource leak detector or by checking that a subsequent region with the same name can be created. |

### 1.8 Error-Case Tests (by CommError Variant)

These tests verify that each backend correctly reports errors via the `CommError` variants defined in [Communicator Trait §1.4](./communicator-trait.md) and [Communicator Trait §4.6](./communicator-trait.md). Error-case behavior varies significantly by backend.

#### 1.8.1 CommError::InvalidBufferSize

| Test Name                                  | Ranks | Input Scenario                                                                                                                                           | Expected Observable Behavior                                                                                                                                                                              |
| ------------------------------------------ | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_{backend}_allreduce_buffer_mismatch` | 2     | `send.len() = 4`, `recv.len() = 3` (mismatch violating [Communicator Trait §2.2](./communicator-trait.md) precondition).                                 | Returns `Err(CommError::InvalidBufferSize { operation: "allreduce", expected: 4, actual: 3 })` on all backends (local, ferrompi, tcp, shm). All backends validate buffer size preconditions (see §1.9.1). |
| `test_{backend}_allgatherv_recv_too_small` | 2     | `recv.len()` is smaller than `displs[R-1] + counts[R-1]` (receive buffer too small per [Communicator Trait §2.1](./communicator-trait.md) precondition). | Returns `Err(CommError::InvalidBufferSize { ... })` on all backends that validate buffer sizes (local, ferrompi, tcp, shm).                                                                               |

#### 1.8.2 CommError::InvalidRoot

| Test Name                               | Ranks | Input Scenario                                | Expected Observable Behavior                                                                                                                            |
| --------------------------------------- | ----- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_{backend}_broadcast_invalid_root` | 2     | `broadcast(buf, root=5)` where `size() == 2`. | Returns `Err(CommError::InvalidRoot { root: 5, size: 2 })` on all backends (local, ferrompi, tcp, shm). All backends validate root values (see §1.9.1). |

#### 1.8.3 CommError::CollectiveFailed

| Test Name                  | Ranks | Input Scenario                                                                                                                               | Expected Observable Behavior                                                                                                                                                                                                                                            |
| -------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_tcp_barrier_timeout` | 2     | One rank calls `barrier()`; the other rank does not (simulating a crash). `COBRE_TCP_TIMEOUT_SECS=2`.                                        | The calling rank receives `Err(CommError::CollectiveFailed { operation: "barrier", mpi_error_code: 0, message: ... })` within approximately 2 seconds. The `mpi_error_code` is 0 per [TCP Backend §5.3](./backend-tcp.md).                                              |
| `test_shm_barrier_timeout` | 2     | One rank calls `barrier()`; the other rank never reaches the barrier (process exits). The shm backend's barrier timeout is set to 2 seconds. | The waiting rank receives `Err(CommError::CollectiveFailed { operation: "barrier", ... })` within approximately 2 seconds. Verifies that the generation counter mechanism ([Shared Memory Backend §1.4](./backend-shm.md)) does not block indefinitely on rank failure. |

#### 1.8.4 CommError::InvalidCommunicator

| Test Name                              | Ranks | Input Scenario                                                                                           | Expected Observable Behavior                                                                                                                                                                    |
| -------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_ferrompi_finalized_communicator` | 2     | Attempt to call `allgatherv` after `MPI_Finalize` has been called (communicator is in an invalid state). | Returns `Err(CommError::InvalidCommunicator)`. This error is MPI-specific per [Communicator Trait §1.4](./communicator-trait.md); the local, TCP, and shm backends do not produce this variant. |

#### 1.8.5 CommError::AllocationFailed

| Test Name                                     | Ranks | Input Scenario                                                                           | Expected Observable Behavior                                                                                                                                                                                                                                                                                   |
| --------------------------------------------- | ----- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_{backend}_shared_region_excessive_size` | 1     | Call `create_shared_region::<f64>(usize::MAX / 2)` (request exceeding available memory). | Returns `Err(CommError::AllocationFailed { requested_bytes: ..., message: ... })` for true shared memory backends (ferrompi, shm). `HeapFallback` backends (local, TCP) follow Rust's standard allocation failure semantics (abort on OOM by default), per [Communicator Trait §4.6](./communicator-trait.md). |

### 1.9 Backend-Specific Conformance Notes

#### 1.9.1 Local Backend

The local backend validates preconditions and returns errors on violations: it returns `Err(CommError::InvalidBufferSize)` when buffer sizes do not match the operation's requirements, and `Err(CommError::InvalidRoot)` when a root value other than 0 is passed. It cannot produce `CommError::CollectiveFailed` (no MPI calls) or `CommError::InvalidCommunicator` (no communicator state to invalidate).

**Testing implication:** Error-case tests (§1.8) for `InvalidBufferSize` and `InvalidRoot` apply to the local backend and must assert `Err(...)`. Only `CollectiveFailed` and `InvalidCommunicator` error tests are not applicable to the local backend.

#### 1.9.2 Ferrompi Backend

Precondition checks (buffer sizes, root validity) are delegated to the ferrompi layer per [Ferrompi Backend §1.2](./backend-ferrompi.md). The `FerrompiBackend` does not duplicate these checks. Conformance tests for error cases must verify that the ferrompi-generated MPI error codes are correctly mapped to `CommError` variants via the error conversion function in [Ferrompi Backend §5.2](./backend-ferrompi.md).

**Testing implication:** Ferrompi error-case tests depend on the MPI implementation's error reporting behavior, which may vary across MPI vendors (OpenMPI, MPICH, Intel MPI). Tests should assert on the `CommError` variant (e.g., `InvalidBufferSize`, `InvalidRoot`) rather than on the specific `mpi_error_code` value.

#### 1.9.3 TCP Backend

For `CommError::CollectiveFailed`, the `mpi_error_code` field is always 0 per [TCP Backend §5.3](./backend-tcp.md), because there is no MPI error code to report. The `message` field contains the specific TCP error description.

**Testing implication:** TCP error-case tests must assert `mpi_error_code == 0` for all `CollectiveFailed` errors. Timeout-based failure detection (§1.8.3) is controlled via the `COBRE_TCP_TIMEOUT_SECS` environment variable.

#### 1.9.4 Shared Memory Backend

The shm backend uses a two-barrier protocol for `allreduce` ([Shared Memory Backend §3.2](./backend-shm.md)): all ranks write to their per-rank slots, execute a first barrier, then rank 0 performs the reduction and writes to the output slot, and a second barrier ensures the result is visible to all ranks.

**Testing implication:** The conformance test `test_shm_allreduce_sum` (§1.2) implicitly validates the two-barrier protocol. An additional structural test should verify that rank 0 is the rank performing the reduction (not any other rank), by instrumenting or observing that non-zero ranks do not write to the output slot between the two barriers. This is validated indirectly: if a non-zero rank performed the reduction, the result would be incorrect for non-commutative edge cases or the timing guarantee would be violated.

## 2. Interchangeability Tests

These tests verify that swapping communication backends produces equivalent SDDP results. The training loop is generic over `C: Communicator` ([Communicator Trait §3](./communicator-trait.md)), and any conformant backend must produce the same policy.

### 2.1 Reference Test Case

The interchangeability tests use a small SDDP problem defined by the following properties:

| Property                     | Value       | Rationale                                                                              |
| ---------------------------- | ----------- | -------------------------------------------------------------------------------------- |
| Stages                       | 2--3        | Minimum for meaningful cut generation (backward pass requires at least 2 stages)       |
| Hydro plants                 | 2           | Minimum for reservoir coupling and non-trivial state space                             |
| Forward passes per iteration | 10          | Small enough for fast execution; large enough for meaningful upper bound statistics    |
| Iterations                   | 5           | Sufficient to generate a non-trivial cut pool; small enough for sub-5-second execution |
| Openings per stage           | 3--5        | Small scenario tree for fast backward pass                                             |
| Execution time (single core) | < 5 seconds | Hard constraint: interchangeability tests run in CI without HPC resources              |

The test case is defined by properties, not by a specific dataset. Any dataset satisfying the above properties is acceptable, provided it is deterministic (fixed random seed) and produces a non-trivial policy (at least one cut generated per stage).

### 2.2 Backend Comparison Matrix

The reference test case is executed on each backend configuration and the results are compared pairwise.

| Configuration  | Backend  | Size | Description                                                                |
| -------------- | -------- | ---- | -------------------------------------------------------------------------- |
| `ref_local`    | local    | 1    | Reference result. Single-process execution with identity collectives.      |
| `ref_shm`      | shm      | 2    | Two-process execution on the same node via POSIX shared memory.            |
| `ref_tcp`      | tcp      | 2    | Two-process execution via TCP coordinator pattern.                         |
| `ref_ferrompi` | ferrompi | 2    | Two-process execution via MPI collectives (CI environments with MPI only). |

### 2.3 Comparison Criteria

| Quantity          | Comparison Method                  | Expected Result                                                                               | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------- | ---------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lower bound trace | Bit-for-bit identical              | All multi-rank backends match each other and `ref_local`                                      | Lower bound is computed from the first-stage LP objective, which is deterministic given identical cut pools. The rank-ordered `allgatherv` ensures identical cut pool construction across backends. Per [Shared Memory Aggregation §3.4](./shared-memory-aggregation.md).                                                                                                                                                                                                                |
| Final cut pool    | Bit-for-bit identical              | All multi-rank backends match each other and `ref_local`                                      | Cut coefficients are computed from LP dual multipliers (deterministic) and aggregated via rank-ordered `allgatherv` (deterministic). Cut slot assignment is deterministic from `(iteration, forward_pass_index)` per [Shared Memory Aggregation §3.2](./shared-memory-aggregation.md).                                                                                                                                                                                                   |
| Policy output     | Bit-for-bit identical              | All multi-rank backends match each other and `ref_local`                                      | The policy is fully determined by the cut pool. Identical cut pools produce identical policies.                                                                                                                                                                                                                                                                                                                                                                                          |
| Upper bound mean  | Within tolerance (~1e-12 relative) | Multi-rank backends match each other; may differ from `ref_local` by floating-point tolerance | `allreduce` with `ReduceOp::Sum` may produce results that vary with the reduction tree shape ([Communicator Trait §2.2](./communicator-trait.md)). TCP and shm use fixed rank-order sequential reduction (deterministic), so they match each other exactly. Ferrompi uses MPI's implementation-defined reduction tree, which may diverge from TCP/shm by ~1e-12 relative error. The local backend (size=1) has no reduction tree, so its upper bound may differ from multi-rank results. |

See [Shared Memory Aggregation §3.4](./shared-memory-aggregation.md) for the full reproducibility verification criteria.

### 2.4 Python Multi-Process Interchangeability

This test verifies that the Python multi-process execution path produces results equivalent to single-process execution. Unlike the Rust-level tests in §2.2--§2.3, this test exercises the full Python worker lifecycle ([Python Bindings SS2.1a](../interfaces/python-bindings.md)), including process spawning, backend initialization, result collection, and `WorkerInfo` metadata.

**Test procedure:**

1. Execute `result_1 = cobre.train(case, num_workers=1)` (single-process, local backend).
2. Execute `result_n = cobre.train(case, num_workers=N, backend="shm")` where $N \geq 2$ (multi-process, shm backend).
3. Compare results using the criteria from §2.3:
   - Lower bound trace: bit-for-bit identical.
   - Final cut pool: bit-for-bit identical.
   - Policy output: bit-for-bit identical.
   - Upper bound mean: within ~1e-12 relative tolerance.

**WorkerInfo metadata assertions:**

4. Verify that `result_n.workers` returns a list of exactly $N$ `WorkerInfo` instances.
5. Verify that the `WorkerInfo.rank` values form a permutation of `0..N` (all distinct).
6. Verify that all `WorkerInfo.backend` values are `"shm"` (matching the requested backend).
7. Verify that `result_1.workers` returns `None` (single-process mode does not produce worker metadata, per [Python Bindings SS2.7](../interfaces/python-bindings.md)).

**Error handling assertion:**

8. If a worker process fails during test execution, verify that `cobre.WorkerError` is raised with the `rank` property identifying the failed worker (per [Python Bindings SS6.1a](../interfaces/python-bindings.md)). This is tested by intentionally causing a worker to fail (e.g., invalid case data passed to one rank only) and verifying the exception structure.

## 3. Performance Regression Tests

These tests ensure that the `Communicator` trait abstraction layer introduces no measurable overhead and that backend performance remains within expected bounds across releases.

### 3.1 Ferrompi Overhead Baseline

The primary performance claim of the backend abstraction is zero-cost abstraction via monomorphization ([Backend Selection §1.4](./backend-selection.md)): when the training loop is generic over `C: Communicator` and the binary is built with a single backend feature, the compiler resolves `C` to the concrete type at compile time, producing identical assembly to direct API calls.

**Test methodology:**

1. **Direct ferrompi calls.** Execute the SDDP training loop calling ferrompi API functions directly (no `Communicator` trait indirection). Record wall-clock time per iteration at worst-case scale ($R = 16$ ranks, $T = 120$ stages — hypothetical maximum; DEC-009 baseline is $T = 60$, $M = 192$ forward passes, ~587 MB/iteration per [Communication Patterns §3.1](./communication-patterns.md)).
2. **Trait wrapper calls.** Execute the same training loop using `train::<FerrompiBackend>(comm, ...)` with the `Communicator` trait. Record wall-clock time per iteration at the same scale.
3. **Compare.** The trait wrapper must introduce < 1% overhead relative to direct calls.

**Rationale:** Monomorphization eliminates the trait abstraction at compile time ([Ferrompi Backend §4.1](./backend-ferrompi.md)). Each trait method call compiles to a direct call to the corresponding ferrompi function. The `match` on `ReduceOp` in `allreduce` is eliminated when the variant is known at the call site (constant propagation). The expected overhead is zero; the 1% threshold accounts for measurement noise.

**Note:** This test is only applicable to the ferrompi backend. The local, TCP, and shm backends do not have a "direct call" baseline -- the `Communicator` trait is their only API. Their performance tests are integration-level (§3.2, §3.3).

### 3.2 Per-Operation Measurement

Per-operation timing measurements capture the communication cost of each collective operation in isolation, enabling regression detection at the operation level rather than the iteration level.

| Operation                       | Payload                                     | Rank Count | Measurement                                 | Regression Threshold           |
| ------------------------------- | ------------------------------------------- | ---------- | ------------------------------------------- | ------------------------------ |
| `allgatherv` (trial points)     | ~206 MB total ($R = 16$, ~12.9 MB per rank) | 16         | Median wall-clock time over 10 repetitions  | < 5% regression from baseline  |
| `allgatherv` (cuts per stage)   | ~3.2 MB total ($R = 16$, ~200 KB per rank)  | 16         | Median wall-clock time over 10 repetitions  | < 5% regression from baseline  |
| `allreduce` (convergence stats) | 32 bytes (4 x `f64`)                        | 16         | Median wall-clock time over 100 repetitions | < 5% regression from baseline  |
| `barrier`                       | 0 bytes                                     | 16         | Median wall-clock time over 100 repetitions | < 10% regression from baseline |
| `broadcast` (initialization)    | ~10 KB                                      | 16         | Median wall-clock time over 10 repetitions  | < 10% regression from baseline |

**Notes:**

- `allgatherv` is the dominant communication cost (~587 MB/iteration), so its regression threshold is tighter (5% vs 10%).
- `allreduce` is latency-sensitive and on the convergence checking critical path.
- `barrier` and `broadcast` are off the hot path (checkpoint and initialization only).
- Measurements are per-backend; the local backend is excluded because its operations are identity copies or no-ops.

### 3.3 Python Multi-Process Smoke Test

This test verifies that multi-process execution from Python achieves a meaningful speedup over single-process execution, confirming that the inter-process communication overhead does not negate the parallelism benefit.

**Test procedure:**

1. Execute `t1 = time(cobre.train(case, num_workers=1))` -- single-process wall-clock time.
2. Execute `t2 = time(cobre.train(case, num_workers=2, backend="shm"))` -- two-process wall-clock time.
3. Assert `t2 < 0.70 * t1` -- two workers complete in less than 70% of single-worker wall-clock time.

**Rationale:** With 2 workers, perfect scaling would achieve 50% of single-process time. The 70% threshold allows for:

- Process spawn overhead (Python `multiprocessing.Process` with `"spawn"` start method).
- Backend initialization overhead (shm segment creation and mapping).
- Communication overhead (shm collectives, barriers).
- Load imbalance between workers.

This is a smoke test, not a microbenchmark. It catches gross regressions (e.g., communication deadlocks, accidental serialization) rather than measuring fine-grained performance. The test case from §2.1 (< 5 seconds single-core) is used to keep CI execution time reasonable.

## 4. Determinism Verification

This section verifies a stronger property than interchangeability (§2): that the same backend produces **bit-for-bit identical** results across different rank counts, thread counts, and repeated executions. The invariant is defined in [Shared Memory Aggregation §3.1](./shared-memory-aggregation.md).

### 4.1 Reproducibility Invariant

Given the same inputs and random seed, Cobre produces **bit-for-bit identical** results regardless of:

- (a) the number of ranks,
- (b) the number of OpenMP threads per rank,
- (c) execution timing and OS scheduling.

**Bit-for-bit identical quantities (all backends):**

| Quantity                          | Determinism Guarantee | Rationale                                                                                                                                                                                                                                                              |
| --------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lower bound trace                 | Bit-for-bit identical | Computed from the first-stage LP objective, which is deterministic given identical cut pools. Cut pools are synchronized via rank-ordered `allgatherv` (deterministic on all backends). Per [Shared Memory Aggregation §3.2](./shared-memory-aggregation.md).          |
| Final cut pool coefficients       | Bit-for-bit identical | Cut coefficients are LP dual multipliers (deterministic), aggregated via rank-ordered `allgatherv` (deterministic). Cut slot assignment is deterministic from `(iteration, forward_pass_index)`. Per [Shared Memory Aggregation §3.2](./shared-memory-aggregation.md). |
| Policy output (FlatBuffers file)  | Bit-for-bit identical | The policy is fully determined by the cut pool. Identical cut pools produce identical policies. Serialization is deterministic.                                                                                                                                        |
| Convergence termination iteration | Bit-for-bit identical | The stopping rule evaluates the lower bound (bit-for-bit identical) against deterministic thresholds. All backends terminate at the same iteration number.                                                                                                             |

**Exempt quantities:**

| Quantity                   | Determinism Guarantee        | Rationale                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upper bound mean/variance  | Backend-dependent (see §4.2) | Upper bound is computed via `allreduce` with `ReduceOp::Sum`. For local, TCP, and shm backends, the reduction order is fixed (sequential rank 0..R-1), producing identical results. For ferrompi, the MPI reduction tree shape is implementation-defined, so the floating-point result may differ across rank counts or MPI libraries. Per [Communicator Trait §2.2](./communicator-trait.md). |
| Wall-clock timing metrics  | Non-deterministic            | Timing depends on OS scheduling, system load, and hardware state. Not comparable across runs.                                                                                                                                                                                                                                                                                                  |
| Per-rank timing breakdowns | Non-deterministic            | Per-rank timing varies with OS scheduling, NUMA placement, and inter-process contention. Not comparable across runs.                                                                                                                                                                                                                                                                           |

### 4.2 Per-Backend Determinism Properties

The following table classifies each collective operation's determinism properties per backend. An operation is **deterministic** if it produces bit-for-bit identical results given identical inputs, regardless of timing or scheduling.

| Backend  | allgatherv                                           | allreduce Sum                                                                              | allreduce Min                           | broadcast             | barrier               |
| -------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------- | --------------------- | --------------------- |
| local    | Identity copy (deterministic)                        | Identity copy (deterministic)                                                              | Identity copy (deterministic)           | No-op (deterministic) | No-op (deterministic) |
| tcp      | Rank-ordered assembly by coordinator (deterministic) | Fixed sequential reduction rank 0..R-1 (deterministic)                                     | Min is comparison-based (deterministic) | Deterministic         | Deterministic         |
| shm      | Rank-ordered writes to shared buffer (deterministic) | Rank 0 reduces sequentially rank 0..R-1 (deterministic)                                    | Min is comparison-based (deterministic) | Deterministic         | Deterministic         |
| ferrompi | Rank-ordered per MPI spec (deterministic)            | Implementation-defined tree shape (non-deterministic across MPI libraries and rank counts) | Min is comparison-based (deterministic) | Deterministic         | Deterministic         |

All four backends produce deterministic `allgatherv` results -- data is assembled in rank order (rank 0, rank 1, ..., rank R-1), ensuring identical cut pools and trial point sets on all ranks. This is the foundation of the reproducibility invariant.

For `allreduce` with `ReduceOp::Sum`, the local, TCP, and shm backends are deterministic because their reduction order is fixed:

- **local**: Identity copy -- no reduction, no floating-point arithmetic ([Local Backend §5](./backend-local.md)).
- **tcp**: The coordinator reduces sequentially in rank order (rank 0, 1, ..., R-1) ([TCP Backend §3.2](./backend-tcp.md)).
- **shm**: Rank 0 performs the reduction sequentially in rank order (rank 0, 1, ..., R-1) ([Shared Memory Backend §3.2](./backend-shm.md)).

The ferrompi backend delegates to `MPI_Allreduce`, whose reduction tree shape is implementation-defined ([Communicator Trait §2.2](./communicator-trait.md)). Different MPI libraries (OpenMPI, MPICH, Intel MPI) and different rank counts may produce different floating-point summation results due to the non-associativity of IEEE 754 addition. This affects only the upper bound statistics -- all other SDDP quantities are determined by `allgatherv` (deterministic) and LP solves (deterministic given identical inputs).

For `allreduce` with `ReduceOp::Min`, all backends are deterministic because minimum is a comparison-based operation with no floating-point arithmetic ambiguity. Given identical inputs, `min(a, b) = min(b, a)` regardless of evaluation order.

### 4.3 Determinism Test Matrix

The determinism tests use the same reference test case defined in §2.1 (2-3 stages, 2 hydro plants, 10 forward passes, 5 iterations, 3-5 openings per stage, < 5 seconds single-core). All tests use the same fixed random seed.

| Test ID | Backend  | Ranks |  Threads/Rank | Reference | Purpose                                                       |
| ------- | -------- | ----: | ------------: | --------- | ------------------------------------------------------------- |
| D1      | local    |     1 |             1 | --        | Baseline: single-rank, single-thread reference result         |
| D2      | local    |     1 |             4 | D1        | Thread independence: same result with different thread count  |
| D3      | shm      |     2 |             1 | D1        | Multi-rank: 2 ranks match single-rank result                  |
| D4      | shm      |     4 |             1 | D1        | Rank scaling: 4 ranks match single-rank result                |
| D5      | shm      |     2 |             4 | D1        | Combined: multi-rank + multi-thread                           |
| D6      | tcp      |     2 |             1 | D1        | TCP backend: 2 ranks match single-rank result                 |
| D7      | tcp      |     4 |             1 | D1        | TCP rank scaling: 4 ranks match single-rank result            |
| D8      | ferrompi |     2 |             1 | D1        | MPI backend: 2 ranks match single-rank on exact quantities    |
| D9      | ferrompi |     4 |             2 | D1        | MPI combined: 4 ranks, 2 threads                              |
| D10     | ferrompi |     2 | 1 (run twice) | D10-self  | MPI reproducibility: same config run twice, bit-for-bit match |

**Comparison criteria per quantity:**

| Quantity                          | D2-D7 (local/shm/tcp vs D1) | D8-D9 (ferrompi vs D1) | D10 (ferrompi run A vs run B) |
| --------------------------------- | --------------------------- | ---------------------- | ----------------------------- |
| Lower bound trace                 | Bit-for-bit identical       | Bit-for-bit identical  | Bit-for-bit identical         |
| Final cut pool coefficients       | Bit-for-bit identical       | Bit-for-bit identical  | Bit-for-bit identical         |
| Policy output (FlatBuffers file)  | Bit-for-bit identical       | Bit-for-bit identical  | Bit-for-bit identical         |
| Convergence termination iteration | Bit-for-bit identical       | Bit-for-bit identical  | Bit-for-bit identical         |
| Upper bound mean                  | Bit-for-bit identical       | Unconstrained          | Bit-for-bit identical         |

**Notes on comparison modes:**

- **D2-D5 (local/shm vs D1):** All quantities including upper bound are bit-for-bit identical. Neither the local backend's identity copy nor the shm backend's sequential reduction introduces floating-point non-determinism.

- **D6-D7 (tcp vs D1):** All quantities including upper bound are bit-for-bit identical. The TCP coordinator reduces sequentially in rank order, producing the same floating-point results as the shm backend.

- **D8-D9 (ferrompi vs D1):** Lower bound, cut pool, policy, and convergence iteration are bit-for-bit identical (dependent only on `allgatherv` and LP solves). Upper bound mean is unconstrained: `MPI_Allreduce` uses an implementation-defined reduction tree, so the ferrompi upper bound may be computed in a different floating-point summation order than D1.

- **D10 (ferrompi run-twice):** Both runs must be bit-for-bit identical on all quantities including upper bound. The MPI reduction tree is deterministic for a fixed (library, rank count) configuration; the non-determinism in §4.2 applies only when comparing across different rank counts or MPI libraries.

### 4.4 Python Multi-Process Determinism Tests

These tests verify that the Python multi-process execution path preserves determinism and correctly handles error conditions. Unlike §4.3 which tests the Rust-level backends directly, these tests exercise the full Python worker lifecycle from [Python Bindings SS2.1a](../interfaces/python-bindings.md).

#### 4.4.1 Fork Rejection

**Test procedure:**

1. Set `multiprocessing.set_start_method("fork")` in the test process.
2. Call `cobre.train(case, num_workers=2)`.

**Expected result:** Raises `cobre.CobreError` with `kind="IncompatibleSettings"` and a message explaining that the `"fork"` start method is prohibited.

**Reference:** [Python Bindings SS2.1a](../interfaces/python-bindings.md) step 1: "If it is already set to `\"fork\"`, the library raises `cobre.CobreError` with `kind=\"IncompatibleSettings\"`."

**Rationale:** The `"fork"` start method is unsafe for multi-threaded processes (OpenMP threads, solver threads) and can cause deadlocks or corrupted state. Cobre requires `"spawn"` for multi-process execution.

#### 4.4.2 Worker Error Propagation

**Test variant A -- Worker raises `CobreError`:**

1. Call `cobre.train(case_with_invalid_data, num_workers=2, backend="shm")` where `case_with_invalid_data` is constructed to trigger a `CobreError` in the worker during training (e.g., an LP infeasibility that surfaces after the first iteration).
2. The parent process raises `cobre.WorkerError`.
3. Assert: `WorkerError.inner` is an instance of `cobre.CobreError` (or a subclass) containing the original error from the failed worker.
4. Assert: `WorkerError.rank` identifies the failed worker (0 or 1).
5. Assert: All remaining workers are terminated (no zombie processes).

**Test variant B -- Worker crashes at OS level:**

1. Call `cobre.train(case, num_workers=2, backend="shm")` with a test hook that causes one worker to crash at the OS level (e.g., by sending `SIGKILL` to the worker process after it starts).
2. The parent process raises `cobre.WorkerError`.
3. Assert: `WorkerError.inner` is `None` (the worker crashed without raising a Python-level exception).
4. Assert: `WorkerError.exit_code` is signal-based (negative on Unix, reflecting the signal number).
5. Assert: All remaining workers are terminated.

**Reference:** [Python Bindings SS6.1a](../interfaces/python-bindings.md): `WorkerError` with `rank`, `exit_code`, and `inner` properties.

#### 4.4.3 Run-Twice Reproducibility

**Test procedure:**

1. Execute `result_a = cobre.train(case, config_overrides={"random_seed": 42}, num_workers=2, backend="shm")`.
2. Execute `result_b = cobre.train(case, config_overrides={"random_seed": 42}, num_workers=2, backend="shm")`.
3. Compare lower bound traces: bit-for-bit identical.
4. Compare final cut pool coefficients: bit-for-bit identical.
5. Compare policy output: bit-for-bit identical.
6. Compare convergence termination iteration: identical.

**Expected result:** All exact quantities are bit-for-bit identical across the two runs.

**Rationale:** This test verifies that the Python `multiprocessing.Process` spawn path, `Communicator` creation, worker coordination, and result collection introduce no non-determinism. The `"spawn"` start method creates fresh Python interpreters, and any state leakage between runs would manifest as differing results.

#### 4.4.4 Progress Event Multiplexing

**Test procedure:**

1. Define a `progress_callback` that appends each `ProgressEvent` to a list.
2. Execute `cobre.train(case, num_workers=2, backend="shm", progress_callback=progress_callback)`.
3. Let `N` be the number of training iterations (convergence termination iteration from the result).

**Assertions:**

- Both `worker_id=0` and `worker_id=1` appear in the collected events.
- For each worker, events with that `worker_id` are in iteration order: if event $e_i$ has `iteration=i` and event $e_j$ has `iteration=j` with $i < j$, then $e_i$ appears before $e_j$ in the list.
- Total events count equals `2 * N` (one event per iteration per worker).

**Reference:** [Python Bindings SS2.9](../interfaces/python-bindings.md): `ProgressEvent.worker_id` disambiguates the source; "events from the same worker are in iteration order."

#### 4.4.5 TCP Auto-Detection

**Test procedure:**

1. Set the environment variables `COBRE_TCP_COORDINATOR=127.0.0.1` and `COBRE_TCP_PORT=<port>` where `<port>` is a free ephemeral port (see [TCP Backend §8.1](./backend-tcp.md) for the split-variable convention).
2. Execute `result = cobre.train(case, num_workers=2, backend="auto")`.
3. Inspect `result.workers` metadata.

**Assertions:**

- `result.workers` is a list of 2 `WorkerInfo` instances.
- `WorkerInfo.backend == "tcp"` for both workers (the auto-detection logic selected TCP because `COBRE_TCP_COORDINATOR` was set).

**Reference:** [Python Bindings SS7.5](../interfaces/python-bindings.md): "When `backend=\"auto\"` and `num_workers > 1`: selects `\"tcp\"` if the `COBRE_TCP_COORDINATOR` environment variable is set."

### 4.5 Scope of the Reproducibility Invariant

The invariant is defined in [Shared Memory Aggregation §3.1](./shared-memory-aggregation.md) (requirement statement), with mechanisms in §3.2 (deterministic seeding, rank-ordered `allgatherv`, deterministic cut slots), floating-point considerations in §3.3, and quantity-by-quantity verification criteria in §3.4.

The test matrix in §4.3 and the Python multi-process tests in §4.4 provide the concrete configurations that exercise this invariant across all four backends and both execution modes (Rust-level and Python-level). No modifications to [Shared Memory Aggregation](./shared-memory-aggregation.md) are required -- the invariant definition is backend-agnostic by design.

## Cross-References

- [Communicator Trait §1](./communicator-trait.md) -- `Communicator` trait definition, `CommData`, `ReduceOp`, `CommError` type definitions (§1)
- [Communicator Trait §2](./communicator-trait.md) -- Method contracts verified by conformance tests §1.1--§1.6
- [Communicator Trait §2.2](./communicator-trait.md) -- `allreduce` floating-point non-determinism: MPI reduction tree shape is implementation-defined (§4.2)
- [Communicator Trait §3](./communicator-trait.md) -- Generic parameterization `train<C: Communicator>` enabling the interchangeability claim (§2)
- [Communicator Trait §4](./communicator-trait.md) -- `SharedMemoryProvider` trait, `SharedRegion<T>` lifecycle, leader/follower pattern (§1.7)
- [Communicator Trait §4.4](./communicator-trait.md) -- `HeapFallback` semantics for local and TCP backends (§1.7)
- [Communicator Trait §4.6](./communicator-trait.md) -- `CommError::AllocationFailed` variant (§1.8.5)
- [Ferrompi Backend §1.2](./backend-ferrompi.md) -- Precondition checks delegated to ferrompi layer (§1.9.2)
- [Ferrompi Backend §4.1](./backend-ferrompi.md) -- Zero-cost abstraction via monomorphization (§3.1)
- [Ferrompi Backend §5](./backend-ferrompi.md) -- Error mapping from MPI error codes to `CommError` variants (§1.8)
- [Local Backend §1.2](./backend-local.md) -- Precondition validation: returns `InvalidBufferSize` / `InvalidRoot` on violations (§1.9.1)
- [Local Backend §2](./backend-local.md) -- Identity semantics verified by conformance tests at size=1
- [Local Backend §5](./backend-local.md) -- Reproducibility: trivially satisfied for single-rank execution (§4.2)
- [TCP Backend §3](./backend-tcp.md) -- Coordinator-mediated collective protocols (TCP conformance tests)
- [TCP Backend §3.2](./backend-tcp.md) -- Fixed sequential reduction rank 0..R-1, deterministic `allreduce Sum` (§4.2)
- [TCP Backend §5](./backend-tcp.md) -- Timeout-based failure detection (§1.8.3); `mpi_error_code: 0` for `CollectiveFailed` (§1.9.3)
- [Shared Memory Backend §1.4](./backend-shm.md) -- Barrier generation counter (§1.4)
- [Shared Memory Backend §3](./backend-shm.md) -- Shared buffer collective protocols (shm conformance tests)
- [Shared Memory Backend §3.2](./backend-shm.md) -- Rank 0 sequential reduction, deterministic `allreduce Sum` (§4.2)
- [Shared Memory Aggregation §3.1](./shared-memory-aggregation.md) -- Reproducibility invariant definition (§4.3)
- [Shared Memory Aggregation §3.2](./shared-memory-aggregation.md) -- Component-level reproducibility mechanisms (§4)
- [Shared Memory Aggregation §3.3](./shared-memory-aggregation.md) -- Floating-point considerations; basis for exempt quantities in §4.1
- [Shared Memory Aggregation §3.4](./shared-memory-aggregation.md) -- Reproducibility verification criteria used in §2.3 and §4.3
- [Backend Selection §1.4](./backend-selection.md) -- Monomorphization guarantee for single-feature builds (§3.1)
- [Python Bindings SS2.1a](../interfaces/python-bindings.md) -- Worker lifecycle (spawn, rank assignment, result collection); fork rejection (§2.4, §4.4, §4.4.1)
- [Python Bindings SS2.7](../interfaces/python-bindings.md) -- `WorkerInfo` dataclass with `rank` and `backend` properties (§2.4, §4.4.5)
- [Python Bindings SS2.9](../interfaces/python-bindings.md) -- `ProgressEvent.worker_id` for multi-process progress multiplexing (§4.4.4)
- [Python Bindings SS6.1a](../interfaces/python-bindings.md) -- `WorkerError` with `rank`, `exit_code`, and `inner` properties (§2.4, §4.4.2)
- [Python Bindings SS7.5](../interfaces/python-bindings.md) -- Backend selection and auto-detection logic (§2.4, §4.4.5)
