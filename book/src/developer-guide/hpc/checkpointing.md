# Checkpointing

## Purpose

This spec defines how Cobre persists training state for fault tolerance (checkpointing) and supports resuming training or warm-starting from a previously trained policy. For the serialization format and policy directory structure, see [Binary Formats §3](../data-model/binary-formats.md). For output generation (simulation results, timing data), see [Output Infrastructure](../data-model/output-infrastructure.md) and [Output Schemas](../data-model/output-schemas.md).

## 1. Checkpoint Strategy

### 1.1 Goals

| Goal                | Requirement                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| Fault tolerance     | Resume training after SLURM preemption, wall-time limit, or node failure               |
| Checkpoint overhead | < 5% of iteration time (rank 0 writes while other ranks wait at barrier)               |
| Warm-start          | Start new training from a previously trained policy's cuts                             |
| Reproducibility     | Resume from checkpoint must produce bit-for-bit identical results to uninterrupted run |

### 1.2 Checkpoint Triggers

| Trigger     | Condition                                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| Periodic    | Every $N$ iterations (configurable via `checkpoint_interval`; disabled by default — must be explicitly enabled) |
| Signal      | SIGTERM/SIGINT sets shutdown flag; checkpoint written from last completed iteration's state                     |
| Convergence | Final checkpoint on training completion                                                                         |

Signal handling follows the protocol in [CLI and Lifecycle §7](../architecture/cli-and-lifecycle.md): the handler sets a global flag, checkpoints the **last fully completed iteration** (not the in-progress one), and exits. The training loop checks the flag at iteration boundaries.

## 2. Checkpoint Contents

### 2.1 What Must Be Serialized

| Component                | Serialization                                  | Why Required                                                   |
| ------------------------ | ---------------------------------------------- | -------------------------------------------------------------- |
| Cut pool (all stages)    | FlatBuffers `StageCuts` per stage              | Primary policy data — the trained cost-to-go approximation     |
| Cut activity/slot state  | `is_active` flags and slot indices per cut     | LP row structure must be reconstructed identically             |
| Solver basis (per stage) | FlatBuffers `StageBasis` per stage             | Exact warm-start — avoids full re-solve on resume              |
| Iteration counter        | `PolicyMetadata.completed_iterations`          | Resume continues from correct iteration                        |
| RNG state                | `PolicyMetadata.rng_state` (full state vector) | Scenario reproducibility — next iteration generates same noise |
| Convergence history      | Lower/upper bound traces                       | Convergence monitoring continues with correct history          |
| Config hash              | `PolicyMetadata.config_hash`                   | Detect config changes between runs                             |
| System hash              | `PolicyMetadata.system_hash`                   | Detect input data changes between runs                         |

For the complete FlatBuffers schema (`StageCuts`, `StageBasis`, `PolicyMetadata`), see [Binary Formats §3.1](../data-model/binary-formats.md). For the reproducibility requirements on checkpoint/resume, see [Binary Formats §4.1](../data-model/binary-formats.md).

### 2.2 What Is NOT Serialized

| Component                 | Reason Not Serialized                                                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Opening tree              | Deterministically regenerable from `rng_seed` + opening indices (see [Scenario Generation §2.3](../architecture/scenario-generation.md)) |
| Solver workspaces         | Thread-local, rebuilt at initialization (see [Solver Workspaces §1.3](../architecture/solver-workspaces.md))                             |
| MPI communication buffers | Allocated at initialization, not training state                                                                                          |
| Forward pass state        | Ephemeral — consumed within each iteration                                                                                               |

### 2.3 Checkpoint Write Protocol

1. Training loop completes an iteration and checks checkpoint triggers (§1.2)
2. All ranks synchronize at an MPI barrier
3. Rank 0 writes the checkpoint to the policy directory (FlatBuffers format per [Binary Formats §3.2](../data-model/binary-formats.md))
4. Rank 0 updates the `latest` symlink to point to the new checkpoint
5. Rank 0 removes old checkpoints beyond the retention limit (default: keep last 3)
6. All ranks synchronize at an MPI barrier
7. Training continues

Only rank 0 performs I/O. All other ranks wait at the barrier. At production scale, the checkpoint write takes a few seconds (cut pool is pre-allocated and contiguous — see [Binary Formats §3.3](../data-model/binary-formats.md) for memory layout).

### 2.4 Checkpoint Sizing

Checkpoint size is dominated by the cut pool. At production scale (per [Binary Formats §4.3](../data-model/binary-formats.md)):

| Component             | Size at capacity                                                                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cut pool (all stages) | 60 stages × up to 15K cuts × ~17 KB per cut — up to ~14.3 GB at maximum capacity (cut coefficients absorbed into StageLpCache SharedRegion at ~22.3 GB node-wide) |
| Solver basis          | 60 stages × ~87 KB per basis ≈ ~5 MB                                                                                                                              |
| Metadata + history    | < 1 MB                                                                                                                                                            |

Early iterations produce much smaller checkpoints (only populated slots are serialized). The cut pool pre-allocates slots but only populated ones are written.

## 3. Execution Modes

### 3.1 Mode Definitions

Three execution modes determine how training initializes (per [Binary Formats §4.2](../data-model/binary-formats.md)):

| Mode         | Cut Loading            | RNG State                | Cut Pool Capacity       | Result Guarantee          |
| ------------ | ---------------------- | ------------------------ | ----------------------- | ------------------------- |
| `fresh`      | None                   | From config seed         | `max_iter × fwd_passes` | Deterministic from seed   |
| `warm_start` | All cuts from policy   | Fresh from config seed   | loaded + new training   | Different from original   |
| `resume`     | All cuts + exact state | Restored from checkpoint | Same as checkpoint      | **Bit-for-bit identical** |

### 3.2 Resume Protocol

On resume, the following state must be restored exactly:

1. **Cut pool**: All cuts (active and inactive) with their original slot indices — LP row structure must match
2. **RNG state**: Full state vector, not just seed — ensures next iteration generates identical scenarios
3. **Convergence history**: Lower/upper bound traces — convergence monitoring continues correctly
4. **Iteration counter**: Resume from `completed_iterations + 1`
5. **Solver basis**: Per-stage basis vectors — exact warm-start avoids different pivot sequences

After restoration, the resumed run must produce bit-for-bit identical results to an uninterrupted run. See [Binary Formats §4.1](../data-model/binary-formats.md) and [Shared Memory Aggregation §3](./shared-memory-aggregation.md) for the reproducibility guarantee.

### 3.3 Warm-Start Protocol

Warm-start loads cuts from a previous policy but starts training with fresh RNG state:

1. Load cuts from policy directory into the cut pool (these become the "warm-start" cuts)
2. Initialize RNG from config seed (not restored — new scenario sequence)
3. Allocate additional cut pool capacity for new training cuts
4. Begin training from iteration 0 with pre-populated cost-to-go approximation

Warm-start produces different results from the original training because the scenario sequence differs.

### 3.4 Compatibility Validation

Before loading cuts (resume or warm-start), the system validates that the policy is compatible with the current input data:

| Validation Check     | What Is Compared                                      | Failure Mode |
| -------------------- | ----------------------------------------------------- | ------------ |
| State dimension      | Policy `state_dimension` vs. computed from input data | Hard error   |
| Stage count          | Policy `num_stages` vs. input stage count             | Hard error   |
| Config hash (resume) | Policy `config_hash` vs. current config hash          | Hard error   |
| System hash (resume) | Policy `system_hash` vs. current input hash           | Hard error   |

> **Note**: Comprehensive policy compatibility validation (block modes, hydro counts, AR orders, cascade topology, penalty configuration) is deferred to [Deferred Features §C.9](../deferred.md). The checks above are the minimum required for initial implementation.

## 4. Signal Handling Integration

### 4.1 SLURM Preemption

SLURM sends `SIGTERM` when a job approaches its wall-time limit. The graceful shutdown protocol (defined in [CLI and Lifecycle §7](../architecture/cli-and-lifecycle.md)) ensures checkpoint integrity:

| Step | Action                                                                |
| ---- | --------------------------------------------------------------------- |
| 1    | Signal handler sets global shutdown flag                              |
| 2    | Training loop detects flag at next iteration boundary                 |
| 3    | Checkpoint written from last fully completed iteration's policy state |
| 4    | Training manifest updated with `status: interrupted`                  |
| 5    | Process exits with code 0 (clean shutdown)                            |

The checkpoint is written from the **last completed iteration**, not the in-progress one. This avoids serializing partially-updated state (e.g., cuts from an incomplete backward pass).

### 4.2 Resume After Preemption

The next SLURM job invocation detects the checkpoint via the `latest` symlink and resumes:

1. Load checkpoint (§3.2 resume protocol)
2. Regenerate opening tree from persisted `rng_seed` (deterministic — see [Scenario Generation §2.3](../architecture/scenario-generation.md))
3. Rebuild solver workspaces with first-touch NUMA allocation (see [Solver Workspaces §1.3](../architecture/solver-workspaces.md))
4. Restore solver basis per stage for warm-start
5. Continue training from `completed_iterations + 1`

## Cross-References

- [Binary Formats §3](../data-model/binary-formats.md) — FlatBuffers schema (StageCuts, StageBasis, PolicyMetadata), policy directory structure, encoding guidelines
- [Binary Formats §4](../data-model/binary-formats.md) — Cut pool persistence: checkpoint reproducibility, execution modes, cut pool sizing
- [Output Infrastructure §1.2](../data-model/output-infrastructure.md) — Training manifest with status values (completed, interrupted)
- [Output Schemas §6.2-§6.3](../data-model/output-schemas.md) — Timing output schemas (iterations.parquet, mpi_ranks.parquet)
- [CLI and Lifecycle §5](../architecture/cli-and-lifecycle.md) — Execution phases and exit codes
- [CLI and Lifecycle §7](../architecture/cli-and-lifecycle.md) — Signal handling and graceful shutdown protocol
- [Convergence Monitoring §1](../architecture/convergence-monitoring.md) — Convergence criteria, bound computation, history that must be checkpointed
- [Scenario Generation §2.3](../architecture/scenario-generation.md) — Opening tree is deterministically regenerable from seed (not checkpointed)
- [Solver Workspaces §1.3](../architecture/solver-workspaces.md) — NUMA-aware workspace initialization on resume
- [Training Loop §3](../architecture/training-loop.md) — Iteration structure, checkpoint integration points
- [Shared Memory Aggregation §3](./shared-memory-aggregation.md) — Reproducibility guarantees (bit-for-bit identical results)
- [Memory Architecture §4](./memory-architecture.md) — Pre-allocated components that are rebuilt (not checkpointed)
- [SLURM Deployment](./slurm-deployment.md) — Job scripts with checkpoint/resume configuration
- [Deferred Features §C.9](../deferred.md) — Comprehensive policy compatibility validation (deferred)
