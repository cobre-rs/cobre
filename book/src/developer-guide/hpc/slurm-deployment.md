# SLURM Deployment

## Purpose

This spec defines SLURM job scripts and deployment patterns for Cobre on HPC clusters: single-node development jobs, multi-node production runs with NUMA binding, alternative deployment configurations, and parameter study sweeps via job arrays. For performance monitoring and diagnostics, see [Output Schemas §6.2-§6.3](../data-model/output-schemas.md) and [Shared Memory Aggregation §4](./shared-memory-aggregation.md).

## Shell / Rust Boundary

> **Important**: SLURM scripts are shell scripts that configure the job environment — they are preserved as-is (not Rust code). The Rust binary launched by `srun` uses `ferrompi` to detect placement:
>
> - **SLURM scripts** → set `--ntasks`, `--cpus-per-task`, `--mem-bind`, bind policies, module loads
> - **Rust startup** → calls `ferrompi::Mpi::init_thread(Funneled)` to initialize MPI with thread support (only the main thread calls MPI), and `cobre_comm::slurm::local_rank()` to read SLURM topology variables (`SLURM_LOCALID`, `SLURM_CPUS_PER_TASK`, etc.)
>
> See [Hybrid Parallelism §5](./hybrid-parallelism.md) for the full initialization sequence and [Hybrid Parallelism §3](./hybrid-parallelism.md) for `ParallelConfig::from_environment()` which delegates to `cobre_comm::slurm` helpers.

## 1. Single-Node Job (Development/Testing)

For development, debugging, and small-scale testing on a single node.

```bash
#!/bin/bash
#SBATCH --job-name=cobre-dev
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --output=cobre_%j.log

module load openmpi/4.1.5

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun cobre run /path/to/case
```

**Notes:**

- 4 ranks × 24 threads = 96 cores (single-socket or half a dual-socket node)
- `OMP_NUM_THREADS` is derived from `SLURM_CPUS_PER_TASK` — never hardcoded
- `debug` partition typically has shorter queue wait and 1-hour wall time limits
- The Rust binary reads `SLURM_CPUS_PER_TASK` via `cobre_comm::slurm::cpus_per_task()`

## 2. Multi-Node Production Job (Recommended)

Production configuration with one MPI rank per NUMA domain — the recommended deployment per [Hybrid Parallelism §4.4](./hybrid-parallelism.md) and [Memory Architecture §3.2](./memory-architecture.md).

```bash
#!/bin/bash
#SBATCH --job-name=cobre-prod
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=cobre_%j.log
#SBATCH --error=cobre_%j.err

module load openmpi/4.1.5

# OpenMP configuration
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# MPI tuning for large jobs
export OMPI_MCA_mpi_yield_when_idle=1

# Checkpoint signal: SIGTERM 60s before kill
srun --signal=TERM@60 \
     --cpu-bind=verbose \
     --distribution=block:block \
     cobre run /scratch/user/case
```

**Design choices:**

| Choice                           | Rationale                                                                                            |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--ntasks-per-node=8`            | One rank per NUMA domain on dual-socket EPYC (8 NUMA domains per node)                               |
| `--cpus-per-task=24`             | All cores within the NUMA domain available to OpenMP threads                                         |
| `--mem=0` + `--exclusive`        | Full node memory; avoids contention with other jobs                                                  |
| `--signal=TERM@60`               | SIGTERM 60s before kill — enables graceful checkpoint (see [Checkpointing §4.1](./checkpointing.md)) |
| `--cpu-bind=verbose`             | Logs binding decisions for debugging NUMA placement                                                  |
| `--distribution=block:block`     | Keeps rank-to-node mapping predictable for reproducibility                                           |
| `OMP_PROC_BIND=close`            | Threads stay within NUMA domain — local memory access                                                |
| `OMPI_MCA_mpi_yield_when_idle=1` | Reduces CPU waste when ranks wait at MPI barriers                                                    |

**Scale:** 8 nodes × 8 ranks/node × 24 threads/rank = 1,536 cores, 64 MPI ranks total.

## 3. Alternative: One Rank Per Node

For memory-constrained scenarios where maximizing per-rank memory is more important than NUMA locality, a single rank per node with all cores available to OpenMP:

```bash
#!/bin/bash
#SBATCH --job-name=cobre-1ppn
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=cobre_%j.log

module load openmpi/4.1.5

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun --signal=TERM@60 \
     cobre run /scratch/user/case
```

**Trade-offs vs. recommended deployment (§2):**

| Aspect               | 1 rank/node (§3)                              | 8 ranks/node (§2, recommended)                         |
| -------------------- | --------------------------------------------- | ------------------------------------------------------ |
| MPI rank count       | Fewer ranks (lower communication overhead)    | More ranks (higher `MPI_Allgatherv` participant count) |
| NUMA locality        | Threads span all NUMA domains — remote access | Each rank's threads stay within one NUMA domain        |
| Memory per rank      | Full node memory available                    | ~1/8 of node memory per rank                           |
| SharedWindow savings | No savings (1 rank = 1 node)                  | Shared read-only data avoids per-rank replication      |
| LP solve performance | Potentially slower (NUMA cross-access)        | Better locality for solver working data                |

The recommended deployment (§2) is preferred because LP solve performance is NUMA-sensitive — solver working data (LU factors, pricing vectors) benefits strongly from local NUMA access (see [Memory Architecture §3.4](./memory-architecture.md)).

## 4. Job Arrays for Parameter Studies

Parameterized job submission for sweeping configuration values (e.g., forward scenario counts):

```bash
#!/bin/bash
#SBATCH --job-name=cobre-sweep
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=cobre_%A_%a.log

module load openmpi/4.1.5

SCENARIOS=(100 200 500 1000 2000)
N_SCENARIOS=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

CASE_DIR=/scratch/user/sweep_${N_SCENARIOS}
cp -r /home/user/base_case "$CASE_DIR"

jq ".training.forward_passes = ${N_SCENARIOS}" \
    "$CASE_DIR/config.json" > "$CASE_DIR/config_tmp.json"
mv "$CASE_DIR/config_tmp.json" "$CASE_DIR/config.json"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun cobre run "$CASE_DIR"
```

**Notes:**

- `%A` = parent array job ID, `%a` = array task index — produces unique log files per sweep point
- Each array element runs independently with its own SLURM allocation
- `jq` modifies `config.json` for each sweep point

## 5. Environment Variables Reference

### 5.1 SLURM Variables Read by Cobre

The Rust binary reads these SLURM environment variables via `cobre_comm::slurm` helpers during initialization (see [Hybrid Parallelism §3](./hybrid-parallelism.md)):

| Variable                | Usage                                             |
| ----------------------- | ------------------------------------------------- |
| `SLURM_CPUS_PER_TASK`   | Sets OpenMP thread count per rank                 |
| `SLURM_LOCALID`         | Local rank index within node (for NUMA placement) |
| `SLURM_NTASKS_PER_NODE` | Ranks per node (for SharedWindow leader election) |
| `SLURM_JOB_NUM_NODES`   | Total node count (for deployment diagnostics)     |

### 5.2 OpenMP Variables

| Variable          | Recommended Value        | Rationale                               |
| ----------------- | ------------------------ | --------------------------------------- |
| `OMP_NUM_THREADS` | `${SLURM_CPUS_PER_TASK}` | Match SLURM allocation — never hardcode |
| `OMP_PROC_BIND`   | `close`                  | Keep threads within NUMA domain         |
| `OMP_PLACES`      | `cores`                  | One thread per physical core (no SMT)   |

### 5.3 MPI Tuning Variables (OpenMPI)

| Variable                       | Recommended Value | Rationale                        |
| ------------------------------ | ----------------- | -------------------------------- |
| `OMPI_MCA_mpi_yield_when_idle` | `1`               | Reduce CPU waste at MPI barriers |

## 6. Checkpoint/Resume Integration

### 6.1 Signal Configuration

SLURM's `--signal=TERM@N` sends SIGTERM $N$ seconds before killing the job. This gives Cobre time to write a checkpoint from the last completed iteration (see [Checkpointing §4.1](./checkpointing.md) and [CLI and Lifecycle §7](../architecture/cli-and-lifecycle.md)).

Recommended: `--signal=TERM@60` (60 seconds is sufficient for checkpoint writes at production scale).

### 6.2 Resume Job Script

To resume from a checkpoint, submit the same job script pointing to the same case directory. The Rust binary detects the `latest` checkpoint symlink and resumes automatically (see [Checkpointing §3.2](./checkpointing.md)):

```bash
# Same script as §2 — resume is automatic if checkpoint exists
srun --signal=TERM@60 \
     cobre run /scratch/user/case
```

No script changes are needed. The execution mode (`fresh` vs. `resume`) is determined at runtime by the presence of a checkpoint in the case directory.

## Cross-References

- [Hybrid Parallelism §3](./hybrid-parallelism.md) — `ParallelConfig::from_environment()`, `cobre_comm::slurm` helpers
- [Hybrid Parallelism §4.4](./hybrid-parallelism.md) — NUMA binding policy, one rank per NUMA domain recommendation
- [Hybrid Parallelism §5](./hybrid-parallelism.md) — Full initialization sequence (MPI init, thread discovery)
- [Memory Architecture §3](./memory-architecture.md) — NUMA-aware allocation principles, NUMA latency impact
- [Checkpointing §1.2](./checkpointing.md) — Checkpoint triggers (periodic, signal, convergence)
- [Checkpointing §4](./checkpointing.md) — Signal handling integration, SLURM preemption protocol
- [CLI and Lifecycle §7](../architecture/cli-and-lifecycle.md) — Signal handling and graceful shutdown protocol
- [Shared Memory Aggregation §1](./shared-memory-aggregation.md) — SharedWindow leader election using intra-node communicator
- [Shared Memory Aggregation §4](./shared-memory-aggregation.md) — Performance monitoring, diagnostic interpretation
- [Output Schemas §6.2-§6.3](../data-model/output-schemas.md) — Timing output Parquet schemas (iterations, per-rank)
- [Work Distribution §3.1](./work-distribution.md) — Static contiguous block distribution (determines load balance characteristics)
