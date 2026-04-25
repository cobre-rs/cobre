# Policy Management

Cobre stores the trained future-cost function (cuts), LP basis, and visited
states in a _policy directory_. The `policy` section of `config.json` controls
where that directory lives, whether training starts from scratch or from a
prior checkpoint, and how often intermediate checkpoints are written during
training.

---

## Policy Modes

The `policy.mode` field selects one of three initialization strategies. The
default is `"fresh"`.

### Fresh (Default)

Training starts from an empty future-cost function. All prior cuts in
`policy.path` are ignored (or the directory does not yet exist).

```json
{ "policy": { "mode": "fresh" } }
```

Use `"fresh"` for new studies or when you want a clean training run with no
influence from earlier iterations.

### Warm Start

Cobre loads the cuts from an existing policy checkpoint before training begins.
Training then continues, adding new cuts on top of the loaded ones. The loaded
cuts count as the initial future-cost approximation.

```json
{ "policy": { "mode": "warm_start", "path": "./policy" } }
```

Use `"warm_start"` when you have a policy from a previous run (possibly with
different parameters) and want to accelerate convergence by reusing its cuts.
Set `policy.validate_compatibility` to `true` (the default) to have Cobre
verify that the state dimension and entity layout of the saved policy match the
current system before loading.

### Resume

Cobre reads the checkpoint metadata to determine how many iterations were
completed, then resumes training from that point. The RNG seed and iteration
counter are restored so the noise sequences are identical to an uninterrupted
run.

```json
{ "policy": { "mode": "resume", "path": "./policy" } }
```

Use `"resume"` after an interrupted training run (power loss, job timeout, or
manual cancellation) to continue exactly where training stopped. Requires that
checkpointing was enabled in the interrupted run.

---

## Simulation-Only Mode

To evaluate a previously trained policy without re-running training, disable
training and load the policy in warm-start mode:

```json
{
  "training": { "enabled": false },
  "policy": { "mode": "warm_start", "path": "./policy" }
}
```

Cobre loads the cuts from `policy.path`, skips the training phase entirely, and
runs the post-training simulation using the loaded future-cost function. This
is useful for running additional simulation scenarios on a policy that has
already converged, or for comparing multiple saved policies on the same
scenarios.

---

## Checkpointing Configuration

The `policy.checkpointing` section controls periodic checkpointing during
training. All fields are optional; omitting a field leaves the solver default
in effect.

| Field                 | Type            | Description                                                                                                      |
| --------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| `enabled`             | boolean or null | Enable periodic checkpointing. When `null` or omitted, checkpointing is disabled.                                |
| `initial_iteration`   | integer or null | First iteration at which a checkpoint is written. When `null`, the first checkpoint uses `interval_iterations`.  |
| `interval_iterations` | integer or null | Number of iterations between successive checkpoints. When `null`, defaults to the solver's built-in interval.    |
| `store_basis`         | boolean or null | Include LP basis files in checkpoints. Enables faster basis warm-start on resume. When `null`, basis is omitted. |
| `compress`            | boolean or null | Compress checkpoint binary files. Reduces disk usage at the cost of slightly slower reads and writes.            |

Example enabling checkpointing every 50 iterations starting at iteration 100,
with basis storage and compression:

```json
{
  "policy": {
    "path": "./policy",
    "checkpointing": {
      "enabled": true,
      "initial_iteration": 100,
      "interval_iterations": 50,
      "store_basis": true,
      "compress": true
    }
  }
}
```

---

## Checkpoint Directory Contents

A written checkpoint has the following layout under `policy.path`:

```text
policy/
  metadata.json          -- run metadata and compatibility hashes (written last)
  cuts/
    stage_000.bin        -- cut coefficients and intercepts for stage 0
    stage_001.bin        -- cut coefficients and intercepts for stage 1
    ...
  basis/
    stage_000.bin        -- LP basis for stage 0 (when store_basis is enabled)
    stage_001.bin
    ...
  states/
    stage_000.bin        -- visited states for dominated cut selection, stage 0
    stage_001.bin
    ...
```

`metadata.json` is written **last**. Its presence signals that the checkpoint
is complete and safe to load. An interrupted write leaves `metadata.json`
absent; Cobre treats a directory without `metadata.json` as an incomplete
checkpoint and refuses to load it.

The `metadata.json` file records the number of completed iterations,
lower-bound and upper-bound values, state dimension, number of stages,
configuration and system hashes (used by `validate_compatibility`), forward
passes per iteration, and the RNG seed. These fields allow Cobre to verify
that a saved policy is compatible with the current system before loading it
in `"warm_start"` or `"resume"` mode.

---

## Boundary Cuts

Boundary cuts allow a Cobre study to load terminal-stage future cost function
(FCF) approximations from a different Cobre policy checkpoint. This is the
mechanism for **model coupling** — a short-horizon study (e.g., weekly+monthly
DECOMP) can use the long-horizon policy (e.g., a monthly NEWAVE-equivalent)
as its terminal boundary condition, ensuring that end-of-horizon decisions
account for the long-term future cost of water.

### How it works

1. **Run a monthly study** and produce a policy checkpoint (the "outer" model).
2. **Run a weekly+monthly study** with `policy.boundary` pointing to the
   monthly checkpoint. Cobre loads cuts from the specified stage and injects
   them into the terminal stage's row pool as fixed boundary conditions.

The imported boundary cuts are **not updated** by the SDDP training algorithm.
They remain fixed throughout training and simulation, providing a floor on
the terminal-stage future cost.

### Configuration

Add a `boundary` object to the `policy` section of `config.json`:

```json
{
  "policy": {
    "mode": "fresh",
    "boundary": {
      "path": "../monthly_study/policy",
      "source_stage": 2
    }
  }
}
```

| Field          | Type    | Description                                                     |
| -------------- | ------- | --------------------------------------------------------------- |
| `path`         | string  | Path to the source Cobre policy checkpoint directory.           |
| `source_stage` | integer | 0-based stage index in the source checkpoint to load cuts from. |

When `boundary` is absent or `null`, no boundary cuts are loaded (the default).

### Compatibility requirements

The source checkpoint must have the same state dimension (number of hydro
plants and maximum PAR order) as the current study. Cobre validates this
automatically when `validate_compatibility` is `true`. If the dimensions
don't match, loading fails with a descriptive error.

### Production DECOMP workflow

The typical production DECOMP pipeline uses boundary cuts as follows:

```text
Monthly Cobre study (12 stages)
  └─ policy checkpoint: cuts for stages 0–11

Weekly+monthly DECOMP study (W1, W2, W3, W4, M2)
  └─ policy.boundary.path = "../monthly/policy"
  └─ policy.boundary.source_stage = 2  (March cuts → terminal FCF)
```

The DECOMP study's terminal stage (M2) receives the monthly model's March
cuts as its future cost function. The lag accumulation mechanism ensures that
the state vector's lag values at the terminal stage are monthly averages,
making the imported cut coefficients evaluate correctly.

### Interaction with warm-start

Boundary cuts and warm-start are independent features. You can combine them:

```json
{
  "policy": {
    "mode": "warm_start",
    "path": "./policy",
    "boundary": {
      "path": "../monthly/policy",
      "source_stage": 2
    }
  }
}
```

This loads the previous DECOMP policy's own cuts via warm-start AND loads
the monthly model's boundary cuts at the terminal stage. Both sets of cuts
contribute to the lower bound.

---

## See Also

- [Configuration](./configuration.md) — every `config.json` field documented
- [Running Studies](./running-studies.md) — common workflows including training-only and simulation-only runs
- [Output Format](../reference/output-format.md) — detailed description of every output file
