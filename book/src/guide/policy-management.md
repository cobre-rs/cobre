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

## See Also

- [Configuration](./configuration.md) — every `config.json` field documented
- [Running Studies](./running-studies.md) — common workflows including training-only and simulation-only runs
- [Output Format](../reference/output-format.md) — detailed description of every output file
