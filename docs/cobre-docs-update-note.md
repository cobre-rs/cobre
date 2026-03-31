# cobre-docs Update Note: Dominated Cut Selection

Items to add or update in the methodology reference (`cobre-rs/cobre-docs`).

## New Section: Dominated Cut Selection

The `"domination"` cut selection method is now fully implemented.
Add a methodology section covering:

### Algorithm

1. For each active cut $k$ at stage $t$, evaluate whether it is
   **dominated** at every visited trial point $\hat{x}$ in the archive:
   a cut is dominated at $\hat{x}$ if there exists another active cut
   whose value at $\hat{x}$ is at least as large.

2. If a cut is dominated at all visited points, increment its
   `domination_count`. If `domination_count > threshold`, deactivate it.

3. Cuts generated in the current iteration are exempt from deactivation
   (protection window).

4. Stage 0 is exempt from cut selection because its cuts have no
   backward-pass activity tracking.

### Visited States Archive

- The archive collects all forward-pass trial points across all training
  iterations, regardless of the cut selection method.
- Storage is flat `Vec<f64>` per stage: `count * state_dimension` elements.
- After per-stage state exchange in the backward pass, the gathered states
  (from all MPI ranks) are appended to the archive.
- The archive is always allocated during training. Persistence to the
  policy checkpoint is controlled by `exports.states` (default: `false`).

### Configuration

Document the `config.json` fields under `training.cut_selection`:

```json
{
  "training": {
    "cut_selection": {
      "enabled": true,
      "method": "domination",
      "threshold": 3,
      "check_frequency": 5,
      "cut_activity_tolerance": 1e-6
    }
  }
}
```

- `method: "domination"` -- enables dominated cut selection.
- `threshold` -- number of consecutive domination checks a cut must fail
  before deactivation. Higher values are less aggressive.
- `check_frequency` -- iterations between pruning checks.

### Relationship to Other Methods

| Method       | Criterion                       | Aggressiveness |
| ------------ | ------------------------------- | -------------- |
| `level1`     | Cumulative binding count        | Least          |
| `lml1`       | Binding within sliding window   | Moderate       |
| `domination` | Dominated at all visited points | Most           |

### Policy Checkpoint Format

The visited states are serialized per stage as FlatBuffers files
under `policy/states/stage_NNN.bin`. Each file contains:

- `stage_id: u32`
- `state_dimension: u32`
- `count: u32`
- `data: [f64]` (flat, `count * state_dimension` elements, row-major)

The `metadata.json` includes `total_visited_states: u64` summing
the count across all stages.

## Existing Sections to Update

- **Cut selection overview**: replace "domination -- stub; not yet
  implemented" with the algorithm description above.
- **Policy checkpoint format**: add the `states/` subdirectory and
  `total_visited_states` metadata field.
- **Configuration reference**: add `"domination"` as a valid `method`
  value with its semantics.
