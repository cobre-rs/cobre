# Terminal UI

> **Status: Not Implemented.** This specification describes planned behavior. The current codebase contains an empty library crate.

## Purpose

This spec defines the Cobre terminal UI (`cobre-tui`): a ratatui-based interactive terminal interface for real-time monitoring of SDDP training runs. The TUI consumes the same event stream as the [Structured Output](./structured-output.md) JSON-lines writer, rendering convergence plots, iteration timing breakdowns, cut statistics, resource usage, and MPI communication metrics. It supports interactive features -- pause/resume, cut inspection, scenario comparison, stopping rule adjustment, and snapshot export -- all operating at iteration boundaries to preserve training loop atomicity. The TUI operates in two modes: co-hosted within the `cobre` binary via an in-process broadcast channel, or standalone reading JSON-lines from stdin for monitoring remote or already-running jobs.

## 1. Crate Architecture

### 1.1 Crate Identity

| Attribute              | Value                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| **Crate name**         | `cobre-tui`                                                                                        |
| **Crate type**         | Library crate (consumed by `cobre-cli`), optionally compiled as standalone binary                  |
| **Execution mode**     | Single-threaded rendering. Receives events via channel (co-hosted) or JSON-lines pipe (standalone) |
| **Dependencies**       | `cobre-core` (event type definitions only)                                                         |
| **External deps**      | `ratatui`, `crossterm`                                                                             |
| **Does NOT depend on** | `cobre-sddp`, `cobre-io`, `cobre-solver`, `cobre-stochastic`, `ferrompi`                           |
| **MPI relationship**   | Never touches MPI. In co-hosted mode, runs on rank 0 only                                          |

### 1.2 Dependency Graph

```
cobre-cli
  +-- cobre-tui [feature = "tui"]
  |     +-- cobre-core           (event type definitions only)
  |     +-- ratatui              (terminal rendering)
  |     +-- crossterm            (terminal backend)
  +-- cobre-sddp
  +-- cobre-io
  +-- cobre-core
```

The TUI depends only on `cobre-core` for event type deserialization. It does not depend on `cobre-sddp` directly. In co-hosted mode, the `cobre-cli` binary mediates the connection by subscribing the TUI to the event broadcast channel. In standalone pipe mode, the TUI reads JSON-lines from stdin and deserializes event payloads using the types defined in `cobre-core`.

### 1.3 Deployment Modes

> **Design decision (Q-2)**: The TUI is both a library crate consumed by `cobre-cli` via the `tui` Cargo feature flag, and can be compiled as a standalone binary for pipe mode. Co-hosted is the primary mode; standalone is secondary.

| Mode       | Activation                                                        | Binary                       |
| ---------- | ----------------------------------------------------------------- | ---------------------------- |
| Co-hosted  | `cobre run --tui /path/to/case`                                   | `cobre` (with `tui` feature) |
| Standalone | `mpiexec cobre run --output-format json-lines /case \| cobre-tui` | `cobre-tui`                  |
| Standalone | `cobre-tui < progress.jsonl`                                      | `cobre-tui`                  |
| Offline    | `cobre-tui --replay /path/to/output/training/convergence.parquet` | `cobre-tui`                  |

## 2. Event Consumption

The TUI consumes training events defined in `cobre-core`. These are the same event types consumed by the JSON-lines writer ([Structured Output](./structured-output.md) SS3.4).

### 2.1 Co-Hosted Mode (Broadcast Channel)

In co-hosted mode, the TUI runs on a dedicated thread within the `cobre` binary (rank 0 only). It subscribes to the `broadcast::Sender<TrainingEvent>` channel alongside the text logger and Parquet writer. Event delivery is in-process with no serialization overhead.

```
Training Loop (cobre-sddp, rank 0)
    |
    v
broadcast::Sender<TrainingEvent>
    |
    +---> Text Logger          [main thread, --output-format human]
    +---> TUI Renderer         [TUI thread, --tui flag]
    +---> Parquet Writer       [always active]
```

The TUI thread runs the ratatui event loop, polling both the broadcast channel receiver and terminal input events (keyboard, resize) via crossterm.

### 2.2 Standalone Pipe Mode (JSON-Lines Stdin)

In standalone mode, the TUI binary reads JSON-lines from stdin, deserializing each line into the `TrainingEvent` enum using `serde_json`. The event types are imported from `cobre-core`.

```
mpiexec cobre run --output-format json-lines /case
    |
    | (stdout pipe)
    v
cobre-tui (stdin reader)
    |
    v
serde_json::from_str::<TrainingEvent>(line)
    |
    v
TUI Renderer
```

**Deserialization**: Each JSON-lines envelope has a `type` field that discriminates the event kind. The standalone TUI maps the 4 JSON-lines envelope types (`started`, `progress`, `terminated`, `result`) to the corresponding `TrainingEvent` variants:

| JSON-Lines Envelope Type | `TrainingEvent` Variant |
| ------------------------ | ----------------------- |
| `started`                | `TrainingStarted`       |
| `progress`               | `ConvergenceUpdate`     |
| `terminated`             | `TrainingFinished`      |
| `result`                 | (ignored by TUI)        |

> **Note**: In pipe mode, the TUI receives only the 4 JSON-lines envelope types, not the full 7 iteration-scoped events. Detailed per-step events (`ForwardPassComplete`, `BackwardPassComplete`, etc.) are available only in co-hosted mode. The pipe-mode TUI gracefully degrades: the Iteration Detail and Communication views show reduced information, and the Convergence Plot and Summary Dashboard remain fully functional.

### 2.3 Backpressure and Gap Tolerance

Following the broadcast channel design:

1. **Dropped events**: If the TUI falls behind (e.g., slow terminal rendering), the broadcast channel drops old events rather than blocking the training loop. The TUI must tolerate gaps in the event sequence.
2. **Lagging indicator**: When the TUI detects a gap in iteration numbers (e.g., receives iteration 45 after iteration 42), it renders a `LAGGING` indicator in the status bar and interpolates missing data points in the convergence plot.
3. **No crash on gaps**: Missing events never cause a panic or render error. The TUI renders the last known state for any view that depends on a missing event.
4. **Pipe EOF handling**: When stdin reaches EOF in pipe mode, the TUI enters "replay complete" mode: all views remain visible and interactive (scrolling, inspection), but the status bar shows `REPLAY COMPLETE` and no further updates are expected.

### 2.4 Event-to-View Mapping

All 7 iteration-scoped events from [Training Loop](../architecture/training-loop.md) SS2.1 and the 4 lifecycle events are mapped to TUI views:

| Event Type             | Lifecycle Step | Consuming Views                          |
| ---------------------- | -------------- | ---------------------------------------- |
| `ForwardPassComplete`  | Step 1         | Iteration Detail, Communication          |
| `ForwardSyncComplete`  | Step 2         | Communication                            |
| `BackwardPassComplete` | Step 3         | Iteration Detail                         |
| `CutSyncComplete`      | Step 4         | Cut Statistics, Communication            |
| `ConvergenceUpdate`    | Step 5         | Convergence Plot, Summary Dashboard      |
| `CheckpointComplete`   | Step 6         | Summary Dashboard (checkpoint indicator) |
| `IterationSummary`     | Step 7         | Iteration Detail, Summary Dashboard      |
| `TrainingStarted`      | Lifecycle      | Summary Dashboard (header)               |
| `TrainingFinished`     | Lifecycle      | Summary Dashboard (termination reason)   |
| `SimulationProgress`   | Lifecycle      | Summary Dashboard (simulation progress)  |
| `SimulationFinished`   | Lifecycle      | Summary Dashboard (simulation complete)  |

## 3. Monitoring Views

The TUI provides 5 monitoring views, each occupying the full terminal area below a persistent status bar. Views are switched via keyboard shortcuts (SS6).

### 3.1 Convergence Plot

**Purpose**: Visualize the convergence of lower and upper bounds across iterations.

**Primary data source**: `ConvergenceUpdate` events. The data model is specified in SS4.

**Update frequency**: Once per iteration (on `ConvergenceUpdate` event).

**Layout**:

```
+--[ Convergence Plot ]--------------------------------------------+
|                                                                   |
|  Cost ($)                                                         |
|  74000 |                                           * * * * UB     |
|        |                                     * *                  |
|  73000 |                               * *         +---+          |
|        |                         * *               | * | CI band  |
|  72000 |                   * *                     +---+          |
|        |             # # # # # # # # # # # # # # # # # LB        |
|  71000 |       # # #                                              |
|        |   # #                                                    |
|  70000 | #                                                        |
|        +--+-----+-----+-----+-----+-----+-----+-----+-----+---> |
|           10    20    30    40    50    60    70    80    90       |
|                            Iteration                              |
|                                                                   |
|  Gap: 1.51%  |  LB: 72105.4  |  UB: 73211.8 +/- 361.1           |
+------------------------------------------------------------------+
```

**Widget mapping**:

| Element        | ratatui Widget       | Data Source Field             |
| -------------- | -------------------- | ----------------------------- |
| LB series      | `Dataset` (line)     | `lower_bound` (accumulated)   |
| UB series      | `Dataset` (line)     | `upper_bound` (accumulated)   |
| CI error band  | `Dataset` (area)     | `upper_bound +/- ci_95`       |
| X-axis         | `Axis`               | `iteration` or `wall_time_ms` |
| Y-axis         | `Axis`               | Cost value (linear or log)    |
| Gap annotation | `Paragraph` (footer) | `gap` (latest)                |
| Bound summary  | `Paragraph` (footer) | Latest LB, UB, CI values      |

### 3.2 Iteration Detail

**Purpose**: Display per-iteration timing breakdown and resource metrics.

**Primary data source**: `IterationSummary` events.

**Update frequency**: Once per iteration (on `IterationSummary` event).

**Layout**:

```
+--[ Iteration Detail ]--------------------------------------------+
|                                                                   |
|  Iteration: 87 / --      Wall Time: 18m 02s     Avg: 12.4s/iter |
|                                                                   |
|  Timing Breakdown (iteration 87):                                 |
|  +-----------------------------------------------------------+   |
|  | Forward Pass  [=============================        ] 7.2s |   |
|  | Backward Pass [==================                   ] 4.5s |   |
|  | Fwd Sync      [=                                    ] 0.2s |   |
|  | Cut Sync      [==                                   ] 0.4s |   |
|  | Checkpoint    [                                     ] 0.0s |   |
|  | Other         [                                     ] 0.1s |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  LP Solves: 23,400  |  Memory Peak: 4,812 MB                     |
|                                                                   |
|  Iteration History (last 20):                                     |
|  +--iter--+--total--+---fwd---+---bwd---+--sync--+--lp_solves--+ |
|  |   87   | 12.4s   |  7.2s   |  4.5s   | 0.6s   |   23,400    | |
|  |   86   | 12.1s   |  7.0s   |  4.3s   | 0.6s   |   23,400    | |
|  |   85   | 11.8s   |  6.9s   |  4.2s   | 0.6s   |   23,400    | |
|  |  ...   |  ...    |  ...    |  ...    |  ...   |    ...      | |
|  +--------+---------+---------+---------+--------+-------------+ |
+------------------------------------------------------------------+
```

**Widget mapping**:

| Element         | ratatui Widget | Data Source Field                          |
| --------------- | -------------- | ------------------------------------------ |
| Progress header | `Paragraph`    | `iteration`, `wall_time_ms`                |
| Timing bars     | `BarChart`     | `forward_ms`, `backward_ms` (from summary) |
| LP solve count  | `Paragraph`    | `lp_solves`                                |
| Memory peak     | `Paragraph`    | `memory_peak_mb`                           |
| Iteration table | `Table`        | Accumulated `IterationSummary` history     |

### 3.3 Cut Statistics

**Purpose**: Display cut pool health -- active, total, and removed cuts across stages.

**Primary data source**: `CutSyncComplete` events.

**Update frequency**: Once per iteration (on `CutSyncComplete` event).

**Layout**:

```
+--[ Cut Statistics ]----------------------------------------------+
|                                                                   |
|  Iteration: 87  |  Total Active: 8,340  |  Removed This Iter: 12 |
|                                                                   |
|  Per-Stage Active Cuts:                                           |
|  +-----------------------------------------------------------+   |
|  |  1 [====                 ]  72                             |   |
|  |  2 [======               ]  94                             |   |
|  |  3 [=======              ] 108                             |   |
|  |  4 [========             ] 120                             |   |
|  | .. [                     ]                                 |   |
|  | 60 [=====                ]  82                             |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Cut History:                                                     |
|  +---iter---+--generated--+---active---+---removed---+            |
|  |    87    |     120     |   8,340    |      12     |            |
|  |    86    |     120     |   8,232    |       8     |            |
|  |    85    |     120     |   8,120    |       5     |            |
|  |   ...    |     ...     |    ...     |      ...    |            |
|  +----------+-------------+------------+-------------+            |
+------------------------------------------------------------------+
```

**Widget mapping**:

| Element           | ratatui Widget | Data Source Field                                     |
| ----------------- | -------------- | ----------------------------------------------------- |
| Summary header    | `Paragraph`    | `iteration`, `cuts_active`, `cuts_removed`            |
| Per-stage bar     | `BarChart`     | Per-stage cut counts (derived from `CutSyncComplete`) |
| Cut history table | `Table`        | Accumulated `CutSyncComplete` history                 |

### 3.4 Resource Usage

**Purpose**: Display memory and CPU utilization across MPI ranks.

**Primary data source**: OS-level telemetry (polled), supplemented by `IterationSummary.memory_peak_mb`.

**Update frequency**: Continuous (polled at rendering tick rate).

> **Note**: This is the only view that does not update exclusively from iteration events. In co-hosted mode, the TUI polls OS-level process metrics (heap size via `jemalloc_ctl` or `/proc/self/statm`). In standalone pipe mode, this view shows only the `memory_peak_mb` field from `IterationSummary` events and displays a notice that real-time polling is unavailable.

**Layout**:

```
+--[ Resource Usage ]----------------------------------------------+
|                                                                   |
|  Process Memory:                                                  |
|  Heap:     4,812 MB  |  RSS: 5,120 MB  |  Peak: 5,240 MB        |
|                                                                   |
|  Memory Breakdown (estimated):                                    |
|  +-----------------------------------------------------------+   |
|  | Cut Pool    [=========================          ] 2,400 MB |   |
|  | Solver WS   [==============                     ] 1,200 MB |   |
|  | Scenario    [======                             ]   600 MB |   |
|  | System Data [====                               ]   400 MB |   |
|  | Other       [==                                 ]   212 MB |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Memory Over Time:                                                |
|  5500 |                                                           |
|  5000 |              * * * * * * * * * * * * * *                   |
|  4500 |        * * *                                              |
|  4000 |    * *                                                    |
|  3500 | *                                                         |
|       +--+-----+-----+-----+-----+-----+-----+----->             |
|          10    20    30    40    50    60    70                    |
|                         Iteration                                 |
+------------------------------------------------------------------+
```

**Widget mapping**:

| Element            | ratatui Widget | Data Source                                     |
| ------------------ | -------------- | ----------------------------------------------- |
| Memory summary     | `Paragraph`    | OS telemetry (co-hosted) or `memory_peak_mb`    |
| Breakdown bars     | `BarChart`     | Estimated from allocator stats (co-hosted only) |
| Memory time series | `Chart`        | Accumulated `memory_peak_mb` from summaries     |

### 3.5 Communication

**Purpose**: Display MPI synchronization timing and overhead.

**Primary data source**: `ForwardSyncComplete` and `CutSyncComplete` events.

**Update frequency**: Once per iteration.

**Layout**:

```
+--[ Communication ]-----------------------------------------------+
|                                                                   |
|  Iteration: 87  |  Total Sync Time: 0.6s  |  Overhead: 4.8%      |
|                                                                   |
|  Sync Breakdown (iteration 87):                                   |
|  +-----------------------------------------------------------+   |
|  | Fwd Allreduce  [====                              ] 0.2s  |   |
|  | Cut Allgatherv [=========                         ] 0.4s  |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Sync Overhead History:                                           |
|  10% |                                                            |
|   8% |  *                                                         |
|   6% |    * *                                                     |
|   4% |        * * * * * * * * * * * * * * * * *                    |
|   2% |                                                            |
|      +--+-----+-----+-----+-----+-----+-----+----->              |
|         10    20    30    40    50    60    70                     |
|                         Iteration                                 |
|                                                                   |
|  Sync History:                                                    |
|  +--iter--+--fwd_sync--+--cut_sync--+--total--+--overhead%--+     |
|  |   87   |   0.20s    |   0.40s    | 0.60s   |    4.8%     |     |
|  |   86   |   0.19s    |   0.38s    | 0.57s   |    4.7%     |     |
|  |  ...   |    ...     |    ...     |  ...    |    ...      |     |
|  +---------+-----------+-----------+---------+-------------+     |
+------------------------------------------------------------------+
```

**Widget mapping**:

| Element            | ratatui Widget | Data Source Field                                                  |
| ------------------ | -------------- | ------------------------------------------------------------------ |
| Summary header     | `Paragraph`    | Derived from sync times and iteration time                         |
| Sync breakdown     | `BarChart`     | `ForwardSyncComplete.sync_time_ms`, `CutSyncComplete.sync_time_ms` |
| Overhead chart     | `Chart`        | `(fwd_sync + cut_sync) / iteration_time` accumulated               |
| Sync history table | `Table`        | Accumulated sync event history                                     |

## 4. Convergence Plot Data Model

**Canonical data source**: The convergence plot data is the per-iteration output record defined in [Convergence Monitoring](../architecture/convergence-monitoring.md) SS2.4. The TUI does not require any fields beyond what SS2.4 already defines. The `ConvergenceUpdate` Rust struct is the shared type definition.

### 4.1 Field-to-Axis Mapping

| SS2.4 Field         | Type  | Plot Element        | Axis Assignment    | Rendering Notes                                                      |
| ------------------- | ----- | ------------------- | ------------------ | -------------------------------------------------------------------- |
| `iteration`         | `i32` | X-axis tick         | Primary X-axis     | 1-based integer; used as default X-axis                              |
| `wall_time_ms`      | `i64` | X-axis tick         | Alternative X-axis | Toggled via `t` key; displayed as `HH:MM:SS`                         |
| `lower_bound`       | `f64` | Line series         | Primary Y-axis     | Monotonically non-decreasing; rendered as solid line with `#` marker |
| `upper_bound`       | `f64` | Line series         | Primary Y-axis     | Rendered as solid line with `*` marker                               |
| `upper_bound_std`   | `f64` | Error band width    | (derived)          | Used to compute CI band: `upper_bound +/- ci_95`                     |
| `ci_95`             | `f64` | Error band extent   | Primary Y-axis     | Rendered as shaded area around upper bound series                    |
| `gap`               | `f64` | Annotation / series | Secondary Y-axis   | Displayed as percentage in footer; optionally as separate series     |
| `iteration_time_ms` | `i64` | (not plotted)       | N/A                | Used in Iteration Detail view, not in convergence plot               |

### 4.2 Scale Options

| Option           | Default | Toggle Key | Description                                            |
| ---------------- | ------- | ---------- | ------------------------------------------------------ |
| Linear Y-axis    | Yes     | `l`        | Standard linear scale for cost values                  |
| Logarithmic Y    | No      | `l`        | Log scale; useful when bounds span orders of magnitude |
| Iteration X-axis | Yes     | `t`        | X-axis shows iteration number                          |
| Wall-time X-axis | No      | `t`        | X-axis shows cumulative wall-clock time                |

### 4.3 Error Bar Rendering

The 95% confidence interval is rendered as a band around the upper bound series:

- **Upper edge**: `upper_bound + ci_95`
- **Lower edge**: `upper_bound - ci_95`
- **Rendering**: In terminals with 256+ colors, the band is rendered as a dim-colored area. In 16-color terminals, the band boundaries are rendered as dotted lines.

### 4.4 Auto-Scaling

The Y-axis auto-scales to fit all visible data points with 5% padding above and below. When the user scrolls to a subset of iterations, the Y-axis rescales to the visible range. The X-axis always starts at iteration 1 (or time 0) and extends to the latest received iteration.

## 5. Interactive Features

All interactive features that read training state operate at **iteration boundaries only**. The training loop is not designed for mid-iteration inspection -- solver workspaces and cut pools are in flux during forward/backward passes.

### 5.1 Pause/Resume Training

| Attribute           | Value                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| **Trigger**         | `p` key                                                                                           |
| **Mechanism**       | Sets a pause flag checked at the iteration boundary (between Step 7 and Step 1 of next iteration) |
| **Training impact** | Training loop blocks until resume. No mid-iteration pause -- iterations are atomic                |
| **Resume**          | `p` key again (toggle)                                                                            |
| **Status bar**      | Shows `PAUSED` indicator when paused                                                              |
| **Availability**    | Co-hosted mode only. In pipe mode, training is in a separate process and cannot be paused         |

### 5.2 Inspect Cuts at a Specific Stage

| Attribute           | Value                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------- |
| **Trigger**         | `c` key, then enter stage number                                                         |
| **Mechanism**       | Read-only access to the in-memory cut pool for the selected stage                        |
| **Constraint**      | Requires training to be paused, or reads a snapshot taken at the last iteration boundary |
| **Display**         | Overlay panel showing cut count, most recent cut coefficients, active/inactive status    |
| **Training impact** | None (read-only)                                                                         |
| **Availability**    | Co-hosted mode only                                                                      |

### 5.3 Compare Scenario Forward Paths

| Attribute           | Value                                                                                |
| ------------------- | ------------------------------------------------------------------------------------ |
| **Trigger**         | `s` key, then select scenario indices                                                |
| **Mechanism**       | Read-only access to visited states from the most recent forward pass                 |
| **Display**         | Side-by-side table of storage volumes and costs across stages for selected scenarios |
| **Training impact** | None (read-only)                                                                     |
| **Availability**    | Co-hosted mode only                                                                  |

### 5.4 Adjust Stopping Rules

| Attribute           | Value                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Trigger**         | `r` key, then select parameter to modify                                                                       |
| **Mechanism**       | Modifies stopping rule parameters (tolerance, iteration limit) at runtime                                      |
| **Constraint**      | The convergence monitor reads updated parameters at each iteration; changes take effect at the next evaluation |
| **Adjustable**      | `iteration_limit`, `time_limit.seconds`, `bound_stalling.tolerance`, `bound_stalling.iterations`               |
| **Training impact** | Changes when the training terminates; does not affect computation                                              |
| **Availability**    | Co-hosted mode only                                                                                            |

### 5.5 Export Snapshot

| Attribute           | Value                                                                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trigger**         | `e` key                                                                                                                                        |
| **Mechanism**       | Writes current convergence history and cut pool to files using the existing checkpoint mechanism from [Checkpointing](../hpc/checkpointing.md) |
| **Output**          | Checkpoint directory with FCF cuts, convergence history, and iteration metadata                                                                |
| **Training impact** | Brief I/O pause for serialization (same as periodic checkpoint)                                                                                |
| **Availability**    | Co-hosted mode. In pipe mode, exports the locally accumulated convergence history as JSON                                                      |

### 5.6 Feature Availability by Mode

| Feature               | Co-Hosted | Standalone Pipe | Offline Replay |
| --------------------- | --------- | --------------- | -------------- |
| Pause/resume          | Yes       | No              | N/A            |
| Inspect cuts          | Yes       | No              | No             |
| Compare scenarios     | Yes       | No              | No             |
| Adjust stopping rules | Yes       | No              | N/A            |
| Export snapshot       | Yes       | Partial (JSON)  | N/A            |
| View switching        | Yes       | Yes             | Yes            |
| Scrolling/navigation  | Yes       | Yes             | Yes            |
| Scale toggles         | Yes       | Yes             | Yes            |

## 6. Input Model

### 6.1 Keyboard Shortcuts

| Key          | Action                                             | Context            |
| ------------ | -------------------------------------------------- | ------------------ |
| `1`          | Switch to Convergence Plot view                    | Global             |
| `2`          | Switch to Iteration Detail view                    | Global             |
| `3`          | Switch to Cut Statistics view                      | Global             |
| `4`          | Switch to Resource Usage view                      | Global             |
| `5`          | Switch to Communication view                       | Global             |
| `Tab`        | Cycle to next view                                 | Global             |
| `Shift+Tab`  | Cycle to previous view                             | Global             |
| `p`          | Toggle pause/resume training                       | Global (co-hosted) |
| `c`          | Open cut inspection overlay                        | Global (co-hosted) |
| `s`          | Open scenario comparison overlay                   | Global (co-hosted) |
| `r`          | Open stopping rule adjustment dialog               | Global (co-hosted) |
| `e`          | Export snapshot                                    | Global             |
| `t`          | Toggle X-axis: iteration / wall-time               | Convergence Plot   |
| `l`          | Toggle Y-axis: linear / logarithmic                | Convergence Plot   |
| `j` / `Down` | Scroll down in tables                              | Table views        |
| `k` / `Up`   | Scroll up in tables                                | Table views        |
| `g`          | Jump to top of table                               | Table views        |
| `G`          | Jump to bottom of table                            | Table views        |
| `?`          | Toggle help overlay                                | Global             |
| `q`          | Quit TUI (training continues if co-hosted)         | Global             |
| `Ctrl+c`     | Force quit (sends SIGINT to training if co-hosted) | Global             |

### 6.2 Navigation Model

Navigation follows vi-style conventions:

- **View switching**: Number keys `1`--`5` for direct access; `Tab`/`Shift+Tab` for cycling
- **Scrolling**: `j`/`k` or arrow keys for line-by-line; `g`/`G` for top/bottom
- **Overlays**: `Esc` closes any open overlay (cut inspection, scenario comparison, stopping rule dialog, help)
- **Input fields**: When an overlay has a text input (e.g., stage number for cut inspection), `Enter` confirms and `Esc` cancels

## 7. Deployment Modes

### 7.1 Co-Hosted Mode

The primary deployment mode. The TUI is a library crate linked into the `cobre` binary via the `tui` Cargo feature flag. Activation: `cobre run --tui /path/to/case`.

**Lifecycle**:

1. `cobre-cli` initializes the training loop with an event broadcast channel
2. `cobre-cli` spawns the TUI on a dedicated thread, passing the broadcast receiver
3. The TUI thread initializes crossterm raw mode and the ratatui terminal
4. The TUI event loop polls: (a) broadcast channel for training events, (b) crossterm for input events, (c) tick timer for rendering
5. On `TrainingFinished` or `q` key, the TUI restores the terminal and the thread exits
6. If the user presses `q`, training continues in the background (the TUI thread detaches from the broadcast channel)

**Feature flag**: When compiled without the `tui` feature, `--tui` is rejected with a clear error message. The `cobre-tui` dependency and `ratatui`/`crossterm` transitive dependencies are excluded from the build.

### 7.2 Standalone Pipe Mode

The secondary deployment mode. The TUI is compiled as a standalone `cobre-tui` binary that reads JSON-lines from stdin.

**Lifecycle**:

1. `cobre-tui` initializes crossterm raw mode and the ratatui terminal
2. The event loop polls: (a) stdin for JSON-lines, (b) crossterm for input events, (c) tick timer for rendering
3. Each stdin line is deserialized as a JSON-lines envelope (SS2.2)
4. On stdin EOF, the TUI enters "replay complete" mode
5. On `q` key, the TUI restores the terminal and exits

**Reduced feature set**: Interactive features that require in-process access to training state (pause, cut inspection, scenario comparison, stopping rule adjustment) are unavailable. The TUI displays a notice when the user attempts these features: "Feature unavailable in pipe mode".

### 7.3 Offline Replay Mode

The TUI can render convergence data from a completed study by reading the `convergence.parquet` file from an output directory.

**Activation**: `cobre-tui --replay /path/to/output/training/convergence.parquet`

**Behavior**: The TUI loads all rows from the Parquet file into memory, populates the `ConvergenceUpdate` history, and renders the Convergence Plot view. The user can scroll through iterations and toggle scale options. Views that require live events (Iteration Detail timing breakdown, Cut Statistics, Communication) show a "data not available in replay mode" notice.

## 8. Rendering Architecture

### 8.1 Immediate-Mode Rendering

The TUI uses ratatui's immediate-mode rendering pattern: the entire screen is redrawn on every tick. There is no persistent widget state -- the view functions read the current event history and produce a frame.

```
loop {
    // 1. Poll events (non-blocking, up to tick deadline)
    poll_training_events(&mut state);
    poll_input_events(&mut state);

    // 2. Render current state
    terminal.draw(|frame| {
        render_status_bar(frame, &state);
        match state.active_view {
            View::Convergence => render_convergence(frame, &state),
            View::Iteration   => render_iteration(frame, &state),
            View::Cuts         => render_cuts(frame, &state),
            View::Resource     => render_resource(frame, &state),
            View::Communication => render_communication(frame, &state),
        }
        if state.show_help {
            render_help_overlay(frame);
        }
    });

    // 3. Sleep until next tick
    sleep_until(next_tick);
}
```

### 8.2 Tick Rate

The rendering tick rate is **10 Hz** (100 ms per frame). This provides smooth visual updates without excessive CPU usage. Event processing occurs between ticks -- all events received since the last tick are batched and applied before rendering.

### 8.3 Terminal Size Requirements

| Dimension     | Minimum | Recommended | Maximum  |
| ------------- | ------- | ----------- | -------- |
| Width (cols)  | 80      | 120         | No limit |
| Height (rows) | 24      | 40          | No limit |

**Undersized terminal handling**: If the terminal is resized below the minimum dimensions, the TUI replaces all views with a centered message:

```
+----------------------------------------------+
|                                              |
|   Terminal too small.                        |
|   Minimum: 80 x 24                          |
|   Current: 60 x 18                          |
|   Please resize your terminal.              |
|                                              |
+----------------------------------------------+
```

The TUI does not crash or produce corrupted output on undersized terminals. It resumes normal rendering as soon as the terminal is resized above the minimum.

### 8.4 Color Support

| Level     | Detection                               | Rendering                                                        |
| --------- | --------------------------------------- | ---------------------------------------------------------------- |
| 16-color  | Minimum supported                       | LB line: green, UB line: yellow, CI: dotted boundaries, gap: red |
| 256-color | Preferred                               | CI band as dim area, gradient coloring for bar charts            |
| Truecolor | Optional                                | Smooth CI gradient, richer chart coloring                        |
| No color  | `NO_COLOR` env var or `--no-color` flag | Monochrome: LB as `#`, UB as `*`, CI as `+/- ` text              |

Color capability is detected from the `TERM` and `COLORTERM` environment variables. The `NO_COLOR` convention (https://no-color.org/) is respected.

### 8.5 Unicode Requirements

The TUI requires Unicode box-drawing characters for borders and chart axes:

| Character Set             | Usage                       | Required                                  |
| ------------------------- | --------------------------- | ----------------------------------------- |
| Box Drawing (U+2500)      | View borders, table grid    | Yes                                       |
| Block Elements (U+2580)   | Bar chart fills             | Yes                                       |
| Braille Patterns (U+2800) | High-resolution plot points | Preferred (fallback to ASCII `*` and `#`) |

**Fallback**: If the terminal does not support Unicode (detected via locale settings or `--ascii` flag), the TUI falls back to ASCII box-drawing characters (`+`, `-`, `|`) and ASCII plot markers.

### 8.6 Terminal Resize Handling

On terminal resize (detected via crossterm `Event::Resize`):

1. The layout is recalculated for the new dimensions
2. The current frame is immediately redrawn at the new size
3. If the new size is below minimum, the "terminal too small" message is shown
4. No event data is lost during resize

## Cross-References

- [Structured Output](./structured-output.md) -- JSON-lines streaming protocol (SS3) that defines the event envelope types consumed by the TUI in pipe mode; shared event type definitions
- [Convergence Monitoring](../architecture/convergence-monitoring.md) -- Per-iteration output record (SS2.4) that is the canonical data source for the convergence plot; stopping rules (SS1) that the TUI can adjust at runtime
- [Training Loop](../architecture/training-loop.md) -- Iteration lifecycle (SS2.1) defining the 7 steps that produce the 7 iteration-scoped event types consumed by the TUI
- [Checkpointing](../hpc/checkpointing.md) -- Checkpoint mechanism used by the snapshot export feature (SS5.5)
- [Configuration Reference](../configuration/configuration-reference.md) -- Stopping rule parameters that the TUI can adjust at runtime
