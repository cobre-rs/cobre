# cobre-sddp

Stochastic Dual Dynamic Programming (SDDP) algorithm for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

Implements the SDDP algorithm (Pereira & Pinto, 1991) for long-term hydrothermal
dispatch and energy planning. The crate covers the full solve cycle: forward
pass scenario simulation, backward pass Benders cut generation, cut management
with cut selection (Level-1, LML1, dominated-cut pruning, and Dynamic Cut Selection (DCS)), CVaR risk
measures, convergence monitoring, policy warm-start and resume from checkpoint,
and annual discount rate support. Designed for hybrid MPI + thread-level
parallelism via [ferrompi](https://github.com/cobre-rs/ferrompi).

## When to Use

Use `cobre-sddp` when you need programmatic access to the SDDP algorithm —
embedding it in a custom orchestration layer, running parameter sweeps, or
integrating the solver into a larger application. For single-study command-line
use, prefer `cobre-cli`, which wraps this crate.

## Risk aggregation contract

`RiskMeasure::CVaR { alpha, lambda }` evaluates
`(1-lambda) E[Z] + lambda CVaR_alpha[Z]`, where `alpha` is the upper-tail
probability fraction. Both scalar evaluation and backward-cut aggregation
reserve `(1-lambda)*p[i]` on every scenario, then allocate the remaining
`lambda` mass in descending cost order with an additional cap
`lambda*p[i]/alpha`. The final weights sum to one and stay between the
expectation floor and that floor plus the cap. Scratch buffers are reused.

```rust
use cobre_sddp::risk_measure::RiskMeasure;

let risk = RiskMeasure::CVaR { alpha: 0.15, lambda: 0.4 };
let value = risk.evaluate_risk(&[0.0, 100.0], &[0.5, 0.5]);
assert!((value - 70.0).abs() < 1e-10); // 0.6 * 50 + 0.4 * 100
```

The regression tests `cvar_mixture_preserves_expectation_floor_in_value_and_cut`
and `cvar_mixture_matches_primal_tail_formula_and_envelope` pin the floor and
compare the result with the primal CVaR formula, including partial tail atoms.
Retrain mixed-risk policies computed without this floor; their existing cuts
can overestimate the intended nested risk objective. A sampled forward mean
still estimates risk-neutral policy cost and is not a compatible upper bound
for a risk-adjusted optimality gap. The [methodology reference](https://docs.cobre-rs.dev/math/risk-measures/)
owns the full derivation.

## Key Types

- **`TrainingConfig`** — algorithm parameters: iteration budget, forward scenario
  count, checkpoint cadence, warm-start cut count, and cut selection strategy
- **`TrainingContext`** — runtime state shared across all iterations of the
  training loop (cut pools, workspaces, convergence monitor)
- **`CutPool`** — pre-allocated storage for Benders cuts with active/inactive
  bookkeeping
- **`CutSelectionStrategy`** — enum controlling cut pool pruning:
  - `Level1` — deactivates cuts below `tie_tolerance` of the per-state max at every visited state
  - `Lml1` — deactivates cuts that are not the oldest eligible within `tie_tolerance` at any visited state
  - `Dominated` — geometric dominance at visited states, using `threshold` as the tolerance
  - `Dynamic` — lazy incremental scheme (DCS): adds at most `nadic` cuts per inner re-solve round
    (the inner loop repeats up to `max_inner_iterations` rounds per backward solve) that violate the
    current LP solution by more than `epsilon_viol` (the `CutSelectionStrategy::Dynamic` API field;
    the `config.json` key is `violation_tolerance`, which falls back to `tie_tolerance` when absent);
    never deactivates cuts from the pool

  Both `Level1` and `Lml1` use `tie_tolerance` (default `1e-10`) to control how closely a cut must
  approach the per-state maximum to be retained. The `memory_window` config field is deprecated and
  silently ignored; use `tie_tolerance` instead.

- **`SimulationConfig`** — parameters for the post-training simulation run
- **`ConvergenceMonitor`** — tracks lower bound, statistical upper bound, and
  gap closure across iterations

## Error handling (`SddpError`)

All fallible operations return `Result<T, SddpError>` (`Send + Sync + 'static`).

| Variant               | Trigger                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `Solver`              | An LP subproblem solve failed (wraps `cobre_solver::SolverError`) after all retries                                       |
| `Communication`       | An MPI collective operation failed (wraps `cobre_comm::CommError`)                                                        |
| `Stochastic`          | Scenario generation or PAR model validation failed                                                                        |
| `Io`                  | Case directory loading or validation failed (wraps `cobre_io::LoadError`)                                                 |
| `Validation`          | Algorithm configuration is semantically invalid                                                                           |
| `Infeasible`          | An LP subproblem was provably infeasible (carries `stage`, `iteration`, `scenario`) — distinct from `Solver`, a hard stop |
| `Simulation`          | A simulation-phase operation failed (LP failure, I/O, or policy issue)                                                    |
| `WireVersionMismatch` | A postcard-encoded broadcast payload's wire `version` does not match this binary — restart all ranks with the same binary |

## Feature flags

| Feature      | Default | Description                                                                                                                                                                     |
| ------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `highs`      | on      | Selects the HiGHS LP solver backend (`cobre-solver/highs`); gates the per-phase `HighsProfile` impl and the `FORWARD_PROFILE`/`BACKWARD_PROFILE`/`SIMULATION_PROFILE` constants |
| `clp`        | off     | Selects the CLP/CoinUtils LP solver backend (`cobre-solver/clp`); gates the per-phase `ClpProfile` impl                                                                         |
| `dhat-heap`  | off     | Enables DHAT heap profiling — replaces the global allocator, so never enable by default; only for the `dhat_baseline` example                                                   |
| `slow-tests` | off     | Opts in to the slow test suite (D-case sweep, FPHA plane-selection, forward-sampler convergence), ignored by default so `cargo test --workspace` stays fast                     |

## Testing

```
cargo test -p cobre-sddp
```

No external system libraries are required beyond the workspace default (HiGHS
is always available; MPI is optional via the `mpi` feature of `cobre-comm`).
The suite covers unit tests for each module's core logic, integration tests
using `LocalBackend` (single-rank) for the communication-involving modules,
and doc-tests for all public types with constructible examples.

## Links

| Resource                    | URL                                                        |
| --------------------------- | ---------------------------------------------------------- |
| Docs site                   | <https://docs.cobre-rs.dev/>                               |
| API docs                    | <https://docs.rs/cobre-sddp/latest/cobre_sddp/>            |
| Repository                  | <https://github.com/cobre-rs/cobre>                        |
| Changelog                   | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable. See the [main repository](https://github.com/cobre-rs/cobre) for the current release.

## License

Apache-2.0
