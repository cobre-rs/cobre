# cobre-sddp

Stochastic Dual Dynamic Programming (SDDP) algorithm for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

Implements the SDDP algorithm (Pereira & Pinto, 1991) for long-term hydrothermal
dispatch and energy planning. The crate covers the full solve cycle: forward
pass scenario simulation, backward pass Benders cut generation, cut management
with cut selection (Level-1, LML1, and dominated-cut pruning), CVaR risk
measures, convergence monitoring, policy warm-start and resume from checkpoint,
and annual discount rate support. Designed for hybrid MPI + thread-level
parallelism via [ferrompi](https://github.com/cobre-rs/ferrompi).

## When to Use

Use `cobre-sddp` when you need programmatic access to the SDDP algorithm —
embedding it in a custom orchestration layer, running parameter sweeps, or
integrating the solver into a larger application. For single-study command-line
use, prefer `cobre-cli`, which wraps this crate.

## Key Types

- **`TrainingConfig`** — algorithm parameters: iteration budget, forward scenario
  count, checkpoint cadence, warm-start cut count, and cut selection strategy
- **`TrainingContext`** — runtime state shared across all iterations of the
  training loop (cut pools, workspaces, convergence monitor)
- **`CutPool`** — pre-allocated storage for Benders cuts with active/inactive
  bookkeeping
- **`CutSelectionStrategy`** — enum controlling cut pool pruning: `Level1`
  (activity threshold), `Lml1` (memory window), and `Dominated` (geometric
  dominance at visited states)
- **`SimulationConfig`** — parameters for the post-training simulation run
- **`ConvergenceMonitor`** — tracks lower bound, statistical upper bound, and
  gap closure across iterations

## Links

| Resource                    | URL                                                        |
| --------------------------- | ---------------------------------------------------------- |
| Book — SDDP crate reference | <https://cobre-rs.github.io/cobre/crates/sddp.html>        |
| API docs                    | <https://docs.rs/cobre-sddp/latest/cobre_sddp/>            |
| Repository                  | <https://github.com/cobre-rs/cobre>                        |
| Changelog                   | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable. See the [main repository](https://github.com/cobre-rs/cobre) for the current release.

## License

Apache-2.0
