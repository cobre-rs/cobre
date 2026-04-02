# cobre-solver

LP/MIP solver abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
power systems ecosystem.

Defines a backend-agnostic `SolverInterface` trait for LP and MIP problem
construction, solving, dual extraction, and basis warm-starting. The default
backend is [HiGHS](https://highs.dev), a production-grade open-source solver
well-suited to the iterative LP workloads of power system optimization. The
crate includes a 12-level retry escalation strategy for numerically difficult
LPs: when HiGHS returns an infeasible or numerically unstable status, the
solver retries with progressively more aggressive scaling, presolve, and
simplex strategy options before propagating failure. Additional backends (Clp,
CPLEX, Gurobi) can be added behind feature flags without changing algorithm
crates.

## When to Use

Depend on `cobre-solver` directly when you are writing an optimization
algorithm that needs to build and solve LP subproblems and you want
backend-portability without coupling to HiGHS internals. If you only need to
run the full SDDP pipeline, depend on `cobre-sddp` instead, which manages the
solver lifecycle for you.

## Key Types

- **`SolverInterface`** — the core trait every backend must implement; defines
  problem construction, solve, and dual/basis extraction methods
- **`HighsSolver`** — the default HiGHS backend; feature-gated behind `highs`
  (enabled by default)
- **`Basis`** — LP basis snapshot for warm-starting a subsequent solve on a
  structurally related problem
- **`LpSolution`** — solved LP result carrying primal values, duals, and
  objective value
- **`SolverStatistics`** — per-solve diagnostics including iteration count,
  wall-clock time, basis rejections, and retry escalation level reached
- **`StageTemplate`** — pre-built LP structure for a single time stage,
  cloned and patched each iteration to avoid repeated matrix assembly

## Links

| Resource   | URL                                                        |
| ---------- | ---------------------------------------------------------- |
| Book       | <https://cobre-rs.github.io/cobre/crates/solver.html>      |
| API Docs   | <https://docs.rs/cobre-solver/latest/cobre_solver/>        |
| Repository | <https://github.com/cobre-rs/cobre>                        |
| CHANGELOG  | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable.

## License

Apache-2.0
