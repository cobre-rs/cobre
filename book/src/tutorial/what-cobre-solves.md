# What Cobre Solves

## The Problem

Power systems with large hydroelectric capacity face a fundamental dilemma:
water stored in reservoirs today could generate cheap electricity now, but
saving it might avoid burning expensive fuel months from now. The decision is
complicated by uncertainty -- nobody knows how much rain will fall next month.

This is the **hydrothermal dispatch problem**: given a network of hydro plants,
thermal generators, transmission lines, and uncertain future inflows, find the
least-cost operating policy over a multi-year horizon. It is one of the central
problems in energy planning for countries like Brazil, Colombia, and Norway.

The problem is hard because decisions are coupled across time (water used today
is gone tomorrow), across space (reservoirs in a cascade share the same river),
and across scenarios (a drought year requires completely different decisions
than a wet year).

## How SDDP Works (Conceptual)

Stochastic Dual Dynamic Programming (SDDP) solves this problem by iterating
between two phases:

1. **Forward pass** -- Simulate the system from the first stage to the last,
   making decisions at each stage under sampled uncertainty (random inflows).
   Record the resulting costs and state transitions.

2. **Backward pass** -- Starting from the last stage and working backwards,
   use the forward decisions to build "cuts" -- linear approximations of the
   future cost. These cuts capture the trade-off: "if you use this much water
   now, the expected future cost is at least this much."

Each iteration improves the policy. After enough iterations, the lower bound
(from cuts) and the upper bound (from forward simulations) converge, producing
a near-optimal dispatch policy.

## What Cobre Provides

- **System modeling** -- Define hydro plants (with cascades, variable-head
  production, evaporation), thermal units, transmission lines, non-controllable
  sources, and user-defined constraints.
- **Stochastic scenario generation** -- Fit periodic autoregressive (PAR)
  models to historical inflow records and generate correlated scenarios.
- **SDDP solver** -- Train a dispatch policy with configurable stopping rules,
  risk measures, and cut selection strategies.
- **Simulation** -- Evaluate the trained policy across thousands of scenarios,
  producing per-scenario cost breakdowns and operational trajectories.
- **Multiple interfaces** -- Use the CLI for batch runs, Python for interactive
  analysis, or the MCP server for AI agent workflows.
