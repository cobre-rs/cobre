# Roadmap

The Cobre post-v0.1.0 roadmap documents deferred features, HPC optimizations,
and post-MVP crates planned for future releases. It covers implementation-level
work items only; for methodology-level roadmap pages (algorithm theory, spec
evolution), see the
[cobre-docs roadmap](https://cobre-rs.github.io/cobre-docs/roadmap/).

The roadmap is maintained in the repository at
[`docs/ROADMAP.md`](https://github.com/cobre-rs/cobre/blob/main/docs/ROADMAP.md).

## Sections

The roadmap is organized into four areas:

- **Inflow Truncation Methods** -- Two additional non-negativity treatment
  methods (`Truncation` and `TruncationWithPenalty`) deferred from v0.1.0.
- **HPC Optimizations** -- Performance improvements beyond the rayon baseline,
  grouped into near-term (v0.1.x/v0.2.x) and longer-term (v0.3+) items.
- **Post-MVP Crates** -- Implementation plans for the three stubbed workspace
  crates: `cobre-mcp`, `cobre-python`, and `cobre-tui`.
- **Algorithm Extensions** -- Deferred solver variants: CVaR risk measure,
  multi-cut formulation, and infinite-horizon policy graphs.
