# cobre-tui

Interactive terminal UI for the [Cobre](https://github.com/cobre-rs/cobre) power systems solver.

Provides real-time SDDP training monitoring, convergence visualization, cut
inspection, and simulation progress tracking using `ratatui` and `crossterm`.
Consumed by `cobre-cli`; also usable standalone via stdin JSON-lines pipe.

## When to Use

Depend on `cobre-tui` directly when you are building a custom CLI binary that
needs Cobre's terminal dashboard without pulling in the full `cobre-cli`
command set. For end-user terminal monitoring, use `cobre-cli` which embeds
`cobre-tui` automatically.

## Key Types

None yet -- this crate is a stub.

## Links

| Resource   | URL                                                        |
| ---------- | ---------------------------------------------------------- |
| API Docs   | <https://docs.rs/cobre-tui/latest/cobre_tui/>              |
| Repository | <https://github.com/cobre-rs/cobre>                        |
| CHANGELOG  | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

Not yet implemented -- this crate is a stub.

## License

Apache-2.0
