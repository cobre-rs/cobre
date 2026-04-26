# Interfaces

This section specifies the four interface layers through which external agents, tools, and users interact with Cobre beyond the traditional MPI batch execution path. Each interface layer is implemented as a separate crate with its own responsibility boundary, but all share the same event types defined in `cobre-core` and operate in single-process mode (no MPI).

| Interface                                              | Crate        | Purpose                                                                                           |
| ------------------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------- |
| [Structured Output](./interfaces/structured-output.md) | cobre-cli    | JSON and JSON-lines output protocol for CLI responses and progress streaming                      |
| [MCP Server](./interfaces/mcp-server.md)               | cobre-mcp    | Model Context Protocol server exposing Cobre operations as tools, resources, and prompts          |
| [Python Bindings](./interfaces/python-bindings.md)     | cobre-python | PyO3-based Python module for programmatic access to validation, training, simulation, and results |
| [Terminal UI](./interfaces/terminal-ui.md)             | cobre-tui    | Interactive terminal dashboard for real-time training monitoring and convergence visualization    |

All four interfaces consume the shared event stream architecture defined in the [Training Loop](./architecture/training-loop.md) and [Convergence Monitoring](./architecture/convergence-monitoring.md) specs. The structured output spec (CLI JSON protocol) serves as the foundational schema layer -- the MCP server, Python bindings, and TUI all reuse its error schema and event type definitions.
