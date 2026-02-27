//! # cobre-mcp
//!
//! Model Context Protocol (MCP) server for the [Cobre](https://github.com/cobre-rs/cobre)
//! power systems solver.
//!
//! Exposes Cobre operations as MCP tools, data artifacts as resources, and
//! guided workflows as prompts. Enables AI coding assistants and automation
//! agents to validate cases, run training, query results, and compare policies
//! without parsing CLI text output.
//!
//! ## Transport modes
//!
//! - **stdio** — local integration with Claude Desktop, VS Code extensions,
//!   and other MCP clients that spawn the server as a subprocess.
//! - **streamable-HTTP/SSE** — remote or multi-client deployments.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.

fn main() {
    todo!("cobre-mcp: MCP server not yet implemented")
}
