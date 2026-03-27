# Installation

This page gets you to a working `cobre` installation in the shortest path possible.
For alternative methods — including `cargo install`, building from source, and the
full platform support table — see [Installation (User Guide)](../guide/installation.md).

---

## Fastest Path: Pre-built Binary

Download and install the pre-built binary for your platform with a single command.

### Linux and macOS

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cobre-rs/cobre/releases/latest/download/cobre-cli-installer.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/cobre-rs/cobre/releases/latest/download/cobre-cli-installer.ps1 | iex"
```

The installer places `cobre` in `$CARGO_HOME/bin` (typically `~/.cargo/bin`). Ensure
that directory is in your `PATH`.

---

## Verify the Installation

```bash
cobre version
```

Expected output:

```
cobre   v0.2.0
solver: HiGHS
comm:   local
zstd:   enabled
arch:   x86_64-linux
build:  release (lto=thin)
```

The exact version, arch, and build fields will vary by platform and release.

---

## Next Steps

- [Quickstart](./quickstart.md) — run your first study in three commands
- [Installation (User Guide)](../guide/installation.md) — `cargo install`, build from source, and platform table
