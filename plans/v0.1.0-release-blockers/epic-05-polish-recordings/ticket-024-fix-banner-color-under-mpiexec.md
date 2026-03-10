# ticket-024: Fix Banner Color/Style Loss Under mpiexec

## Context

### Background

When the CLI is launched via `mpiexec`, the banner loses all ANSI color and styling because `console::colors_enabled_stderr()` returns `false` -- MPI redirects stderr through a pipe, not a TTY. Users running distributed SDDP studies see a plain-text banner even though their terminal supports color. Adding a `--color` CLI flag allows forcing color output, following the same env-var override pattern established by `--threads` / `COBRE_THREADS`.

### Relation to Epic

Epic 05 is about polish for v0.1.0. The banner is the first thing users see; maintaining its visual identity under MPI improves the production experience.

### Current State

- `banner.rs` calls `console::colors_enabled_stderr() && std::env::var_os("NO_COLOR").is_none()` to decide whether to emit ANSI escapes
- `console::set_colors_enabled_stderr(bool)` exists in the `console` crate (v0.16) and overrides the auto-detection
- The `Cli` struct in `main.rs` has subcommands `Init`, `Run`, `Validate`, `Report`, `Version` -- there are no global flags (flags are per-subcommand)
- `RunArgs` has per-subcommand flags: `--quiet`, `--no-banner`, `--verbose`, `--threads`, `--skip-simulation`, `--output`
- `run::execute()` calls `banner::print_banner()` and `progress::run_progress_thread()` which uses `indicatif::ProgressBar`
- `indicatif` uses its own terminal detection for progress bar rendering
- The `error.rs` module uses `console::style()` for colored error labels -- these also respect `console::colors_enabled_stderr()`

## Specification

### Requirements

1. Add a `--color <WHEN>` global CLI flag to the `Cli` struct with values `auto` (default), `always`, `never`
2. When `always`: call `console::set_colors_enabled_stderr(true)` before any subcommand execution
3. When `never`: call `console::set_colors_enabled_stderr(false)` before any subcommand execution
4. When `auto`: do not call `set_colors_enabled_stderr` -- keep current auto-detection behavior
5. Also honor `COBRE_COLOR` env var (following the existing `COBRE_*` env var pattern): `COBRE_COLOR=always` forces color on, `COBRE_COLOR=never` forces it off. The `--color` flag takes precedence over `COBRE_COLOR`.
6. Also honor `FORCE_COLOR=1` env var as a color-on signal (complementary to `NO_COLOR`), but only when `--color` is `auto` and `COBRE_COLOR` is not set. This follows the [force-color.org](https://force-color.org/) convention.

### Inputs/Props

- `--color <WHEN>` flag with `WHEN` being one of `auto`, `always`, `never`
- `COBRE_COLOR` env var (optional, overridden by `--color` if provided)
- `FORCE_COLOR` env var (optional, lowest priority)

### Outputs/Behavior

- When color is forced on, all stderr output (banner, progress bars, error messages) uses ANSI color escapes regardless of whether stderr is a TTY
- When color is forced off, no ANSI escapes are emitted
- The progress bar (`indicatif`) should also respect the color setting via `ProgressBar::with_draw_target` using `ProgressDrawTarget::term_like` or by setting `console::set_colors_enabled_stderr` early (which indicatif respects since it uses the `console` crate internally)

### Error Handling

- Invalid `--color` values are rejected by clap's `ValueEnum` derive (no manual validation needed)
- Invalid `COBRE_COLOR` values are silently ignored (fall back to `auto` behavior)

## Acceptance Criteria

- [ ] Given the `Cli` struct in `main.rs`, when reading its fields, then it has a `color` field of type `ColorWhen` with `#[arg(long, global = true, default_value = "auto")]`
- [ ] Given the command `cobre run --color always examples/1dtoy/ 2>&1 | cat`, when the output is inspected, then it contains ANSI escape sequences (`\x1b[`) in the banner output (piped through `cat` to prove color is forced even without a TTY)
- [ ] Given the command `cobre run --color never examples/1dtoy/`, when the stderr output is inspected, then it does not contain any `\x1b[` escape sequences
- [ ] Given `COBRE_COLOR=always` is set and no `--color` flag is provided, when `cobre run examples/1dtoy/ 2>&1 | cat` is run, then the output contains ANSI escape sequences
- [ ] Given `FORCE_COLOR=1` is set and no `--color` or `COBRE_COLOR` is set, when `cobre run examples/1dtoy/ 2>&1 | cat` is run, then the output contains ANSI escape sequences

## Implementation Guide

### Suggested Approach

1. Define a `ColorWhen` enum in `main.rs` (or a new `color.rs` module) with `#[derive(Clone, Copy, Debug, clap::ValueEnum)]` and variants `Auto`, `Always`, `Never`.
2. Add `#[arg(long, global = true, default_value = "auto")] color: ColorWhen` to the `Cli` struct. The `global = true` attribute makes it available on all subcommands.
3. In `main()`, after parsing args but before dispatching to the subcommand, resolve the color setting:
   ```rust
   fn resolve_color(cli_color: ColorWhen) {
       match cli_color {
           ColorWhen::Always => console::set_colors_enabled_stderr(true),
           ColorWhen::Never => console::set_colors_enabled_stderr(false),
           ColorWhen::Auto => {
               // Check COBRE_COLOR env var
               if let Ok(val) = std::env::var("COBRE_COLOR") {
                   match val.to_ascii_lowercase().as_str() {
                       "always" => console::set_colors_enabled_stderr(true),
                       "never" => console::set_colors_enabled_stderr(false),
                       _ => { /* ignore invalid values, keep auto */ }
                   }
               } else if std::env::var_os("FORCE_COLOR").is_some() {
                   console::set_colors_enabled_stderr(true);
               }
               // Otherwise: keep console's auto-detection
           }
       }
   }
   ```
4. Call `resolve_color(cli.color)` in `main()` before `logging::init_logging()`.
5. Add unit tests for `resolve_color` behavior using the `ENV_LOCK` pattern from Epic 01 learnings (mutex-guarded env var manipulation).
6. Add integration tests using `assert_cmd`:
   - `cobre run --color always` piped to check for `\x1b[` in stderr
   - `cobre run --color never` piped to check absence of `\x1b[` in stderr

### Key Files to Modify

- `crates/cobre-cli/src/main.rs` -- add `color` field to `Cli`, add `ColorWhen` enum, add `resolve_color()`, call it in `main()`
- `crates/cobre-cli/src/banner.rs` -- no changes needed (it already reads `console::colors_enabled_stderr()` which will be set by `resolve_color`)

### Patterns to Follow

- Follow the `resolve_thread_count` pattern from `run.rs` for resolution order: CLI flag > env var > auto
- Follow the `ENV_LOCK: Mutex<()>` pattern from Epic 01 for thread-safe env var tests
- Use `clap::ValueEnum` derive for the enum (same as how `--threads` uses `value_parser`)

### Pitfalls to Avoid

- Do not modify `banner.rs` or `error.rs` -- they already use `console::colors_enabled_stderr()` which will be set globally by `resolve_color`
- The `console::set_colors_enabled_stderr()` call must happen before ANY output -- place it at the very start of `main()` after arg parsing
- `indicatif` respects the `console` crate's color setting, so no separate configuration is needed for progress bars
- The `global = true` flag in clap means `--color` can appear before or after the subcommand name -- both `cobre --color always run` and `cobre run --color always` work

## Testing Requirements

### Unit Tests

- Test `resolve_color` with `ColorWhen::Always` sets color enabled
- Test `resolve_color` with `ColorWhen::Never` sets color disabled
- Test `resolve_color` with `ColorWhen::Auto` + `COBRE_COLOR=always` sets color enabled
- Test `resolve_color` with `ColorWhen::Auto` + `FORCE_COLOR=1` sets color enabled
- All env var tests use `ENV_LOCK` mutex guard

### Integration Tests

- `assert_cmd` test: `cobre run --color always --skip-simulation examples/1dtoy/` with stderr captured -- assert stderr contains `\x1b[`
- `assert_cmd` test: `cobre run --color never --skip-simulation examples/1dtoy/` with stderr captured -- assert stderr does not contain `\x1b[`

### E2E Tests (if applicable)

Not applicable.

## Dependencies

- **Blocked By**: None
- **Blocks**: ticket-023-final-review-pass.md

## Effort Estimate

**Points**: 2
**Confidence**: High
