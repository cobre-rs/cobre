# Terminal Recordings

This directory holds the VHS tape files that render the terminal-recording GIFs
demonstrating the Cobre CLI in action, plus the two scripts that install the
toolchain and generate the GIFs.

Generated GIFs are committed here as the source of truth. The documentation site
([cobre-docs](https://github.com/cobre-rs/cobre-docs)) vendors them from a tagged
release via `npm run refresh:recordings -- --ref <tag>` — so after regenerating a
GIF for a CLI change, cut a release and re-run that command in cobre-docs. The
temporary `demo/` case created while recording is gitignored.

## Quick start

```sh
./recordings/setup.sh      # one-time: install the vhs toolchain
./recordings/generate.sh   # regenerate every GIF
```

## Prerequisites

- **The `cobre` binary on PATH** — the subject of every recording. Install it
  from the workspace root and confirm it resolves:

  ```sh
  cargo install --path crates/cobre-cli
  cobre version
  ```

- **The VHS toolchain** — `vhs` (the recorder), `ttyd` (the terminal it drives),
  and `ffmpeg` (the GIF encoder), plus `jq` (two tapes mutate the demo case with
  it). `setup.sh` installs all of these: it downloads a pinned `vhs` release into
  `~/.local/bin` (override with `$BIN_DIR`) and ensures that directory is on your
  PATH, and installs `ttyd` + `ffmpeg` via your package manager (`dnf` / `apt` /
  `brew`). Install `jq` from your package manager if it is missing.

## Generating the recordings

```sh
./recordings/generate.sh              # all tapes
./recordings/generate.sh quickstart   # just quickstart.tape (name or path)
```

`generate.sh` runs each tape from within `recordings/`, so every GIF is written
next to its tape and the `demo/` case that `cobre init demo/` creates lands in
`recordings/demo/` (gitignored). It removes that `demo/` — and any stray
`tmp.json` a tape's `jq` step leaves — **before and after every tape**. Cleaning
outside the tape is deliberate: it keeps helper commands out of the capture, so
each GIF shows only the demo it is meant to show, and re-runs always start from a
clean case (never a `cobre init` "directory exists" prompt).

The tapes:

| Tape                    | GIF                    | Demo                                                               |
| ----------------------- | ---------------------- | ------------------------------------------------------------------ |
| `quickstart.tape`       | `quickstart.gif`       | `init` → `run` → `jq` metadata query                               |
| `validation.tape`       | `validation.gif`       | `init` → `validate` on a valid case                                |
| `validation-error.tape` | `validation-error.gif` | `init` → corrupt with `jq` → `validate` surfacing the schema error |
| `multithreading.tape`   | `multithreading.gif`   | `run --threads 1` vs `--threads 4`, back-to-back timing            |

`validation-error.tape` uses `jq` to corrupt the 1dtoy case on the fly (no
pre-built broken directory is committed). It applies two mutations:

- the `reservoir` object is deleted from `hydros.json`, producing a missing
  required field error in the schema validation layer;
- the first thermal's `bus_id` in `thermals.json` is set to a bus that does not
  exist, which the reference validation layer would report.

Validation runs the layers in strict dependency order and stops after the schema
layer once it collects errors, because the later layers consume parsed data the
schema layer failed to produce. So the recording surfaces the schema error — a
red `error:` label and a non-zero exit code — while the reference mutation is
never reached. To observe the reference error on its own, corrupt only
`bus_id` and leave the schema intact.

`multithreading.tape` runs the same 1dtoy case twice in sequence — first with
`--threads 1`, then with `--threads 4` — so the post-run summary timing lines
appear back-to-back for a direct comparison.

## Brand Theme

Each tape opens with the same block of `Set` directives: a `Set Theme` JSON object
carries the palette, and the other `Set` lines set the font and the window frame.
Copy that block from an existing tape when you add one.

The full brand palette is documented in `docs/internal/BRAND-GUIDELINES.md`.

## Embedding

Reference GIF output in Markdown:

```markdown
![Quick Start](recordings/quickstart.gif)
```

For SVG output (smaller file size, no browser autoplay restrictions):

```markdown
![Quick Start](recordings/quickstart.svg)
```
