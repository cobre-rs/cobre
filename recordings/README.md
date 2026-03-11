# Terminal Recordings

This directory contains scripts and tape files for generating terminal recordings
that demonstrate Cobre in action. The recordings are intended for use in the README
and documentation pages.

Generated recordings (`.gif` and `.cast` files) are committed so they can be
embedded in the README and documentation. Temporary files (`demo/`) are gitignored.

## Prerequisites

### Cobre binary

Install the `cobre` binary from the workspace root:

```sh
cargo install --path crates/cobre-cli
```

Verify it is on your PATH:

```sh
cobre version
```

### VHS (for `.tape` files)

VHS generates GIF and SVG animations from `.tape` script files.

```sh
# macOS
brew install vhs

# Go toolchain (any platform)
go install github.com/charmbracelet/vhs@latest
```

### asciinema (for `.sh` scripts)

asciinema records terminal sessions into `.cast` files.

```sh
pip install asciinema
```

### jq (for report.sh)

```sh
# macOS
brew install jq

# Debian / Ubuntu
sudo apt-get install jq

# Fedora
sudo dnf install jq
```

## VHS Recordings

Run the tape files from the repository root. VHS writes output next to the tape file.

```sh
# Quick Start demo (init → run → report)
vhs recordings/quickstart.tape
# Output: recordings/quickstart.gif

# Validate demo (init → validate on a valid case)
vhs recordings/validation.tape
# Output: recordings/validation.gif

# Validation error demo (init → corrupt JSON with jq → validate showing errors)
vhs recordings/validation-error.tape
# Output: recordings/validation-error.gif
```

The `validation-error.tape` uses `jq` to corrupt the 1dtoy case on the fly (no
pre-built broken directory is committed). It injects two distinct error categories:

- A schema error: the `reservoir` object is deleted from `hydros.json`, causing a
  missing required field error in the structural validation layer.
- A semantic constraint violation: `max_turbined_m3s` is set to a negative value,
  triggering a constraint check in the semantic validation layer.

Both errors appear in the `cobre validate` output with red `error:` labels and the
command exits with a non-zero exit code.

## asciinema Recordings

Run the shell scripts from the repository root. Scripts write `.cast` files into
`recordings/` and clean up temporary directories on exit.

```sh
# Training run (progress bar and banner)
bash recordings/training.sh
# Output: recordings/training.cast

# Report output piped through jq
bash recordings/report.sh
# Output: recordings/report.cast
```

To replay a cast locally:

```sh
asciinema play recordings/training.cast
```

## Embedding in the README

Reference GIF output in Markdown:

```markdown
![Quick Start](recordings/quickstart.gif)
```

For SVG output (smaller file size, no browser autoplay restrictions):

```markdown
![Quick Start](recordings/quickstart.svg)
```

To share via asciinema.org, upload a cast file:

```sh
asciinema upload recordings/training.cast
```
