#!/usr/bin/env bash
set -euo pipefail

# Check prerequisites
command -v asciinema >/dev/null 2>&1 || { echo "asciinema not found. Install with: pip install asciinema"; exit 1; }
command -v cobre >/dev/null 2>&1 || { echo "cobre not found. Install with: cargo install --path crates/cobre-cli"; exit 1; }

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

cobre init --template 1dtoy "$WORKDIR/demo"

asciinema rec recordings/training.cast \
  --title "Cobre Training Run" \
  --command "cobre run $WORKDIR/demo/" \
  --overwrite
