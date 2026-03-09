#!/usr/bin/env bash
set -euo pipefail

# Check prerequisites
command -v asciinema >/dev/null 2>&1 || { echo "asciinema not found. Install with: pip install asciinema"; exit 1; }
command -v cobre >/dev/null 2>&1 || { echo "cobre not found. Install with: cargo install --path crates/cobre-cli"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq not found. Install with: brew install jq  (macOS) or sudo apt-get install jq (Debian/Ubuntu)"; exit 1; }

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

cobre init --template 1dtoy "$WORKDIR/demo"
cobre run "$WORKDIR/demo/"

asciinema rec recordings/report.cast \
  --title "Cobre Report Output" \
  --command "cobre report $WORKDIR/demo/output/ | jq ." \
  --overwrite
