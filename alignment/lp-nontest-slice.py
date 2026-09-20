#!/usr/bin/env python3
"""Measure the lp/ kernel boundary: the brace-aware non-test slice, the two-pass vocabulary
grep, the module-universe reconciliation, and the classification artifacts built on them.

The non-test slice treats only a line whose stripped form starts with the `#[cfg(test)]`
attribute as a test boundary (the same token inside a `//!` doc comment is prose), then
drops exactly the attributed item: by brace balance when the item opens a block, through
the terminating `;` when it is a statement item (`mod tests;`, `use …;`). Production lines
after the item are kept, so a test-only fn in the middle of a module does not swallow the
code that follows it. Sibling `tests.rs` / `test_support.rs` modules count zero.

Subcommands (all read the tree at a sha through `git show`; nothing here touches lp/):
  self-test [--baseline SHA]      the four trap assertions the ticket pins
  slice --baseline SHA PATH       kept-line count + excluded ranges for one module
  functions --baseline SHA PATH   production functions with their spans and geometry hits
  proof --baseline SHA            lp-grep-proof.json over the 30-module universe
  classify --baseline SHA         lp-classification.json + lp-classification.md (+ --append-register)
Exit: 0 clean, 1 universe drift / failing assertion / failing checker, 2 register unreadable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve()
ALIGN = HERE.parent
AUDIT = HERE.parents[1]
TOOLS = AUDIT / "tools"
sys.path.insert(0, str(TOOLS))

from lib import backlog_parse as bp  # noqa: E402


ROOT = AUDIT.parents[1]
REGISTER = AUDIT / "BACKLOG.md"
INVENTORY = AUDIT / "measurements" / "lp-inventory.json"
ROADMAP = ROOT / "plans" / "generalizing" / "beyond-sddp-generalization.md"
LP_ROOT = "crates/cobre-sddp/src/lp"
TICKET_BASELINE = "a136840d4f2ea137f685f0af6dac04254b983b60"
ATTR = "#[cfg(test)]"
SIBLING_SUFFIXES = ("/tests.rs", "/test_support.rs")
VOCAB = r"state_space|cost_to_go|theta|cut|ring"
SUBSTR = re.compile(VOCAB)
WORD = re.compile(r"\b(?:" + VOCAB + r")\b")
CAMEL_GEOMETRY = re.compile(
    r"\b(?:StateSpace|StateDim|StateRegion|CutSlot|CutStateProjection|InCol|OutCol|DeliveryRing|AnticipatedLocal|AnticipatedLayout)\b"
)
DECL = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r"(?P<kind>fn|struct|enum|trait|type|const|static|mod)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
SECTION_TITLE = "generalization-alignment"
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_REGISTER = 2


def register_pin() -> str:
    """The sha pinned in the register header — the default tree every subcommand measures."""
    return bp.parse_baseline(bp.read_register(REGISTER))


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def show_lines(sha: str, path: str) -> list[str]:
    proc = git("show", f"{sha}:{path}")
    if proc.returncode != 0:
        raise FileNotFoundError(f"{path} at {sha[:8]}")
    return proc.stdout.splitlines()


def line_of(text: str | list[str], pattern: str) -> int | None:
    lines = text.splitlines() if isinstance(text, str) else text
    for idx, raw in enumerate(lines, 1):
        if re.search(pattern, raw):
            return idx
    return None


def decl_line(lines: list[str], symbol: str) -> int | None:
    return line_of(
        lines,
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
        r"(?:fn|struct|enum|trait|type|const|static|mod)\s+"
        + re.escape(symbol)
        + r"\b",
    )


def tree_modules(sha: str) -> list[str]:
    out = git("ls-tree", "-r", "--name-only", sha, "--", LP_ROOT).stdout.split()
    return sorted(p for p in out if p.endswith(".rs"))


def is_sibling(path: str) -> bool:
    return path.endswith(SIBLING_SUFFIXES)


def nontest_slice(lines: list[str]) -> tuple[list[tuple[int, str]], list[list[int]]]:
    """Kept (1-based line, text) pairs and the excluded 1-based inclusive ranges."""
    kept: list[tuple[int, str]] = []
    excluded: list[list[int]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(ATTR):
            start = i
            j = i + 1
            while (
                j < len(lines)
                and "{" not in lines[j]
                and not lines[j].rstrip().endswith(";")
            ):
                j += 1
            if j < len(lines) and "{" not in lines[j]:
                end = j + 1
            else:
                depth = 0
                while j < len(lines):
                    depth += lines[j].count("{") - lines[j].count("}")
                    j += 1
                    if depth <= 0:
                        break
                end = j
            excluded.append([start + 1, end])
            i = end
            continue
        kept.append((i + 1, lines[i]))
        i += 1
    return kept, excluded


def naive_first_marker_loc(lines: list[str]) -> int:
    for idx, raw in enumerate(lines):
        if ATTR in raw:
            return idx
    return len(lines)


def module_slice(sha: str, path: str) -> dict[str, Any]:
    lines = show_lines(sha, path)
    if is_sibling(path):
        return {
            "module": path,
            "grossLoc": len(lines),
            "nonTestLoc": 0,
            "excludedRanges": [[1, len(lines)]] if lines else [],
            "sliceMethod": "sibling-test-file",
            "naiveFirstMarkerLoc": 0,
            "kept": [],
        }
    kept, excluded = nontest_slice(lines)
    return {
        "module": path,
        "grossLoc": len(lines),
        "nonTestLoc": len(kept),
        "excludedRanges": excluded,
        "sliceMethod": "brace-aware",
        "naiveFirstMarkerLoc": naive_first_marker_loc(lines),
        "kept": kept,
    }


def vocabulary_hits(kept: list[tuple[int, str]]) -> dict[str, Any]:
    substr = [[n, line.rstrip()] for n, line in kept if SUBSTR.search(line)]
    word = [[n, line.rstrip()] for n, line in kept if WORD.search(line)]
    word_lines = {n for n, _ in word}
    queue = []
    for n, line in substr:
        if n not in word_lines:
            tokens = sorted(
                {
                    m.group(0)
                    for m in re.finditer(
                        r"[A-Za-z_-]*(?:" + VOCAB + r")[A-Za-z_-]*", line
                    )
                }
            )
            queue.append(
                [n, ", ".join(f"{t} -> {SUBSTR.search(t).group(0)}" for t in tokens)]
            )
    return {
        "substringHits": substr,
        "wordBoundaryHits": word,
        "falsePositiveQueue": queue,
        "camelCaseGeometryHits": [
            [n, line.rstrip()] for n, line in kept if CAMEL_GEOMETRY.search(line)
        ],
    }


def function_spans(kept: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """Production declarations with the kept-line span each owns (until the next declaration).

    An indented `fn` inside an `impl` block is attributed to the impl's type (`Type::fn`), so
    the partition can name a type once and cover its methods.
    """
    impl_re = re.compile(
        r"^\s*impl(?:<[^>]*>)?\s+(?:[A-Za-z_][A-Za-z0-9_:<>, ]*?\s+for\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    )
    decls: list[tuple[int, str, str, str | None]] = []
    owner: str | None = None
    for idx, (_, line) in enumerate(kept):
        im = impl_re.match(line)
        if im:
            owner = im.group("name")
            continue
        m = DECL.match(line)
        if m:
            indented = line[:1].isspace()
            decls.append(
                (idx, m.group("kind"), m.group("name"), owner if indented else None)
            )
            if not indented and m.group("kind") in ("struct", "enum", "trait"):
                owner = m.group("name")
    spans = []
    for k, (idx, kind, name, own) in enumerate(decls):
        stop = decls[k + 1][0] if k + 1 < len(decls) else len(kept)
        body = kept[idx:stop]
        spans.append(
            {
                "kind": kind,
                "name": name,
                "owner": own,
                "qualified": f"{own}::{name}" if own else name,
                "line": kept[idx][0],
                "keptLines": len(body),
                "wordHits": sum(1 for _, l in body if WORD.search(l)),
                "camelGeometryHits": sum(
                    1 for _, l in body if CAMEL_GEOMETRY.search(l)
                ),
            }
        )
    return spans


def universe(sha: str) -> tuple[list[str], list[str]]:
    """(tree modules, drift lines); drift is a non-empty list when inventory and tree disagree."""
    tree = tree_modules(sha)
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    listed = sorted(f["path"] for f in inventory["files"])
    drift = [f"inventory-only {p}" for p in sorted(set(listed) - set(tree))]
    drift += [f"tree-only {p}" for p in sorted(set(tree) - set(listed))]
    return tree, drift


def build_proof(sha: str) -> dict[str, Any]:
    tree, drift = universe(sha)
    modules: dict[str, Any] = {}
    total = naive_total = gross = 0
    for path in tree:
        s = module_slice(sha, path)
        hits = vocabulary_hits(s["kept"])
        gross += s["grossLoc"]
        total += s["nonTestLoc"]
        naive_total += s["naiveFirstMarkerLoc"]
        modules[path] = {
            "grossLoc": s["grossLoc"],
            "nonTestLoc": s["nonTestLoc"],
            "excludedRanges": s["excludedRanges"],
            "sliceMethod": s["sliceMethod"],
            "naiveFirstMarkerLoc": s["naiveFirstMarkerLoc"],
            **hits,
        }
    return {
        "baseline": sha,
        "vocabulary": VOCAB,
        "sliceRule": "a line whose stripped form starts with `#[cfg(test)]` opens a test boundary; the attributed item is dropped by brace balance (or through its `;` for a statement item) and production lines after it are kept; the same token inside a `//!` doc comment is prose; sibling tests.rs / test_support.rs count zero; `#[cfg(any(test, feature = \"test-support\"))]` items are production (the ticket's rule), unlike the sddp station's loc-stats rule",
        "universe": {
            "modules": len(tree),
            "inventory": str(INVENTORY.relative_to(ROOT)),
            "drift": drift,
        },
        "totals": {
            "grossLoc": gross,
            "nonTestLoc": total,
            "naiveFirstMarkerLoc": naive_total,
        },
        "modules": modules,
    }


def cmd_self_test(sha: str) -> int:
    failures: list[str] = []

    def expect(label: str, got: Any, want: Any) -> None:
        ok = got == want
        print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")
        if not ok:
            failures.append(label)

    bg = module_slice(sha, f"{LP_ROOT}/indexer/block_grid.rs")
    expect(
        "block_grid.rs non-test LOC (doc-comment token ignored)", bg["nonTestLoc"], 126
    )
    expect("block_grid.rs test tail starts at", bg["excludedRanges"][0][0], 127)
    expect("block_grid.rs naive first-marker LOC", bg["naiveFirstMarkerLoc"], 20)
    ag = module_slice(sha, f"{LP_ROOT}/indexer/anticipated_gate.rs")
    expect(
        "anticipated_gate.rs non-test LOC (test-only fn excluded, code after it kept)",
        ag["nonTestLoc"],
        112,
    )
    expect(
        "anticipated_gate.rs excluded ranges",
        ag["excludedRanges"],
        [[51, 81], [144, 368]],
    )
    expect("anticipated_gate.rs naive first-marker LOC", ag["naiveFirstMarkerLoc"], 50)
    rc = module_slice(sha, f"{LP_ROOT}/indexer/range_cursor.rs")
    hits = vocabulary_hits(rc["kept"])
    expect(
        "range_cursor.rs substring hit (consecutive -> cut) on line",
        [h[0] for h in hits["substringHits"]],
        [10],
    )
    expect("range_cursor.rs word-boundary hits", hits["wordBoundaryHits"], [])
    pb = module_slice(sha, f"{LP_ROOT}/builder/patch.rs")
    ph = vocabulary_hits(pb["kept"])
    expect("patch.rs non-test LOC", pb["nonTestLoc"], 413)
    expect(
        "patch.rs zero-hit on both passes",
        (len(ph["substringHits"]), len(ph["wordBoundaryHits"])),
        (0, 0),
    )
    expect(
        "patch.rs CamelCase geometry lines the vocabulary cannot see",
        len(ph["camelCaseGeometryHits"]),
        8,
    )
    if sha.startswith(TICKET_BASELINE[:8]):
        proof = build_proof(sha)
        expect(
            "corpus non-test LOC at the ticket's scaffold pin",
            proof["totals"]["nonTestLoc"],
            11675,
        )
        expect(
            "corpus naive first-marker LOC at the ticket's scaffold pin",
            proof["totals"]["naiveFirstMarkerLoc"],
            11396,
        )
        expect(
            "corpus gross LOC at the ticket's scaffold pin",
            proof["totals"]["grossLoc"],
            44374,
        )
    print(f"self-test: {'FAIL' if failures else 'ok'} ({len(failures)} failing)")
    return EXIT_FAIL if failures else EXIT_OK


def cmd_slice(sha: str, path: str) -> int:
    s = module_slice(sha, path)
    print(json.dumps({k: v for k, v in s.items() if k != "kept"}, indent=2))
    return EXIT_OK


def cmd_functions(sha: str, path: str) -> int:
    s = module_slice(sha, path)
    print(
        f"{path} @ {sha[:8]}: {s['nonTestLoc']} non-test lines, excluded {s['excludedRanges']}"
    )
    for f in function_spans(s["kept"]):
        print(
            f"  {f['line']:>5} {f['kind']:<6} {f['name']:<48} kept {f['keptLines']:>4}  word {f['wordHits']:>3}  camel {f['camelGeometryHits']:>3}"
        )
    return EXIT_OK


def cmd_proof(sha: str, out: pathlib.Path) -> int:
    proof = build_proof(sha)
    out.write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    t = proof["totals"]
    print(
        f"{len(proof['modules'])} modules @ {sha[:8]}: gross {t['grossLoc']} / non-test {t['nonTestLoc']} / naive {t['naiveFirstMarkerLoc']}"
    )
    for line in proof["universe"]["drift"]:
        print(f"UNIVERSE DRIFT: {line}", file=sys.stderr)
    return EXIT_FAIL if proof["universe"]["drift"] else EXIT_OK


DEFAULT_SECTION_HEADING = (
    "## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — generalization-alignment"
)
LP = LP_ROOT + "/"
NEUTRAL, GEOMETRY, MIXED, SIBLING = (
    "engine-neutral",
    "sddp-geometry",
    "mixed",
    "test-sibling",
)
REPARAM = (
    "neutral emission that is anchored after the state region today and would be re-parameterized off "
    "the θ-anchored layout on extraction (IV.2)"
)

# One decision per production module: the class, the anchor symbol, the reading-level rationale,
# and for a mixed module the function-level partition (neutral / shared / geometry; every span
# must land in exactly one list). The grep result is a filter, never the verdict.
CLASSES: dict[str, dict[str, Any]] = {
    LP + "indexer/study_dimensions.rs": {
        "class": NEUTRAL,
        "anchor": "StudyDimensions",
        "reading": "the study-invariant non-state LP shape (entity counts, presence flags, the anticipated-thermal identity list); its two CamelCase lines are the doc's ownership split against StateSpace, not geometry",
    },
    LP + "indexer/entity_index.rs": {
        "class": NEUTRAL,
        "anchor": "HydroSys",
        "reading": "typed system/local entity index vocabulary carrying no arithmetic; AnticipatedLocal is a local-list position, not ring geometry",
    },
    LP + "indexer/layout.rs": {
        "class": NEUTRAL,
        "anchor": "EvaporationIndices",
        "reading": "per-stage evaporation column/row and FPHA row-block satellites; the one CamelCase line is a doc pointer to StateSpace",
    },
    LP + "indexer/storage_boundary_grid.rs": {
        "class": NEUTRAL,
        "anchor": "StorageBoundaryGrid",
        "reading": "block/cursor primitive for the chronological storage-boundary column formula; it takes StateSpace column bases as inputs ("
        + REPARAM
        + ")",
    },
    LP + "indexer/block_grid.rs": {
        "class": NEUTRAL,
        "anchor": "BlockGrid",
        "reading": "the block-major stride primitive shared by every fill path",
        "dismissal": "the only substring hit is the English participle `mirroring` in the module doc",
    },
    LP + "indexer/range_cursor.rs": {
        "class": NEUTRAL,
        "anchor": "RangeCursor",
        "reading": "the running offset allocator; its two CamelCase lines are doc pointers to StateSpace::new as one of its two callers",
        "dismissal": "the only substring hit is `consecutive` in the module doc, which contains `cut`",
    },
    LP + "indexer/hydro_cell.rs": {
        "class": NEUTRAL,
        "anchor": "HydroCellIndex",
        "reading": "the study-scope partition of hydro unit groups into bus cells (entity indexing)",
        "dismissal": "the substring hits are the participles `filtering`, `renumbering` and `reordering` in doc comments",
    },
    LP + "builder/fpha_cursor.rs": {
        "class": NEUTRAL,
        "anchor": "for_each_fpha_plane",
        "reading": "the FPHA plane visitor over a stage's hydro cells",
        "dismissal": "the only substring hit is `load-bearing` in the doc comment on the row-block nesting",
    },
    LP + "indexer/index.rs": {
        "class": MIXED,
        "anchor": "Col",
        "reading": "typed address vocabulary: Col, Row, BlockIdx and Boundary are LP-address newtypes any engine needs; StateDim, CutSlot, InCol and OutCol are state-vector and cut-render vocabulary",
        "splitNote": "move `Col`, `Row`, `BlockIdx` and `Boundary` (plus `Boundary::from_index`) to the neutral substrate; `StateDim`, `CutSlot`, `InCol` and `OutCol` stay with the engine; the module doc's cut-path paragraphs go with them",
        "neutral": ["Col", "Row", "BlockIdx", "Boundary"],
        "geometry": ["StateDim", "CutSlot", "InCol", "OutCol"],
        "shared": [],
    },
    LP + "indexer/anticipated_gate.rs": {
        "class": GEOMETRY,
        "anchor": "is_anticipated_decision_active_for_delivery",
        "reading": "anticipated-decision horizon and commissioning-window gating keyed on StateSpace's lead stages and resolution (anticipated ring geometry); the two substring hits (`load-bearing`, `maturing`) are prose and irrelevant to the class",
    },
    LP + "indexer/cut_state_projection.rs": {
        "class": GEOMETRY,
        "anchor": "CutStateProjection",
        "reading": "storage-scoped projection of StateSpace onto the enabled cut-state dimensions (cut storage, subgradient extraction, cut rendering)",
    },
    LP + "indexer/state_space.rs": {
        "class": GEOMETRY,
        "anchor": "StateSpace",
        "reading": "the stage-invariant state-vector layout and its incoming/outgoing LP-column resolvers — the SDDP state region itself",
    },
    LP + "indexer/mod.rs": {
        "class": MIXED,
        "anchor": "BlockGrid",
        "anchorPath": LP + "indexer/block_grid.rs",
        "reading": "module root: a column-layout narrative owned by StateSpace plus the `mod` and re-export lines of both the neutral primitives and the geometry types",
        "splitNote": "the `mod`/`pub use` lines for block_grid, entity_index, hydro_cell, layout, range_cursor, storage_boundary_grid and study_dimensions (and the Col/Row/BlockIdx/Boundary half of the `index` re-export) move with the neutral substrate; the state_space, cut_state_projection and anticipated_gate lines and the state-layout narrative stay",
        "neutralHalf": [18, 29],
    },
    LP + "builder/patch.rs": {
        "class": GEOMETRY,
        "anchor": "PatchBuffer",
        "reading": "PatchBuffer owns the column-bound state region: fill_col_state_patches walks StateSpace::state_to_lp_incoming_column over every state-vector index and fill_forward_patches rewrites the stochastic row bounds before each solve",
        "dismissal": "the zero-hit on both passes over 413 non-test lines is not a neutral signal: the snake_case vocabulary cannot see the 8 CamelCase `StateSpace`/`StateDim` lines, so a zero-hit is never on its own a class-1 proof",
    },
    LP + "builder/delivery_ring.rs": {
        "class": GEOMETRY,
        "anchor": "DeliveryRing",
        "reading": "the lagged-delivery ring skeleton (Markov-1 slot per stage) shared by the travel-time bucket ring and the anticipated-thermal ring — state geometry borrowed from StateSpace",
    },
    LP + "builder/commitment_reconcile.rs": {
        "class": GEOMETRY,
        "anchor": "reconcile_commitment",
        "reading": "reconciles a pinned anticipated-commitment ring slot against the delivery-stage bounds through StateSpace and StageGeometry",
    },
    LP + "builder/rows.rs": {
        "class": MIXED,
        "anchor": "fill_stage_rows",
        "reading": "row-bound fills: the water-balance, filling, load-balance, FPHA, evaporation and operational-violation rows are equipment rows; the anticipated fishing/state-out/slot rows, the transit-bucket definition rows and the z_inflow rows are ring and noise geometry; the three substring hits (`mirroring`, `maturing`) are prose",
        "splitNote": "move `fill_water_balance_rows` / `fill_parallel_water_rows` / `fill_chronological_water_rows`, `fill_filling_target_rows`, `fill_filled_min_storage_floor_rows`, `fill_load_balance_rows`, `fill_fpha_rows`, `fill_evaporation_rows` and `fill_operational_violation_rows`; `fill_anticipated_*_rows`, `fill_transit_bucket_definition_rows` and `fill_z_inflow_rows` stay; `fill_stage_rows` (the orchestrator) and `fill_zero_equality_rows` are shared",
        "neutral": [
            "fill_water_balance_rows",
            "fill_parallel_water_rows",
            "fill_chronological_water_rows",
            "fill_filling_target_rows",
            "fill_filled_min_storage_floor_rows",
            "fill_load_balance_rows",
            "fill_fpha_rows",
            "fill_evaporation_rows",
            "fill_operational_violation_rows",
        ],
        "geometry": [
            "fill_transit_bucket_definition_rows",
            "fill_z_inflow_rows",
            "fill_anticipated_fishing_rows",
            "fill_anticipated_state_out_def_rows",
            "fill_anticipated_slot_definition_rows",
        ],
        "shared": ["fill_stage_rows", "fill_zero_equality_rows"],
    },
    LP + "builder/columns.rs": {
        "class": MIXED,
        "anchor": "fill_stage_columns",
        "reading": "column fills: turbine, spillage, diversion, thermal, line, deficit, FPHA, evaporation, slack, NCS, pumping, contract and filling columns are plain equipment columns; storage (the state region), AR-lag, transit-bucket, anticipated slot/state, theta, inflow-slack and z_inflow columns are state-space geometry",
        "splitNote": "move the `fill_<equipment>_columns` family and `GroupBoundLookup` ("
        + REPARAM
        + "); `fill_storage_columns`, `fill_ar_lag_columns`, `fill_transit_bucket_columns`, `fill_anticipated_*_columns`, `fill_theta_column`, `fill_inflow_slack_columns` and `fill_z_inflow_columns` stay; `ColumnBufs` and `fill_stage_columns` are shared",
        "neutral": [
            "GroupBoundLookup",
            "new",
            "max_turbined",
            "max_generation",
            "min_turbined",
            "min_generation",
            "cell_max_turbined",
            "cell_min_turbined",
            "fill_turbine_columns",
            "fill_spillage_columns",
            "fill_diversion_columns",
            "fill_thermal_columns",
            "fill_line_columns",
            "fill_deficit_and_excess_columns",
            "cell_max_generation",
            "cell_min_generation",
            "fill_fpha_generation_columns",
            "fill_evaporation_columns",
            "fill_withdrawal_slack_columns",
            "BlockSlackFamily",
            "CellSlackFamily",
            "fill_operational_slack_columns",
            "fill_block_family",
            "fill_cell_block_family",
            "fill_ncs_columns",
            "fill_pumping_columns",
            "fill_contract_columns",
            "fill_filling_target_columns",
            "fill_filled_min_storage_floor_columns",
        ],
        "geometry": [
            "fill_storage_columns",
            "fill_transit_bucket_columns",
            "fill_anticipated_slot_columns",
            "fill_ar_lag_columns",
            "fill_anticipated_state_columns",
            "fill_theta_column",
            "fill_anticipated_columns",
            "fill_inflow_slack_columns",
            "fill_z_inflow_columns",
        ],
        "shared": ["ColumnBufs", "fill_stage_columns"],
    },
    LP + "builder/entries.rs": {
        "class": MIXED,
        "anchor": "fill_load_balance_entries",
        "reading": "CSC entry fills: the water-balance, arc-release, filling, pumping, load-balance, FPHA, evaporation, generic-constraint, NCS and operational-violation entries are equipment/physics rows; the anticipated ring, transit-bucket ring and z_inflow entries are state geometry",
        "splitNote": "seam by row family, not by block mode: move `fill_load_balance_entries`, `fill_fpha_entries`, `fill_evaporation_entries`, `fill_generic_constraint_entries`, `fill_ncs_load_balance_entries`, `fill_operational_violation_entries`, `fill_pumping_water_entries`, the filling entries and `assemble_csc`, with the water-balance family (the two block-mode water fills, the arc-release fills, `resolve_chrono_arrival_density`, `fill_prefilling_shortcircuit`) moving only after its transit-bucket deposits are re-parameterized — the parallel-versus-chronological pair is a retracted, deliberate-by-design pair of formulations and is NOT the seam; `anticipated_ring`, `fill_anticipated_*_entries`, `transit_bucket_ring`, `fill_transit_bucket_definition_entries`, `plant_transit_bucket_range` and `fill_z_inflow_entries` stay (the transit-bucket topology is a ratified reserved seam, cited as sanctioned, not a carve candidate); `fill_state_and_water_entries` and `build_stage_matrix_entries` are shared orchestrators",
        "neutral": [
            "is_prefilling",
            "resolve_shortcircuit_target",
            "fill_parallel_water_entries",
            "push_plant_release",
            "fill_arc_release_block_entries",
            "fill_chronological_water_entries",
            "fill_arc_release_chrono_block_entries",
            "resolve_chrono_arrival_density",
            "fill_prefilling_shortcircuit",
            "fill_filling_target_entries",
            "fill_filled_min_storage_floor_entries",
            "fill_pumping_water_entries",
            "fill_load_balance_entries",
            "fill_fpha_entries",
            "fill_evaporation_entries",
            "LpMatrixBuffers",
            "fill_generic_constraint_entries",
            "fill_ncs_load_balance_entries",
            "fill_operational_violation_entries",
            "assemble_csc",
        ],
        "geometry": [
            "anticipated_ring",
            "fill_anticipated_fishing_entries",
            "fill_anticipated_state_out_def_entries",
            "fill_anticipated_slot_definition_entries",
            "transit_bucket_plant_ranges",
            "transit_bucket_ring",
            "fill_transit_bucket_definition_entries",
            "plant_transit_bucket_range",
            "fill_z_inflow_entries",
        ],
        "shared": ["fill_state_and_water_entries", "build_stage_matrix_entries"],
    },
    LP + "builder/layout.rs": {
        "class": MIXED,
        "anchor": "StageLayout",
        "reading": "StageLayout allocates both regions: the equipment column/row chains, slack families, generic-constraint rows and their accessors are neutral; AnticipatedLayout, the anticipated/transit-bucket row-position builders and the state-region accessors (col_theta, col_storage_in_start, col_inflow_lags_start, col_z_inflow_start, col_anticipated_state_start, n_state, anticipated_decision) are geometry",
        "splitNote": "move `EquipmentColumns`, `SlackColumns`, `ConstraintRows`, `OperViolationRanges`, `GenericConstraintLayout`, `FillingLayout`, the `identify_*_hydros` / `build_evap_indices` / `allocate_generic_slack_cols` / `enumerate_generic_constraint_rows` builders, `StageProductionRole` and the equipment accessors; `AnticipatedLayout`, `build_transit_bucket_row_pos`, `build_anticipated_*_row_pos` and the state-region accessors stay; `TemplateBuildCtx`, `StageLayout` itself, `StageLayout::new`, `geometry` and `resolver_geom` are shared and would split",
        "neutral": [
            "ResolvedTables",
            "EquipmentColumns",
            "OperViolationRanges",
            "SlackColumns",
            "ConstraintRows",
            "FillingLayout",
            "GenericConstraintLayout",
            "build_evap_indices",
            "hydro_phase",
            "identify_fpha_hydros",
            "identify_evap_hydros",
            "identify_filling_target_hydros",
            "identify_filled_min_storage_floor_hydros",
            "allocate_generic_slack_cols",
            "expression_collapses_to_stage_level",
            "resolve_affine",
            "fold_endpoint",
            "bound_affine_is_block_varying",
            "enumerate_generic_constraint_rows",
            "StageProductionRole",
            "block_col",
            "block_grid",
            "turbine_col",
            "spillage_col",
            "diversion_col",
            "generation_col",
            "fpha_local_first_cell",
            "stage_production_role",
            "line_fwd_col",
            "line_rev_col",
            "outflow_below_col",
            "outflow_above_col",
            "turbine_below_col",
            "generation_below_col",
            "evap_triple_base",
            "evap_flow_col",
            "evap_f_plus_col",
            "evap_f_minus_col",
            "deficit_col",
            "storage_boundary_grid",
            "block_storage_col",
            "row_fpha_start",
            "row_evap_start",
            "filling_target",
            "filling_target_col",
            "filled_min_storage_floor",
            "filled_min_storage_floor_col",
        ],
        "geometry": [
            "AnticipatedLayout",
            "build_transit_bucket_row_pos",
            "build_anticipated_slot_row_pos",
            "build_anticipated_decision_row_pos",
            "build_anticipated_fishing_row_pos",
            "col_theta",
            "col_storage_in_start",
            "col_inflow_lags_start",
            "col_z_inflow_start",
            "col_anticipated_state_start",
            "n_state",
            "anticipated_decision",
        ],
        "shared": [
            "TemplateBuildCtx",
            "StageLayout",
            "new",
            "geometry",
            "resolver_geom",
        ],
    },
    LP + "builder/template.rs": {
        "class": MIXED,
        "anchor": "build_stage_templates",
        "reading": "the stage-template factory IV.2 names: discount factors, pre-baked noise (models_from_normal / load_models_from_normal), build_single_stage_template and build_stage_templates are SDDP-shaped; collect_load_bus_indices, StageBuildOutput and block_storage_col are neutral; StageGeometry, build_template_build_ctx and assemble_stage_templates_output are shared",
        "splitNote": "move `collect_load_bus_indices`, `StageBuildOutput` and `StageGeometry::block_storage_col`; `StageTemplates` (discount factors), `build_single_stage_template`, `models_from_normal`, `load_models_from_normal`, `build_stage_templates`, `build_stage_templates_resolving_layout` and `build_filling_v_target` stay; `StageGeometry`, `build_template_build_ctx` and `assemble_stage_templates_output` are shared and would split",
        "neutral": [
            "collect_load_bus_indices",
            "StageBuildOutput",
            "block_storage_col",
        ],
        "geometry": [
            "StageTemplates",
            "empty",
            "discount_factors",
            "cumulative_discount_factors",
            "set_discount_factors",
            "build_single_stage_template",
            "models_from_normal",
            "load_models_from_normal",
            "build_stage_templates",
            "build_stage_templates_resolving_layout",
            "build_filling_v_target",
        ],
        "shared": [
            "StageGeometry",
            "build_template_build_ctx",
            "assemble_stage_templates_output",
        ],
    },
    LP + "builder/scaling.rs": {
        "class": MIXED,
        "anchor": "compute_col_scale",
        "reading": "geometric-mean LP prescaling is engine-neutral conditioning; the bucket and commitment-hold unscaling and the noise pre-scaling are ring and noise geometry, and apply_col_scale asserts coverage through theta",
        "splitNote": "move `compute_col_scale`, `compute_row_scale`, `apply_row_scale` and `apply_col_scale` (once its theta coverage assertion is re-parameterized); `apply_bucket_col_scale`, `apply_commitment_hold_col_scale_unscale` and `compute_noise_scale` stay",
        "neutral": [
            "compute_col_scale",
            "apply_col_scale",
            "compute_row_scale",
            "apply_row_scale",
        ],
        "geometry": [
            "apply_bucket_col_scale",
            "apply_commitment_hold_col_scale_unscale",
            "compute_noise_scale",
        ],
        "shared": [],
    },
    LP + "builder/mod.rs": {
        "class": MIXED,
        "anchor": "GenericConstraintRowEntry",
        "reading": "module root: the per-solve patch sequence and state-pinning narrative is geometry; the shared physics constants (M3S_TO_HM3, the evaporation column offsets) and GenericConstraintRowEntry are neutral; the `mod` and re-export lines split with their targets",
        "splitNote": "move the shared constants and `GenericConstraintRowEntry` with the neutral substrate; the patch-buffer/state-pinning narrative and the `PatchBuffer`, `BoundRelaxations`, `DeliveryRing` re-exports stay; the `mod` lines and the scaling re-export split with their targets",
        "neutralHalf": [76, 91],
    },
    LP + "generic_constraints.rs": {
        "class": MIXED,
        "anchor": "resolve_variable_ref",
        "reading": "generic-constraint lowering onto the indexed column layout is neutral emission, except the role-(a) state region reached through GenericResolverGeom's StateSpace handle (resolve_hydro_storage, resolve_hydro_inflow, resolve_anticipated_decision); the two word-boundary hits are the doc's cut-path paragraph on that role split",
        "splitNote": "move `resolve_variable_ref` and every role-(b) equipment resolver (turbine cells, storage boundary, evaporation, outflow, generation, line exchange, bus deficit, pumping, contracts, block variables); `resolve_hydro_storage`, `resolve_hydro_inflow` and `resolve_anticipated_decision` stay with the engine behind a state-column callback; `GenericResolverGeom` is the shared view that splits into two",
        "neutral": [
            "block_grid",
            "block_storage_col",
            "EntityPositionMaps",
            "CascadeRefs",
            "PumpingRefs",
            "ContractRefs",
            "resolve_variable_ref",
            "variable_ref_is_block_independent",
            "expression_is_block_independent",
            "resolve_turbine_cells",
            "resolve_hydro_storage_boundary",
            "resolve_hydro_evaporation",
            "resolve_hydro_outflow",
            "resolve_hydro_generation",
            "resolve_line_exchange",
            "resolve_bus_deficit",
            "resolve_pumping_column",
            "contract_family_slot",
            "resolve_contract_column",
            "resolve_block_variable",
            "ElementKind",
            "block_col_range",
        ],
        "geometry": [
            "resolve_hydro_storage",
            "resolve_hydro_inflow",
            "resolve_anticipated_decision",
        ],
        "shared": ["GenericResolverGeom"],
    },
    LP + "mod.rs": {
        "class": MIXED,
        "anchor": "resolve_variable_ref",
        "anchorPath": LP + "generic_constraints.rs",
        "reading": "directory-module root: three `mod` lines and a doc that narrates the state-pinning column-bound contract (geometry) beside the generic-constraint lowering (neutral); the only substring hit is `lowering`",
        "splitNote": "the `mod generic_constraints` line and its sentence move with the neutral substrate; the state-pinning and template-factory sentences stay",
        "neutralHalf": [4, 8],
    },
}


def classify_module(sha: str, path: str, proof_row: dict[str, Any]) -> dict[str, Any]:
    if is_sibling(path):
        return {
            "module": path,
            "class": SIBLING,
            "nonTestLoc": 0,
            "anchor": f"{path}:1",
            "anchorSymbol": None,
            "grepResult": "n/a (sibling test file)",
            "grepPass": {"substring": 0, "wordBoundary": 0},
            "dismissal": None,
            "reading": "sibling test file: zero non-test lines by rule",
            "splitNote": None,
            "neutralHalf": None,
            "baseline": sha,
        }
    d = CLASSES[path]
    anchor_path = d.get("anchorPath", path)
    text = show_lines(sha, anchor_path)
    line = decl_line(text, d["anchor"])
    if line is None:
        raise SystemExit(
            f"anchor {d['anchor']} does not resolve in {anchor_path} at {sha[:8]}"
        )
    sub, word = len(proof_row["substringHits"]), len(proof_row["wordBoundaryHits"])
    if word:
        grep = f"{word} word-boundary hit{'s' if word != 1 else ''} (substring {sub})"
    elif sub:
        grep = (
            f"zero word-boundary hits; {sub} substring-only (substring {sub} / word 0)"
        )
    else:
        grep = "zero-hit (substring 0 / word 0)"
    dismissal = d.get("dismissal")
    if dismissal and proof_row["falsePositiveQueue"]:
        quoted = "; ".join(f"line {n}: {t}" for n, t in proof_row["falsePositiveQueue"])
        dismissal = f"{dismissal} — {quoted}"
    if d["class"] == NEUTRAL and word:
        raise SystemExit(
            f"{path}: engine-neutral with {word} word-boundary hits — reclassify or argue"
        )
    if d["class"] == NEUTRAL and sub and not d.get("dismissal"):
        raise SystemExit(
            f"{path}: engine-neutral with substring-only hits and no argued dismissal"
        )
    neutral_half = None
    partition = None
    if d["class"] == MIXED:
        if "neutralHalf" in d:
            neutral_half = list(d["neutralHalf"])
        else:
            kept = module_slice(sha, path)["kept"]
            spans = function_spans(kept)
            first = min(s["line"] for s in spans)
            preamble = sum(1 for n, _ in kept if n < first)
            preamble_hits = sum(1 for n, l in kept if n < first and WORD.search(l))
            lists = {
                NEUTRAL: set(d["neutral"]),
                GEOMETRY: set(d["geometry"]),
                "shared": set(d["shared"]),
            }
            used: set[str] = set()

            def resolve(span: dict[str, Any]) -> str:
                for key in (span["qualified"], span["name"], span["owner"]):
                    if key is None:
                        continue
                    for klass, names in lists.items():
                        if key in names:
                            used.add(key)
                            return klass
                raise SystemExit(
                    f"{path}: span {span['qualified']} at :{span['line']} is in no partition list"
                )

            assigned = {
                span["qualified"] + f"@{span['line']}": resolve(span) for span in spans
            }
            unknown = (lists[NEUTRAL] | lists[GEOMETRY] | lists["shared"]) - used
            if unknown:
                raise SystemExit(
                    f"{path}: partition names that match no span: {sorted(unknown)}"
                )
            neutral_spans = [
                s_
                for s_ in spans
                if assigned[s_["qualified"] + f"@{s_['line']}"] == NEUTRAL
            ]
            shared_spans = [
                s_
                for s_ in spans
                if assigned[s_["qualified"] + f"@{s_['line']}"] == "shared"
            ]
            geometry_spans = [
                s_
                for s_ in spans
                if assigned[s_["qualified"] + f"@{s_['line']}"] == GEOMETRY
            ]
            low = sum(s["keptLines"] for s in neutral_spans if s["wordHits"] == 0)
            high = (
                sum(s["keptLines"] for s in neutral_spans)
                + sum(s["keptLines"] for s in shared_spans if s["wordHits"] == 0)
                + (preamble if preamble_hits == 0 else 0)
            )
            neutral_half = [low, high]
            partition = {
                "preambleLines": preamble,
                "neutralLines": sum(s["keptLines"] for s in neutral_spans),
                "neutralLinesTouchingVocabulary": sum(
                    s["keptLines"] for s in neutral_spans if s["wordHits"]
                ),
                "sharedLines": sum(s["keptLines"] for s in shared_spans),
                "geometryLines": sum(s["keptLines"] for s in geometry_spans),
                "neutral": [s_["qualified"] for s_ in neutral_spans],
                "shared": [s_["qualified"] for s_ in shared_spans],
                "geometry": [s_["qualified"] for s_ in geometry_spans],
            }
    return {
        "module": path,
        "class": d["class"],
        "nonTestLoc": proof_row["nonTestLoc"],
        "anchor": f"{anchor_path}::{d['anchor']} (:{line})",
        "anchorSymbol": d["anchor"],
        "anchorPath": anchor_path,
        "anchorLine": line,
        "grepResult": grep,
        "grepPass": {"substring": sub, "wordBoundary": word},
        "camelCaseGeometryLines": len(proof_row["camelCaseGeometryHits"]),
        "dismissal": dismissal,
        "reading": d["reading"],
        "splitNote": d.get("splitNote"),
        "neutralHalf": neutral_half,
        "partition": partition,
        "baseline": sha,
    }


def roadmap_band(corpus: int) -> tuple[list[int], int]:
    """The 'fifth to a quarter' band over `corpus` and the roadmap line stating it."""
    text = ROADMAP.read_text(encoding="utf-8")
    line = line_of(
        text, re.escape("Roughly a fifth to a quarter of the existing `lp/` code")
    )
    if line is None:
        raise SystemExit("roadmap IV.2 'fifth to a quarter' sentence not found")
    return [round(corpus / 5), round(corpus / 4)], line


def build_classification(sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    proof = build_proof(sha)
    if proof["universe"]["drift"]:
        raise SystemExit("universe drift: " + "; ".join(proof["universe"]["drift"]))
    rows = [
        classify_module(sha, path, proof["modules"][path]) for path in proof["modules"]
    ]
    corpus = proof["totals"]["nonTestLoc"]
    neutral = sum(r["nonTestLoc"] for r in rows if r["class"] == NEUTRAL)
    mixed_lo = sum(r["neutralHalf"][0] for r in rows if r["class"] == MIXED)
    mixed_hi = sum(r["neutralHalf"][1] for r in rows if r["class"] == MIXED)
    lo, hi = neutral + mixed_lo, neutral + mixed_hi
    band, band_line = roadmap_band(corpus)
    ticket_band = [round(11675 / 5), round(11675 / 4)]
    overlap = hi >= band[0] and lo <= band[1]
    ticket_proof = build_proof(TICKET_BASELINE)
    totals = {
        "corpusNonTestLoc": corpus,
        "productionModules": sum(1 for r in rows if r["class"] != SIBLING),
        "siblingTestFiles": sum(1 for r in rows if r["class"] == SIBLING),
        "classCounts": {
            c: sum(1 for r in rows if r["class"] == c)
            for c in (NEUTRAL, GEOMETRY, MIXED, SIBLING)
        },
        "engineNeutralLoc": neutral,
        "sddpGeometryLoc": sum(r["nonTestLoc"] for r in rows if r["class"] == GEOMETRY),
        "mixedLoc": sum(r["nonTestLoc"] for r in rows if r["class"] == MIXED),
        "mixedNeutralHalf": [mixed_lo, mixed_hi],
        "measuredBand": [lo, hi],
        "measuredShare": [round(lo / corpus, 3), round(hi / corpus, 3)],
        "roadmapBand": band,
        "roadmapBandBasis": f"{ROADMAP.relative_to(ROOT)}:{band_line} ('a fifth to a quarter') over the measured corpus of {corpus}",
        "ticketBand": ticket_band,
        "ticketBandBasis": "the ticket's 2,336-2,919 over its 11,675 corpus at a136840d",
        "verdict": "agreement" if overlap else "amended",
        "amendedFigure": None
        if overlap
        else f"the fifth-to-a-quarter estimate ({band[0]}-{band[1]} of {corpus}) is superseded by the measured {lo}-{hi} ({round(100 * lo / corpus)}-{round(100 * hi / corpus)}%) at {sha[:8]} on 2026-09-19; it undercounts the plain equipment column/row/entry fills and the generic-constraint lowering, which alone exceed it; extraction stays priced as a rewrite (IV.2) because every neutral fill is anchored after the θ-anchored state region today",
        "baseline": sha,
        "ticketBaseline": TICKET_BASELINE,
        "ticketBaselineTotals": ticket_proof["totals"],
    }
    classification = {
        "baseline": sha,
        "baselineSource": "plans/architecture-debt-audit/BACKLOG.md header (lib.backlog_parse.parse_baseline)",
        "ticketBaseline": TICKET_BASELINE,
        "universe": proof["universe"],
        "rubric": {
            "engine-neutral": "clean word-boundary slice AND a reading showing only entity indexing, block/cursor primitives, plain equipment columns and rows, generic-constraint lowering or FPHA planes; every substring-only hit carries an argued dismissal",
            "sddp-geometry": "state-space columns, cost-to-go anchoring, cut projection, anticipated rings, noise patching — stays in the L3 engine; a zero-hit never overrides the reading (builder/patch.rs)",
            "mixed": "both, with a function-level split note; neutralHalf = [neutral spans with zero word-boundary hits, neutral spans + vocabulary-free shared spans + a vocabulary-free preamble]",
            "test-sibling": "tests.rs / test_support.rs: zero non-test lines by rule",
        },
        "rows": rows,
        "totals": totals,
        "deviations": [
            f"The ticket pins a136840d; the register header pins {sha[:8]}, so every row is measured at the register pin; the same file set (30 modules) exists at both pins and the ticket's per-module figures (block_grid 126, anticipated_gate 112, patch.rs 413 with 8 CamelCase lines, naive 11,396, gross 44,374) reproduce exactly at a136840d.",
            f"The ticket's corpus total of 11,675 does not reproduce from its stated slice rule: the rule yields {ticket_proof['totals']['nonTestLoc']} at a136840d and {corpus} at {sha[:8]} (columns.rs +2, entries.rs +14 since the scaffold pin). The 10-line gap is consistent with the ticket snippet's handling of `#[cfg(test)] mod tests;`, which scans forward to the next `{{` and would swallow builder/mod.rs's re-export block as test code; this slicer ends a statement item at its `;`, as the rule's own words require.",
            "The sddp station's measurements/lp-inventory.json (non_test_lines 11,655) follows scripts/loc-stats.sh, which also excludes `#[cfg(any(test, feature = \"test-support\"))]` items; the ticket's rule keeps them (builder/mod.rs 155 vs 145, builder/template.rs 1327 vs 1293, lp/mod.rs 27 vs 25). The inventory is used for the module universe only.",
            "plans/ is tracked, so the artifacts are committed rather than gitignored.",
        ],
    }
    return classification, proof


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| module | class | non-test LOC | anchor | grep result | split note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["class"] == SIBLING:
            anchor = f"`{r['anchor']}`"
            note = "-"
            grep = r["grepResult"]
        else:
            anchor = f"`{r['anchorPath']}::{r['anchorSymbol']}` (:{r['anchorLine']})"
            grep = r["grepResult"] + (
                f"; DISMISSED: {r['dismissal']}" if r["dismissal"] else ""
            )
            note = r["splitNote"] or "-"
            if r["neutralHalf"]:
                note += f" — neutral half {r['neutralHalf'][0]}-{r['neutralHalf'][1]}"
        cell = lambda s: str(s).replace("|", "\\|")
        out.append(
            f"| `{r['module']}` | {r['class']} | {r['nonTestLoc']} | {anchor} | {cell(grep)} | {cell(note)} |"
        )
    return out


def render_totals(t: dict[str, Any]) -> list[str]:
    verdict = (
        "**agreement**"
        if t["verdict"] == "agreement"
        else f"**amended (dated 2026-09-19, baseline `{t['baseline'][:8]}`)** — {t['amendedFigure']}"
    )
    return [
        f"**Totals.** corpus {t['corpusNonTestLoc']:,} non-test lines over {t['productionModules']} production modules "
        f"(30 rows; {t['siblingTestFiles']} sibling test files at 0; classes: {t['classCounts'][NEUTRAL]} engine-neutral, "
        f"{t['classCounts'][GEOMETRY]} sddp-geometry, {t['classCounts'][MIXED]} mixed). engine-neutral: {t['engineNeutralLoc']:,} lines; "
        f"sddp-geometry: {t['sddpGeometryLoc']:,}; mixed: {t['mixedLoc']:,} with neutral half {t['mixedNeutralHalf'][0]:,}-{t['mixedNeutralHalf'][1]:,}. "
        f"Measured share {t['measuredBand'][0]:,}-{t['measuredBand'][1]:,} of {t['corpusNonTestLoc']:,} "
        f"({round(100 * t['measuredShare'][0])}-{round(100 * t['measuredShare'][1])}%) versus the roadmap's fifth-to-a-quarter band "
        f"{t['roadmapBand'][0]:,}-{t['roadmapBand'][1]:,} (the ticket's {t['ticketBand'][0]:,}-{t['ticketBand'][1]:,} over 11,675): {verdict}.",
        "",
        "**Method.** non-test slice = brace-aware exclusion of every `#[cfg(test)]`-attributed item (a statement item ends at its `;`; "
        "production lines after an attributed item are kept; the token inside a `//!` doc comment is prose; sibling `tests.rs` / "
        "`test_support.rs` count zero), not first-marker truncation; grep run twice over the slice, substring per the rubric and "
        "word-boundary to separate false positives; every substring-only hit on an engine-neutral row carries an argued dismissal; "
        "a zero-hit never classifies on its own (CamelCase geometry is invisible to the snake_case vocabulary). Extraction priced as a rewrite per Part IV.2.",
    ]


def render_markdown(c: dict[str, Any]) -> str:
    t = c["totals"]
    out = [
        f"# lp/ kernel boundary — classification at the register pin `{c['baseline'][:8]}`",
        "",
        f"Universe: {c['universe']['modules']} `.rs` modules under `{LP_ROOT}` (reconciled against `{c['universe']['inventory']}`; drift: {c['universe']['drift'] or 'none'}). "
        f"Machine-readable rows: alignment/lp-classification.json; grep evidence: alignment/lp-grep-proof.json; producer: alignment/lp-nontest-slice.py.",
        "",
        "## Rubric",
        "",
    ]
    out += [f"- **{k}** — {v}" for k, v in c["rubric"].items()]
    out += ["", "## Rows (tree order)", ""]
    out += render_table(c["rows"])
    out += ["", "## Totals", ""]
    out += render_totals(t)
    out += ["", "## Readings", ""]
    out += [
        f"- `{r['module']}` — {r['reading']}."
        for r in c["rows"]
        if r["class"] != SIBLING
    ]
    out += ["", "## Deviations recorded (no spec edit)", ""]
    out += [f"- {d}" for d in c["deviations"]]
    return "\n".join(out) + "\n"


def render_register_block(c: dict[str, Any]) -> list[str]:
    out = [f"#### lp/ kernel boundary (measured at baseline `{c['baseline'][:8]}`)", ""]
    out += render_table(c["rows"])
    out += [""]
    out += render_totals(c["totals"])
    out += [
        "",
        "Companion table with the reading-level rationale per module: alignment/lp-classification.md (rows also in alignment/lp-classification.json; grep evidence in alignment/lp-grep-proof.json).",
        "",
    ]
    return out


def append_register(block: list[str], heading: str) -> None:
    lines = REGISTER.read_text(encoding="utf-8").splitlines()
    marker = "#### lp/ kernel boundary (measured at baseline"
    if any(l.startswith(marker) for l in lines):
        raise SystemExit(
            "the lp/ kernel boundary H4 is already present in the register"
        )
    if heading in lines:
        idx = lines.index(heading)
        end = next(
            (j for j in range(idx + 1, len(lines)) if lines[j].startswith("## ")),
            len(lines),
        )
        while end > idx + 1 and not lines[end - 1].strip():
            end -= 1
        body = lines[idx + 1 : end]
        body = [l for l in body if l.strip() and l.strip() != "_(no entries yet)_"]
        new = (
            lines[: idx + 1]
            + [""]
            + (body + [""] if body else [])
            + block
            + lines[end:]
        )
    else:
        new = lines + ["", heading, ""] + block
    REGISTER.write_text("\n".join(new).rstrip("\n") + "\n", encoding="utf-8")


def cmd_classify(sha: str, out_dir: pathlib.Path, append: bool, heading: str) -> int:
    classification, proof = build_classification(sha)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lp-grep-proof.json").write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "lp-classification.json").write_text(
        json.dumps(classification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "lp-classification.md").write_text(
        render_markdown(classification), encoding="utf-8"
    )
    t = classification["totals"]
    print(
        f"{len(classification['rows'])} rows @ {sha[:8]}: corpus {t['corpusNonTestLoc']}, neutral {t['engineNeutralLoc']}, "
        f"mixed half {t['mixedNeutralHalf']}, band {t['measuredBand']} vs roadmap {t['roadmapBand']} -> {t['verdict']}"
    )
    if append:
        append_register(render_register_block(classification), heading)
        print(f"register: appended the lp/ kernel boundary H4 under {heading!r}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("self-test")
    p.add_argument("--baseline", default=TICKET_BASELINE)
    p = sub.add_parser("slice")
    p.add_argument("--baseline", default=None)
    p.add_argument("path")
    p = sub.add_parser("functions")
    p.add_argument("--baseline", default=None)
    p.add_argument("path")
    p = sub.add_parser("proof")
    p.add_argument("--baseline", default=None)
    p.add_argument("--out", type=pathlib.Path, default=ALIGN / "lp-grep-proof.json")
    p = sub.add_parser("classify")
    p.add_argument("--baseline", default=None)
    p.add_argument("--append-register", action="store_true")
    p.add_argument("--out-dir", type=pathlib.Path, default=ALIGN)
    p.add_argument("--section-heading", default=DEFAULT_SECTION_HEADING)
    args = ap.parse_args(argv)
    if getattr(args, "baseline", None) is None and args.cmd != "self-test":
        args.baseline = register_pin()
    if args.cmd == "self-test":
        return cmd_self_test(args.baseline)
    if args.cmd == "slice":
        return cmd_slice(args.baseline, args.path)
    if args.cmd == "functions":
        return cmd_functions(args.baseline, args.path)
    if args.cmd == "proof":
        return cmd_proof(args.baseline, args.out)
    if args.cmd == "classify":
        return cmd_classify(
            args.baseline, args.out_dir, args.append_register, args.section_heading
        )
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
