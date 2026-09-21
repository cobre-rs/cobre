#!/usr/bin/env python3
"""Assemble measurements/claim-table.json — the performance sweep's schedule — and render its blocks.

Reads the seven station perf queues (an absent perf-queue.json is a station with zero rows,
recorded, never omitted), derives every row's layout and deck from its claim-type tag and its
`requires` list (collective -> 2x2 on the sampled deck; single-process -> 4t on the sampled
deck, or 2t on the enumerated deck when the row requires `enumerated` traversal or an
`external-library`), refuses a stated layout that contradicts the derived one unless the owner
recorded an allowance for that id, refuses a queue row that asserts a timing figure, re-verifies
every anchor at the register pin through the shared parser's `git show` (never the worktree),
copies claim text verbatim, and appends the two register-parked items (PD-004 under the epic's
pending-a-profile exemption, the Wave-2 PD-005 residual). Alignment is read from the
adjudicated ledger (alignment/alignment-ledger.json), the register's single authority since the
alignment epic; PD-004 is not a station entry and carries the neutral tag the register gives it.

Usage: claim-table.py [--accept-derived ID ...] [--out PATH] [--render-section] [--render-log]
Exit: 0 written · 1 a contradiction, a timing assertion or a rejected item · 2 an input is missing.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve()
MEAS = HERE.parents[1]
AUDIT = MEAS.parent
TOOLS = AUDIT / "tools"
sys.path.insert(0, str(TOOLS))
from lib import backlog_parse as bp  # noqa: E402  (the harness owns every read of BACKLOG.md)

REGISTER = AUDIT / "BACKLOG.md"
LEDGER = AUDIT / "alignment" / "alignment-ledger.json"
OUT = MEAS / "claim-table.json"
CASE = MEAS / "_case"
STATIONS = (
    "core-io",
    "stochastic",
    "solver-comm",
    "sddp",
    "cli-python",
    "build-ci",
    "test-corpus",
)
ENUMERATED_DECK_REQS = {"enumerated", "external-library"}
DECK_OF = {"4t": "reduzido", "2x2": "reduzido", "2t": "mar-26-enumerated"}
DECK_DIR = {
    "reduzido": "measurements/_case/deck",
    "mar-26-enumerated": "measurements/_case/deck-enumerated",
}
TICKET_DECK_NAMES = {
    "reduzido": "reduzido-2 (the tickets' lost cobre_reduzido_2; the owner re-sanctioned cobre_reduzido)",
    "mar-26-enumerated": "mar-26-enumerated",
}
DO_NOT_TOUCH_ALLOWED = {"PD-004"}
TIMING_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:ms|µs|us|ns|s|sec|secs|seconds?|minutes?|min)\b(?!-)"
)
DECL = r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(fn|struct|enum|trait|type|const|static|mod|impl)\s+{sym}\b"
TICKET_BASE = "a136840d"


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def requires_list(value: Any) -> list[str]:
    if value in (None, "none", "", []):
        return []
    if isinstance(value, str):
        return [
            v.strip()
            for v in re.split(r"[,+/]", value)
            if v.strip() and v.strip() != "none"
        ]
    return [str(v) for v in value if str(v) != "none"]


def derive_layout(tag: str, requires: list[str]) -> str:
    if tag == "collective":
        return "2x2"
    return "2t" if ENUMERATED_DECK_REQS & set(requires) else "4t"


def anchor_string(row: dict[str, Any]) -> str | None:
    if row.get("anchor"):
        return row["anchor"]
    t = row.get("target")
    if isinstance(t, dict) and t.get("path"):
        if t.get("symbol"):
            return f"{t['path']}::{t['symbol']}"
        if t.get("line"):
            return f"{t['path']}:{t['line']}"
        return t["path"]
    return None


def anchor_resolves(pin: str, anchor: str) -> tuple[bool, str]:
    """(resolves, reason) for `path`, `path:line` or `path::symbol` at the pin, via git show."""
    if "::" in anchor:
        path, sym = anchor.split("::", 1)
        text = bp.git_show(pin, path)
        if text is None:
            return False, "path-missing"
        return (
            re.search(DECL.format(sym=re.escape(sym)), text, re.M) is not None,
            "symbol-missing",
        )
    path, _, tail = anchor.partition(":")
    text = bp.git_show(pin, path)
    if text is None:
        return False, "path-missing"
    if tail:
        first = int(re.match(r"\d+", tail).group(0)) if re.match(r"\d+", tail) else None
        if first is None:
            return False, "malformed-line"
        return (1 <= first <= len(text.splitlines()), "line-past-eof")
    return True, ""


def register_entry_text(lines: list[str], entry_id: str) -> tuple[str, int]:
    """The verbatim claim text of a register entry (heading tail + body up to the first blank or arrow line)."""
    for i, ln in enumerate(lines):
        if ln.startswith(f"**{entry_id} ·"):
            body: list[str] = []
            head = ln.split("**", 2)[2].strip(" —-")
            if head:
                body.append(head)
            for nxt in lines[i + 1 :]:
                if (
                    not nxt.strip()
                    or nxt.startswith("**→")
                    or nxt.startswith("**")
                    or nxt.startswith("#")
                ):
                    break
                body.append(nxt.strip())
            return " ".join(body), i + 1
    raise SystemExit(f"{entry_id}: not found in the register")


def retired_ids(pin: str) -> set[str]:
    """Finding ids named by the harness's retired corpus (do-not-touch list, Cleared lines, mirror,
    resolved forks) — loaded through check-reraise.py itself, never re-derived here."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_reraise", TOOLS / "check-reraise.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("tools/check-reraise.py not importable")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_reraise"] = (
        mod  # its frozen dataclasses resolve annotations via sys.modules
    )
    spec.loader.exec_module(mod)
    ids: set[str] = set()
    for item in mod.load_retired_corpus(pin):
        ids.update(re.findall(r"\b(?:CD|PD|OD|TD)-\d{3}\b", f"{item.ref} {item.title}"))
    return ids


def parked_rows(
    lines: list[str], pin: str, alignment: dict[str, str]
) -> list[dict[str, Any]]:
    pd004_claim, pd004_line = register_entry_text(lines, "PD-004")
    pd005_claim, pd005_line = register_entry_text(lines, "PD-005")
    return [
        {
            "id": "PD-004",
            "station": "sddp",
            "origin": "register (Part II do-not-touch list; also queued by the sddp station as an existence row)",
            "claim": pd004_claim,
            "claimSource": f"BACKLOG.md:{pd004_line} (verbatim)",
            "claimType": "single-process",
            "requires": ["enumerated"],
            "statedLayout": "2t",
            "layout": "2t",
            "case": "mar-26-enumerated",
            "deckDir": DECK_DIR["mar-26-enumerated"],
            "profiledSymbol": "run_enumerated_backward",
            "anchors": ["crates/cobre-sddp/src/training/backward_pass_state.rs:919"],
            "exercisingCallSites": [
                "crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward"
            ],
            "severity": "C",
            "alignment": alignment.get("PD-004") or "neutral",
            "alignmentSource": "register do-not-touch list (not a station entry; absent from the alignment ledger)",
            "measured": False,
            "status": "queued",
            "parked": True,
            "note": "do-not-touch, deferred 2026-08-18 and re-confirmed 2026-08-19 pending a profile; admitted here under this epic's explicit exemption and nowhere else; the sampled deck never reaches the enumerated arm, hence the enumerated deck",
        },
        {
            "id": "PD-005-residual",
            "station": "sddp",
            "origin": "register (Wave-2 residual of PD-005)",
            "claim": pd005_claim,
            "claimSource": f"BACKLOG.md:{pd005_line} (verbatim)",
            "claimType": "single-process",
            "requires": ["enumerated"],
            "statedLayout": None,
            "layout": "2t",
            "case": "mar-26-enumerated",
            "deckDir": DECK_DIR["mar-26-enumerated"],
            "profiledSymbol": "nested_ub_recursion",
            "anchors": [
                "crates/cobre-sddp/src/training/forward/stats_aggregation.rs::nested_ub_recursion",
                "crates/cobre-sddp/src/setup/node_graph.rs::NestedUbTopology",
            ],
            "exercisingCallSites": [
                "crates/cobre-sddp/src/training/forward/stats_aggregation.rs::nested_ub_recursion (once per iteration on the forward path)"
            ],
            "severity": "B",
            "alignment": alignment.get("PD-005") or "neutral",
            "alignmentSource": "register entry PD-005 (Wave-2 fixed; the residual carries its tag)",
            "measured": False,
            "status": "queued",
            "parked": True,
            "note": "Wave-2 residual: the NestedUbTopology once-per-iteration precompute, recorded as a noted residual, never as a closed item",
        },
    ]


def assemble(
    accept_derived: set[str], stations_dir: pathlib.Path = AUDIT / "stations"
) -> dict[str, Any]:
    lines = bp.read_register(REGISTER)
    pin = bp.parse_baseline(lines)
    ledger = read_json(LEDGER)
    alignment = {e["entryId"]: e["decided"] for e in ledger["ledger"]}
    retired = retired_ids(pin)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    queue_baselines: dict[str, str] = {}
    bounced: list[dict[str, str]] = []
    allowances: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for station in STATIONS:
        path = stations_dir / station / "perf-queue.json"
        if not path.exists():
            counts[station] = 0
            queue_baselines[station] = "(no perf-queue.json)"
            continue
        q = read_json(path)
        entries = q.get("queue") or q.get("entries") or []
        counts[station] = len(entries)
        queue_baselines[station] = q.get("baseline", "")[:8]
        for e in entries:
            eid = e["id"]
            if eid in seen:
                raise SystemExit(f"{eid}: queued twice")
            seen.add(eid)
            tag = e["claimType"]
            reqs = requires_list(e.get("requires"))
            layout = derive_layout(tag, reqs)
            stated = e.get("layout")
            allowance = None
            if stated not in (None, layout):
                if eid not in accept_derived:
                    raise SystemExit(
                        f"{eid}: stated layout {stated} contradicts claimType {tag} / requires {reqs} (derived {layout})"
                    )
                allowance = f"owner allowance 2026-09-20: derived {layout} accepted over the station's stated {stated}; the slip is returned to {station}"
                allowances.append(
                    {"id": eid, "station": station, "stated": stated, "derived": layout}
                )
            if e.get("assertedTiming") is not None or TIMING_RE.search(
                e.get("claim") or ""
            ):
                m = TIMING_RE.search(e.get("claim") or "")
                raise SystemExit(
                    f"{eid}: station asserted a timing figure ({m.group(0) if m else e.get('assertedTiming')}); queues carry claims only"
                )
            anchor = anchor_string(e)
            claim = e.get("claim")
            if claim is None and eid == "PD-004":
                continue  # the sddp existence row for PD-004: the parked row below carries the register's verbatim text
            if claim is None:
                raise SystemExit(f"{eid}: queue row without claim text")
            row = {
                "id": eid,
                "station": station,
                "origin": f"stations/{station}/perf-queue.json",
                "candidateRef": e.get("candidateRef"),
                "claim": claim,
                "claimSource": "queue row, verbatim",
                "claimType": tag,
                "requires": reqs,
                "statedLayout": stated,
                "layout": layout,
                "layoutAllowance": allowance,
                "case": DECK_OF[layout],
                "deckDir": DECK_DIR[DECK_OF[layout]],
                "profiledSymbol": e.get("profiledSymbol")
                or (e.get("target") or {}).get("symbol"),
                "anchors": [anchor] if anchor else [],
                "exercisingCallSites": e.get("exercisingCallSites", []),
                "severity": e.get("severity"),
                "stationStatus": e.get("status"),
                "alignment": alignment.get(eid),
                "alignmentSource": "alignment/alignment-ledger.json (decided)"
                if eid in alignment
                else None,
                "measured": False,
                "status": "queued",
            }
            if row["alignment"] is None:
                raise SystemExit(f"{eid}: no decided Alignment in the ledger")
            if not row["anchors"]:
                row["status"] = "anchor-missing"
                bounced.append(
                    {"id": eid, "station": station, "reason": "no anchor recorded"}
                )
            for a in row["anchors"]:
                ok, reason = anchor_resolves(pin, a)
                if not ok:
                    row["status"] = "anchor-missing"
                    bounced.append(
                        {"id": eid, "station": station, "anchor": a, "reason": reason}
                    )
            rows.append(row)
    for p in parked_rows(lines, pin, alignment):
        for a in p["anchors"]:
            ok, reason = anchor_resolves(pin, a)
            if not ok:
                p["status"] = "anchor-missing"
                bounced.append(
                    {
                        "id": p["id"],
                        "station": p["station"],
                        "anchor": a,
                        "reason": reason,
                    }
                )
        rows.append(p)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate id in the claim table"
    for r in rows:
        if (
            r["id"].startswith("PD-")
            and r["id"] in retired
            and r["id"] not in DO_NOT_TOUCH_ALLOWED
            and r["id"] != "PD-005-residual"
        ):
            rejected.append(
                {
                    "id": r["id"],
                    "reason": "retired / Cleared / reserved-seam item queued by a station",
                }
            )
    if rejected:
        raise SystemExit(f"rejected items in a queue: {rejected}")
    schedulable = [r for r in rows if r["status"] == "queued"]
    return {
        "artifact": "claim-table",
        "baseline": pin,
        "ticketBaseline": TICKET_BASE,
        "generatedBy": "measurements/_tools/claim-table.py",
        "layoutRule": "collective -> 2x2 on the sampled deck; single-process -> 4t on the sampled deck, or 2t on the enumerated deck when requires includes enumerated or external-library; the tag picks the layout, never the operator",
        "decks": {
            "reduzido": {
                "source": "~/git/cobre-bridge/example/cobre_reduzido",
                "dir": DECK_DIR["reduzido"],
                "ticketName": TICKET_DECK_NAMES["reduzido"],
            },
            "mar-26-enumerated": {
                "source": "~/git/cobre-bridge/example/cobre-mar-26-rv2-reduced",
                "dir": DECK_DIR["mar-26-enumerated"],
                "ticketName": TICKET_DECK_NAMES["mar-26-enumerated"],
            },
        },
        "stationCounts": counts,
        "queueBaselines": queue_baselines,
        "parked": ["PD-004", "PD-005-residual"],
        "allowances": allowances,
        "bounced": bounced,
        "schedulable": len(schedulable),
        "byLayout": dict(collections.Counter(r["layout"] for r in schedulable)),
        "byStation": dict(collections.Counter(r["station"] for r in rows)),
        "rows": rows,
    }


def cell(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).replace("|", "\\|").strip()


def render_table(table: dict[str, Any]) -> list[str]:
    out = [
        "| ID | Claim | Symbol | Tag | Layout | Anchor | Exercising call sites | Station |",
        "|----|-------|--------|-----|--------|--------|-----------------------|---------|",
    ]
    for r in table["rows"]:
        if r["status"] != "queued":
            continue
        anchors = "; ".join(f"`{a}`" for a in r["anchors"])
        sites = "; ".join(cell(s) for s in r.get("exercisingCallSites") or []) or "—"
        layout = r["layout"] + (
            " (derived; station stated " + r["statedLayout"] + ")"
            if r.get("layoutAllowance")
            else ""
        )
        sym = r.get("profiledSymbol") or "—"
        if "::" in str(sym):
            sym = str(sym).rsplit("::", 1)[1]
        out.append(
            f"| {r['id']} | {cell(r['claim'])} | `{cell(sym)}` | {r['claimType']} | {layout} | {anchors} | {sites} | {r['station']} |"
        )
    return out


def render_section(table: dict[str, Any], log_path: pathlib.Path) -> list[str]:
    log = {}
    for ln in log_path.read_text(encoding="utf-8").splitlines():
        if "\t" in ln:
            k, v = ln.split("\t", 1)
            log.setdefault(k, v)
    sampled_cfg = read_json(CASE / "deck" / "config.json")
    enum_cfg = read_json(CASE / "deck-enumerated" / "config.json")
    s_stages = len(read_json(CASE / "deck" / "stages.json")["stages"])
    e_stages = len(read_json(CASE / "deck-enumerated" / "stages.json")["stages"])
    s_tr = sampled_cfg["training"]
    e_tr = enum_cfg["training"]
    it = next(
        r["limit"] for r in s_tr["stopping_rules"] if r["type"] == "iteration_limit"
    )
    gap = next(
        r["relative_tolerance"] for r in e_tr["stopping_rules"] if r["type"] == "gap"
    )
    cap = next(
        r["limit"] for r in e_tr["stopping_rules"] if r["type"] == "iteration_limit"
    )
    ext = [
        k
        for k in ("inflow", "load", "ncs")
        if e_tr["scenario_source"].get(k, {}).get("scheme") == "external"
    ]
    c = table["stationCounts"]
    b4 = log["protocol_bound_4t"]
    b2 = log["protocol_bound_2t"]
    body = [
        "",
        f"Opened {'2026-09-20'} at the register pin `{table['baseline'][:8]}` (the tickets quote the scaffold pin `{TICKET_BASE}`; crates/ and the Cargo manifests at HEAD are byte-identical to the register pin, so every later figure is a register-pin figure). "
        "No timing figure, no measurement block and no finding id is written by the opening ticket; the measuring tickets fill the blocks below. Sweep log: measurements/sweep-log.md; schedule: measurements/claim-table.json.",
        "",
        '**Binary.** target/profiling/cobre — `cargo build --profile profiling --features mpi --bin cobre` (release codegen, `debug = 1`, `strip = "none"`; `file` reports **not stripped**, `readelf -S` shows `.debug_line`; '
        f"build id `{log['build_id']}`). One binary serves 4t, 2t and 2x2: the `mpi` feature links cobre-comm/mpi for the collective layout and `--comm-backend local` keeps 4t/2t single-process on the same codegen. "
        "`release` strips symbols and `dist` adds thin LTO, so neither is substituted and no figure here is a dist number.",
        "",
        f"**Host.** {log['cpu_model']} hybrid, governor `powersave` (disclosed, not changed). cpus 0-15 are the eight SMT P-cores, cpus 16-19 the E-cores at 3.8 GHz; E-cores never enter a timed run. "
        "4t pins to physical P-cores `0,2,4,6`; 2t pins to `0,2`; 2x2 pins rank 0 to `0,2` and rank 1 to `4,6` **inside the wrapper** (measurements/_wrap/rank-wrapper.sh, created by the collective ticket), never around `mpiexec`. "
        "perf-run.sh already pins these sets (the tickets' 'default PCORES 0-15 overridden' premise is stale; the harness is not edited). "
        f"`{log['perf']}` with `kernel.perf_event_paranoid = {log['perf_event_paranoid']}`: user-space samples of our own process resolve, kernel symbols do not, so **no claim may attribute cost to a kernel frame**. "
        "No `flamegraph`, `inferno-flamegraph` or `cargo-flamegraph` on PATH and this epic installs nothing, so the deliverable is `perf report --stdio --no-children` text; an SVG is optional garnish and never a gate. "
        f"`mpiexec` is present ({log['mpiexec'].split(' (')[0]}), so a 2x2 row that cannot run is `case-infeasible`, not `mpi-unavailable`.",
        "",
        f"**Cases.** Sampled deck `cobre_reduzido` staged once to measurements/_case/deck (case id `reduzido`; {s_stages} stages, {s_tr['selection']['method']} selection, {s_tr['selection']['forward_passes']} forward passes, iteration limit {it}, "
        f"cut selection {s_tr['cut_selection']['selection']['method']} with check_frequency {s_tr['cut_selection']['selection']['check_frequency']}, PAR max_order {sampled_cfg['estimation']['max_order']}, {sampled_cfg['simulation']['selection']['num_scenarios']} simulation scenarios). "
        "The tickets name `cobre_reduzido_2`, which was lost from the gitignored cobre-bridge example tree; the owner re-sanctioned `cobre_reduzido`, which the harness and the CAL run already use. "
        f"Enumerated deck `cobre-mar-26-rv2-reduced` staged once to measurements/_case/deck-enumerated (case id `mar-26-enumerated`; {e_stages} stages, enumerated selection, external {'/'.join(ext)} libraries, gap {gap} with a {cap}-iteration cap; its dated output/ tree is part of the digest). "
        "Both source digests are fenced in measurements/_case/source-sha256.txt (sha256 before == after == staged); every run writes under its own scratch output and never into a deck. "
        "The deck copies are gitignored inside a tracked plans/ tree; the digest file is committed.",
        "",
        f"**Bounds.** 4t/2x2: {b4}. 2t: {b2}. A run past 3x its bound is killed and tagged `UNMEASURED timeout`. UNMEASURED reasons: `timeout` / `unexercised-path` / `mpi-unavailable` / `case-infeasible`. "
        "Material at >= 3% of the median phase wall, or an allocation site holding >= 1% of user-space samples; below that a claim closes `not-material` rather than deferring again.",
        "",
        "### Claim-type table",
        "",
        f"Station queues read: core-io {c['core-io']}, stochastic {c['stochastic']}, solver-comm {c['solver-comm']}, sddp {c['sddp']} (its PD-004 existence row is folded into the parked PD-004 row), cli-python {c['cli-python']}, build-ci {c['build-ci']} (no queue), test-corpus {c['test-corpus']} (no queue); "
        f"plus the two register-parked rows PD-004 (admitted under this epic's pending-a-profile exemption, the only do-not-touch item the table may carry) and PD-005-residual. "
        f"{table['schedulable']} schedulable rows ({', '.join(f'{v} {k}' for k, v in sorted(table['byLayout'].items()))}); bounced (anchor-missing): {len(table['bounced'])}; "
        f"layout allowances: {', '.join(a['id'] + ' (' + a['stated'] + ' → ' + a['derived'] + ')' for a in table['allowances']) or 'none'}. "
        "Claim text is the station's, verbatim. Alignment is the adjudicated ledger's decided value (all rows neutral). The tag picks the layout; the operator never does.",
        "",
        *render_table(table),
        "",
    ]
    return body


def render_log_tail(table: dict[str, Any]) -> list[str]:
    out = ["", "## Claim table", ""]
    out.append(
        f"rows\t{len(table['rows'])} ({table['schedulable']} schedulable; "
        + ", ".join(f"{k} {v}" for k, v in sorted(table["byLayout"].items()))
        + ")"
    )
    out.append(
        "station_counts\t"
        + ", ".join(f"{k} {v}" for k, v in table["stationCounts"].items())
    )
    out.append(
        "parked\tPD-004 (pending-a-profile exemption; the sddp queue's existence row is folded into it), PD-005-residual (Wave-2 residual)"
    )
    for a in table["allowances"]:
        out.append(
            f"layout_allowance\t{a['id']} ({a['station']}): station stated {a['stated']}, derived {a['derived']} accepted by the owner 2026-09-20; the stated-layout slip is returned to the station for correction"
        )
    out += [
        "",
        "## Bounced claims (anchor-missing at the pin; returned to their station; no measurements/<ID>/ directory)",
        "",
    ]
    if not table["bounced"]:
        out.append("none\tevery queued anchor resolves at the register pin")
    for b in table["bounced"]:
        out.append(f"{b['id']}\t{b['station']}\t{b.get('anchor', '')}\t{b['reason']}")
    out += ["", "## Preflight answers inherited by the measuring tickets", ""]
    out.append(
        "mpiexec\tpresent — the collective ticket runs 2x2; a 2x2 row that cannot run is case-infeasible, not mpi-unavailable"
    )
    out.append(
        "rank_wrapper\tmeasurements/_wrap/rank-wrapper.sh is absent; perf-run.sh requires it for 2x2 — the collective ticket creates it (taskset rank 0 -> 0,2, rank 1 -> 4,6) before its first run"
    )
    out.append(
        "enumerated_bound\tmeasured (30.434 s), so no 2t row is pre-tagged UNMEASURED timeout"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--accept-derived",
        action="append",
        default=[],
        help="row id whose stated layout the owner allowed to be superseded by the derived one",
    )
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument(
        "--stations-dir",
        type=pathlib.Path,
        default=AUDIT / "stations",
        help="read the perf queues from another stations tree (guard demonstrations)",
    )
    ap.add_argument(
        "--render-section",
        action="store_true",
        help="print the register section body instead of writing the table",
    )
    ap.add_argument(
        "--render-log", action="store_true", help="print the sweep-log tail blocks"
    )
    args = ap.parse_args(argv)
    try:
        table = assemble(set(args.accept_derived), args.stations_dir)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    if args.render_section:
        print("\n".join(render_section(table, MEAS / "sweep-log.md")))
        return 0
    if args.render_log:
        print("\n".join(render_log_tail(table)))
        return 0
    args.out.write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"claim-table.json: {len(table['rows'])} rows ({table['schedulable']} schedulable; {table['byLayout']}), "
        f"stations {table['stationCounts']}, bounced {len(table['bounced'])}, allowances {[a['id'] for a in table['allowances']]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
