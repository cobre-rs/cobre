#!/usr/bin/env python3
"""Consolidate the stations' Part-I handoffs into the nine-row disposition envelope.

Reads the register pin from BACKLOG.md through the shared parser, resolves every
station handoff under its primary name (partI-handoff.json) or its alias
(alignment-queue.json), seeds one row per Part-I item (I.3 items 1-8 and I.5) with the
roadmap's verbatim claim text, re-resolves every anchor at the pin (and at the
superseded scaffold pin the ticket text quotes, so the drift is recorded rather than
silently corrected), allocates fresh CD ids from max(existing)+1 over the register
minus the alignment section's own entries (so a regeneration after the adjudication
ticket splices them in reuses the ids it minted), and renders the table fragment the
adjudication ticket splices into the register. Every ingest failure lands in
`ingestRejects` and makes the run exit 1; a short table never passes as a complete one.
Writes only under alignment/.

Usage: consolidate_part_i.py [--stations-dir DIR] [--out-dir DIR] [--no-self-check]
Exit: 0 clean, 1 ingest reject(s) or a failing self-check, 2 register unreadable.
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
AUDIT = HERE.parents[1]
TOOLS = AUDIT / "tools"
sys.path.insert(0, str(TOOLS))

from lib import backlog_parse as bp  # noqa: E402

ROOT = AUDIT.parents[1]
REGISTER = AUDIT / "BACKLOG.md"
ROADMAP = ROOT / "plans" / "generalizing" / "beyond-sddp-generalization.md"
ROADMAP_REL = "plans/generalizing/beyond-sddp-generalization.md"
TICKET_BASELINE = "a136840d4f2ea137f685f0af6dac04254b983b60"
SECTION_TITLE = "GENERALIZATION ALIGNMENT"
HANDOFF_NAMES = ("partI-handoff.json", "alignment-queue.json")
OWNING_STATIONS = ("core-io", "solver-comm", "sddp", "cli-python", "build-ci")
DECL = (
    r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?"
    r"(fn|struct|enum|trait|type|const|static|mod|impl)\s+{sym}\b"
)
EXIT_OK = 0
EXIT_REJECT = 1
EXIT_REGISTER = 2


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def show(sha: str, path: str) -> str | None:
    proc = git("show", f"{sha}:{path}")
    return proc.stdout if proc.returncode == 0 else None


def line_of(text: str, pattern: str) -> int | None:
    for idx, raw in enumerate(text.splitlines(), 1):
        if re.search(pattern, raw):
            return idx
    return None


def decl_line(text: str, symbol: str) -> int | None:
    return line_of(text, DECL.format(sym=re.escape(symbol)))


def grep_count(
    sha: str, pattern: str, *pathspec: str, only_matching: bool = False
) -> int:
    args = [
        "grep",
        "-o" if only_matching else "-n",
        "-E",
        pattern,
        sha,
        "--",
        *pathspec,
    ]
    return len([ln for ln in git(*args).stdout.splitlines() if ln])


def grep_files(sha: str, pattern: str, *pathspec: str) -> int:
    return len(
        [
            ln
            for ln in git(
                "grep", "-l", "-E", pattern, sha, "--", *pathspec
            ).stdout.splitlines()
            if ln
        ]
    )


def grep_distinct(sha: str, pattern: str, *pathspec: str) -> int:
    out = git(
        "grep", "-h", "-o", "-E", pattern, sha, "--", *pathspec
    ).stdout.splitlines()
    return len({ln for ln in out if ln})


def roadmap_line(needle: str) -> int:
    n = line_of(ROADMAP.read_text(encoding="utf-8"), re.escape(needle))
    if n is None:
        raise SystemExit(f"roadmap needle not found: {needle!r}")
    return n


def roadmap_items() -> dict[str, str]:
    """Verbatim Part-I claim text: I.3 items 1-8 (numbered paragraphs) and the I.5 opening."""
    lines = ROADMAP.read_text(encoding="utf-8").splitlines()
    sec = bp.find_section(
        lines,
        'I.3 The leakage points — SDDP and stochastic concepts inside the "generic" core',
    )
    claims: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for raw in sec.lines:
        m = re.match(r"^(\d)\.\s+(.*)$", raw)
        if m:
            if current:
                claims[current] = " ".join(buf)
            current, buf = m.group(1), [m.group(2).strip()]
        elif current and raw.startswith("   ") and raw.strip():
            buf.append(raw.strip())
        elif current and not raw.strip():
            claims[current] = " ".join(buf)
            current = None
    if current:
        claims[current] = " ".join(buf)
    i5 = bp.find_section(lines, "I.5 The CLI and orchestration coupling (`cobre-cli`)")
    body: list[str] = []
    for raw in i5.lines:
        if raw.startswith("The insertion points"):
            break
        if raw.strip():
            body.append(raw.strip())
    claims["I.5"] = " ".join(body)
    return claims


def load_handoffs(
    stations_dir: pathlib.Path,
) -> tuple[dict[str, dict], list[str], list[dict]]:
    handoffs: dict[str, dict] = {}
    generated_from: list[str] = []
    rejects: list[dict] = []
    for station in OWNING_STATIONS:
        found = None
        for name in HANDOFF_NAMES:
            candidate = stations_dir / station / name
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            rejects.append(
                {
                    "item": STATION_ITEMS[station],
                    "kind": "handoff-missing",
                    "station": station,
                    "detail": f"neither {' nor '.join(HANDOFF_NAMES)} under {stations_dir / station}",
                }
            )
            continue
        handoffs[station] = json.loads(found.read_text(encoding="utf-8"))
        generated_from.append(f"stations/{station}/{found.name}")
    return handoffs, generated_from, rejects


STATION_ITEMS = {
    "core-io": "1, 2, 3, 4, 6, 7",
    "solver-comm": "8",
    "sddp": "7",
    "cli-python": "7, I.5",
    "build-ci": "6",
}


def next_free_cd(register_lines: list[str]) -> int:
    """max(existing CD id)+1 over the register minus the alignment section's own entries,
    so a regeneration after the adjudication ticket splices them in reuses the ids it minted."""
    lines = list(register_lines)
    try:
        own = bp.find_section(lines, "generalization-alignment")
        lines = lines[: own.start] + lines[own.end :]
    except bp.SectionNotFound:
        pass
    ids = [int(m) for m in re.findall(r"\bCD-(\d{3})\b", "\n".join(lines))]
    return max(ids) + 1


def resolve(
    pin: str, path: str, symbol: str | None = None, needle: str | None = None
) -> dict[str, Any]:
    text = show(pin, path)
    if text is None:
        return {"path": path, "symbol": symbol, "line": None, "status": "path-missing"}
    line = decl_line(text, symbol) if symbol else line_of(text, needle or "")
    return {
        "path": path,
        "symbol": symbol,
        "line": line,
        "status": "ok" if line else ("symbol-missing" if symbol else "needle-missing"),
    }


def anchor_text(a: dict[str, Any]) -> str:
    if a.get("symbol"):
        return f"`{a['path']}::{a['symbol']}`"
    return f"`{a['path']}:{a['line']}`"


def specs() -> list[dict[str, Any]]:
    """The nine rows: anchor specs (resolved at both pins), dispositions and prose."""
    return [
        {
            "item": "1",
            "partIRef": "I.3-1",
            "title": "System stores the stochastic input model as first-class fields",
            "v012Anchor": {"path": "crates/cobre-core/src/system/mod.rs", "line": 107},
            "anchors": [
                ("crates/cobre-core/src/system/mod.rs", "System", None, "container"),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+inflow_models: Vec<InflowModel>,",
                    "claimed field inflow_models",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+load_models: Vec<LoadModel>,",
                    "claimed field load_models",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+ncs_models: Vec<NcsModel>,",
                    "claimed field ncs_models",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+correlation: CorrelationModel,",
                    "claimed field correlation",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+inflow_history: Vec<InflowHistoryRow>,",
                    "widening field inflow_history",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+external_scenarios: Vec<ExternalScenarioRow>,",
                    "widening field external_scenarios",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+external_load_scenarios: Vec<ExternalLoadRow>,",
                    "widening field external_load_scenarios",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+external_ncs_scenarios: Vec<ExternalNcsRow>,",
                    "widening field external_ncs_scenarios",
                ),
                (
                    "crates/cobre-core/src/model/scenario.rs",
                    None,
                    r"^//!",
                    "the stochastic-inflow pipeline module",
                ),
            ],
            "sourceStations": ["core-io"],
            "relatedRegisterIds": [
                "CD-045",
                "OD-013",
                "OD-016",
                "CD-065",
                "CD-070",
                "CD-071",
            ],
            "stationDispositions": {
                "core-io": "sharpen",
                "stochastic": "sharpen (CD-065 / CD-070 / CD-071, the L1 store-vs-config seam)",
            },
            "disposition": "sharpen",
            "sharpenedClaim": "`System` carries eight stochastic-input fields, not four: the claimed `inflow_models`, `load_models`, `ncs_models`, `correlation` plus `inflow_history`, `external_scenarios`, `external_load_scenarios`, `external_ncs_scenarios`, which the v0.12 text folded into 'plus external/historical scenario tables' without naming; a deterministic dispatch or power-flow study reads none of them, and the whole `model/scenario.rs` pipeline (1175 lines) still sits in the generic crate.",
            "changedSinceV012": "The four claimed fields still exist (anchor drift only); the surface is wider than recorded; `System::policy_graph` is item 3's residue, not this item's.",
            "fixShapePhase": "1",
            "fixShape": "Phase 1 purification: the eight stochastic-input fields and the scenario pipeline leave `System` for the `cobre-stochastic` uncertainty store (`Switchable<T>`, by-reference); prose only, no fix executed here.",
            "alignment": (
                "advances-1",
                "IV.5 (`InflowModel`, `LoadModel`, `NcsModel`, `CorrelationModel` and the external-scenario tables leave `System`)",
                "`InflowModel`,",
            ),
            "severity": "B",
            "effort": "L",
            "measurements": [],
        },
        {
            "item": "2",
            "partIRef": "I.3-2",
            "title": "Stage bakes in stochastic and risk configuration",
            "v012Anchor": {
                "path": "crates/cobre-core/src/model/temporal.rs",
                "line": 281,
            },
            "anchors": [
                ("crates/cobre-core/src/model/temporal.rs", "Stage", None, "container"),
                (
                    "crates/cobre-core/src/model/temporal.rs",
                    None,
                    r"^\s+pub state_config: StageStateConfig,",
                    "claimed field state_config",
                ),
                (
                    "crates/cobre-core/src/model/temporal.rs",
                    None,
                    r"^\s+pub risk_config: StageRiskConfig,",
                    "claimed field risk_config",
                ),
                (
                    "crates/cobre-core/src/model/temporal.rs",
                    None,
                    r"^\s+pub scenario_config: ScenarioSourceConfig,",
                    "claimed field scenario_config",
                ),
            ],
            "sourceStations": ["core-io"],
            "relatedRegisterIds": ["CD-044", "CD-050"],
            "stationDispositions": {"core-io": "keep"},
            "disposition": "keep",
            "sharpenedClaim": None,
            "changedSinceV012": "Anchor drift only: `Stage` moved from :281 (v0.12) to the pin's line; `state_config`, `risk_config` and `scenario_config` are all still declared on it.",
            "fixShapePhase": "1",
            "fixShape": "Phase 1 purification: `StageRiskConfig` (CVaR), `ScenarioSourceConfig` and `inflow_lags` leave `Stage` for an SDDP per-node config; prose only.",
            "alignment": (
                "advances-1",
                "IV.5 (`StageRiskConfig`, `ScenarioSourceConfig` and `inflow_lags` leave `Stage`)",
                "`ScenarioSourceConfig`, and `inflow_lags` leave `Stage`",
            ),
            "severity": "B",
            "effort": "M",
            "measurements": [],
        },
        {
            "item": "3",
            "partIRef": "I.3-3",
            "title": "PolicyGraph is framed around forward/backward traversal, discount rates and cyclic convergence",
            "v012Anchor": {
                "path": "crates/cobre-core/src/model/horizon.rs",
                "line": None,
                "symbol": "PolicyGraph",
            },
            "anchors": [
                (
                    "crates/cobre-core/src/model/horizon.rs",
                    "HorizonGraph",
                    None,
                    "the surviving type (renamed from PolicyGraph in bd2cd4c0)",
                ),
                (
                    "crates/cobre-core/src/model/horizon.rs",
                    None,
                    r"^\s+pub graph_type: PolicyGraphType,",
                    "graph_type field",
                ),
                (
                    "crates/cobre-core/src/model/horizon.rs",
                    None,
                    r"^\s+pub annual_discount_rate: f64,",
                    "discount-rate framing",
                ),
                (
                    "crates/cobre-core/src/model/temporal.rs",
                    "PolicyGraphType",
                    None,
                    "residual policy vocabulary (enum)",
                ),
                (
                    "crates/cobre-core/src/model/temporal.rs",
                    "Node",
                    None,
                    "the node type",
                ),
                (
                    "crates/cobre-core/src/system/mod.rs",
                    None,
                    r"^\s+policy_graph: HorizonGraph,",
                    "residual System::policy_graph field name",
                ),
            ],
            "renamedFrom": {
                "symbol": "PolicyGraph",
                "to": "HorizonGraph",
                "commit": "bd2cd4c0",
            },
            "sourceStations": ["core-io"],
            "relatedRegisterIds": [],
            "stationDispositions": {"core-io": "sharpen"},
            "disposition": "sharpen",
            "sharpenedClaim": "`HorizonGraph` (the type `PolicyGraph` became in bd2cd4c0) still frames the horizon as discount-rate plus cyclic convergence (`graph_type: PolicyGraphType`, `annual_discount_rate`), and the residual `System::policy_graph` field name and the `PolicyGraphType` enum keep the policy vocabulary; the forward/backward-traversal framing is gone (zero forward|backward hits in horizon.rs).",
            "changedSinceV012": "The v0.12 type name resolves nowhere at the pin (only cobre-io's `RawPolicyGraph` loader shape carries the word); dispositioned on the rename evidence, never as anchor-missing.",
            "fixShapePhase": "1",
            "fixShape": "Phase 1 purification settles the residual naming (`PolicyGraphType`, `System::policy_graph`) with the horizon model; `HorizonGraph` itself stays in the purified core; prose only.",
            "alignment": (
                "neutral",
                "IV.1 keeps `HorizonGraph` in the purified `cobre-core`; no roadmap row names the residual-vocabulary rename, so the row neither advances nor conflicts",
                "| `cobre-core`                          | **purified**",
            ),
            "severity": "C",
            "effort": "S",
            "measurements": [],
        },
        {
            "item": "4",
            "partIRef": "I.3-4",
            "title": "InitialConditions is SDDP-warm-start-shaped",
            "v012Anchor": {
                "path": "crates/cobre-core/src/constraints/initial_conditions.rs",
                "line": 183,
            },
            "anchors": [
                (
                    "crates/cobre-core/src/constraints/initial_conditions.rs",
                    "InitialConditions",
                    None,
                    "container",
                ),
                (
                    "crates/cobre-core/src/constraints/initial_conditions.rs",
                    None,
                    r"^\s+pub past_anticipated_commitments: Vec<AnticipatedCommitmentHistory>,",
                    "surviving field past_anticipated_commitments",
                ),
                (
                    "crates/cobre-core/src/constraints/initial_conditions.rs",
                    None,
                    r"^\s+pub past_defluences: Vec<HydroPastDefluence>,",
                    "surviving field past_defluences",
                ),
                (
                    "crates/cobre-core/src/constraints/initial_conditions.rs",
                    None,
                    r"^\s+pub recent_observations:",
                    "field added since v0.12",
                ),
            ],
            "sourceStations": ["core-io"],
            "relatedRegisterIds": [],
            "stationDispositions": {"core-io": "sharpen"},
            "disposition": "sharpen",
            "sharpenedClaim": "Two of the three v0.12 warm-start fields remain on `InitialConditions` — `past_anticipated_commitments` and `past_defluences`; `past_inflows` is gone (closed by the windowed-inflow work, c1a360ad, released in v0.13.0), and `recent_observations` was added as paradigm-neutral realized data.",
            "changedSinceV012": "`past_inflows` has zero production references at the pin (it survives only as the rejection test `test_legacy_past_inflows_field_is_rejected` in cobre-io); the container moved from :183 to the pin's line.",
            "fieldDispositions": [
                {
                    "field": "past_inflows",
                    "disposition": "retire",
                    "closedBy": "c1a360ad",
                    "closedIn": "v0.13.0 (CHANGELOG.md: BREAKING — `past_inflows` is gone from `initial_conditions.json`)",
                },
                {
                    "field": "past_anticipated_commitments",
                    "disposition": "keep",
                    "closedBy": None,
                },
                {"field": "past_defluences", "disposition": "keep", "closedBy": None},
            ],
            "fixShapePhase": "1",
            "fixShape": "Phase 1 purification: the anticipated-commitment and defluence seeds follow the commitment and routing state respectively; prose only.",
            "alignment": (
                "advances-1",
                "IV.5 (`InitialConditions`' PAR-lag/defluence seeds follow the uncertainty and routing state)",
                "`InitialConditions`' PAR-lag/defluence seeds",
            ),
            "severity": "C",
            "effort": "M",
            "measurements": [],
        },
        {
            "item": "5",
            "partIRef": "I.3-5",
            "title": "training_event.rs lives in cobre-core",
            "v012Anchor": {
                "path": "crates/cobre-core/src/constraints/training_event.rs",
                "line": 125,
            },
            "anchors": [
                (
                    "crates/cobre-core/src/constraints/training_event.rs",
                    "TrainingEvent",
                    None,
                    "the event enum",
                ),
                (
                    "crates/cobre-core/src/constraints/training_event.rs",
                    "StageRowSelectionRecord",
                    None,
                    "the struct carrying the near-miss field",
                ),
                (
                    "crates/cobre-core/src/constraints/training_event.rs",
                    None,
                    r"Active cuts after budget enforcement",
                    "the lone lexical near-miss doc comment",
                ),
                (
                    "crates/cobre-core/src/constraints/training_event.rs",
                    None,
                    r"^\s+pub active_after_budget:",
                    "field active_after_budget",
                ),
                (
                    "crates/cobre-core/src/constraints/training_event.rs",
                    None,
                    r"can consume events without depending on the algorithm crate",
                    "the placement rationale in the module doc",
                ),
                (
                    "crates/cobre-core/src/lib.rs",
                    None,
                    r"^pub use constraints::training_event::\{",
                    "the crate-root re-export",
                ),
                (
                    "crates/cobre-cli/Cargo.toml",
                    None,
                    r"^cobre-sddp = \{ version",
                    "cobre-cli depends on the algorithm crate",
                ),
                (
                    "crates/cobre-python/Cargo.toml",
                    None,
                    r"^cobre-sddp = \{ version",
                    "cobre-python depends on the algorithm crate",
                ),
                (
                    "crates/cobre-tui/src/lib.rs",
                    None,
                    r".",
                    "the named cobre-tui consumer (stub)",
                ),
                (
                    "crates/cobre-mcp/src/main.rs",
                    None,
                    r".",
                    "the named cobre-mcp consumer (stub)",
                ),
            ],
            "sourceStations": [],
            "directFromTree": True,
            "relatedRegisterIds": [],
            "stationDispositions": {},
            "disposition": "sharpen",
            "sharpenedClaim": "`training_event.rs` (938 lines) still lives in `cobre-core` with `TrainingEvent`'s `ForwardPassComplete` / `BackwardPassComplete` / `ConvergenceUpdate` variants and allreduce fields, and its placement rationale — interface crates consume events without depending on the algorithm crate — is defeated at the pin: the real consumers are `cobre-cli` and `cobre-python`, both of which already depend on `cobre-sddp`, while the named `cobre-tui` and `cobre-mcp` consumers are 1-line and 6-line stubs. The lone lexical near-miss ('Active cuts after budget enforcement' on `active_after_budget`) stands; the gate's word-boundary pattern cannot see the plural.",
            "changedSinceV012": "Line count and the :125 near-miss unchanged; the consumer set the module doc names never materialised beyond stubs.",
            "fixShapePhase": "1",
            "fixShape": "Phase 1: `training_event.rs` moves to a telemetry module the verticals emit into (not core); the :125 doc comment is a rename rider on that move; prose only.",
            "alignment": (
                "advances-1",
                "IV.5 (`training_event.rs` moves to a telemetry module the verticals emit into)",
                "`training_event.rs` moves to a telemetry module",
            ),
            "severity": "C",
            "effort": "M",
            "measurements": "item5",
        },
        {
            "item": "6",
            "partIRef": "I.3-6",
            "title": "cobre-io's policy checkpoint format is literally cut-records",
            "v012Anchor": {"path": "crates/cobre-io/src/output/policy", "line": None},
            "anchors": [
                (
                    "crates/cobre-io/src/output/policy/records.rs",
                    "PolicyCutRecord",
                    None,
                    "surviving coupling",
                ),
                (
                    "crates/cobre-io/src/output/policy/records.rs",
                    "StageCutsPayload",
                    None,
                    "surviving coupling",
                ),
                (
                    "crates/cobre-io/schemas/policy.fbs",
                    None,
                    r"^table StageCuts \{",
                    "the StageCuts root table the gate never reads",
                ),
                (
                    "crates/cobre-io/src/output/policy/checkpoint.rs",
                    None,
                    r'path\.join\("cuts"\)',
                    "the on-disk cuts/ directory",
                ),
                (
                    "scripts/ci/check-infra-genericity.sh",
                    None,
                    r"^EXCLUDED_FILES=\(\)",
                    "the withdrawn exemption",
                ),
            ],
            "sourceStations": ["core-io", "build-ci"],
            "relatedRegisterIds": ["CD-061", "PD-015", "OD-024", "TD-021", "CD-099"],
            "stationDispositions": {
                "core-io": "sharpen",
                "build-ci": "claim-stale (exemption parenthetical only; no disposition set)",
            },
            "disposition": "sharpen",
            "sharpenedClaim": "The gate exemption the v0.12 claim leaned on has been withdrawn (`EXCLUDED_FILES=()`; the four policy files are scanned like any other infra source and the gate exits 0), yet the structural coupling survives the lexical retirement: `PolicyCutRecord` and `StageCutsPayload` evade the gate's word-boundary pattern, the `StageCuts` FlatBuffers root table lives in a schema the gate never reads (it scans only `*.rs` under five src/ trees), and the on-disk artifact is still a `cuts/` directory. The persisted value-function format is still cut-shaped.",
            "changedSinceV012": "Exemption retired by the script itself (header and the comment block above the empty array); the FlatBuffers vocabulary moved to `AffinePiece` and the provenance carrier to a `CheckpointManifest` table — exit 0 is not proof the claim retired.",
            "reRaiseOf": "the 2026-07-23 sanctioned genericity-gate exemption for `output/policy/{mod,records,codec,checkpoint}.rs` (roadmap I.3 parenthetical; retired in the gate's own header, the emptied `EXCLUDED_FILES` sanctioned as the enforced state at the build-ci station 2026-09-19). Live again because the lexical sanction was retired while the structural coupling it covered was not.",
            "fixShapePhase": "1",
            "fixShape": "Generalize the persisted artifact behind an engine-neutral policy artifact (the checkpoint format row of the target crate table); prose only, no fix executed here.",
            "alignment": (
                "advances-1",
                "IV.1 `cobre-io` row (checkpoint format generalized from SDDP cut-records to an engine-neutral policy artifact)",
                "Checkpoint format generalized from SDDP cut-records",
            ),
            "severity": "B",
            "effort": "L",
            "measurements": "item6",
        },
        {
            "item": "7",
            "partIRef": "I.3-7",
            "title": "The config type is SDDP-shaped",
            "v012Anchor": {"path": "crates/cobre-io/src/config/mod.rs", "line": None},
            "anchors": [
                (
                    "crates/cobre-io/src/config/mod.rs",
                    None,
                    r"^\s+pub training: TrainingConfig,",
                    "Config.training (core-io fragment)",
                ),
                (
                    "crates/cobre-sddp/src/setup/params.rs",
                    None,
                    r"^\s+pub fn from_config\(config: &Config\)",
                    "StudyParams::from_config (sddp fragment)",
                ),
                (
                    "crates/cobre-cli/src/commands/broadcast.rs",
                    None,
                    r"^pub\(crate\) struct BroadcastConfig \{",
                    "BroadcastConfig, now pub(crate) in cobre-cli (cli-python fragment)",
                ),
                (
                    "crates/cobre-io/src/broadcast.rs",
                    "BroadcastScalarParameter",
                    None,
                    "the only broadcast type cobre-io exposes",
                ),
                (
                    "crates/cobre-sddp/src/setup/params.rs",
                    "StudyParams",
                    None,
                    "the conversion's owner",
                ),
                (
                    "crates/cobre-io/src/config/mod.rs",
                    "Config",
                    None,
                    "the config type",
                ),
            ],
            "sourceStations": ["core-io", "sddp", "cli-python"],
            "mergedFrom": [
                {
                    "station": "core-io",
                    "fragment": "Config.training",
                    "anchor": "crates/cobre-io/src/config/mod.rs::training",
                },
                {
                    "station": "sddp",
                    "fragment": "StudyParams::from_config",
                    "anchor": "crates/cobre-sddp/src/setup/params.rs::from_config",
                },
                {
                    "station": "cli-python",
                    "fragment": "BroadcastConfig",
                    "anchor": "crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig",
                },
            ],
            "relatedRegisterIds": ["CD-004", "CD-082", "CD-051", "OD-020", "OD-043"],
            "stationDispositions": {
                "core-io": "sharpen (Config.training half)",
                "sddp": "sharpen (params.rs half; CD-004 carrier)",
                "cli-python": "sharpen (front-end half)",
            },
            "disposition": "sharpen",
            "sharpenedClaim": "The config leak moved layers rather than closing: `cobre_io::Config.training` is still SDDP-shaped (tree_seed, stopping_rules, cut_selection, the forward-pass mode, per-phase backward/forward solver profiles, the backward scheduler selector), the real config-to-domain conversion is still `StudyParams::from_config` in `cobre-sddp`, and the MPI-broadcastable subset `BroadcastConfig` is now a `pub(crate)` type inside `cobre-cli` (grown by backward_scheduler, cost_scale_factor, three per-phase solver profiles and the boundary requirements) while `cobre-io` exposes only the generic `BroadcastScalarParameter`. The front ends reach the conversion through two paths (run via `BroadcastConfig::from_config`, validate directly and via `StudySetup::new_with_boundary_requirements`).",
            "supersededWording": "'the MPI-broadcastable BroadcastConfig is a hand-maintained SDDP-parameter subset' — true, but no longer a cobre-io type.",
            "changedSinceV012": "`ConstructionConfig` and `into_construction_config` were deleted (4075c4e8); `StudyParams::from_config` now also captures export_states; the `Phase` enum stayed in `cobre-sddp` and `cobre-solver` stayed phase-agnostic, so the containment the item notes still holds.",
            "fixShapePhase": "0a",
            "fixShape": "Phase 0a: one config projection behind the study admission gate; CLI and Python both wire through it instead of hand-mirroring a broadcast subset; prose only.",
            "alignment": (
                "advances-0a",
                "V.1 Phase 0 (the `study`/config schema, the typed admission gate and per-engine solver-profile scoping land first and alone)",
                "surfaces — the seam, the `study`/config schema, the admission gate",
            ),
            "severity": "B",
            "effort": "M",
            "measurements": [],
        },
        {
            "item": "8",
            "partIRef": "I.3-8",
            "title": "The leakage reaches one layer below cobre-core: StageTemplate",
            "v012Anchor": {"path": "crates/cobre-solver/src/types.rs", "line": None},
            "anchors": [
                (
                    "crates/cobre-solver/src/types.rs",
                    "StageTemplate",
                    None,
                    "the L0 container",
                ),
                (
                    "crates/cobre-solver/src/types.rs",
                    None,
                    r"^\s+pub n_state: usize,",
                    "field n_state",
                ),
                (
                    "crates/cobre-solver/src/types.rs",
                    None,
                    r"^\s+pub n_transfer: usize,",
                    "field n_transfer",
                ),
                (
                    "crates/cobre-solver/src/types.rs",
                    None,
                    r"^\s+pub n_dual_relevant: usize,",
                    "field n_dual_relevant",
                ),
                (
                    "crates/cobre-solver/src/types.rs",
                    None,
                    r"^\s+pub n_hydro: usize,",
                    "field n_hydro",
                ),
                (
                    "crates/cobre-solver/src/types.rs",
                    None,
                    r"^\s+pub max_par_order: usize,",
                    "field max_par_order",
                ),
                (
                    "crates/cobre-solver/src/freeze.rs",
                    None,
                    r"out\.n_state = base\.n_state;",
                    "the one production propagation site",
                ),
            ],
            "sourceStations": ["solver-comm"],
            "relatedRegisterIds": ["CD-079", "TD-045"],
            "stationDispositions": {
                "solver-comm": "sharpen (claim holds and is sharper: doc-prose vocabulary, one propagation site, all five fields write-only)"
            },
            "disposition": "keep",
            "sharpenedClaim": None,
            "mergeRule": "Item-level keep: all five fields are still declared on `StageTemplate`, so the v0.12 claim holds verbatim; the solver-comm station's sharper evidence (write-only fields, one production propagation site, multistage vocabulary in L0 doc prose) narrows nothing and is recorded as evidence, and its per-field `retire` verdicts describe what the Phase-1 fix-shape does to each field, not a closure at the pin — a field-level disposition inside a surviving claim is subordinate to the claim and is never promoted to an item-level retire.",
            "changedSinceV012": "Nothing structural: the five fields and their doc vocabulary are unchanged; `n_dual_relevant`'s doc comment still claims it equals n_state while its only writer hard-codes 0 (documented drift rider).",
            "fieldDispositions": "item8",
            "fixShapePhase": "1",
            "fixShape": "Phase 1: `StageTemplate` sheds the five fields to the layer that owns the multistage layout (`StateSpace`, `StageRowLayout` on the kernel/engine side), leaving the L0 container pure CSC; prose only.",
            "alignment": (
                "advances-1",
                "IV.5 (`StageTemplate` sheds n_state/n_transfer/n_dual_relevant/n_hydro/max_par_order) and the IV.1 `cobre-solver` purified-plus-capabilities row",
                "`StageTemplate` sheds `n_state`/`n_transfer`/`n_dual_relevant`/`n_hydro`/",
            ),
            "severity": "B",
            "effort": "M",
            "measurements": [],
        },
        {
            "item": "I.5",
            "partIRef": "I.5",
            "title": "The CLI and orchestration coupling",
            "v012Anchor": {"path": "crates/cobre-cli/src", "line": None},
            "anchors": [
                (
                    "crates/cobre-cli/src/commands/run/outputs.rs",
                    "write_training_outputs",
                    None,
                    "the CLI output-writer layer",
                ),
                (
                    "crates/cobre-cli/src/commands/run/outputs.rs",
                    "write_simulation_outputs",
                    None,
                    "the CLI output-writer layer",
                ),
                (
                    "crates/cobre-python/src/run.rs",
                    None,
                    r"^pub\(crate\) fn run_via_study\(",
                    "the Python output path",
                ),
                (
                    "crates/cobre-cli/src/commands/broadcast.rs",
                    None,
                    r"^\s+pub\(crate\) fn from_config\(config: &Config\)",
                    "the run path's conversion call",
                ),
                (
                    "crates/cobre-cli/src/commands/run/mod.rs",
                    None,
                    r"if training_enabled \{",
                    "the two-phase lifecycle gate",
                ),
                (
                    "crates/cobre-cli/src/main.rs",
                    "Command",
                    None,
                    "the command surface",
                ),
            ],
            "sourceStations": ["cli-python"],
            "relatedRegisterIds": [],
            "stationDispositions": {
                "cli-python": "8 sub-claims: I.5-1a retire, I.5-1b keep, I.5-2 keep, I.5-3..7 sharpen"
            },
            "disposition": "sharpen",
            "sharpenedClaim": "The run subcommand is still SDDP-hydrothermal with no algorithm-selection seam: the lifecycle is the train-then-simulate two-phase shape gated on training_enabled and n_scenarios, the run is typed on the concrete `cobre_sddp::StudySetup` with inherent train/simulate methods, and config conversion reaches `StudyParams::from_config` through `BroadcastConfig::from_config`. The import figure is measured, not estimated: 79 `use cobre_sddp` lines / 95 `cobre_sddp::` references / 46 distinct symbols / 11 files at the ticket's scaffold pin and 81 / 97 / 48 / 10 at the register pin (each figure with its command below; the cli-python station's 70 / 97 / 88 / 10 use different commands and are recorded, not discarded). The help text no longer names SDDP (retired against main.rs), `report` and `summary` no longer exist (797ba443), `validate` is SDDP-shaped end to end, and the output tree gained `generic_constraints`. The CLI and Python output paths invoke different writer sets (the writer-mirror measurement below).",
            "changedSinceV012": "One sub-claim retired (the 'train an SDDP policy' help text), two hold verbatim, five sharpened by the cli-python station; report/summary removed in 797ba443.",
            "fixShapePhase": "0a",
            "fixShape": "Phase 0a: the `Engine` enum and dispatch in `cobre-cli` / `cobre-python` (the seam covers the full command surface) plus the shared output-orchestration entry point in `cobre-io`; prose only.",
            "alignment": (
                "advances-0a",
                "V.1 Phase 0 (introduce the `Engine` enum + dispatch in `cobre-cli`/`cobre-python`; the shared output-orchestration entry point)",
                "Introduce the `Engine` enum + dispatch",
            ),
            "severity": "B",
            "effort": "L",
            "measurements": "i5",
        },
    ]


def item5_measurements(pin: str) -> list[dict[str, Any]]:
    out = []
    for crate in ("cobre-cli", "cobre-python", "cobre-tui", "cobre-mcp"):
        out.append(
            {
                "figure": grep_count(
                    pin, "TrainingEvent", f"crates/{crate}", only_matching=True
                ),
                "unit": f"TrainingEvent occurrences in crates/{crate}",
                "command": f"git grep -o TrainingEvent {pin[:8]} -- crates/{crate} | wc -l",
            }
        )
    for path in (
        "crates/cobre-tui/src/lib.rs",
        "crates/cobre-mcp/src/main.rs",
        "crates/cobre-core/src/constraints/training_event.rs",
    ):
        text = show(pin, path) or ""
        out.append(
            {
                "figure": text.count("\n"),
                "unit": f"lines in {path}",
                "command": f"git show {pin[:8]}:{path} | wc -l",
            }
        )
    return out


def item6_measurements(pin: str) -> list[dict[str, Any]]:
    gate = subprocess.run(
        ["bash", "scripts/ci/check-infra-genericity.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    fbs = grep_files(
        pin, ".", "crates/cobre-io/src/*.fbs", "crates/cobre-io/src/**/*.fbs"
    )
    return [
        {
            "figure": gate.returncode,
            "unit": "genericity gate exit code (worktree)",
            "command": "bash scripts/ci/check-infra-genericity.sh; echo $?",
        },
        {
            "figure": fbs,
            "unit": ".fbs files under the scanned crates/cobre-io/src tree",
            "command": f"git grep -l . {pin[:8]} -- 'crates/cobre-io/src/**/*.fbs' | wc -l",
        },
    ]


def i5_measurements(pin: str) -> list[dict[str, Any]]:
    rows = []
    for sha, label in ((TICKET_BASELINE, "ticket scaffold pin"), (pin, "register pin")):
        s = sha[:8]
        rows += [
            {
                "figure": grep_count(sha, "use cobre_sddp", "crates/cobre-cli/src"),
                "unit": f"`use cobre_sddp` lines ({label} {s})",
                "command": f"git grep -n 'use cobre_sddp' {s} -- crates/cobre-cli/src | wc -l",
            },
            {
                "figure": grep_count(
                    sha, "cobre_sddp::", "crates/cobre-cli/src", only_matching=True
                ),
                "unit": f"`cobre_sddp::` references ({label} {s})",
                "command": f"git grep -o 'cobre_sddp::' {s} -- crates/cobre-cli/src | wc -l",
            },
            {
                "figure": grep_distinct(
                    sha, "cobre_sddp::[A-Za-z_][A-Za-z0-9_]*", "crates/cobre-cli/src"
                ),
                "unit": f"distinct `cobre_sddp::<symbol>` ({label} {s})",
                "command": f"git grep -hoE 'cobre_sddp::[A-Za-z_][A-Za-z0-9_]*' {s} -- crates/cobre-cli/src | sort -u | wc -l",
            },
            {
                "figure": grep_files(sha, "cobre_sddp", "crates/cobre-cli/src"),
                "unit": f"files naming cobre_sddp ({label} {s})",
                "command": f"git grep -l cobre_sddp {s} -- crates/cobre-cli/src | wc -l",
            },
        ]
    rows += [
        {
            "figure": 70,
            "unit": "`^use cobre_sddp` lines (cli-python station, register pin)",
            "command": "cli-python partI-handoff.json I.5-3 survivingClaim (station command)",
        },
        {
            "figure": 97,
            "unit": "`cobre_sddp::` occurrences over 10 files (cli-python station, register pin)",
            "command": "cli-python partI-handoff.json I.5-3 survivingClaim (station command)",
        },
        {
            "figure": 88,
            "unit": "distinct imported leaf symbols, re-exports included (cli-python station, register pin)",
            "command": "cli-python partI-handoff.json I.5-3 survivingClaim (station command)",
        },
    ]
    cli = show(pin, "crates/cobre-cli/src/commands/run/outputs.rs") or ""
    py = show(pin, "crates/cobre-python/src/run.rs") or ""
    call = re.compile(r"\b([a-z_]*write[a-z_]*)\(")
    cli_names = sorted(set(call.findall(cli)))
    py_names = sorted(set(call.findall(py)))
    rows.append(
        {
            "figure": f"{len(cli_names)} vs {len(py_names)}",
            "unit": "writer names invoked in crates/cobre-cli/src/commands/run/outputs.rs vs crates/cobre-python/src/run.rs (register pin)",
            "command": f"git show {pin[:8]}:<file> | grep -oE '\\b[a-z_]*write[a-z_]*\\(' | sort -u",
            "cliOnly": sorted(set(cli_names) - set(py_names)),
            "pythonOnly": sorted(set(py_names) - set(cli_names)),
            "shared": sorted(set(cli_names) & set(py_names)),
        }
    )
    return rows


def item8_field_dispositions(handoff: dict) -> list[dict[str, Any]]:
    out = []
    for f in handoff.get("perFieldDisposition", []):
        out.append(
            {
                "field": f["field"],
                "disposition": f["disposition"],
                "semantics": "what the Phase-1 fix-shape does to the field (the template sheds it); not a closure at the pin",
                "closedBy": None,
                "anchor": f["anchor"],
                "ownerAfterShed": f.get("ownerAfterShed"),
                "reason": f.get("dispositionReason"),
            }
        )
    return out


def build(stations_dir: pathlib.Path) -> tuple[dict[str, Any], int]:
    register_lines = bp.read_register(REGISTER)
    pin = bp.parse_baseline(register_lines)
    handoffs, generated_from, rejects = load_handoffs(stations_dir)
    claims = roadmap_items()
    first_free = next_free_cd(register_lines)
    allocated: list[str] = []
    rows: list[dict[str, Any]] = []
    rejected_items = {r["item"] for r in rejects}
    for spec in specs():
        item = spec["item"]
        if any(
            item in r["item"].replace(" ", "").split(",")
            for r in rejects
            if r["kind"] == "handoff-missing"
        ):
            continue
        resolved = [
            dict(resolve(pin, p, s, n), role=role) for p, s, n, role in spec["anchors"]
        ]
        superseded = [
            dict(resolve(TICKET_BASELINE, p, s, n), role=role)
            for p, s, n, role in spec["anchors"]
        ]
        missing = [a for a in resolved if a["status"] != "ok"]
        if missing:
            rejects.append(
                {
                    "item": item,
                    "kind": "anchor-missing",
                    "detail": "; ".join(
                        f"{a['path']}::{a['symbol'] or a['role']} {a['status']}"
                        for a in missing
                    ),
                }
            )
            rejected_items.add(item)
            continue
        disposition = spec["disposition"]
        register_id = None
        if disposition in ("keep", "sharpen"):
            register_id = f"CD-{first_free + len(allocated):03d}"
            allocated.append(register_id)
        alignment_value, alignment_cite, cite_needle = spec["alignment"]
        cite_line = roadmap_line(cite_needle)
        fields = spec.get("fieldDispositions")
        if fields == "item8":
            fields = item8_field_dispositions(handoffs.get("solver-comm", {}))
        measurements = spec.get("measurements")
        if measurements == "item5":
            measurements = item5_measurements(pin)
        elif measurements == "item6":
            measurements = item6_measurements(pin)
        elif measurements == "i5":
            measurements = i5_measurements(pin)
        row = {
            "item": item,
            "partIRef": spec["partIRef"],
            "title": spec["title"],
            "claim": claims[item],
            "v012Anchor": spec["v012Anchor"],
            "baselineAnchors": [
                {k: a[k] for k in ("path", "symbol", "line", "role")} for a in resolved
            ],
            "ticketBaselineAnchors": [
                {k: a[k] for k in ("path", "symbol", "line", "role")}
                for a in superseded
            ],
            "disposition": disposition,
            "closedBy": None,
            "sharpenedClaim": spec.get("sharpenedClaim"),
            "supersededWording": spec.get("supersededWording"),
            "changedSinceV012": spec.get("changedSinceV012"),
            "renamedFrom": spec.get("renamedFrom"),
            "mergeRule": spec.get("mergeRule"),
            "fieldDispositions": fields or [],
            "fixShapePhase": spec["fixShapePhase"],
            "fixShape": spec["fixShape"],
            "alignment": alignment_value,
            "alignmentCites": f"{ROADMAP_REL}:{cite_line} — {alignment_cite}",
            "severity": spec["severity"],
            "reviewerRating": spec["severity"],
            "category": "paradigm-leakage",
            "effort": spec["effort"],
            "confidence": "high",
            "registerId": register_id,
            "clearedLine": None,
            "reRaiseOf": spec.get("reRaiseOf"),
            "sourceStations": spec["sourceStations"],
            "stationDispositions": spec.get("stationDispositions", {}),
            "mergedFrom": spec.get("mergedFrom", []),
            "directFromTree": bool(spec.get("directFromTree")),
            "relatedRegisterIds": spec.get("relatedRegisterIds", []),
            "measurements": measurements or [],
        }
        rows.append(row)
    envelope = {
        "baseline": pin,
        "baselineSource": "plans/architecture-debt-audit/BACKLOG.md header (lib.backlog_parse.parse_baseline)",
        "ticketBaseline": TICKET_BASELINE,
        "generatedFrom": generated_from,
        "idBlock": {"firstFree": f"CD-{first_free:03d}", "allocated": allocated},
        "dispositions": rows,
        "ingestRejects": rejects,
        "_needsHuman": [],
        "mergeRules": {
            "baseline": "anchors and figures are resolved at the register pin; the ticket text's a136840d lines and figures are recorded per row (ticketBaselineAnchors, measurements) as superseded values, never used as the pin",
            "rename": "a v0.12 type name that resolves nowhere is dispositioned sharpen or retire with the rename as evidence (item 3: PolicyGraph → HorizonGraph, bd2cd4c0), never rejected as anchor-missing",
            "fragments": "fragments of one item from several stations land as ONE row with mergedFrom (item 7: core-io / sddp / cli-python)",
            "subordination": "a field-level disposition inside a surviving claim narrows the claim, it does not close it; it is recorded in fieldDispositions and never promoted to an item-level retire (item 8)",
            "directFromTree": "an item no station owns is dispositioned from the baseline tree (item 5), never dropped for want of a handoff",
            "figures": "a figure two stations measured differently is recorded once per command, no winner picked (I.5)",
            "ids": "register ids are allocated from max(existing CD id)+1 over the whole register at ingest, in item order, to keep and sharpen rows only",
            "rejects": "handoff-missing / anchor-missing / dup-of / needs-human land in ingestRejects, write no row for the item, and make the run exit 1",
        },
        "deviations": [
            "The ticket pins a136840d; the register header pins the baseline read here. Every a136840d line the ticket quotes that drifted is recorded next to the pin's line (records.rs :98/:176 → :195/:306, policy.fbs :140 → :163, checkpoint.rs :212 → :323, broadcast.rs BroadcastScalarParameter :48 → :24, temporal.rs Node :559 → :564, system/mod.rs policy_graph :92 → :96) and the I.5 figures 79/95/46/11 are the a136840d values (81/97/48/10 at the pin).",
            "plans/ is tracked (only plans/* outside architecture-debt-audit is gitignored), so the envelope and the table fragment are committed artifacts, not gitignored files.",
            "Item 8: the solver-comm station dispositioned the claim `sharpen`; consolidated as `keep` under the subordination rule (the station's sharper evidence narrows nothing; its per-field `retire` means the Phase-1 shed, not a closure).",
            "cobre-mcp's stub is crates/cobre-mcp/src/main.rs (6 lines); the ticket names a lib.rs that does not exist at either pin.",
        ],
    }
    return envelope, (EXIT_REJECT if rejects else EXIT_OK)


def render(envelope: dict[str, Any]) -> str:
    pin = envelope["baseline"]
    short = pin[:8]
    out: list[str] = []
    out.append(f"## {SECTION_TITLE}")
    out.append("")
    out.append(
        f"Part-I re-verification — the nine roadmap claims (I.3 items 1-8 and I.5) consolidated from the "
        f"five owning stations' handoffs and dispositioned at the register pin `{short}` "
        f"(envelope: alignment/part-i-dispositions.json beside this fragment). "
        f"Fresh ids start at {envelope['idBlock']['firstFree']}; no fix is executed and no tracked source is written."
    )
    out.append("")
    out.append(
        "| # | Claim (v0.12) | v0.12 anchor | Baseline anchor | Disposition | Phase | Register |"
    )
    out.append(
        "|---|---------------|--------------|-----------------|-------------|-------|----------|"
    )
    for r in envelope["dispositions"]:
        v = r["v012Anchor"]
        v_txt = (
            v["path"]
            + (f":{v['line']}" if v.get("line") else "")
            + (f" `{v['symbol']}`" if v.get("symbol") else "")
        )
        primary = r["baselineAnchors"][0]
        reg = r["registerId"] or "Cleared"
        out.append(
            f"| {r['item']} | {r['title']} | {v_txt} | {anchor_text(primary)} | {r['disposition']} | {r['fixShapePhase']} | {reg} |"
        )
    out.append("")
    out.append("### Entries (keep and sharpen rows)")
    out.append("")
    for r in envelope["dispositions"]:
        if r["disposition"] == "retire":
            out.append(
                f"**↩︎ CLEARED — Part-I item {r['item']}:** retired at {r['closedBy']}; {r['clearedLine']}"
            )
            out.append("")
            continue
        out.append(
            f"**{r['registerId']} · Sev {r['severity']} · {r['category']} · effort {r['effort']} · confidence {r['confidence']}**"
        )
        out.append(
            r["sharpenedClaim"]
            or f"{r['title']} — the v0.12 claim holds verbatim at the pin. {r['changedSinceV012']}"
        )
        out.append("")
        stations = (
            f"handoffs from {', '.join(r['sourceStations'])}"
            if r["sourceStations"]
            else "no owning station — dispositioned directly from the baseline tree"
        )
        out.append(f"- **Station:** alignment (Part-I re-verification; {stations})")
        out.append(f"- **Baseline:** `{pin}`")
        out.append(
            "- **Anchors:** " + ", ".join(anchor_text(a) for a in r["baselineAnchors"])
        )
        evidence = []
        for a, t in zip(r["baselineAnchors"], r["ticketBaselineAnchors"]):
            drift = (
                f" (:{t['line']} at the ticket's {TICKET_BASELINE[:8]})"
                if t["line"] and t["line"] != a["line"]
                else ""
            )
            evidence.append(f"{anchor_text(a)} — {a['role']}{drift}")
        if r.get("renamedFrom"):
            rf = r["renamedFrom"]
            evidence.append(
                f"`{rf['symbol']}` resolves nowhere at the pin; renamed to `{rf['to']}` in {rf['commit']} (rename evidence, not anchor-missing)"
            )
        if r.get("changedSinceV012"):
            evidence.append(f"Changed since v0.12: {r['changedSinceV012']}")
        if r.get("mergeRule"):
            evidence.append(f"Merge rule: {r['mergeRule']}")
        out.append(
            "- **Evidence:** "
            + " ".join(f"({i}) {e}" for i, e in enumerate(evidence, 1))
        )
        out.append(f"- **Fix-shape:** {r['fixShape']}")
        out.append(
            f"- **Alignment:** {r['alignment']} ({r['alignmentCites']}; provisional until the alignment adjudication)"
        )
        if r.get("reRaiseOf"):
            out.append(f"- **Re-raise-of:** {r['reRaiseOf']}")
        out.append(
            f"- **Part-I:** {r['partIRef']} — disposition {r['disposition']}, phase {r['fixShapePhase']}; station verdicts: "
            + (
                "; ".join(f"{k}: {v}" for k, v in r["stationDispositions"].items())
                or "none (no owning station)"
            )
            + "."
        )
        if r["mergedFrom"]:
            out.append(
                "- **Merged from:** "
                + "; ".join(
                    f"{m['station']} — {m['fragment']} (`{m['anchor']}`)"
                    for m in r["mergedFrom"]
                )
            )
        if r["fieldDispositions"]:
            semantics = next(
                (f["semantics"] for f in r["fieldDispositions"] if f.get("semantics")),
                None,
            )
            label = (
                f"- **Field dispositions ({semantics}):** "
                if semantics
                else "- **Field dispositions:** "
            )
            out.append(
                label
                + "; ".join(
                    f"`{f['field']}` {f['disposition']}"
                    + (
                        f" (closed by {f['closedBy']}, {f['closedIn']})"
                        if f.get("closedBy")
                        else ""
                    )
                    for f in r["fieldDispositions"]
                )
            )
        if r["measurements"]:
            out.append(
                "- **Measurements:** "
                + "; ".join(
                    f"{m['figure']} = {m['unit']} (`{m['command']}`)"
                    for m in r["measurements"]
                )
            )
            mirror = [m for m in r["measurements"] if "cliOnly" in m]
            if mirror:
                m = mirror[0]
                out.append(
                    f"- **Writer mirror:** CLI-only {', '.join('`' + n + '`' for n in m['cliOnly']) or 'none'}; Python-only {', '.join('`' + n + '`' for n in m['pythonOnly']) or 'none'}; shared {len(m['shared'])}."
                )
        if r["relatedRegisterIds"]:
            out.append(
                "- **Related register ids:** "
                + ", ".join(r["relatedRegisterIds"])
                + " (station findings that cross-reference this item; none is this claim, so none is a dup-of)"
            )
        out.append("")
    if envelope["ingestRejects"]:
        out.append("### Ingest rejects")
        out.append("")
        for rej in envelope["ingestRejects"]:
            out.append(f"- item {rej['item']} — {rej['kind']}: {rej['detail']}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def self_check(table: pathlib.Path, envelope: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    checks = (
        ("check-anchors.py", ["--register", str(table), SECTION_TITLE]),
        ("check-reraise.py", ["--register", str(table), SECTION_TITLE]),
        (
            "fields-check.py",
            ["--register", str(table), "--require", "Alignment", SECTION_TITLE],
        ),
    )
    for tool, args in checks:
        proc = subprocess.run(
            [sys.executable, str(TOOLS / tool), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            problems.append(
                f"{tool} exit {proc.returncode}: {(proc.stdout + proc.stderr).strip()[:600]}"
            )
    headers = len(
        re.findall(r"^\*\*CD-\d{3} ", table.read_text(encoding="utf-8"), re.M)
    )
    non_retire = sum(
        1 for r in envelope["dispositions"] if r["disposition"] != "retire"
    )
    if headers != non_retire:
        problems.append(f"{headers} entry headers vs {non_retire} keep/sharpen rows")
    status = git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "crates",
        "docs",
        "scripts",
        ".github",
        "schemas",
        "Cargo.toml",
    ).stdout.strip()
    if status:
        problems.append(f"tracked evaluated surface modified: {status}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--stations-dir", type=pathlib.Path, default=AUDIT / "stations")
    ap.add_argument("--out-dir", type=pathlib.Path, default=AUDIT / "alignment")
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args(argv)
    try:
        envelope, code = build(args.stations_dir)
    except (LookupError, OSError) as exc:
        print(f"register unreadable: {exc}", file=sys.stderr)
        return EXIT_REGISTER
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env_path = args.out_dir / "part-i-dispositions.json"
    table_path = args.out_dir / "part-i-disposition-table.md"
    env_path.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table_path.write_text(render(envelope), encoding="utf-8")
    for rej in envelope["ingestRejects"]:
        print(
            f"ingest-reject[{rej['kind']}] item {rej['item']}: {rej['detail']}",
            file=sys.stderr,
        )
    print(
        f"baseline {envelope['baseline'][:8]} / first free id {envelope['idBlock']['firstFree']} / "
        f"{len(envelope['dispositions'])} rows / {len(envelope['ingestRejects'])} rejects / allocated {', '.join(envelope['idBlock']['allocated']) or 'none'}"
    )
    if code != EXIT_OK:
        return code
    if not args.no_self_check:
        problems = self_check(table_path, envelope)
        for p in problems:
            print(f"self-check FAIL: {p}", file=sys.stderr)
        if problems:
            return EXIT_REJECT
        print(
            "self-check: anchors, re-raise, fields (Alignment), header count, scoped git status — all clean"
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
