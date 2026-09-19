#!/usr/bin/env python3
"""Station-local validator for the test-corpus attacker envelopes (the test-bloat profile).

Layers the station's profile on top of the shared tools/validate-envelope.py, which is
imported as a library and never edited (the core-io ingest ticket couples to its contract).
The profile adds the four mandatory per-candidate fields — `yardstickRef`, `measuredValue`,
`measurementDefinition`, `claimKind` — plus the optional `seedRef` / `dupOf` links, the
per-lens extras (reservedSeamsCheck for over-engineering; informational / UNMEASURED /
costMechanism for performance) and the worker's seed-disposition ledger.

Validation is SHAPE-ONLY and INVENTORY-KEY-ONLY. Anchor RESOLUTION at the baseline is
deliberately not checked here: tools/check-anchors.py owns that at ingest, so a stale
anchor is rejected there once with the anchor-missing code instead of being silently
dropped by this gate.

Usage:
  validate-tb.py envelope <file.json> [--worker <name>]
      shared attacker shape (station test-corpus) + the test-bloat profile; with --worker
      the anchors must sit inside that worker's scope and its seed ledger must be complete.
      Exit 0 = valid (no output); exit 1 = one diagnostic per line naming the JSON path.
  validate-tb.py partition
      the nine-row scope table in attacker-prompt.md partitions the corpus at the pin;
      prints PARTITION OK or the two-way diff and exits 1.
  validate-tb.py keys
      the inventory.json keys a measuredValue may cite (scalars only).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from typing import Any

STATION_DIR = pathlib.Path(__file__).resolve().parent
AUDIT = STATION_DIR.parents[1]
TOOLS = AUDIT / "tools"
sys.path.insert(0, str(TOOLS))
from lib import station_checks as sc  # noqa: E402

STATION = "test-corpus"
LENSES = ("test-bloat", "architecture", "over-engineering", "performance")
PRIMARY_LENS = "test-bloat"
CLAIM_KINDS = {"tree-fact", "target-gap", "prose-drift"}
DEFINITIONS = {"binary", "file", "n/a"}
DISPOSITIONS = {"confirmed", "sharpened", "dup-of", "dropped"}
MIRROR_ITEMS = (
    "Oracle test-harness duplication",
    "Python-binding Rust tests invisible to CI",
    "Mega-file / inline-test-giant asymmetry",
)
YARDSTICK_RE = re.compile(r"^ta-(\d(?:\.\d+)?)$")
SEED_REF_RE = re.compile(r"^stations/[a-z-]+/td-queue\.json#queue\[\d+\]$")
# A bare count in a title: an integer that is not a line reference (`:77`, `L77`),
# a section (`§5.2`, `ta-2.3`, `5.1`), a register id (`TD-047`) or a version.
BARE_COUNT_RE = re.compile(r"(?<![:\.\-\w§])\b\d{1,6}\b(?![\.\-]\d)")
TIMING_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|µs|us|ns|s|sec|secs|seconds|minutes|min|hours)\b"
    r"|\b\d+(?:\.\d+)?\s*x\s*faster|\b\d+(?:\.\d+)?x\b|speed-?up",
    re.I,
)
SHARED_DOC_ANCHORS = (
    "docs/design/testing-architecture.md",
    "docs/design/reserved-seams-and-deferred-debt.md",
    ".claude/rules/testing.md",
    "Cargo.toml",
    "ARCHITECTURE.md",
    "CLAUDE.md",
)
WORKFLOWS = (
    ".github/workflows/ci.yml",
    ".github/workflows/invariance-shuffle.yml",
    ".github/workflows/mpi-slurm.yml",
)
ZERO_SURFACE_CRATES = (
    "crates/cobre",
    "crates/cobre-mcp",
    "crates/cobre-tui",
    "crates/cobre-flow",
    "crates/cobre-uc",
    "crates/cobre-emt",
)
SCOPE_ROW_RE = re.compile(
    r"^\| (?P<worker>tb-[a-z0-9-]+) \| (?P<lens>[a-z-]+) \|(?P<paths>[^|]*)\|(?P<extra>[^|]*)\|",
    re.M,
)


def load_shared():
    spec = importlib.util.spec_from_file_location(
        "validate_envelope", TOOLS / "validate-envelope.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("tools/validate-envelope.py not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pin() -> str:
    for ln in (AUDIT / "BACKLOG.md").read_text(encoding="utf-8").splitlines()[:40]:
        m = re.match(r"^Baseline:\s+`?([0-9a-f]{40})`?", ln)
        if m:
            return m.group(1)
    raise RuntimeError("no Baseline line in BACKLOG.md")


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted keys; a list of records keyed by `id` or `crate` uses that value as the segment."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if obj and all(
            isinstance(x, dict) and ("id" in x or "crate" in x) for x in obj
        ):
            for x in obj:
                out.update(flatten(x, f"{prefix}.{x.get('id') or x.get('crate')}"))
        else:
            out[prefix] = obj
    else:
        out[prefix] = obj
    return out


def inventory_keys() -> dict[str, Any]:
    return flatten(sc.load_json(STATION_DIR / "inventory.json"))


def scope_rows(prompt_text: str) -> list[dict[str, Any]]:
    rows = []
    for m in SCOPE_ROW_RE.finditer(prompt_text):
        rows.append(
            {
                "worker": m.group("worker"),
                "lens": m.group("lens").strip(),
                "prefixes": [p for p in m.group("paths").replace("`", "").split() if p],
                "extra": [p for p in m.group("extra").replace("`", "").split() if p],
            }
        )
    return rows


def corpus_paths(tree: sc.Tree) -> list[str]:
    """The test corpus at the pin: crates/*/tests/**/*.{rs,py}, src files carrying
    `#[cfg(test)]`, sibling src/**/tests.rs, cobre-sddp benches, the three test-facing workflows."""
    files = tree.files()
    corpus: set[str] = set()
    for f in files:
        parts = f.split("/")
        if (
            len(parts) >= 4
            and parts[0] == "crates"
            and parts[2] == "tests"
            and f.endswith((".rs", ".py"))
        ):
            corpus.add(f)
        elif (
            len(parts) >= 4
            and parts[0] == "crates"
            and parts[2] == "src"
            and f.endswith("/tests.rs")
        ):
            corpus.add(f)
        elif (
            len(parts) >= 4
            and parts[0] == "crates"
            and parts[2] == "benches"
            and f.endswith(".rs")
        ):
            corpus.add(f)
    for f in tree.ls_files("crates/*/src/**/*.rs", "crates/*/src/*.rs"):
        if "#[cfg(test)]" in tree.read_text(f):
            corpus.add(f)
    corpus.update(w for w in WORKFLOWS if w in files)
    return sorted(corpus)


def check_partition(prompt_text: str, tree: sc.Tree) -> tuple[bool, list[str]]:
    rows = scope_rows(prompt_text)
    report: list[str] = []
    ok = True
    if len(rows) != 9:
        report.append(f"expected 9 test-bloat scope rows, found {len(rows)}")
        ok = False
    corpus = corpus_paths(tree)
    files = tree.files()
    prefixes: dict[str, str] = {}
    for r in rows:
        for p in r["prefixes"]:
            if p in prefixes:
                report.append(f"prefix {p} claimed by {prefixes[p]} and {r['worker']}")
                ok = False
            prefixes[p] = r["worker"]
            if not (tree.is_dir(p.rstrip("/")) or tree.is_file(p)):
                report.append(
                    f"phantom scope entry (absent at the pin): {p} ({r['worker']})"
                )
                ok = False
    matched: dict[str, list[str]] = {}
    hit_count = {p: 0 for p in prefixes}
    for path in corpus:
        owners = [w for p, w in prefixes.items() if path == p or path.startswith(p)]
        for p in prefixes:
            if path == p or path.startswith(p):
                hit_count[p] += 1
        matched[path] = sorted(set(owners))
    unswept = [p for p, o in matched.items() if not o]
    double = [p for p, o in matched.items() if len(o) > 1]
    empty_prefix = [p for p, n in hit_count.items() if n == 0]
    if unswept:
        ok = False
        report.append("left-only = unswept corpus paths:")
        report.extend(f"  {p}" for p in unswept)
    if double:
        ok = False
        report.append("double-swept corpus paths:")
        report.extend(f"  {p} <- {matched[p]}" for p in double)
    if empty_prefix:
        ok = False
        report.append("right-only = scope prefixes matching no corpus path (phantom):")
        report.extend(f"  {p}" for p in empty_prefix)
    topo = [r for r in rows if r["worker"] == "tb-workspace-topology"]
    if len(topo) != 1:
        ok = False
        report.append("exactly one tb-workspace-topology row required")
    else:
        row = topo[0]
        if sorted(row["prefixes"]) != sorted(WORKFLOWS):
            ok = False
            report.append(
                f"topology row must sweep exactly the three workflows, got {row['prefixes']}"
            )
        extra = set(row["extra"])
        if ".config/nextest.toml" not in extra:
            ok = False
            report.append("topology row must own the absent .config/nextest.toml")
        elif tree.exists(".config/nextest.toml"):
            ok = False
            report.append(
                ".config/nextest.toml exists at the pin; the absence premise fails"
            )
        for crate in ZERO_SURFACE_CRATES:
            if crate not in extra:
                ok = False
                report.append(f"topology row must own the zero-surface crate {crate}")
                continue
            if not tree.is_dir(crate):
                ok = False
                report.append(f"zero-surface crate absent at the pin: {crate}")
            elif tree.is_dir(f"{crate}/tests") or any(
                f.startswith(f"{crate}/src/") and "#[cfg(test)]" in tree.read_text(f)
                for f in files
                if f.endswith(".rs")
            ):
                ok = False
                report.append(f"{crate} carries a test surface; it is not zero-surface")
    report.insert(
        0,
        f"corpus paths {len(corpus)} across {len(prefixes)} scope prefixes / {len(rows)} workers",
    )
    return ok, report


def worker_scope(prompt_text: str, worker: str) -> list[str] | None:
    """Scope prefixes for a worker; None for a whole-corpus (secondary) worker."""
    for r in scope_rows(prompt_text):
        if r["worker"] == worker:
            return r["prefixes"]
    return None


def _has_timing(blob: str) -> bool:
    return bool(TIMING_RE.search(blob))


def validate_profile(
    env: dict[str, Any],
    err,
    *,
    keys: dict[str, Any],
    yardstick_sections: set[str],
    seeds: dict[str, Any] | None,
    worker: str | None,
    scope: list[str] | None,
) -> None:
    lens = env.get("lens")
    if lens not in LENSES:
        err(f"$.lens: {lens!r} not in {list(LENSES)}")
    if env.get("worker") != env.get("subStation"):
        err("$.worker: must equal $.subStation (the worker name)")
    if worker and env.get("subStation") != worker:
        err(f"$.subStation: {env.get('subStation')!r} != dispatched worker {worker!r}")
    titles_by_seed: dict[str, str] = {}
    for i, c in enumerate(env.get("candidates") or []):
        if not isinstance(c, dict):
            continue
        at = f"$.candidates[{i}]"
        title = str(c.get("title", ""))
        if c.get("claimKind") not in CLAIM_KINDS:
            err(f"{at}.claimKind: {c.get('claimKind')!r} not in {sorted(CLAIM_KINDS)}")
        d = c.get("measurementDefinition")
        if d not in DEFINITIONS:
            err(f"{at}.measurementDefinition: {d!r} not in {sorted(DEFINITIONS)}")
        mv = c.get("measuredValue")
        if mv is None and d in DEFINITIONS and d != "n/a":
            err(f"{at}.measuredValue: required unless measurementDefinition is n/a")
        if mv is not None:
            if d == "n/a":
                err(f"{at}.measurementDefinition: n/a but the claim states a count")
            if not isinstance(mv, dict) or "key" not in mv or "value" not in mv:
                err(f"{at}.measuredValue: needs key and value")
            else:
                key = mv.get("key")
                if key not in keys:
                    err(f"{at}.measuredValue.key: {key!r} not in inventory.json")
                else:
                    have = keys[key]
                    if isinstance(have, (int, float, str, bool)) and have != mv.get(
                        "value"
                    ):
                        err(
                            f"{at}.measuredValue.value: {mv.get('value')!r} != inventory {have!r} "
                            f"for {key}"
                        )
        if d == "n/a" and BARE_COUNT_RE.search(title):
            err(f"{at}.measurementDefinition: n/a but the title states a count")
        yr = str(c.get("yardstickRef", ""))
        m = YARDSTICK_RE.match(yr)
        if not m:
            err(f"{at}.yardstickRef: expected ta-<section>, got {yr!r}")
        elif m.group(1) not in yardstick_sections:
            err(f"{at}.yardstickRef: section {m.group(1)} is not a yardstick section")
        if "targetNotDefect" in c:
            err(f"{at}.targetNotDefect: attackers classify, defenders adjudicate")
        if c.get("dupOf") is not None and c["dupOf"] not in MIRROR_ITEMS:
            err(f"{at}.dupOf: {c['dupOf']!r} is not one of the three mirror items")
        sr = c.get("seedRef")
        if sr is not None:
            if not SEED_REF_RE.match(str(sr)):
                err(
                    f"{at}.seedRef: {sr!r} is not stations/<slug>/td-queue.json#queue[<i>]"
                )
            elif seeds is not None and sr not in seeds:
                err(f"{at}.seedRef: {sr!r} not in seeds.json")
            titles_by_seed.setdefault(str(sr), title)
        for j, a in enumerate(c.get("anchors") or []):
            if not isinstance(a, dict):
                continue
            path = str(a.get("path", ""))
            if scope is not None and not (
                any(path == p or path.startswith(p) for p in scope)
                or path in SHARED_DOC_ANCHORS
                or re.match(r"^crates/[a-z-]+/Cargo\.toml$", path)
                and any(
                    path.startswith(p.split("/tests/")[0].split("/src/")[0])
                    for p in scope
                )
            ):
                err(f"{at}.anchors[{j}].path: {path} is outside the worker's scope")
        if lens == "performance":
            if c.get("informational") is not True:
                err(
                    f"{at}.informational: performance candidates must set informational true"
                )
            if c.get("status") != "UNMEASURED":
                err(f"{at}.status: performance candidates stay UNMEASURED")
            if not str(c.get("statusReason", "")).strip():
                err(f"{at}.statusReason: UNMEASURED needs a reason")
            cm = str(c.get("costMechanism", ""))
            if not cm.strip():
                err(f"{at}.costMechanism: required")
            elif not re.search(r"binar|static solver link|cadence|wall", cm, re.I):
                err(
                    f"{at}.costMechanism: must name binaries x static solver link or job cadence"
                )
            if str(c.get("mechanism", "")).strip() != cm.strip():
                err(
                    f"{at}.mechanism: must equal costMechanism (the shared contract's field)"
                )
            if _has_timing(json.dumps(c)):
                err(f"{at}: performance candidates carry no timing literal")
        if lens == "over-engineering":
            rsc = c.get("reservedSeamsCheck")
            if (
                not isinstance(rsc, dict)
                or rsc.get("checked") is not True
                or rsc.get("result")
                not in {"sanctioned", "not-found", "not-applicable"}
            ):
                err(
                    f"{at}.reservedSeamsCheck: {{checked: true, result: sanctioned|not-found|not-applicable}} required"
                )
    for i, p in enumerate(env.get("positives") or []):
        if isinstance(p, dict) and not str(p.get("why", "")).strip():
            err(f"$.positives[{i}].why: required")
    if not (env.get("candidates") or []) and not isinstance(
        env.get("cleanVerdict"), dict
    ):
        err("$.cleanVerdict: an empty candidates list needs a cleanVerdict object")
    disp = env.get("seedDispositions")
    if disp is None:
        err("$.seedDispositions: required (empty list when no seed was routed)")
        disp = []
    if not isinstance(disp, list):
        err("$.seedDispositions: must be a list")
        disp = []
    seen: set[str] = set()
    for i, sd in enumerate(disp):
        at = f"$.seedDispositions[{i}]"
        if not isinstance(sd, dict):
            err(f"{at}: not an object")
            continue
        sr = str(sd.get("seedRef", ""))
        if not SEED_REF_RE.match(sr):
            err(f"{at}.seedRef: malformed")
        elif seeds is not None and sr not in seeds:
            err(f"{at}.seedRef: {sr!r} not in seeds.json")
        if sr in seen:
            err(f"{at}.seedRef: duplicated disposition for {sr}")
        seen.add(sr)
        dv = sd.get("disposition")
        if dv not in DISPOSITIONS:
            err(f"{at}.disposition: {dv!r} not in {sorted(DISPOSITIONS)}")
        if dv in {"dropped", "dup-of"} and not str(sd.get("reason", "")).strip():
            err(f"{at}.reason: required for {dv}")
        if dv in {"confirmed", "sharpened"}:
            ct = str(sd.get("candidateTitle", ""))
            if not ct.strip():
                err(f"{at}.candidateTitle: required for {dv}")
            elif titles_by_seed.get(sr) != ct:
                err(
                    f"{at}.candidateTitle: no candidate carries seedRef {sr} with that title"
                )
        if dv == "dup-of" and sd.get("dupOf") not in MIRROR_ITEMS:
            err(f"{at}.dupOf: dup-of needs one of the three mirror items")
    if worker and seeds is not None:
        routed = {ref for ref, s in seeds.items() if s.get("routedTo") == worker}
        missing = sorted(routed - seen)
        stray = sorted(seen - routed)
        if missing:
            err(
                f"$.seedDispositions: {len(missing)} routed seed(s) without a disposition: {missing[:3]}…"
            )
        if stray:
            err(f"$.seedDispositions: seed(s) not routed to this worker: {stray[:3]}…")


def validate_file(path: pathlib.Path | str, worker: str | None = None) -> list[str]:
    errs: list[str] = []
    try:
        env = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"$ not parseable JSON: {exc}"]
    if not isinstance(env, dict):
        return ["$ not an object"]
    shared = load_shared()
    shared.validate_attacker(env, errs, STATION)
    if env.get("baseline") != pin():
        errs.append(
            f"$.baseline: {env.get('baseline')!r} is not the register pin {pin()[:8]}"
        )
    claims = sc.load_json(STATION_DIR / "claim-classes.json")
    sections = {r["section"] for r in claims["rows"]} | {"3.1", "4.1", "4.2", "8"}
    seeds_path = STATION_DIR / "seeds.json"
    seeds = None
    if seeds_path.exists():
        seeds = {s["seedRef"]: s for s in sc.load_json(seeds_path)["seeds"]}
    prompt_path = STATION_DIR / "attacker-prompt.md"
    scope = None
    if worker and prompt_path.exists():
        scope = worker_scope(prompt_path.read_text(encoding="utf-8"), worker)
    validate_profile(
        env,
        errs.append,
        keys=inventory_keys(),
        yardstick_sections=sections,
        seeds=seeds,
        worker=worker,
        scope=scope,
    )
    return errs


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, *rest = argv
    if cmd == "keys":
        for k, v in sorted(inventory_keys().items()):
            if isinstance(v, (int, float, str, bool)):
                print(f"{k} = {v}")
        return 0
    if cmd == "partition":
        prompt = (STATION_DIR / "attacker-prompt.md").read_text(encoding="utf-8")
        ok, report = check_partition(prompt, sc.Tree(pin()))
        print("\n".join(report))
        print(
            "PARTITION OK"
            if ok
            else "PARTITION FAIL: left-only = unswept, right-only = phantom or double-swept"
        )
        return 0 if ok else 1
    if cmd == "envelope":
        worker = None
        if "--worker" in rest:
            i = rest.index("--worker")
            worker = rest[i + 1]
            del rest[i : i + 2]
        if len(rest) != 1:
            print(__doc__, file=sys.stderr)
            return 2
        errs = validate_file(rest[0], worker)
        for e in errs:
            print(f"{rest[0]}: {e}", file=sys.stderr)
        return 1 if errs else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
