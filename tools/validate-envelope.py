#!/usr/bin/env python3
"""Shape-only validator for station worker envelopes. No jsonschema dependency.

Anchor RESOLUTION is deliberately NOT checked here: check-anchors.py owns that
at ingest, so a shape-valid envelope with a stale anchor still reaches the
ingest step and is rejected there, once, with the anchor-missing taxonomy code
instead of being silently dropped by this gate.

Usage: validate-envelope.py [--role attacker|defender] [--station <slug>] <file.json>
Exit 0 = shape valid (no output); exit 1 = one diagnostic per line on stderr,
each naming the JSON path of the violation.
"""
from __future__ import annotations

import json
import re
import sys

LENSES = {"architecture", "perf", "over-engineering", "test-bloat"}
SUBS = {"A", "B", "C", "D"}
SEV = {"A", "B", "C"}
ALIGN = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
VERDICT = {"confirmed", "dismissed"}
DIFF_MARKERS = ("---", "+++", "@@", "diff ", "```")
TIMING_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|µs|us|ns|s|sec|secs|seconds|minutes|min)\b", re.I)


def validate_attacker(env: dict, errs: list[str], station: str | None) -> None:
    for key in ("station", "subStation", "baseline", "lens", "candidates", "positives", "_needsHuman"):
        if key not in env:
            errs.append(f"$: missing key {key}")
    if station and env.get("station") != station:
        errs.append(f"$.station: {env.get('station')!r} != {station!r}")
    if env.get("lens") not in LENSES:
        errs.append(f"$.lens: {env.get('lens')!r} not in {sorted(LENSES)}")
    if env.get("subStation") not in SUBS:
        errs.append(f"$.subStation: {env.get('subStation')!r} not in A-D")
    if not re.fullmatch(r"[0-9a-f]{7,40}", str(env.get("baseline", ""))):
        errs.append("$.baseline: not a git sha")
    for key in ("candidates", "positives", "_needsHuman"):
        if key in env and not isinstance(env[key], list):
            errs.append(f"$.{key}: must be a list")
    for i, cand in enumerate(env.get("candidates") or []):
        at = f"$.candidates[{i}]"
        if not isinstance(cand, dict):
            errs.append(f"{at}: not an object")
            continue
        if not str(cand.get("title", "")).strip():
            errs.append(f"{at}.title: empty")
        anchors = cand.get("anchors")
        if not anchors or not isinstance(anchors, list):
            errs.append(f"{at}.anchors: at least one anchor required")
        for j, anc in enumerate(anchors or []):
            if not isinstance(anc, dict):
                errs.append(f"{at}.anchors[{j}]: not an object")
                continue
            path = str(anc.get("path", ""))
            if not path or path.startswith("/") or ".." in path or not re.match(r"(crates|scripts|docs|schemas|examples|tests|\.github|\.claude|plans)/", path):
                errs.append(f"{at}.anchors[{j}].path: must be repo-relative")
            if not anc.get("symbol") and anc.get("line") is None:
                errs.append(f"{at}.anchors[{j}]: needs symbol or line")
            if anc.get("line") is not None and (not isinstance(anc["line"], int) or anc["line"] < 1):
                errs.append(f"{at}.anchors[{j}].line: must be a positive integer")
        if cand.get("proposedSeverity") not in SEV:
            errs.append(f"{at}.proposedSeverity: not in A/B/C")
        if cand.get("alignmentHint") not in ALIGN:
            errs.append(f"{at}.alignmentHint: not in {sorted(ALIGN)}")
        ev = cand.get("evidence")
        if not isinstance(ev, dict) or not str(ev.get("command", "")).strip():
            errs.append(f"{at}.evidence.command: required")
        fix = str(cand.get("fixShape", ""))
        if len(fix.strip()) < 20:
            errs.append(f"{at}.fixShape: too short to be a fix shape")
        if fix.lstrip().startswith(DIFF_MARKERS) or "\n+" in fix or "\n-" in fix or "```" in fix:
            errs.append(f"{at}.fixShape: looks like a diff or a code block; prose only")
        if cand.get("partIRef") is not None and not re.fullmatch(r"I\.3-[1-8]", str(cand["partIRef"])):
            errs.append(f"{at}.partIRef: not an I.3-n reference")
        if env.get("lens") == "perf":
            if not str(cand.get("mechanism", "")).strip():
                errs.append(f"{at}.mechanism: perf candidates must state the cost mechanism")
            blob = " ".join(str(cand.get(k, "")) for k in ("title", "mechanism", "fixShape")) + " " + str((ev or {}).get("reading", ""))
            if TIMING_RE.search(blob):
                errs.append(f"{at}: perf candidates stay UNMEASURED; a timing number was found")
    for i, pos in enumerate(env.get("positives") or []):
        if not isinstance(pos, dict) or not str(pos.get("subject", "")).strip():
            errs.append(f"$.positives[{i}].subject: required")


def validate_defender(env: dict, errs: list[str], station: str | None) -> None:
    for key in ("station", "subStation", "baseline", "verdicts"):
        if key not in env:
            errs.append(f"$: missing key {key}")
    if station and env.get("station") != station:
        errs.append(f"$.station: {env.get('station')!r} != {station!r}")
    for i, ver in enumerate(env.get("verdicts") or []):
        at = f"$.verdicts[{i}]"
        if not isinstance(ver, dict):
            errs.append(f"{at}: not an object")
            continue
        if not ver.get("candidateRef"):
            errs.append(f"{at}.candidateRef: required")
        if ver.get("verdict") not in VERDICT:
            errs.append(f"{at}.verdict: not in confirmed/dismissed")
        if not str(ver.get("argument", "")).strip():
            errs.append(f"{at}.argument: required")
        if ver.get("verdict") == "dismissed" and not (ver.get("argument") or ver.get("sanctionedBy")):
            errs.append(f"{at}: dismissal needs argument or sanctionedBy")
        if ver.get("verdict") == "confirmed" and not str(ver.get("survivingClaim", "")).strip():
            errs.append(f"{at}.survivingClaim: a confirmed verdict needs a surviving claim")


def main(argv: list[str]) -> int:
    role, station = "attacker", None
    args = list(argv)
    if "--role" in args:
        i = args.index("--role")
        role = args[i + 1]
        del args[i:i + 2]
    if "--station" in args:
        i = args.index("--station")
        station = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1 or role not in ("attacker", "defender"):
        print(__doc__, file=sys.stderr)
        return 2
    path = args[0]
    errs: list[str] = []
    try:
        with open(path, encoding="utf-8") as fh:
            env = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{path}: $ not parseable JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(env, dict):
        print(f"{path}: $ not an object", file=sys.stderr)
        return 1
    (validate_attacker if role == "attacker" else validate_defender)(env, errs, station)
    for err in errs:
        print(f"{path}: {err}", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
