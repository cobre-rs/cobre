#!/usr/bin/env python3
"""Flag register entries that re-raise a retired item without a `Re-raise-of:` line.

Retired corpus (four real sources): the BACKLOG do-not-touch list, the BACKLOG
cleared verdicts, the committed mirror's H3 headings (read at the pinned
baseline), and the resolved Part-VI forks of the generalization roadmap.
Exit codes: 0 clean, 1 unjustified hit(s), 2 corpus missing, 3 section not found.
Read-only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lib.backlog_parse import (  # noqa: E402
    ANCHOR_RE,
    ENTRY_RE,
    SECTION_RE,
    Entry,
    Section,
    SectionNotFound,
    all_evaluation_sections,
    find_section,
    git_show,
    iter_entries,
    parse_baseline,
    read_register,
    repo_root,
)

EXIT_OK = 0
EXIT_HIT = 1
EXIT_CORPUS_MISSING = 2
EXIT_SECTION_NOT_FOUND = 3

ROOT = repo_root()
REGISTER = ROOT / "plans" / "architecture-debt-audit" / "BACKLOG.md"
ROADMAP = ROOT / "plans" / "generalizing" / "beyond-sddp-generalization.md"
REFINEMENT_TODO = ROOT / "plans" / "generalizing" / "refinement-todo.md"
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"

RESOLVED_FORKS = ("D1", "D2", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15")
RETIRED_ID_RE = re.compile(r"\b(?:CD|PD|OD|TD)-\d{3}\b|\bD(?:1|2|[7-9]|1[0-5])\b")
JACCARD_MIN = 0.6
STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its no not of on or that the
this to was were with without vs via per than then their there these those over under
""".split())
FILE_RE = re.compile(r"`(?P<path>[\w./-]+\.(?:rs|py|sh|toml|md|json|fbs|yml|yaml))(?::\d+(?:-\d+)?)?`")
IDENT_RE = re.compile(r"`(?P<sym>[A-Za-z_][A-Za-z0-9_]{3,})`")
DO_NOT_TOUCH_RE = re.compile(r"^\*\*Do-not-touch list[^*]*:\*\*")
CLEARED_RE = re.compile(r"^\*\*↩︎ CLEARED[^*]*\*\*|^\*\*↩︎ CLEARED")
FORK_RE = re.compile(r"^\*\*(?P<id>D\d+)\s+—\s+(?P<title>.+)$")
TOKEN_SPLIT_RE = re.compile(r"[^\w\-]+")


@dataclass(frozen=True)
class RetiredItem:
    ref: str
    corpus: str
    title: str
    tokens: frozenset[str]
    anchors: frozenset[str]
    source: str


@dataclass(frozen=True)
class Hit:
    id: str
    title: str
    retired_ref: str
    corpus: str
    kind: str
    justified: bool


class CorpusMissing(RuntimeError):
    """A retirement corpus file cannot be read; maps to exit 2."""


def tokenize(text: str) -> frozenset[str]:
    words = TOKEN_SPLIT_RE.split(text.replace("`", " ").casefold())
    return frozenset(w.strip("-_") for w in words
                     if len(w.strip("-_")) >= 3 and w not in STOPWORDS and not w.isdigit())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def slug(text: str) -> str:
    return re.sub(r"-+", "-", TOKEN_SPLIT_RE.sub("-", text.replace("`", "").casefold())).strip("-")


def anchor_keys(text: str) -> frozenset[str]:
    """`path` keys (basename and, when rooted, the full path) and `sym:` keys."""
    keys: set[str] = set()
    for m in ANCHOR_RE.finditer(text):
        keys.add(pathlib.PurePosixPath(m.group("path")).name)
        keys.add(m.group("path"))
        if m.group("symbol"):
            keys.add(f"sym:{m.group('symbol')}")
    for m in FILE_RE.finditer(text):
        keys.add(pathlib.PurePosixPath(m.group("path")).name)
        keys.add(m.group("path"))
    for m in IDENT_RE.finditer(text):
        keys.add(f"sym:{m.group('sym')}")
    return frozenset(keys)


def read_corpus(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CorpusMissing(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {exc.strerror}") from exc


def read_mirror(baseline: str) -> list[str]:
    blob = git_show(baseline, MIRROR)
    if blob is None:
        raise CorpusMissing(f"{MIRROR} at {baseline[:12]}")
    return blob.splitlines()


def block_after(lines: list[str], start: int, stop) -> list[str]:
    out = [lines[start]]
    for raw in lines[start + 1:]:
        if stop(raw):
            break
        out.append(raw)
    return out


def _entry_block(lines: list[str], entry_id: str) -> tuple[int, list[str]] | None:
    for idx, raw in enumerate(lines):
        if raw.startswith(f"**{entry_id}"):
            return idx, block_after(lines, idx, lambda r: ENTRY_RE.match(r) or SECTION_RE.match(r) or r.strip() == "---")
    return None


def do_not_touch_items(lines: list[str], source: str) -> list[RetiredItem]:
    items: list[RetiredItem] = []
    for idx, raw in enumerate(lines):
        if not DO_NOT_TOUCH_RE.match(raw):
            continue
        para = " ".join(block_after(lines, idx, lambda r: not r.strip()))
        para = para.split("**Resume protocol", 1)[0]
        for ref in RETIRED_ID_RE.findall(para):
            found = _entry_block(lines, ref)
            if found is None:
                items.append(RetiredItem(ref, "do-not-touch", ref, frozenset(), frozenset(), f"{source}:{idx + 1}"))
                continue
            at, block = found
            head = block[0].strip("*").strip()
            items.append(RetiredItem(ref, "do-not-touch", head, tokenize(head),
                                     anchor_keys("\n".join(block)), f"{source}:{at + 1}"))
        if "reserved-seam census" in para:
            items.append(RetiredItem("reserved-seam-census", "do-not-touch",
                                     "sanctioned reserved-seam census",
                                     tokenize("sanctioned reserved-seam census"), frozenset(),
                                     f"{source}:{idx + 1}"))
        break
    return items


def cleared_items(lines: list[str], source: str) -> list[RetiredItem]:
    items: list[RetiredItem] = []
    for idx, raw in enumerate(lines):
        if CLEARED_RE.match(raw):
            block = block_after(lines, idx, lambda r: not r.strip())
            title = raw.split("**", 2)[1].replace("↩︎ CLEARED", "").strip(" —-()") if raw.count("**") >= 2 else raw
            items.append(RetiredItem(f"cleared:{slug(title)[:60]}", "cleared", title, tokenize(title),
                                     anchor_keys("\n".join(block)), f"{source}:{idx + 1}"))
    try:
        section = find_section(lines, "Fix-wave CLEAN verdicts (checked-and-clear — do not re-raise)")
    except SectionNotFound as exc:
        raise CorpusMissing(f"{source}: section 'Fix-wave CLEAN verdicts' not found") from exc
    bullet: list[str] = []
    bullets: list[tuple[int, list[str]]] = []
    for offset, raw in enumerate(section.lines):
        if raw.startswith("- "):
            if bullet:
                bullets.append((start, bullet))
            start, bullet = section.start + 2 + offset, [raw]
        elif raw.startswith("  ") and bullet:
            bullet.append(raw)
        elif bullet:
            bullets.append((start, bullet))
            bullet = []
    if bullet:
        bullets.append((start, bullet))
    for at, block in bullets:
        text = " ".join(b.strip() for b in block)
        title = text.split("**", 2)[1] if text.count("**") >= 2 else text[:80]
        items.append(RetiredItem(f"cleared:{slug(title)[:60]}", "cleared", title, tokenize(title),
                                 anchor_keys(text), f"{source}:{at}"))
    return items


def mirror_items(lines: list[str], source: str) -> list[RetiredItem]:
    items: list[RetiredItem] = []
    for idx, raw in enumerate(lines):
        m = SECTION_RE.match(raw)
        if not m or len(m.group("hashes")) != 3:
            continue
        title = m.group("title").replace("`", "")
        block = block_after(lines, idx, lambda r: SECTION_RE.match(r) is not None)
        items.append(RetiredItem(f"mirror:{slug(title)}", "mirror", title, tokenize(title),
                                 anchor_keys("\n".join(block)), f"{source}:{idx + 1}"))
    return items


def resolved_fork_items(lines: list[str], source: str) -> list[RetiredItem]:
    items: list[RetiredItem] = []
    for idx, raw in enumerate(lines):
        m = FORK_RE.match(raw)
        if not m or m.group("id") not in RESOLVED_FORKS:
            continue
        block = block_after(lines, idx, lambda r: FORK_RE.match(r) is not None or SECTION_RE.match(r) is not None)
        title = re.split(r"\*\*|\. |_\(", m.group("title"), maxsplit=1)[0].strip(" .")
        items.append(RetiredItem(m.group("id"), "resolved-fork", title, tokenize(title),
                                 anchor_keys("\n".join(block)), f"{source}:{idx + 1}"))
    return items


def load_retired_corpus(baseline: str) -> list[RetiredItem]:
    register = read_corpus(REGISTER)
    read_corpus(REFINEMENT_TODO)
    roadmap = read_corpus(ROADMAP)
    mirror = read_mirror(baseline)
    return (do_not_touch_items(register, "BACKLOG.md")
            + cleared_items(register, "BACKLOG.md")
            + mirror_items(mirror, MIRROR)
            + resolved_fork_items(roadmap, "beyond-sddp-generalization.md"))


def entry_title(entry: Entry) -> str:
    for raw in entry.body:
        stripped = raw.strip()
        if stripped and not stripped.startswith(("-", "|", "*")):
            return stripped
    return entry.fields.get("Evidence", "")


def entry_anchor_keys(entry: Entry) -> frozenset[str]:
    return anchor_keys(entry.fields.get("Anchors", ""))


def anchors_overlap(item: RetiredItem, keys: frozenset[str]) -> bool:
    """Same file AND same symbol, or the same rooted file for a file-scoped retired item."""
    shared = item.anchors & keys
    paths = {k for k in shared if not k.startswith("sym:")}
    syms = {k for k in shared if k.startswith("sym:")}
    if paths and syms:
        return True
    item_syms = {k for k in item.anchors if k.startswith("sym:")}
    return bool(paths and not item_syms and any("/" in p for p in paths))


def match_entry(entry: Entry, corpus: list[RetiredItem]) -> list[Hit]:
    raw = "\n".join([entry.heading, *entry.body])
    body_ids = set(RETIRED_ID_RE.findall(raw))
    justified = set(RETIRED_ID_RE.findall(entry.fields.get("Re-raise-of", "")))
    justified |= {slug(x) for x in re.split(r"[;,]", entry.fields.get("Re-raise-of", ""))}
    keys = entry_anchor_keys(entry)
    title = entry_title(entry)
    tokens = tokenize(title)
    hits: list[Hit] = []
    for item in corpus:
        if item.ref in body_ids:
            kind = "explicit-id"
        elif anchors_overlap(item, keys):
            kind = "anchor-overlap"
        elif jaccard(item.tokens, tokens) >= JACCARD_MIN:
            kind = "title-overlap"
        else:
            continue
        ok = item.ref in justified or item.ref.split(":", 1)[-1] in justified
        hits.append(Hit(entry.id, title, item.ref, item.corpus, kind, ok))
    return hits


def resolve_target(arg: str, register: pathlib.Path | None) -> tuple[pathlib.Path, str | None]:
    candidate = pathlib.Path(arg)
    if candidate.is_file():
        return candidate, None
    return register or REGISTER, arg


def evaluate(register: pathlib.Path, section_name: str | None, baseline: str) -> tuple[int, dict]:
    lines = read_register(register)
    try:
        if section_name is None:
            found = all_evaluation_sections(lines)
            if not found:
                raise SectionNotFound(str(register))
            section: Section = found[0]
        else:
            section = find_section(lines, section_name)
    except SectionNotFound as exc:
        print(f"section-not-found: {exc}", file=sys.stderr)
        return EXIT_SECTION_NOT_FOUND, {}
    corpus = load_retired_corpus(baseline)
    counts: dict[str, int] = {}
    for item in corpus:
        counts[item.corpus] = counts.get(item.corpus, 0) + 1
    entries = iter_entries(section)
    hits = [h for e in entries for h in match_entry(e, corpus)]
    report = {"section": section.heading, "baseline": baseline, "corpora": counts,
              "checked": len(entries),
              "hits": [{"id": h.id, "title": h.title, "retiredRef": h.retired_ref,
                        "corpus": h.corpus, "kind": h.kind, "justified": h.justified}
                       for h in hits]}
    code = EXIT_HIT if any(not h.justified for h in hits) else EXIT_OK
    return code, report


def run(arg: str, register: pathlib.Path | None, baseline: str | None, as_json: bool) -> int:
    baseline = baseline or parse_baseline(read_register(REGISTER))
    target, section_name = resolve_target(arg, register)
    try:
        code, report = evaluate(target, section_name, baseline)
    except CorpusMissing as exc:
        print(f"corpus-missing: {exc}", file=sys.stderr)
        return EXIT_CORPUS_MISSING
    if code == EXIT_SECTION_NOT_FOUND:
        return code
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        for h in report["hits"]:
            if not h["justified"]:
                print(f"re-raise[{h['kind']}] {h['id']} -> {h['retiredRef']} (corpus {h['corpus']})")
        unjustified = sum(1 for h in report["hits"] if not h["justified"])
        print(f"checked {report['checked']} entries, {unjustified} unjustified re-raise(s)")
    return code


def self_test(baseline: str) -> int:
    global MIRROR
    fx = pathlib.Path(__file__).resolve().parent / "fixtures"
    failed = 0

    def expect(label: str, got: int, want: int, extra_ok: bool = True) -> None:
        nonlocal failed
        ok = got == want and extra_ok
        failed += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {label}: exit={got} (want {want})")

    code, _ = evaluate(fx / "good-section.md", None, baseline)
    expect("good-section.md clean", code, EXIT_OK)
    code, report = evaluate(fx / "reraise-seeded.md", None, baseline)
    unjust = [h for h in report.get("hits", []) if not h["justified"]]
    corpora = sorted(h["corpus"] for h in unjust)
    expect("reraise-seeded.md two unjustified hits (do-not-touch + mirror)", code, EXIT_HIT,
           len(unjust) == 2 and corpora == ["do-not-touch", "mirror"])
    saved, MIRROR = MIRROR, "docs/design/no-such-mirror.md"
    try:
        evaluate(fx / "good-section.md", None, baseline)
        expect("missing mirror corpus", EXIT_OK, EXIT_CORPUS_MISSING)
    except CorpusMissing:
        expect("missing mirror corpus", EXIT_CORPUS_MISSING, EXIT_CORPUS_MISSING)
    finally:
        MIRROR = saved
    print(f"self-test: {'FAIL' if failed else 'ok'} ({failed} failing cases)")
    return EXIT_HIT if failed else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("section", nargs="?", help="register section name, or a fixture file path")
    ap.add_argument("--register", type=pathlib.Path, help="register file when `section` is a name")
    ap.add_argument("--baseline", help="override the SHA pinned in the register header")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(args.baseline or parse_baseline(read_register(REGISTER)))
    if not args.section:
        ap.error("section is required unless --self-test is given")
    return run(args.section, args.register, args.baseline, args.json)


if __name__ == "__main__":
    sys.exit(main())
