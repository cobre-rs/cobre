"""Shared reader for plans/architecture-debt-audit/BACKLOG.md.

Every python checker in tools/ reads the register through this module, so a
register-format change is one edit. Read-only: nothing here writes a file.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

SECTION_RE = re.compile(r"^(?P<hashes>#{2,4})\s+(?P<title>.+?)\s*$")
ENTRY_RE = re.compile(r"^\*\*(?P<id>(?:CD|PD|OD|TD)-\d{3})\s*(?:·|\||-)")
FIELD_RE = re.compile(r"^\s*-\s+\*\*(?P<label>[^*]+?):\*\*\s*(?P<value>.*?)\s*$")
BASELINE_RE = re.compile(
    r"^Baseline:\s+(?P<sha>[0-9a-f]{8,40})\s+\(pinned\s+(?P<pinned>\d{4}-\d{2}-\d{2})\)"
)

_PATH_ROOTS = r"crates|scripts|docs|plans|schemas|examples|tests|\.github|\.claude"
_EXTS = r"rs|toml|md|json|fbs|sh|py|yml|yaml|txt|csv|parquet|lock"
ANCHOR_RE = re.compile(
    r"`(?P<path>(?:" + _PATH_ROOTS + r")/[\w./-]+?\.(?:" + _EXTS + r"))"
    r"(?:::(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)|:(?P<line>\d+))?`"
)

EVALUATED_SURFACES = (
    "crates", "docs", "scripts", ".github", "schemas",
    "Cargo.toml", "Cargo.lock", "examples", "tests",
)


@dataclass(frozen=True, slots=True)
class Section:
    name: str
    heading: str
    level: int
    start: int
    end: int
    lines: list[str]


@dataclass(frozen=True, slots=True)
class Anchor:
    raw: str
    path: str
    line: int | None
    symbol: str | None
    entry_id: str | None
    lineno: int


@dataclass(frozen=True, slots=True)
class Entry:
    id: str
    heading: str
    fields: dict[str, str]
    body: list[str]
    lineno: int


class SectionNotFound(LookupError):
    """The requested register section has no heading."""


class BaselineDrift(RuntimeError):
    """HEAD is not the pinned baseline and --allow-drift was not passed."""


def read_register(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def find_section(lines: list[str], name: str) -> Section:
    """Match a heading by exact title or by the station name after the last em dash.

    The scaffold headings read
    '## ★ QUALITY EVALUATION (2026-09, baseline <sha>) — core-io', so
    find_section(lines, "core-io") resolves without spelling the date or SHA.
    The section spans to the next heading of the same or a higher level.
    """
    want = name.strip().casefold()
    for idx, raw in enumerate(lines):
        m = SECTION_RE.match(raw)
        if not m:
            continue
        title = m.group("title")
        tail = title.rsplit("—", 1)[-1].strip().casefold()
        if want not in (title.strip().casefold(), tail):
            continue
        level = len(m.group("hashes"))
        end = len(lines)
        for j in range(idx + 1, len(lines)):
            nxt = SECTION_RE.match(lines[j])
            if nxt and len(nxt.group("hashes")) <= level:
                end = j
                break
        return Section(name=name, heading=title, level=level, start=idx, end=end,
                       lines=lines[idx + 1:end])
    raise SectionNotFound(name)


def parse_anchors(lines: list[str], first_lineno: int = 1,
                  entry_id: str | None = None) -> list[Anchor]:
    """Every backticked repo-rooted path in `lines`, attributed to its entry ID.

    Accepted forms: `crates/x/lib.rs`, `crates/x/lib.rs:412`, `crates/x/lib.rs::Sym`.
    Prose backticks (`god-fn`, `Sev A`) and the legacy abbreviated anchors
    (`run/setup.rs:405`) are not repo-rooted and never enter the check set.
    `first_lineno` is the 1-based register line of `lines[0]`; an entry heading
    inside `lines` re-attributes the anchors that follow it.
    """
    out: list[Anchor] = []
    for offset, raw in enumerate(lines):
        hit = ENTRY_RE.match(raw)
        if hit:
            entry_id = hit.group("id")
        for m in ANCHOR_RE.finditer(raw):
            line = m.group("line")
            out.append(Anchor(raw=m.group(0).strip("`"), path=m.group("path"),
                              line=int(line) if line else None,
                              symbol=m.group("symbol"), entry_id=entry_id,
                              lineno=first_lineno + offset))
    return out


def all_evaluation_sections(lines: list[str]) -> list[Section]:
    """Every dated `★ QUALITY EVALUATION (…) — <name>` section, in register order."""
    out: list[Section] = []
    for raw in lines:
        m = SECTION_RE.match(raw)
        if m and "QUALITY EVALUATION (" in m.group("title"):
            out.append(find_section(lines, m.group("title")))
    return out


def iter_anchors(section: Section) -> list[Anchor]:
    return parse_anchors(section.lines, first_lineno=section.start + 2)


def iter_entries(section: Section) -> list[Entry]:
    """Split a section into entries headed by a bold `**CD-nnn · …**` line.

    `fields` is keyed by the bold bullet labels below the heading
    ("- **Alignment:** advances-0a (…)" -> fields["Alignment"]); `lineno` is the
    1-based register line of the heading.
    """
    entries: list[Entry] = []
    current: tuple[str, str, list[str], int] | None = None

    def flush() -> None:
        if current is None:
            return
        entry_id, heading, body, lineno = current
        fields: dict[str, str] = {}
        for raw in body:
            f = FIELD_RE.match(raw)
            if f and f.group("label") not in fields:
                fields[f.group("label").strip()] = f.group("value")
        entries.append(Entry(id=entry_id, heading=heading, fields=fields,
                             body=list(body), lineno=lineno))

    for offset, raw in enumerate(section.lines):
        hit = ENTRY_RE.match(raw)
        if hit:
            flush()
            current = (hit.group("id"), raw.strip().strip("*").strip(), [],
                       section.start + 2 + offset)
        elif current is not None:
            current[3 - 1].append(raw)
    flush()
    return entries


def parse_baseline(lines: list[str]) -> str:
    """The pinned SHA from the column-0 `Baseline:` line in the register header."""
    for raw in lines[:60]:
        m = BASELINE_RE.match(raw.strip())
        if m:
            return m.group("sha")
    raise LookupError("no 'Baseline: <sha> (pinned YYYY-MM-DD)' line in the register header")


def parse_tables(section: Section) -> list[list[dict[str, str]]]:
    """Every pipe table in the section as header-keyed row dicts."""
    tables: list[list[dict[str, str]]] = []
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for raw in section.lines + [""]:
        stripped = raw.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if header is None:
                header = cells
            elif set("".join(cells)) <= set("-: "):
                continue
            else:
                rows.append(dict(zip(header, cells, strict=False)))
            continue
        if header is not None:
            tables.append(rows)
        header, rows = None, []
    return tables


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, check=False)


def head_sha() -> str:
    return _git("rev-parse", "HEAD").stdout.decode().strip()


def baseline_matches_head(baseline: str) -> bool:
    """True when HEAD is the pin, or a descendant whose evaluated surfaces equal the pin.

    The evaluation branch carries non-evaluated commits (the ledger itself, the
    mirror document), so equality of the evaluated surfaces is the drift test.
    """
    head = head_sha()
    if head.startswith(baseline) or baseline.startswith(head[:8]):
        return True
    if _git("merge-base", "--is-ancestor", baseline, "HEAD").returncode != 0:
        return False
    return _git("diff", "--quiet", baseline, "HEAD", "--", *EVALUATED_SURFACES).returncode == 0


def ensure_baseline(baseline: str, allow_drift: bool) -> None:
    if baseline_matches_head(baseline):
        return
    head = head_sha()
    if not allow_drift:
        raise BaselineDrift(f"HEAD {head[:12]} != baseline {baseline[:12]}")
    print(f"warning: HEAD {head[:12]} drifted from baseline {baseline[:12]}", file=sys.stderr)


def git_show(baseline: str, path: str) -> str | None:
    """Blob content at the pinned SHA, or None when the path is absent there.

    Never reads the worktree, so a dirty tree or a later commit cannot
    false-green an anchor.
    """
    proc = _git("show", f"{baseline}:{path}")
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class Table:
    header: list[str]
    header_line: int
    body: list[tuple[int, list[str]]]


@dataclass(frozen=True, slots=True)
class Milestone:
    name: str
    wave: int
    trigger: str
    line: int


def _split_cells(stripped: str) -> list[str]:
    return [c.strip() for c in stripped.strip("|").split("|")]


def parse_table(section: Section, index: int = 0) -> Table | None:
    """The `index`-th pipe table of the section with 1-based line numbers and raw cells.

    Unlike parse_tables, rows are not zipped against the header, so a caller can
    detect a row whose cell count disagrees with the header.
    """
    tables: list[Table] = []
    header: list[str] | None = None
    header_line = 0
    body: list[tuple[int, list[str]]] = []
    for offset, raw in enumerate(section.lines + [""]):
        lineno = section.start + 2 + offset
        stripped = raw.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = _split_cells(stripped)
            if header is None:
                header, header_line = cells, lineno
            elif set("".join(cells)) <= set("-: "):
                continue
            else:
                body.append((lineno, cells))
            continue
        if header is not None:
            tables.append(Table(header, header_line, body))
        header, body = None, []
    return tables[index] if index < len(tables) else None


def parse_milestones(lines: list[str], section: Section | None = None) -> dict[str, Milestone]:
    """The waved Milestones table (`Milestone | Wave | Trigger`), keyed by milestone name.

    Searched inside `section` first, then over the whole register. A Milestones
    table without a Wave column (the header vocabulary block) does not qualify.
    """
    scopes: list[Section] = []
    if section is not None:
        scopes.append(section)
    scopes.append(Section(name="*", heading="*", level=0, start=-1, end=len(lines), lines=list(lines)))
    for scope in scopes:
        idx = 0
        while (table := parse_table(scope, idx)) is not None:
            idx += 1
            head = [h.strip("`").casefold() for h in table.header]
            if "milestone" not in head or "wave" not in head:
                continue
            name_col, wave_col = head.index("milestone"), head.index("wave")
            trig_col = head.index("trigger") if "trigger" in head else None
            out: dict[str, Milestone] = {}
            for lineno, cells in table.body:
                if len(cells) <= max(name_col, wave_col):
                    continue
                name = cells[name_col].strip("`")
                wave_text = cells[wave_col].strip("`* ")
                if not wave_text.isdigit():
                    continue
                trigger = cells[trig_col] if trig_col is not None and trig_col < len(cells) else ""
                out[name] = Milestone(name=name, wave=int(wave_text), trigger=trigger, line=lineno)
            if out:
                return out
    return {}


DO_NOT_TOUCH_LINE_RE = re.compile(r"^\*\*Do-not-touch list[^*]*:\*\*")
FINDING_ID_RE = re.compile(r"\b(?:CD|PD|OD|TD)-\d{3}\b")
STATUS_MARKERS = (
    ("RETRACTED", "retracted"), ("REFUTED", "refuted"), ("WONTFIX", "wontfix"),
    ("DEFERRED", "deferred"), ("FIXED", "fixed"), ("CLEARED", "cleared"),
)


def register_findings(lines: list[str]) -> dict[str, str]:
    """Finding id -> status for every `**<ID> · …**` entry in the register.

    An explicit `- **Status:** <token>` bullet wins; otherwise an upper-case marker
    in the heading or in a `**→ MARKER` body line (RETRACTED, REFUTED, WONTFIX,
    DEFERRED, FIXED, CLEARED) sets it; otherwise the entry is `open`. Ids named
    on the do-not-touch list are `do-not-touch` regardless.
    """
    statuses: dict[str, str] = {}
    current: str | None = None
    for raw in lines:
        hit = ENTRY_RE.match(raw)
        if hit:
            current = hit.group("id")
            status = "open"
            for marker, token in STATUS_MARKERS:
                if marker in raw:
                    status = token
                    break
            statuses.setdefault(current, status)
            continue
        if SECTION_RE.match(raw):
            current = None
            continue
        if current is None:
            continue
        field = FIELD_RE.match(raw)
        if field and field.group("label").strip() == "Status":
            statuses[current] = field.group("value").strip("`* ").split()[0].casefold() if field.group("value").strip() else statuses[current]
            continue
        if raw.startswith("**→") and statuses[current] == "open":
            for marker, token in STATUS_MARKERS:
                if marker in raw:
                    statuses[current] = token
                    break
    for idx, raw in enumerate(lines):
        if DO_NOT_TOUCH_LINE_RE.match(raw):
            para = " ".join(lines[idx:idx + 4]).split("**Resume protocol", 1)[0]
            for fid in FINDING_ID_RE.findall(para):
                statuses[fid] = "do-not-touch"
            break
    return statuses
