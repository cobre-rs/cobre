"""Station tests for the test-corpus station (stations/test-corpus/).

InventoryTests binds the E08-1 re-measurement and the E08-2 priors to the pinned
baseline: every figure equals a fresh run of its recorded command, every prior-register
anchor and cited register id resolves at the pin, and every claim class carries its
yardstick section, a verbatim quote found at the stated lines, and a resolving anchor
into the test corpus. Later test-corpus tickets append their stage classes here.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

ID_RE = re.compile(r"\b((?:CD|PD|OD|TD)-\d{3})\b")
SWEEP_ANCHOR_RE = re.compile(
    r"`(?P<path>[A-Za-z0-9_./-]+\.(?:rs|toml|yml|md|py|json|sh|lock))"
    r"(?:::(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)|:(?P<line>\d+)(?:-(?P<line_to>\d+))?)?`"
)
BLOCK_RE = re.compile(
    r"^=== figure=(?P<id>\S+) source=(?P<source>\S+) definition=(?P<definition>.+)$"
)
NEXTEST_LINE_RE = re.compile(r"^\S+ \S+$")
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"
YARDSTICK = "docs/design/testing-architecture.md"
TOOLCHAIN = {"nextest-list", "doctests", "pytest-collect"}


def load_check_anchors():
    spec = importlib.util.spec_from_file_location(
        "check_anchors", sc.TOOLS / "check-anchors.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("tools/check-anchors.py not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_blocks(log: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    lines = log.splitlines()
    i = 0
    while i < len(lines):
        m = BLOCK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fid = m.group("id")
        block = {"command": "", "exit": "", "status": "", "stdout": ""}
        i += 1
        while i < len(lines) and not lines[i].startswith("stdout:"):
            for key in ("command", "exit", "status"):
                if lines[i].startswith(f"{key}: "):
                    block[key] = lines[i][len(key) + 2 :]
            i += 1
        i += 1
        out: list[str] = []
        while i < len(lines) and lines[i] != f"=== end {fid}":
            out.append(lines[i])
            i += 1
        block["stdout"] = "\n".join(out).rstrip("\n")
        blocks[fid] = block
        i += 1
    return blocks


def run(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", cmd], cwd=sc.REPO, capture_output=True, text=True, check=False
    )


class TestCorpusCase(sc.StationCase):
    SLUG = "test-corpus"
    SECTION_TITLE = "test-corpus"

    @classmethod
    def baseline(cls) -> str:
        pinned = sc.load_json(cls.station_dir() / "inventory.json")["baseline"]
        return pinned["sha"] if isinstance(pinned, dict) else str(pinned)


class InventoryTests(TestCorpusCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.claims = sc.load_json(cls.station_dir() / "claim-classes.json")
        cls.prior = (cls.station_dir() / "prior-register.md").read_text(
            encoding="utf-8"
        )
        cls.lens = (cls.station_dir() / "lens-rules.md").read_text(encoding="utf-8")
        cls.log = (cls.station_dir() / "remeasure.log").read_text(encoding="utf-8")
        cls.blocks = parse_blocks(cls.log)
        cls.sha = cls.baseline()
        cls.t = cls.tree()
        cls.figures = {f["id"]: f for f in cls.inv["figures"]}
        cls.register = backlog_parse.read_register(sc.BACKLOG)
        cls.entry_lines: dict[str, int] = {}
        for i, ln in enumerate(cls.register, 1):
            m = backlog_parse.ENTRY_RE.match(ln)
            if m:
                cls.entry_lines.setdefault(m.group("id"), i)

    def lines(self, path: str) -> list[str]:
        return self.t.read_text(path).splitlines()

    # ---- baseline ----

    def test_baseline_is_the_register_pin_everywhere(self) -> None:
        self.assertRegex(self.sha, r"^[0-9a-f]{40}$")
        self.assertTrue(sc.is_register_pin(self.sha))
        self.assertEqual(self.claims["baseline"], self.sha)
        self.assertIn(self.sha, self.prior)
        self.assertIn(self.sha, self.lens)
        blob = run(f"git rev-parse {self.sha}:{YARDSTICK}").stdout.strip()
        self.assertEqual(self.claims["sourceBlob"], blob)

    # ---- E08-1 figures ----

    def test_every_figure_is_measured_and_logged(self) -> None:
        self.assertEqual(
            self.inv["figureStatus"], {"measured": len(self.inv["figures"])}
        )
        for fid, fig in self.figures.items():
            self.assertEqual(fig["status"], "measured", fid)
            parts = [b for b in self.blocks.values() if b["command"] in fig["command"]]
            self.assertTrue(parts, fid)
            for block in parts:
                self.assertEqual(block["exit"], "0", fid)
        self.assertEqual(self.log.count("/target/"), 0)

    def test_git_census_blocks_equal_a_fresh_run(self) -> None:
        reran = 0
        for fid, block in self.blocks.items():
            if fid in TOOLCHAIN or self.sha not in block["command"]:
                continue
            proc = run(block["command"])
            self.assertEqual(proc.returncode, int(block["exit"]), fid)
            self.assertEqual(proc.stdout.rstrip("\n"), block["stdout"], fid)
            reran += 1
        self.assertGreaterEqual(reran, 20)

    def test_toolchain_figures_equal_a_fresh_run(self) -> None:
        nextest = run(self.blocks["nextest-list"]["command"])
        self.assertEqual(nextest.returncode, 0, nextest.stderr[-2000:])
        listed = [ln for ln in nextest.stdout.splitlines() if NEXTEST_LINE_RE.match(ln)]
        self.assertEqual(len(listed), self.figures["nextest-list"]["value"])
        binaries = {ln.split()[0] for ln in listed}
        self.assertEqual(len(binaries), self.figures["nextest-list"]["listedBinaries"])

        doc = run(self.blocks["doctests"]["command"])
        self.assertEqual(doc.returncode, 0, doc.stderr[-2000:])
        running = sum(
            int(m) for m in re.findall(r"^running (\d+) tests?$", doc.stdout, re.M)
        )
        self.assertEqual(running, self.figures["doctests"]["value"])

        pytest = run(self.blocks["pytest-collect"]["command"])
        self.assertEqual(pytest.returncode, 0, pytest.stderr[-2000:])
        m = re.search(r"^(\d+) tests? collected", pytest.stdout, re.M)
        self.assertIsNotNone(m, pytest.stdout[-500:])
        self.assertEqual(int(m.group(1)), self.figures["pytest-collect"]["value"])  # type: ignore[union-attr]

    # ---- E08-2 prior register ----

    def entries(self) -> dict[str, str]:
        body = self.prior.split("## Ticket-figure deviations", 1)[0]
        parts = re.split(r"^### ", body, flags=re.M)[1:]
        return {p.splitlines()[0].strip(): p for p in parts}

    def test_prior_register_has_exactly_three_mirror_items(self) -> None:
        entries = self.entries()
        self.assertEqual(
            set(entries),
            {
                "Oracle test-harness duplication",
                "Python-binding Rust tests invisible to CI",
                "Mega-file / inline-test-giant asymmetry",
            },
        )
        mirror = self.lines(MIRROR)
        for title, body in entries.items():
            anchor_line = next(
                ln for ln in body.splitlines() if ln.startswith("**Mirror anchor.**")
            )
            self.assertIn(f"`{MIRROR}:", anchor_line, title)
            self.assertIn("## Deferred-debt register", anchor_line, title)
            self.assertTrue("### " in anchor_line or "bullet" in anchor_line, title)
            self.assertRegex(body, r"`[A-Za-z0-9_./-]+\.(?:rs|toml|yml|md):\d+`", title)
            self.assertIn("dup-of-likely", body, title)
            self.assertIn("**Owner (from mirror).**", body, title)
            self.assertIn("**Trigger (from mirror).**", body, title)
            for m in re.finditer(rf"`{re.escape(MIRROR)}:(\d+)`", anchor_line):
                line = mirror[int(m.group(1)) - 1]
                self.assertTrue(
                    line.startswith("##") or line.startswith("- **"), f"{title}: {line}"
                )
        self.assertIn("### Oracle test-harness duplication", mirror[307])
        self.assertIn("### Python-binding Rust tests invisible to CI", mirror[346])
        self.assertIn("Mega-file / inline-test-giant asymmetry", mirror[710])
        self.assertIn("## Deferred-debt register", mirror[239])
        self.assertIn("whole-lifecycle audit findings", mirror[552])

    def test_oracle_entry_names_all_four_carriers_at_pin_lines(self) -> None:
        body = self.entries()["Oracle test-harness duplication"]
        sharpening = next(
            ln
            for ln in body.splitlines()
            if ln.startswith("**Sharpening at baseline.**")
        )
        census = run(
            f"git grep -n 'fn close' {self.sha} -- crates/cobre-sddp/tests"
        ).stdout
        hits = {
            (m.group(1), int(m.group(2)))
            for m in re.finditer(r"^[0-9a-f]{40}:(crates/[^:]+):(\d+):", census, re.M)
        }
        self.assertEqual(
            hits,
            {
                ("crates/cobre-sddp/tests/extensive_form_oracle.rs", 77),
                ("crates/cobre-sddp/tests/branching_value_oracle.rs", 93),
                ("crates/cobre-sddp/tests/node_native_backward_gate.rs", 76),
                ("crates/cobre-sddp/tests/mpi_wire.rs", 2816),
            },
        )
        for path, line in hits:
            self.assertIn(f"{pathlib.PurePosixPath(path).name}:{line}", sharpening)
            self.assertIn(f"`{path}:{line}`", body)
        self.assertIn("**Disposition.** `dup-of-likely`", body)
        self.assertIsNone(re.search(r"\bTD-\d{3}\b.*\bowns this item", body))
        self.assertIn("mirror-only", body)

    def test_python_entry_records_the_superseded_premise(self) -> None:
        body = self.entries()["Python-binding Rust tests invisible to CI"]
        ci = self.lines(".github/workflows/ci.yml")
        self.assertIn(
            "cargo test --manifest-path crates/cobre-python/Cargo.toml", ci[566]
        )
        self.assertIn("Run Rust tests for the bindings crate", ci[561])
        self.assertIn("`.github/workflows/ci.yml:567`", body)
        self.assertIn("SN-06", body)
        self.assertIn("does NOT hold at the pin", body)
        manifest = self.lines("crates/cobre-python/Cargo.toml")
        self.assertIn('crate-type = ["cdylib"]', manifest[16])
        self.assertIn("[dev-dependencies]", manifest[33])
        root = self.lines("Cargo.toml")
        self.assertIn("exclude", root[20])
        self.assertIn("crates/cobre-python", root[21])

    def test_inline_giant_entry_line_counts_match_the_pin(self) -> None:
        body = self.entries()["Mega-file / inline-test-giant asymmetry"]
        lp = "crates/cobre-sddp/src/lp/builder"
        counts = {
            f: len(self.lines(f"{lp}/{f}"))
            for f in (
                "entries.rs",
                "columns.rs",
                "template/tests.rs",
                "layout/tests.rs",
                "template.rs",
                "layout.rs",
            )
        }
        self.assertEqual(counts["entries.rs"], 10093)
        self.assertEqual(counts["columns.rs"], 9319)
        self.assertEqual(counts["template/tests.rs"], 5356)
        self.assertEqual(counts["layout/tests.rs"], 3565)
        for n in counts.values():
            self.assertIn(f"{n:,}", body)
        entries = self.lines(f"{lp}/entries.rs")
        columns = self.lines(f"{lp}/columns.rs")
        self.assertEqual(
            [i + 1 for i, ln in enumerate(entries) if ln == "#[cfg(test)]"],
            [1680, 1716, 2242, 3510],
        )
        self.assertEqual(len([ln for ln in columns if ln == "#[cfg(test)]"]), 12)
        self.assertEqual(columns[1304], "#[cfg(test)]")
        self.assertIn("8,413", body)
        self.assertIn("8,014", body)
        self.assertEqual(self.lines(f"{lp}/layout.rs")[1964], "#[cfg(test)]")
        self.assertIn("CD-021", body)
        self.assertIn("CD-007", body)
        self.assertIn("TD-047", body)

    def test_prior_register_ids_exist_and_their_anchors_resolve(self) -> None:
        ids = set(ID_RE.findall(self.prior))
        self.assertTrue(
            {"CD-007", "CD-021", "TD-047", "TD-030", "TD-043", "CD-103"} <= ids
        )
        for id_ in ids:
            self.assertIn(id_, self.entry_lines, id_)
        for m in re.finditer(r"\b((?:CD|PD|OD|TD)-\d{3})\b \(`([^`]+)`", self.prior):
            self.assertTrue(sc.anchor_exists(f"`{m.group(2)}`", self.t), m.group(0))
        for m in re.finditer(r"\bL(\d{3,4})\b", self.prior):
            self.assertLessEqual(int(m.group(1)), len(self.register))

    def test_anchor_sweep_recorded_and_independently_confirmed(self) -> None:
        self.assertIn("## Anchor verification", self.prior)
        head, sweep = self.prior.split("## Anchor verification", 1)
        self.assertNotIn("anchor-missing `", sweep)
        self.assertIn("stays in the dup-of set", sweep)
        self.assertEqual(sweep.count("stays in the dup-of set"), 3)
        ca = load_check_anchors()
        swept = {m.group(0) for m in SWEEP_ANCHOR_RE.finditer(sweep)}
        for m in SWEEP_ANCHOR_RE.finditer(head):
            raw = m.group(0)
            path = m.group("path")
            if path.startswith(("plans/", "stations/")):
                continue
            self.assertIn(raw, swept, raw)
            self.assertTrue(self.t.is_file(path), raw)
            text = self.t.read_text(path)
            if m.group("symbol"):
                self.assertTrue(
                    ca.decl_pattern(path, m.group("symbol")).search(text), raw
                )
            elif m.group("line"):
                hi = int(m.group("line_to") or m.group("line"))
                self.assertLessEqual(hi, len(text.splitlines()), raw)

    def test_dup_of_set_lists_the_three_items(self) -> None:
        section = self.prior.split("## dup-of set handed to the attacker ticket", 1)[1]
        section = section.split("## Anchor verification", 1)[0]
        for title in self.entries():
            self.assertIn(f"`{title}`", section)

    # ---- E08-2 claim classes ----

    def test_claim_classes_schema_and_section_rule(self) -> None:
        rows = self.claims["rows"]
        self.assertEqual(len(rows), self.claims["rowCount"])
        self.assertGreaterEqual(len(rows), 60)
        ids = [r["id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for r in rows:
            for key in (
                "id",
                "section",
                "lines",
                "quote",
                "sentence",
                "class",
                "driftIsFinding",
                "adjudicate",
                "counterAnchors",
                "anchors",
                "note",
            ):
                self.assertIn(key, r, r["id"])
            self.assertIn(r["class"], {"current-state", "target"}, r["id"])
            self.assertEqual(
                r["driftIsFinding"], r["class"] == "current-state", r["id"]
            )
            if r["section"].startswith("2"):
                self.assertEqual(r["class"], "current-state", r["id"])
            if r["section"].startswith("5"):
                self.assertEqual(r["class"], "target", r["id"])
            if r["section"] in {"6", "7"}:
                self.assertEqual(r["class"], "target", r["id"])
            if r["adjudicate"]:
                self.assertTrue(r["counterAnchors"], r["id"])
                self.assertIn("provisional", r["note"], r["id"])
        sections = {r["section"] for r in rows}
        self.assertTrue({"2.1", "2.2", "2.3", "3.2", "5.2", "5.8", "7"} <= sections)

    def test_claim_rows_quote_the_yardstick_at_the_stated_lines(self) -> None:
        doc = self.lines(YARDSTICK)
        for r in self.claims["rows"]:
            lo, _, hi = r["lines"].partition("-")
            lo_i, hi_i = int(lo), int(hi or lo)
            self.assertLessEqual(hi_i, len(doc), r["id"])
            window = "\n".join(doc[lo_i - 1 : hi_i])
            self.assertIn(r["quote"], window, r["id"])

    def test_claim_rows_carry_a_resolving_anchor_into_the_corpus(self) -> None:
        for r in self.claims["rows"]:
            self.assertTrue(r["anchors"], r["id"])
            for a in r["anchors"]:
                self.assertTrue(sc.anchor_exists(f"`{a}`", self.t), f"{r['id']}: {a}")
            if r.get("figure"):
                self.assertIn(r["figure"], self.figures, r["id"])

    def test_adjudicate_rows_are_the_stubcomm_and_test_support_sentences(self) -> None:
        rows = {r["id"]: r for r in self.claims["rows"]}
        adjudicated = {rid for rid, r in rows.items() if r["adjudicate"]}
        self.assertEqual(
            adjudicated,
            {
                "ta-5.8-stubcomm-home",
                "ta-5.2-stubcomm-in-comm",
                "ta-5.2-keep-feature-three-crates",
                "ta-5.2-adoption-uniform",
            },
        )
        mod = self.lines("crates/cobre-sddp/tests/common/mod.rs")
        self.assertEqual(mod[31], "pub struct StubComm;")
        self.assertEqual(mod[85], "pub struct Rank0Of2;")
        self.assertNotIn(
            "test-support", self.t.read_text("crates/cobre-comm/Cargo.toml")
        )
        for rid in (
            "ta-5.8-stubcomm-home",
            "ta-5.2-stubcomm-in-comm",
            "ta-5.2-adoption-uniform",
        ):
            counters = "\n".join(rows[rid]["counterAnchors"])
            self.assertIn("crates/cobre-sddp/tests/common/mod.rs:32", counters, rid)
            self.assertIn("crates/cobre-sddp/tests/common/mod.rs:86", counters, rid)
            self.assertIn(
                "crates/cobre-comm/Cargo.toml declares no test-support", counters, rid
            )
            self.assertEqual(rows[rid]["class"], "target", rid)
            self.assertFalse(rows[rid]["driftIsFinding"], rid)
        doc = self.lines(YARDSTICK)
        self.assertIn("in `cobre-comm`'s `test-support` surface", doc[523])
        self.assertIn("Keep the `test-support` cargo-feature mechanism", doc[397])
        declarers = self.figures["test-support-declarers-vs-consumers"]["value"][
            "declarers"
        ]
        self.assertEqual(len(declarers), 5)
        self.assertNotIn("cobre-comm", declarers)
        self.assertNotIn("cobre-cli", declarers)

    # ---- E08-2 lens rules ----

    def test_lens_rules_state_both_rules_with_sources_and_candidate_tests(self) -> None:
        testing = self.lines(".claude/rules/testing.md")
        self.assertTrue(testing[39].startswith("## Contracts"))
        self.assertTrue(testing[112].startswith("## Cost discipline"))
        self.assertIn("Declaration-order invariance is a sort contract", testing[41])
        doc = self.lines(YARDSTICK)
        self.assertIn("Reducing coverage to shrink the suite", doc[601])
        self.assertIn("No test is deleted, skipped, renamed, or weakened", doc[352])
        for needle in (
            "## Rule 1",
            "## Rule 2",
            "`.claude/rules/testing.md:40`",
            "`.claude/rules/testing.md:113`",
            "`docs/design/testing-architecture.md:602-604`",
            "**Candidate test a lens applies.**",
            "NEVER a fix-shape",
            "recorded under `positives`",
            "per-**binary** and per-**feature-combo**, never per-**test**",
        ):
            self.assertIn(needle, self.lens, needle)
        self.assertEqual(self.lens.count("**Candidate test a lens applies.**"), 2)
        self.assertIn("62 of the 87", self.lens)
        self.assertEqual(self.figures["int-binaries-solver-linking"]["value"], 62)
        self.assertEqual(self.figures["int-binaries"]["value"], 87)
        self.assertIn("6,371", self.lens)
        self.assertEqual(self.figures["nextest-list"]["value"], 6371)

    def test_lens_rule_one_probe_reproduces_at_the_pin(self) -> None:
        claimers = run(
            f"git grep -il 'invarian' {self.sha} -- 'crates/*/tests/*.rs'"
        ).stdout.split()
        files = [c.split(":", 1)[1] for c in claimers]
        with_permute = {f for f in files if "permute" in self.t.read_text(f)}
        self.assertIn(f"# {len(files)} files claim some invariance", self.lens)
        self.assertIn(f"holds for {len(with_permute)}", self.lens)
        self.assertIn(
            f"The {len(files) - len(with_permute)} files without a permute call",
            self.lens,
        )
        self.assertIn(f"The {len(with_permute)} permute-carrying files", self.lens)
        for f in with_permute:
            self.assertIn(f"`{f}`", self.lens, f)
        self.assertIn(
            "fn permute_case",
            self.lines("crates/cobre-sddp/tests/common/permute.rs")[82],
        )


LENSES = ("test-bloat", "architecture", "over-engineering", "performance")
RESERVED_PREFIXES = (
    "crates/cobre/",
    "crates/cobre-mcp/",
    "crates/cobre-tui/",
    "crates/cobre-flow/",
    "crates/cobre-uc/",
    "crates/cobre-emt/",
)
SHARED_DOCS = {
    "docs/design/testing-architecture.md",
    "docs/design/reserved-seams-and-deferred-debt.md",
    ".claude/rules/testing.md",
    "Cargo.toml",
    "ARCHITECTURE.md",
    "CLAUDE.md",
}
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PERF_TIMING_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|seconds|secs|sec)\b|\bx\s*faster|\d+x faster|speed-?up",
    re.I,
)


def load_validate_tb():
    spec = importlib.util.spec_from_file_location(
        "validate_tb", sc.STATIONS / "test-corpus" / "validate-tb.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("validate-tb.py not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CandidateEnvelopeTests(TestCorpusCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vtb = load_validate_tb()
        cls.env = {
            lens: sc.load_json(cls.station_dir() / f"candidates-{lens}.json")
            for lens in LENSES
        }
        cls.prompt = (cls.station_dir() / "attacker-prompt.md").read_text(
            encoding="utf-8"
        )
        cls.log = (cls.station_dir() / "attacker-log.md").read_text(encoding="utf-8")
        cls.seeds = sc.load_json(cls.station_dir() / "seeds.json")
        cls.t = cls.tree()
        cls.files = set(cls.t.files())
        cls.corpus = set(cls.vtb.corpus_paths(cls.t))
        cls.keys = cls.vtb.inventory_keys()

    def all_candidates(self):
        for lens, env in self.env.items():
            for c in env["candidates"]:
                yield lens, c

    def test_every_candidates_file_validates_through_the_shared_tool_and_the_profile(
        self,
    ) -> None:
        for lens in LENSES:
            path = self.station_dir() / f"candidates-{lens}.json"
            r = subprocess.run(
                [
                    sys.executable,
                    str(sc.TOOLS / "validate-envelope.py"),
                    "--role",
                    "attacker",
                    "--station",
                    "test-corpus",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, f"{lens}: {r.stderr}")
            self.assertEqual(self.vtb.validate_file(path), [], lens)

    def test_header_names_the_station_the_pin_and_the_lens(self) -> None:
        for lens, env in self.env.items():
            self.assertEqual((env["station"], env["lens"]), ("test-corpus", lens))
            self.assertEqual(
                env["baseline"], self.sha if hasattr(self, "sha") else self.baseline()
            )
            self.assertEqual(env["primary"], lens == "test-bloat")
            self.assertEqual(env["informational"], lens == "performance")
            self.assertTrue(env["workers"], lens)
            self.assertTrue(env["candidates"] or env["cleanVerdict"], lens)
            self.assertTrue(env["positives"], lens)
            self.assertEqual(env["dispatcher"].split(" (")[0], "adversarial-attacker")
        self.assertEqual(len(self.env["test-bloat"]["workers"]), 9)
        for lens in ("architecture", "over-engineering", "performance"):
            self.assertEqual(list(self.env[lens]["workers"]), [f"sec-{lens}"])

    def test_profile_fields_on_every_candidate(self) -> None:
        seed_refs = {s["seedRef"] for s in self.seeds["seeds"]}
        for lens, c in self.all_candidates():
            self.assertIn(c["claimKind"], self.vtb.CLAIM_KINDS, c["title"])
            self.assertIn(c["measurementDefinition"], self.vtb.DEFINITIONS, c["title"])
            self.assertRegex(c["yardstickRef"], r"^ta-\d(\.\d+)?$", c["title"])
            self.assertNotIn("targetNotDefect", c, c["title"])
            mv = c.get("measuredValue")
            if c["measurementDefinition"] == "n/a":
                self.assertIsNone(mv, c["title"])
            else:
                self.assertIsNotNone(mv, c["title"])
                self.assertIn(mv["key"], self.keys, c["title"])
                have = self.keys[mv["key"]]
                if isinstance(have, (int, float, str, bool)):
                    self.assertEqual(mv["value"], have, c["title"])
            if c.get("seedRef") is not None:
                self.assertIn(c["seedRef"], seed_refs, c["title"])
            if c.get("dupOf") is not None:
                self.assertIn(c["dupOf"], self.vtb.MIRROR_ITEMS, c["title"])
            self.assertTrue(c.get("raisedBy"), c["title"])
            mv = c.get("measuredValue")
            if mv is not None:
                self.assertRegex(
                    mv["key"], r"^(figures|perCrate|harness)\.", c["title"]
                )

    def test_anchor_grammar_points_at_the_test_surface(self) -> None:
        for lens, c in self.all_candidates():
            self.assertTrue(c["anchors"], c["title"])
            for a in c["anchors"]:
                path = a["path"]
                on_surface = (
                    path in self.corpus
                    or re.match(r"^crates/[a-z-]+/tests/", path) is not None
                    or re.match(r"^crates/[a-z-]+/benches/", path) is not None
                    or path in self.vtb.WORKFLOWS
                    or re.match(r"^crates/[a-z-]+/Cargo\.toml$", path) is not None
                    or path in SHARED_DOCS
                )
                self.assertTrue(
                    on_surface, f"{lens}: {path} is not a test-surface anchor"
                )
                if "symbol" in a and a["symbol"] is not None:
                    self.assertTrue(
                        IDENT_RE.match(a["symbol"]), f"{path}::{a['symbol']}"
                    )
                else:
                    self.assertIsInstance(a.get("line"), int, f"{lens}: {path}")
                    self.assertGreaterEqual(a["line"], 1)

    def test_duplicated_harness_candidates_name_both_copies(self) -> None:
        hits = 0
        for lens, c in self.all_candidates():
            text = (c["title"] + " " + c["evidence"].get("reading", "")).lower()
            if (
                c.get("dupOf") == "Oracle test-harness duplication"
                or "fn close" in text
                or ("stubcomm" in text and "rank0of2" in text)
            ):
                hits += 1
                paths = {a["path"] for a in c["anchors"]}
                self.assertGreaterEqual(len(paths), 2, c["title"])
        self.assertGreaterEqual(hits, 1)

    def test_stubcomm_drift_probe_is_recorded_not_adjudicated(self) -> None:
        """AC 8: the §5.8 sentence, the canonical home mod.rs:32/:86, the copy census and the
        cobre-comm manifest absence are all recorded, classified, and left for the defender."""
        doc_side = []
        tree_side = []
        for lens, c in self.all_candidates():
            blob = json.dumps(c)
            paths = {(a["path"], a.get("line")) for a in c["anchors"]}
            if (
                "docs/design/testing-architecture.md",
                524,
            ) in paths and "crates/cobre-comm/Cargo.toml" in blob:
                doc_side.append(c)
            if ("crates/cobre-sddp/tests/common/mod.rs", 32) in paths and (
                "crates/cobre-sddp/tests/common/mod.rs",
                86,
            ) in paths:
                tree_side.append(c)
        self.assertTrue(
            doc_side, "no candidate anchors §5.8 (:524) against the cobre-comm manifest"
        )
        self.assertTrue(
            tree_side, "no candidate anchors the canonical home mod.rs:32 and :86"
        )
        for c in doc_side:
            self.assertIn("mod.rs:32", json.dumps(c))
        for c in tree_side:
            copies = [
                a
                for a in c["anchors"]
                if a["path"].startswith("crates/cobre-sddp/src/")
            ]
            self.assertGreaterEqual(
                len(copies), 2, "the copy census needs private copies anchored"
            )
        for c in doc_side + tree_side:
            self.assertNotIn("targetNotDefect", c)
            self.assertIn(c["claimKind"], self.vtb.CLAIM_KINDS)
            self.assertTrue(c["evidence"].get("reading"))
        self.assertIn("StubComm / Rank0Of2 copy census", self.log)
        self.assertIn(
            "crates/cobre-sddp/src/training/backward_pass_state.rs:2216", self.log
        )
        self.assertIn("StubComm", self.log)

    def test_performance_file_is_informational_and_number_free(self) -> None:
        env = self.env["performance"]
        for c in env["candidates"]:
            self.assertIs(c["informational"], True, c["title"])
            self.assertEqual(c["status"], "UNMEASURED", c["title"])
            self.assertTrue(c["statusReason"], c["title"])
            self.assertRegex(
                c["costMechanism"],
                r"(?i)binar|static solver link|cadence|wall",
                c["title"],
            )
            self.assertEqual(c["mechanism"], c["costMechanism"], c["title"])
        text = (self.station_dir() / "candidates-performance.json").read_text(
            encoding="utf-8"
        )
        self.assertIsNone(
            PERF_TIMING_RE.search(text), "timing literal in the performance file"
        )

    def test_no_candidate_targets_a_reserved_stub_or_the_golden_roster(self) -> None:
        for lens, c in self.all_candidates():
            for a in c["anchors"]:
                self.assertFalse(
                    a["path"].startswith(RESERVED_PREFIXES), f"{lens}: {a['path']}"
                )
                self.assertFalse(
                    str(a.get("symbol", "")).startswith("parity_hash_"), c["title"]
                )
            self.assertFalse(
                re.search(
                    r"\b(delete|remove|drop|skip)\b[^.]{0,60}\btests?\b",
                    c["fixShape"],
                    re.I,
                )
                and c["claimKind"] != "prose-drift"
                and c.get("seedRef") is None,
                f"{lens}: a deletion fix-shape outside a ratified seed: {c['title']}",
            )
        positives = json.dumps(
            [p for env in self.env.values() for p in env["positives"]]
        )
        self.assertIn("parity_hash", positives)
        self.assertIn("mpi_wire", positives)
        self.assertTrue(
            any(stub.rstrip("/") in positives for stub in RESERVED_PREFIXES[1:]),
            "reserved stubs must appear under positives",
        )

    def test_attacker_log_records_every_dispatch_and_the_self_check(self) -> None:
        for w in self.vtb.__dict__.get("TB_WORKERS", []) or []:
            self.assertIn(f"| {w} | test-bloat |", self.log)
        rows = re.findall(
            r"^\| (tb-[a-z0-9-]+|sec-[a-z-]+) \| ([a-z-]+) \| (\d+) \| (valid|EXCLUDED) \| (\d) \|",
            self.log,
            re.M,
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(1 for r in rows if r[0].startswith("tb-")), 9)
        self.assertEqual(
            {r[1] for r in rows if r[0].startswith("sec-")},
            {"architecture", "over-engineering", "performance"},
        )
        for r in rows:
            self.assertLessEqual(int(r[4]), 1, "at most one re-dispatch per worker")
        for section in (
            "## Gaps",
            "## Seed ledger",
            "## Self-check",
            "git status --porcelain",
        ):
            self.assertIn(section, self.log)
        excluded = [
            w
            for env in self.env.values()
            for w, s in env["workers"].items()
            if s.startswith("excluded")
        ]
        for w in excluded:
            self.assertIn(f"| {w} | envelope-invalid", self.log)
        if not excluded:
            self.assertIn("No worker was excluded", self.log)

    def test_seed_ledger_is_complete(self) -> None:
        counts = self.seeds["counts"]
        self.assertEqual(counts["seedsIn"], len(self.seeds["seeds"]))
        self.assertEqual(counts["seedsIn"], counts["seedsAccounted"])
        by_ref = {}
        for c in self.env["test-bloat"]["candidates"]:
            if c.get("seedRef"):
                by_ref.setdefault(c["seedRef"], []).append(c["title"])
        for s in self.seeds["seeds"]:
            self.assertIn(s["disposition"], self.seeds["dispositionSet"], s["id"])
            if s["disposition"] in ("dropped", "dup-of"):
                self.assertTrue(s["dispositionReason"], s["id"])
            if s["disposition"] in ("confirmed", "sharpened"):
                self.assertIn(
                    s["candidateTitle"], by_ref.get(s["seedRef"], []), s["id"]
                )
        self.assertEqual(
            self.env["test-bloat"]["seedLedger"]["seedsIn"], counts["seedsIn"]
        )
        self.assertIn(
            "## Seed ledger (filled by the dispatcher after the merge)", self.prompt
        )

    def test_partition_holds_on_the_committed_prompt(self) -> None:
        ok, report = self.vtb.check_partition(self.prompt, self.t)
        self.assertTrue(ok, "\n".join(report))
        self.assertEqual(len(self.vtb.scope_rows(self.prompt)), 9)
        self.assertIn("PARTITION OK", self.log)

    def test_validator_rejects_the_three_malformed_shapes(self) -> None:
        import tempfile

        good = {
            "station": "test-corpus",
            "subStation": "tb-cobre-comm",
            "worker": "tb-cobre-comm",
            "baseline": self.baseline(),
            "lens": "test-bloat",
            "candidates": [
                {
                    "title": "a shared harness double is redeclared beside its canonical home",
                    "anchors": [
                        {
                            "path": "crates/cobre-comm/tests/local_conformance.rs",
                            "line": 198,
                        }
                    ],
                    "evidence": {
                        "command": "git grep -n StubComm 077dbe2c -- crates/cobre-comm",
                        "output": "",
                        "reading": "none",
                    },
                    "yardstickRef": "ta-5.2",
                    "claimKind": "target-gap",
                    "measurementDefinition": "n/a",
                    "measuredValue": None,
                    "proposedSeverity": "C",
                    "fixShape": "Re-home the double into the shared harness so both binaries consume one definition.",
                    "alignmentHint": "neutral",
                    "seedRef": None,
                    "dupOf": None,
                }
            ],
            "seedDispositions": [],
            "positives": [{"subject": "x", "why": "clean"}],
            "cleanVerdict": None,
            "_needsHuman": [],
        }
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "good.json").write_text(json.dumps(good))
            self.assertEqual(self.vtb.validate_file(p / "good.json"), [])
            bad = json.loads(json.dumps(good))
            bad["candidates"][0]["claimKind"] = "defect"
            (p / "kind.json").write_text(json.dumps(bad))
            self.assertTrue(
                any(
                    "$.candidates[0].claimKind" in e
                    for e in self.vtb.validate_file(p / "kind.json")
                )
            )
            bad = json.loads(json.dumps(good))
            del bad["candidates"][0]["measurementDefinition"]
            bad["candidates"][0]["title"] = (
                "cobre-comm has 6 source files with inline tests"
            )
            (p / "def.json").write_text(json.dumps(bad))
            self.assertTrue(
                any(
                    "$.candidates[0].measurementDefinition" in e
                    for e in self.vtb.validate_file(p / "def.json")
                )
            )
            (p / "prose.json").write_text("Here is the envelope:\n" + json.dumps(good))
            self.assertTrue(
                any(
                    "$ not parseable JSON" in e
                    for e in self.vtb.validate_file(p / "prose.json")
                )
            )


DISPOSITIONS = {
    "defended",
    "anchor-missing",
    "needs-human",
    "coverage-reduction",
    "dup-of",
    "re-raise",
}
STATES = {
    "accepted",
    "rejected-anchor",
    "rejected-defender",
    "dup-of",
    "proposal-adjudicated",
    "needs-human",
    "coverage-reduction",
    "re-raise",
}
BASES = {
    "sanctioned-seam",
    "premise-false-at-pin",
    "deliberate-and-documented",
    "contract",
    "cost-accepted-by-rule",
    "target-not-defect",
}
SHAPES = (
    "consolidation",
    "re-homing",
    "feature-surface unification",
    "cadence tiering",
)
PROBE_TITLE = "INGEST ANCHOR PROBE — test-corpus (2026-09, baseline)"
DELETION_RE = re.compile(
    r"\b(delet\w*|remov\w*|drop\w*|skip\w*|#\[ignore\])\b[^.;]{0,40}\b(tests?|assertions?)\b(?![/_])",
    re.I,
)
NEGATED_RE = re.compile(
    r"\b(no|none|nothing|never|not|without|instead of|rather than|inadmissible|removes none)\b",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    return len(ta & tb) / max(1, len(ta | tb))


class IngestTests(TestCorpusCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vtb = load_validate_tb()
        cls.verdicts_doc = sc.load_json(cls.station_dir() / "verdicts.json")
        cls.verdicts = cls.verdicts_doc["verdicts"]
        cls.log = (cls.station_dir() / "ingest-log.md").read_text(encoding="utf-8")
        cls.probe = (cls.station_dir() / "anchor-probe.md").read_text(encoding="utf-8")
        cls.brief = (cls.station_dir() / "defender-prompt.md").read_text(
            encoding="utf-8"
        )
        cls.cands = {}
        for lens in LENSES:
            doc = sc.load_json(cls.station_dir() / f"candidates-{lens}.json")
            for i, c in enumerate(doc["candidates"]):
                cls.cands[f"{lens}-{i:02d}"] = dict(c, lens=lens)
        cls.t = cls.tree()
        cls.register = backlog_parse.read_register(sc.BACKLOG)
        cls.entry_ids = {
            m.group("id")
            for ln in cls.register
            if (m := backlog_parse.ENTRY_RE.match(ln))
        }

    def anchor_str(self, a: dict) -> str:
        return (
            f"`{a['path']}::{a['symbol']}`"
            if a.get("symbol")
            else f"`{a['path']}:{a['line']}`"
        )

    def test_verdicts_cover_every_candidate_exactly_once_and_counts_sum(self) -> None:
        self.assertEqual(set(self.verdicts), set(self.cands))
        counts = self.verdicts_doc["counts"]
        self.assertEqual(counts["received"], len(self.cands))
        self.assertEqual(self.verdicts_doc["baseline"], self.baseline())
        self.assertEqual(
            counts["received"],
            counts["anchorRejected"]
            + counts["definitionFlipDowngraded"]
            + counts["coverageReductionDismissed"]
            + counts["dupOf"]
            + counts["reRaiseRejected"]
            + counts["defended"],
        )
        self.assertEqual(sum(counts["byState"].values()), counts["received"])
        self.assertEqual(sum(counts["byDisposition"].values()), counts["received"])
        self.assertEqual(
            counts["needsHuman"],
            sum(1 for v in self.verdicts.values() if v["_needsHuman"]),
        )
        self.assertEqual(counts["unresolved"], 0)
        for ref, v in self.verdicts.items():
            self.assertEqual(v["candidateRef"], ref)
            self.assertEqual(v["title"], self.cands[ref]["title"])

    def test_disposition_and_state_vocabularies_agree(self) -> None:
        for ref, v in self.verdicts.items():
            self.assertIn(v["disposition"], DISPOSITIONS, ref)
            self.assertIn(v["state"], STATES, ref)
            d, s = v["disposition"], v["state"]
            if d == "defended":
                self.assertIn(
                    s,
                    {
                        "accepted",
                        "rejected-defender",
                        "proposal-adjudicated",
                        "needs-human",
                    },
                    ref,
                )
                self.assertIn(v["verdict"], {"confirmed", "dismissed", None}, ref)
            elif d == "dup-of":
                self.assertEqual(s, "dup-of", ref)
                self.assertIsNone(v["verdict"], ref)
            elif d == "anchor-missing":
                self.assertEqual(s, "rejected-anchor", ref)
            elif d == "coverage-reduction":
                self.assertEqual(s, "coverage-reduction", ref)
                self.assertIn("testing-architecture.md:602-604", v["dismissedBy"], ref)
                self.assertIn(".claude/rules/testing.md:113", v["dismissedBy"], ref)
            elif d == "needs-human":
                self.assertEqual(s, "needs-human", ref)
                self.assertIsNone(v["proposedSeverity"], ref)
            else:
                self.assertEqual(s, d, ref)

    def test_every_defended_verdict_is_adjudicated_and_well_formed(self) -> None:
        for ref, v in self.verdicts.items():
            if v["disposition"] != "defended" or v["verdict"] is None:
                continue
            self.assertIn(v["claimKind"], self.vtb.CLAIM_KINDS, ref)
            self.assertIsInstance(v["targetNotDefect"], bool, ref)
            self.assertRegex(v["yardstickRef"], r"^ta-\d(\.\d+)?$", ref)
            self.assertIn(v["measurementDefinition"], self.vtb.DEFINITIONS, ref)
            self.assertGreaterEqual(len(v["argument"]), 120, ref)
            self.assertIn(
                v["alignmentHint"],
                {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"},
                ref,
            )
            self.assertEqual(v["conflicts"], v["alignmentHint"] == "conflicts", ref)
            self.assertEqual(
                v["measurement"],
                "UNMEASURED" if v["lens"] == "performance" else "n/a",
                ref,
            )
            if v["verdict"] == "confirmed":
                scl = v["survivingClaim"]
                self.assertTrue(scl and len(scl) >= 40, ref)
                self.assertNotEqual(_norm(scl), _norm(v["title"]), ref)
                self.assertLess(_jaccard(scl, v["title"]), 0.85, ref)
                self.assertTrue(
                    any(
                        str(v["coverageNeutralShape"]).lower().startswith(s)
                        for s in SHAPES
                    ),
                    ref,
                )
                self.assertFalse(v["targetNotDefect"], ref)
                self.assertIsNone(v["dismissalBasis"], ref)
            else:
                self.assertIn(v["dismissalBasis"], BASES, ref)
                self.assertTrue(v["basisCitation"], ref)
                self.assertIsNone(v["survivingClaim"], ref)
                self.assertEqual(
                    v["targetNotDefect"],
                    v["dismissalBasis"] == "target-not-defect",
                    ref,
                )
                if v["dismissalBasis"] == "target-not-defect":
                    self.assertEqual(v["claimKind"], "target-gap", ref)
                if v["dismissalBasis"] == "sanctioned-seam":
                    self.assertIn(
                        v["sanctionedBy"],
                        self.verdicts_doc["sanctionedByClosedSet"],
                        ref,
                    )
            if self.cands[ref].get("seedRef"):
                self.assertEqual(v["seedRef"], self.cands[ref]["seedRef"], ref)
        self.assertIn("targetNotDefect", self.brief)
        self.assertIn("claimKind", self.brief)

    def test_accepted_anchors_resolve_at_the_pin(self) -> None:
        checked = 0
        for ref, v in self.verdicts.items():
            if v["state"] in {"accepted", "proposal-adjudicated"}:
                for a in v["anchors"]:
                    self.assertTrue(
                        sc.anchor_exists(self.anchor_str(a), self.t), f"{ref}: {a}"
                    )
                    checked += 1
        self.assertGreater(checked, 0)

    def test_dup_of_verdicts_name_the_mirror_item_and_a_resolving_sharpened_anchor(
        self,
    ) -> None:
        dups = {
            ref: v for ref, v in self.verdicts.items() if v["disposition"] == "dup-of"
        }
        self.assertTrue(dups)
        for ref, v in dups.items():
            self.assertIn(v["mirrorRef"], self.vtb.MIRROR_ITEMS, ref)
            anchors = sc.anchors_in(v["sharpenedAnchor"])
            self.assertTrue(anchors, ref)
            for a in anchors:
                self.assertTrue(sc.anchor_exists(a, self.t), f"{ref}: {a}")
            for id_ in v["relatedRegisterIds"]:
                self.assertIn(id_, self.entry_ids, ref)
            if not v["relatedRegisterIds"]:
                self.assertIn("mirror-only", self.log)
        blob = json.dumps(self.verdicts_doc)
        self.assertIsNone(
            re.search(r"\bTD-0(7[4-9]|[89]\d)\b", blob), "no fresh TD id may appear"
        )

    def test_mandatory_proposal_adjudications_recorded(self) -> None:
        mandatory = set(self.verdicts_doc["mandatoryAdjudications"])
        self.assertEqual(
            mandatory, {"test-bloat-30", "test-bloat-43", "architecture-00"}
        )
        for ref in mandatory:
            v = self.verdicts[ref]
            self.assertEqual(v["state"], "proposal-adjudicated", ref)
            a = v["adjudication"]
            self.assertIn(a["side"], {"proposal", "tree"}, ref)
            self.assertEqual(
                a["side"], "tree" if v["verdict"] == "confirmed" else "proposal", ref
            )
            self.assertRegex(a["yardstickRef"], r"^ta-\d(\.\d+)?$", ref)
            self.assertIn(a["claimKind"], self.vtb.CLAIM_KINDS, ref)
            self.assertIsInstance(a["targetNotDefect"], bool, ref)
            if v["verdict"] == "confirmed":
                self.assertEqual(v["claimKind"], "prose-drift", ref)
            else:
                self.assertEqual(v["dismissalBasis"], "target-not-defect", ref)
                self.assertTrue(v["targetNotDefect"], ref)
        self.assertEqual(self.verdicts_doc["counts"]["proposalAdjudicated"], 3)

    def test_ingest_log_names_the_pin_and_carries_one_roster_row_per_candidate(
        self,
    ) -> None:
        self.assertIn(f"Baseline: `{self.baseline()}`", self.log)
        for section in (
            "## Candidate census",
            "## Anchor screen",
            "### Anchor rejections",
            "## Definition screen",
            "### Definition-flip downgrades",
            "## Positives (coverage-reduction dismissals)",
            "## Prior register, do-not-touch and re-raise screen",
            "### Dup-of merges (no fresh TD id)",
            "### Re-raise and dup-of rejections",
            "## Defender summary",
            "## Per-candidate roster",
            "## Reconciliation",
        ):
            self.assertIn(section, self.log, section)
        roster = self.log.split("## Per-candidate roster", 1)[1]
        for ref in self.cands:
            self.assertEqual(
                len(re.findall(rf"^\| {re.escape(ref)} \|", roster, re.M)), 1, ref
            )

    def test_anchor_probe_resolves_with_the_harness_checker(self) -> None:
        self.assertEqual(self.probe.count("· probe ·"), len(self.cands))
        r = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "check-anchors.py"),
                PROBE_TITLE,
                "--register",
                str(self.station_dir() / "anchor-probe.md"),
                "--baseline",
                self.baseline(),
                "--json",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
        report = json.loads(r.stdout[r.stdout.index("{") :])
        self.assertEqual(report["failures"], [])
        self.assertGreaterEqual(report["checked"], len(self.cands))
        # the three anchor shapes the ticket names: an integration-test path plus a free function, the shared
        # harness plus its Rank0Of2 declaration line, and a workflow path plus a line. The ticket's example
        # instance for the first shape (extensive_form_oracle.rs::close) is not a candidate anchor at the pin —
        # the oracle item arrived as a dup-of whose sharpened anchor lives in verdicts.json — so the shape is
        # asserted, not the instance.
        self.assertRegex(
            self.probe, r"`crates/cobre-[a-z]+/tests/[a-z0-9_]+\.rs::[a-z0-9_]+`"
        )
        for needle in (
            "`crates/cobre-sddp/tests/common/mod.rs:32`",
            "`.github/workflows/ci.yml:32`",
        ):
            self.assertIn(needle, self.probe, needle)

    def test_coverage_screen_kept_no_test_deleting_fix_shape(self) -> None:
        """Independent oracle: an un-negated 'delete … test' fix shape must not be defended.

        The one sanctioned exception is lens-rules.md § Rule 2 "Boundary": a ratified duplicate-test seed whose
        fold removes exact-duplicate assertions is defended only as a `consolidation` shape that keeps every
        assertion and hands the nextest-count change to the owner gate through `_needsHuman`.
        """
        for ref, v in self.verdicts.items():
            if v["disposition"] != "defended":
                continue
            fix = self.cands[ref]["fixShape"]
            for m in DELETION_RE.finditer(fix):
                sentence_start = (
                    max(fix.rfind(". ", 0, m.start()), fix.rfind("; ", 0, m.start()))
                    + 1
                )
                sentence_end = min(
                    e
                    for e in (
                        fix.find(". ", m.end()),
                        fix.find("; ", m.end()),
                        len(fix),
                    )
                    if e != -1
                )
                sentence = fix[sentence_start:sentence_end]
                if NEGATED_RE.search(sentence):
                    continue
                self.assertIsNotNone(
                    v.get("seedRef"),
                    f"{ref}: un-negated deletion shape outside the ratified seed class: {sentence}",
                )
                self.assertEqual(v["verdict"], "confirmed", ref)
                self.assertTrue(
                    v["coverageNeutralShape"].startswith("consolidation"),
                    f"{ref}: {v['coverageNeutralShape'][:80]}",
                )
                self.assertTrue(
                    any(
                        re.search(r"nextest list|test count", item)
                        for item in v["_needsHuman"]
                    ),
                    f"{ref}: the fold's nextest-count change must be handed to the owner gate",
                )
        self.assertIn("Regex hits:", self.log)

    def test_definition_screen_rows_and_the_majority_claim(self) -> None:
        keys = self.vtb.inventory_keys()
        section = self.log.split("## Definition screen", 1)[1].split("## Positives", 1)[
            0
        ]
        for ref, v in self.verdicts.items():
            mv = v["measuredValue"]
            if (
                mv
                and mv["key"].startswith(("figures.int-binaries", "perCrate."))
                and ("int-binaries" in mv["key"] or "integration" in mv["key"])
            ):
                self.assertIn(f"| {ref} |", section, ref)
        b = (
            keys["figures.int-binaries-solver-linking.perCrate.cobre-sddp"]
            / keys["figures.int-binaries-solver-linking.value"]
        )
        f = keys["figures.int-binaries.altDefinition.perCrate.cobre-sddp"] / sum(
            keys[f"figures.int-binaries.altDefinition.perCrate.{c}"]
            for c in ("cobre-sddp", "cobre-cli", "cobre-solver")
        )
        self.assertEqual(b > 0.5, f > 0.5)
        self.assertNotIn(
            "performance-01 | 40 | 56",
            self.log.split("### Definition-flip downgrades", 1)[1].split(
                "## Positives", 1
            )[0],
        )
        self.assertEqual(
            self.verdicts_doc["counts"]["definitionFlipDowngraded"],
            sum(
                1
                for v in self.verdicts.values()
                if v["disposition"] == "needs-human" and v["defender"] is None
            ),
        )


CAL_SECTION_TITLE = "STATION 8 — test corpus and test-support (2026-09)"
CAL_SCAFFOLD = "★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus"
FINDING_ID_RE = re.compile(r"^(CD|PD|OD|TD)-\d{3}$")
HEADING_ID_RE = re.compile(r"^\*\*((?:CD|PD|OD|TD)-\d{3}) · Sev ([ABC]) ·", re.M)
CAL_LENS_ORDER = ("test-bloat", "architecture", "over-engineering", "performance")
CAL_CLASS_BY_LENS = {
    "test-bloat": "TD",
    "architecture": "CD",
    "over-engineering": "OD",
    "performance": "PD",
}
NEUTRAL_KINDS = {
    "consolidation",
    "re-homing",
    "feature-surface-unification",
    "cadence-tiering",
}
ALIGNMENT_VOCAB = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
SEED_DISPOSITIONS = {
    "delta-confirmed",
    "delta-dismissed",
    "dup-of",
    "stale-at-pin",
    "fix-shape-refused",
}
CAL_HEADINGS = (
    "#### Seed dispositions (owned prior TD ids — reused, never re-minted)",
    "#### Re-measured figures",
    "#### Merged into existing entries (dup-of)",
    "#### Positives (recorded so the report is not a defect-only list)",
    "#### ↩︎ Cleared (dismissed or sanctioned — do not re-raise)",
    "#### Phase-0a gate substrate (informational)",
    "#### Queued out",
    "#### Owner gate — decisions",
    "#### Findings by lens",
    "#### Test-suite bloat (TD)",
    "#### Architecture (CD)",
    "#### Over-engineering (OD)",
    "#### Performance (PD)",
)
_STATION_ORDER = [
    "core-io",
    "stochastic",
    "solver-comm",
    "sddp",
    "cli-python",
    "build-ci",
    "test-corpus",
]


def register_before_station(register: str, own_section: str, slug: str) -> str:
    """The register as it stood when this station minted: its own section and every LATER crate-station section removed."""
    prior = register.replace(own_section, "")
    for later in _STATION_ORDER[_STATION_ORDER.index(slug) + 1 :]:
        marker = f"## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — {later}"
        if marker in prior:
            block = prior.split(marker, 1)[1].split("\n## ", 1)[0]
            prior = prior.replace(marker + block, "")
    return prior


def inventory_lookup(inv: dict, key: str):
    """Resolve a flattened inventory key (`figures.<id>.perCrate.<crate>`): lists keyed by id / crate, dicts by key."""
    node = inv
    for part in key.split("."):
        if isinstance(node, list):
            node = next(x for x in node if (x.get("id") or x.get("crate")) == part)
        else:
            node = node[part]
    return node


def _sentence_hits(text: str) -> list[tuple[str, bool]]:
    out = []
    for m in DELETION_RE.finditer(text):
        start = max(text.rfind(". ", 0, m.start()), text.rfind("; ", 0, m.start())) + 1
        ends = [
            e
            for e in (text.find(". ", m.end()), text.find("; ", m.end()), len(text))
            if e != -1
        ]
        sentence = text[start : min(ends)]
        out.append((sentence, bool(NEGATED_RE.search(sentence))))
    return out


class CalibrationTests(TestCorpusCase):
    """E08-5: id assignment, the claimKind and coverage gates, seed folds, the section, gate-substrate.json."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.assigned = cls.cal["assigned"]
        cls.by_ref = {r["candidateRef"]: r for r in cls.assigned}
        cls.gs = sc.load_json(cls.station_dir() / "gate-substrate.json")
        cls.verdicts = sc.load_json(cls.station_dir() / "verdicts.json")["verdicts"]
        cls.seeds = sc.load_json(cls.station_dir() / "seeds.json")["seeds"]
        cls.inventory = sc.load_json(cls.station_dir() / "inventory.json")
        cls.register = sc.BACKLOG.read_text(encoding="utf-8")
        start = cls.register.index("\n### " + CAL_SECTION_TITLE + "\n")
        end = cls.register.find("\n## ", start + 1)
        cls.section = cls.register[start:end]
        cls.t = cls.tree()

    def block(self, heading: str) -> str:
        start = self.section.index("\n" + heading + "\n")
        nxt = self.section.find("\n#### ", start + 1)
        return self.section[start : nxt if nxt > 0 else len(self.section)]

    def run_tool(
        self, tool: str, title: str, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(sc.TOOLS / tool), *extra, title],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
        )

    def test_ids_well_formed_contiguous_unique_and_fresh(self) -> None:
        self.assertTrue(self.assigned)
        by_class: dict[str, list[int]] = {}
        for r in self.assigned:
            self.assertRegex(r["id"], FINDING_ID_RE)
            self.assertEqual(r["class"], CAL_CLASS_BY_LENS[r["lens"]])
            self.assertNotEqual(
                r["class"],
                "PD",
                "every performance verdict was dismissed; no PD may be minted",
            )
            by_class.setdefault(r["class"], []).append(int(r["id"].split("-")[1]))
        for cls_, nums in by_class.items():
            floor = self.cal["idFloorsSeen"][cls_]
            self.assertEqual(nums, list(range(floor, floor + len(nums))), cls_)
        self.assertGreaterEqual(
            self.cal["idFloorsSeen"]["TD"],
            74,
            "TD-001 is a floor; the crate stations already minted TD-001..073",
        )
        headings = HEADING_ID_RE.findall(self.register)
        ids = [h[0] for h in headings]
        self.assertEqual(
            len(ids), len(set(ids)), "duplicate finding heading in the register"
        )
        section_ids = [h[0] for h in HEADING_ID_RE.findall(self.section)]
        self.assertEqual(section_ids, [r["id"] for r in self.assigned])
        prior = register_before_station(self.register, self.section, "test-corpus")
        for r in self.assigned:
            self.assertNotRegex(prior, rf"\b{r['id']}\b", r["id"])

    def test_assignment_order_is_lens_yardstick_candidate(self) -> None:
        keys = [
            (CAL_LENS_ORDER.index(r["lens"]), r["yardstickRef"], r["candidateRef"])
            for r in self.assigned
        ]
        self.assertEqual(keys, sorted(keys))
        for r in self.assigned:
            v = self.verdicts[r["candidateRef"]]
            self.assertEqual(v["verdict"], "confirmed")
            self.assertEqual(v["disposition"], "defended")
            self.assertNotEqual(_norm(r["survivingClaim"]), _norm(v["title"]))
            self.assertEqual(r["survivingClaim"], v["survivingClaim"])

    def test_claimkind_gate_and_downgrades(self) -> None:
        for r in self.assigned:
            self.assertIn(r["severity"], "ABC")
            self.assertIn(r["claimKind"], ("tree-fact", "prose-drift"))
            if r["claimKind"] != "tree-fact":
                self.assertEqual(r["severity"], "C", r["id"])
                self.assertEqual(
                    r["anchors"][0]["path"],
                    YARDSTICK,
                    "a prose-drift row anchors on the drifting sentence first",
                )
                self.assertIn("- **Drift (doc ↔ tree):**", self.entry(r["id"]))
            if r["reviewerRating"] != r["severity"]:
                self.assertTrue(r["downgradeReason"], r["id"])
                self.assertIn("- **Reviewer rating:**", self.entry(r["id"]))
        self.assertEqual(
            self.cal["counts"]["downgrades"],
            sum(1 for r in self.assigned if r["downgradeReason"]),
        )
        self.assertEqual(
            [
                t["candidateRef"]
                for t in self.cal["targetGaps"]
                if t["candidateRef"] in self.by_ref
            ],
            [],
        )
        for t in self.cal["targetGaps"]:
            self.assertIsNone(t["severity"])
            self.assertFalse(t["actionable"])
            self.assertTrue(self.verdicts[t["candidateRef"]]["targetNotDefect"])

    def entry(self, fid: str) -> str:
        start = self.section.index(f"\n**{fid} · Sev ")
        rest = self.section[start + 1 :]
        m = re.search(r"\n(?:\*\*(?:CD|PD|OD|TD)-\d{3} · Sev |#### )", rest)
        return rest[: m.start()] if m else rest

    def test_seeded_confirmations_fold_and_every_seed_is_dispositioned_once(
        self,
    ) -> None:
        seed_ids = [s["id"] for s in self.seeds]
        rows = self.cal["seeds"]
        self.assertEqual([r["seedId"] for r in rows], seed_ids)
        for r in rows:
            self.assertIn(r["stationDisposition"], SEED_DISPOSITIONS)
            self.assertIsNone(r["mintedId"])
            if r["stationDisposition"] == "delta-confirmed":
                self.assertNotIn(
                    r["candidateRef"],
                    self.by_ref,
                    "a seeded confirmation folds onto its TD id and mints nothing",
                )
                self.assertIn(r["fixShapeKind"], NEUTRAL_KINDS)
            if r["stationDisposition"] == "stale-at-pin":
                self.assertTrue(r["retireProposed"])
        for r in self.assigned:
            self.assertIsNone(self.verdicts[r["candidateRef"]].get("seedRef"), r["id"])
        table = self.block(
            "#### Seed dispositions (owned prior TD ids — reused, never re-minted)"
        )
        table_ids = re.findall(r"^\| (TD-\d{3}) \| ", table, re.M)
        self.assertEqual(table_ids, seed_ids)
        self.assertEqual(
            len(self.cal["retireProposed"]), self.cal["counts"]["retireProposed"]
        )
        self.assertEqual(len(self.cal["fixShapeRefused"]), 4)
        for r in self.cal["fixShapeRefused"]:
            self.assertIn("Cost discipline", r["sanctionedBy"])

    def test_alignment_vocabulary_citation_and_no_conflicts(self) -> None:
        for r in self.assigned:
            self.assertIn(r["alignmentHint"], ALIGNMENT_VOCAB)
            self.assertRegex(r["alignmentCites"], r"Part (IV|V) §")
            self.assertFalse(r["conflicts"])
            body = self.entry(r["id"])
            self.assertIn(f"- **Alignment:** {r['alignmentHint']} (provisional;", body)
            self.assertNotIn("HELD", body)
        self.assertEqual(self.cal["counts"]["conflicts"], 0)

    def test_harness_checkers_exit_zero_over_both_titles(self) -> None:
        for title in (CAL_SECTION_TITLE, CAL_SCAFFOLD):
            for tool, extra in (
                ("check-anchors.py", ()),
                ("check-reraise.py", ()),
                ("fields-check.py", ("--require", "Alignment")),
            ):
                r = self.run_tool(tool, title, *extra)
                self.assertEqual(
                    r.returncode, 0, f"{tool} {title}: {(r.stdout + r.stderr)[-1200:]}"
                )

    def test_fix_shapes_are_coverage_neutral(self) -> None:
        for r in self.assigned:
            self.assertIn(r["fixShapeKind"], NEUTRAL_KINDS, r["id"])
            if r["fixShapeKind"] == "cadence-tiering":
                self.assertTrue(
                    r["targetTier"],
                    f"{r['id']}: cadence tiering must name the tier tests move INTO",
                )
                self.assertIn("target tier:", self.entry(r["id"]))
            self.assertEqual(
                [h for h in r["deletionVerbHits"] if not h["negated"]], [], r["id"]
            )
        fix_lines = [
            ln for ln in self.section.splitlines() if ln.startswith("- **Fix-shape:**")
        ]
        self.assertEqual(len(fix_lines), len(self.assigned))
        for ln in fix_lines:
            self.assertRegex(
                ln,
                r"\(shape: (consolidation|re-homing|feature-surface-unification|cadence-tiering)",
            )
            self.assertEqual(
                [s for s, negated in _sentence_hits(ln) if not negated], [], ln[:120]
            )
        positives = self.block(
            "#### Positives (recorded so the report is not a defect-only list)"
        )
        for r in self.cal["fixShapeRefused"]:
            self.assertIn(f"- **{r['seedId']}**", positives)

    def test_measurement_definitions_resolve_in_inventory(self) -> None:
        for r in self.assigned:
            body = self.entry(r["id"])
            self.assertIn("- **Measurement:**", body)
            self.assertIn("- **Claim kind:**", body)
            if r["measurementKey"]:
                self.assertIn(r["measurementDefinition"], ("binary", "file"))
                self.assertEqual(
                    inventory_lookup(self.inventory, r["measurementKey"]),
                    r["measuredValue"],
                )
                self.assertIn(f"(definition: {r['measurementDefinition']}", body)
                self.assertTrue(r["measurementCommand"])
            else:
                self.assertEqual(r["measurementDefinition"], "n/a")
        self.assertEqual(self.cal["counts"]["definitionFlipNeedsHuman"], 0)
        self.assertEqual(self.cal["needsHuman"], [])

    def test_gate_substrate_is_informational_and_resolves_at_the_pin(self) -> None:
        gs = self.gs
        self.assertTrue(gs["informational"])
        self.assertEqual(
            (gs["findings"], gs["ids"], gs["proposedChanges"]), ([], [], [])
        )
        self.assertEqual(gs["baseline"], self.baseline())

        def line(anchor: str) -> str:
            path, ln = anchor.rsplit(":", 1)
            lines = self.t.read_text(path).splitlines()
            self.assertLessEqual(int(ln), len(lines), anchor)
            return lines[int(ln) - 1]

        roster = gs["goldenRoster"]
        self.assertEqual(roster["cases"], ["D06", "D15", "D30", "D34", "D41"])
        self.assertEqual(
            [b["module"] for b in roster["backends"]],
            ["parity_hash_highs", "parity_hash_clp"],
        )
        for b in roster["backends"]:
            self.assertIn(f"mod {b['module']}", line(b["anchor"]))
            self.assertIn("run_golden_case(", line(b["runGoldenCaseCall"]))
            self.assertEqual(sorted(b["caseTests"]), roster["cases"])
            for case, anchor in b["caseTests"].items():
                self.assertIn(f"fn parity_hash_{case.lower()}", line(anchor))
            self.assertEqual(
                b["caseTestsGated"],
                5,
                "each of the five golden case tests is slow-tests gated",
            )
            self.assertGreaterEqual(b["slowTestsGatedFns"], 5)
        for name, ln in roster["helpers"].items():
            if ln is not None:
                self.assertIn(f"fn {name}", line(f"{roster['helperOwner']}:{ln}"))
        self.assertEqual(roster["helpersAbsentAtPin"], [])
        self.assertEqual(
            {
                k: roster["helpers"][k]
                for k in (
                    "compute_parity_hash",
                    "run_golden_case",
                    "regen_golden_case",
                    "assert_permutation_hash",
                )
            },
            {
                "compute_parity_hash": 48,
                "run_golden_case": 375,
                "regen_golden_case": 390,
                "assert_permutation_hash": 406,
            },
            "the ticket's helper anchors resolve at their stated lines at the pin",
        )
        gates = gs["determinismGates"]
        for m in gates["modules"]:
            self.assertIn(f"mod {m['name']} {{", line(m["anchor"]))
        self.assertTrue(
            set(gates["ticketNamedModules"]) <= {m["name"] for m in gates["modules"]}
        )
        for d in gs["baselineDirs"]:
            files = [f for f in self.t.ls_files(d["path"]) if f.endswith(".sha256")]
            self.assertEqual(len(files), 5, d["path"])
            self.assertEqual(
                sorted(pathlib.Path(f).stem for f in files), roster["cases"]
            )
        ev = roster["gating"]["ciEvidence"]
        self.assertIn("slow-tests", line(ev[0].split(" ")[0]))
        self.assertIn("cargo test --workspace", line(ev[1].split(" ")[0]))
        self.assertTrue(roster["gating"]["inCi"])

    def test_section_structure_and_placeholder_replaced(self) -> None:
        self.assertEqual(self.register.count("\n### " + CAL_SECTION_TITLE + "\n"), 1)
        self.assertIn(
            "## " + CAL_SCAFFOLD + "\n\n### " + CAL_SECTION_TITLE + "\n", self.register
        )
        self.assertNotIn("_(no entries yet)_", self.section)
        positions = [self.section.index("\n" + h + "\n") for h in CAL_HEADINGS]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "_No confirmed finding under this lens at the baseline",
            self.block("#### Performance (PD)"),
        )
        self.assertIn(
            "_(pending — filled by the gate ticket)_",
            self.block("#### Owner gate — decisions"),
        )
        self.assertIn(f"- **Baseline:** `{self.baseline()}`", self.section)

    def test_re_measured_figures_table_covers_the_inventory(self) -> None:
        table = self.block("#### Re-measured figures").replace("\\|", "|")
        for f in self.inventory["figures"]:
            row = next(
                (ln for ln in table.splitlines() if f"(`{f['id']}`)" in ln), None
            )
            self.assertIsNotNone(row, f["id"])
            assert row is not None
            self.assertIn(f"| {f['definition']} |", row)
            self.assertIn(f["command"][:60].replace("\\|", "|"), row)

    def test_dup_of_merges_name_mirror_items_without_td_ids(self) -> None:
        block = self.block("#### Merged into existing entries (dup-of)")
        self.assertEqual(len(self.cal["dupOf"]), 3)
        for d in self.cal["dupOf"]:
            self.assertIn(
                f"- **{d['mirrorRef']}** (`{d['mirrorAnchor']}`) ← {d['candidateRef']}",
                block,
            )
            self.assertNotIn(d["candidateRef"], self.by_ref)
            for anchor in sc.anchors_in(d["sharpenedAnchor"]):
                self.assertTrue(sc.anchor_exists(anchor, self.t), anchor)
        self.assertNotRegex(block, r"\bTD-0(7[4-9]|[89]\d)\b")
        self.assertEqual(self.cal["counts"]["dupOf"], 3)

    def test_cleared_lists_every_dismissal(self) -> None:
        cleared = self.block(
            "#### ↩︎ Cleared (dismissed or sanctioned — do not re-raise)"
        )
        dismissed = [
            ref for ref, v in self.verdicts.items() if v["verdict"] == "dismissed"
        ]
        self.assertEqual(
            sorted(d["candidateRef"] for d in self.cal["cleared"]), sorted(dismissed)
        )
        for ref in dismissed:
            self.assertIn(f"- **{ref}**", cleared)
            self.assertNotIn(ref, self.by_ref)
        self.assertEqual(
            len(self.cal["targetGaps"]),
            sum(1 for v in self.verdicts.values() if v["claimKind"] == "target-gap"),
        )
        for r in self.cal["retireProposed"]:
            self.assertIn(f"- **{r['seedId']}**", cleared)

    def test_counts_sum(self) -> None:
        c = self.cal["counts"]
        self.assertEqual(c["received"], len(self.verdicts))
        self.assertEqual(c["confirmed"], c["minted"] + c["foldedOntoSeeds"])
        self.assertEqual(c["received"], c["confirmed"] + c["dismissed"] + c["dupOf"])
        self.assertEqual(c["minted"], len(self.assigned))
        self.assertEqual(c["seeds"], 72)
        self.assertEqual(sum(c["seedsByDisposition"].values()), 72)
        self.assertEqual(
            c["byClass"],
            {
                k: v
                for k, v in dict(
                    (cls_, sum(1 for r in self.assigned if r["class"] == cls_))
                    for cls_ in ("TD", "CD", "OD")
                ).items()
                if v
            },
        )


VERIFY_MARKER = "## Station-specific checks — test-corpus"
VERIFY_ROWS = (
    "check-anchors",
    "check-reraise",
    "fields-check",
    "fields-check-alignment",
    "figure-completeness",
    "both-definitions-ran",
    "slow-tests-census-tracked",
    "entry-definition-claimkind-provenance",
    "mirror-items-dup-of-only",
    "read-only-workspace",
)


def load_station_verify():
    spec = importlib.util.spec_from_file_location(
        "station_verify", sc.TOOLS / "station_verify.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SectionVerifyTests(TestCorpusCase):
    """E08-6: executable proof of the station verification.

    verify-station.sh runs this module, so nothing here may invoke it (recursion); the station
    driver verify-test-corpus.py is run once with --no-shared for the same reason. Every recount is
    taken from the tree at the station baseline and compared with inventory.json; the shared
    verifier's new count-census branch is proven on a tampered copy.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.driver = cls.station_dir() / "verify-test-corpus.py"
        cls.driver_run = subprocess.run(
            [sys.executable, str(cls.driver), CAL_SCAFFOLD, "--no-shared"],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.t = cls.tree()
        cls.sv = load_station_verify()

    def test_harness_checkers_exit_zero_over_the_scaffold_title(self) -> None:
        for tool in ("check-anchors", "check-reraise", "fields-check"):
            r = subprocess.run(
                [sys.executable, str(sc.TOOLS / f"{tool}.py"), CAL_SCAFFOLD],
                cwd=sc.REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, f"{tool}: {(r.stdout + r.stderr)[-800:]}")

    def test_no_tracked_modification_outside_the_station(self) -> None:
        station = "plans/architecture-debt-audit/stations/test-corpus/"
        self.assertEqual(
            [
                m
                for m in sc.tracked_modifications()
                if not m.split()[-1].startswith(station)
            ],
            [],
        )

    def test_remeasured_census_equals_a_fresh_recount_at_the_pin(self) -> None:
        rows = {r["crate"]: r for r in self.inv["perCrate"]}
        for crate, rec in rows.items():
            tests = self.t.ls_files(f"crates/{crate}/tests/")
            rs = [f for f in tests if f.endswith(".rs")]
            self.assertEqual(rec["integrationFiles"], len(rs), crate)
            self.assertEqual(
                rec["integrationBinaries"],
                sum(1 for f in rs if f.count("/") == 3),
                crate,
            )
            siblings = [
                f
                for f in self.t.ls_files(f"crates/{crate}/src/")
                if f.endswith("tests.rs")
            ]
            self.assertEqual(rec["siblingTestsRs"], len(siblings), crate)
        figs = {f["id"]: f for f in self.inv["figures"]}
        self.assertEqual(
            figs["int-binaries"]["value"],
            sum(r["integrationBinaries"] for r in rows.values()),
        )
        self.assertEqual(
            figs["int-binaries"]["altDefinition"]["integrationFiles"],
            sum(r["integrationFiles"] for r in rows.values()),
        )
        to_bits = [
            f
            for f in self.t.ls_files("crates")
            if f.endswith(".rs") and "to_bits" in self.t.read_text(f)
        ]
        self.assertEqual(figs["to-bits"]["value"], len(to_bits))
        self.assertNotEqual(
            rows["cobre-sddp"]["integrationBinaries"],
            rows["cobre-sddp"]["integrationFiles"],
        )
        self.assertNotEqual(
            rows["cobre-io"]["integrationBinaries"],
            rows["cobre-io"]["integrationFiles"],
        )

    def test_station_verifier_passes_without_the_shared_run(self) -> None:
        out = self.driver_run.stdout + self.driver_run.stderr
        self.assertEqual(self.driver_run.returncode, 0, out[-2500:])
        self.assertIn(VERIFY_MARKER, out)
        for row in VERIFY_ROWS:
            self.assertIsNotNone(
                re.search(
                    rf"^\| \d+ \| {re.escape(row)} \| .* \| 0 \| PASS \|$", out, re.M
                ),
                row,
            )
        self.assertNotIn("| FAIL |", out)
        self.assertIn("Station-specific result: PASS (10/10 checks).", out)

    def test_station_verifier_refuses_an_ambiguous_or_absent_title(self) -> None:
        r = subprocess.run(
            [sys.executable, str(self.driver), "no such heading", "--no-shared"],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("matched 0 headings", r.stderr)

    def test_report_carries_the_station_block_when_rendered(self) -> None:
        report = self.station_dir() / "verification.md"
        if not report.exists() or VERIFY_MARKER not in report.read_text(
            encoding="utf-8"
        ):
            raise unittest.SkipTest(
                "verification.md not yet rendered by a full run (bootstrap inside verify-station.sh)"
            )
        text = report.read_text(encoding="utf-8")
        self.assertIn("# Station verification — test-corpus", text)
        self.assertEqual(text.count(VERIFY_MARKER), 1)
        self.assertIn("verify-station.sh test-corpus", text)
        block = text[text.index(VERIFY_MARKER) :]
        self.assertNotIn("| FAIL |", block)
        self.assertIn("Station-specific result: PASS", block)
        self.assertNotRegex(
            text,
            r"\b20\d\d-\d\d-\d\d\b",
            "the report carries no run date (byte-stable)",
        )

    def test_shared_count_census_branch_and_its_failure_path(self) -> None:
        rows = self.sv.count_census_rows(self.inv)
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(len(rows), len(self.inv["perCrate"]))
        self.assertIsNone(self.sv.count_census_rows({"baseline": "x", "files": []}))
        self.assertIsNone(self.sv.count_census_rows({"baseline": "x", "crates": {}}))
        self.assertEqual(self.sv.census_baseline(self.inv), self.baseline())
        self.assertEqual(self.sv.count_census_diff(sc.REPO, self.baseline(), rows), [])
        tampered = json.loads(json.dumps(rows))
        row = next(r for r in tampered if r["crate"] == "cobre-sddp")
        row["integrationBinaries"] += 1
        bad = self.sv.count_census_diff(sc.REPO, self.baseline(), tampered)
        self.assertEqual(len(bad), 1)
        self.assertIn("cobre-sddp: integrationBinaries recorded", bad[0])
        r = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "station_verify.py"),
                "inventory",
                str(self.station_dir() / "inventory.json"),
                str(sc.REPO),
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("test-file census", r.stdout)

    def test_partI_handoff_owns_no_row_and_genericity_asserts_the_gate_alone(
        self,
    ) -> None:
        handoff = sc.load_json(self.station_dir() / "partI-handoff.json")
        self.assertEqual(handoff["dispositions"], [])
        self.assertEqual(handoff["entries"], [])
        self.assertIn("owns no Part-I row", handoff["resolutionRule"])
        r = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "station_verify.py"),
                "genericity",
                str(sc.REPO),
                str(self.station_dir() / "partI-handoff.json"),
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_file_is_modified(self) -> None:
        station = "plans/architecture-debt-audit/stations/test-corpus/"
        self.assertEqual(
            [
                m
                for m in sc.tracked_modifications()
                if not m.split()[-1].startswith(station)
            ],
            [],
        )
        out = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "crates",
                ".github",
                "docs",
                "schemas",
                "scripts",
                "Cargo.toml",
                "Cargo.lock",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
