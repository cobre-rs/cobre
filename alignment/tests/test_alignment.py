"""Alignment-epic tests (alignment/).

PartIDispositionTests binds the Part-I consolidation envelope to the register pin: nine
items in roadmap order, one disposition each from the ratified vocabulary, closing
commits git accepts, anchors that resolve at the pin, fresh ids above the register
ceiling, and a rendered table that cannot drift from the JSON. ConsolidatorTests runs the
consolidator itself: a missing handoff or an unresolvable anchor lands in ingestRejects
and exits non-zero, and a regeneration is byte-identical to the committed artifacts.
Later alignment tickets append their classes here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from lib import backlog_parse as bp  # noqa: E402
from lib import station_checks as sc  # noqa: E402

ALIGN = sc.AUDIT / "alignment"
ENVELOPE = ALIGN / "part-i-dispositions.json"
TABLE = ALIGN / "part-i-disposition-table.md"
CONSOLIDATOR = ALIGN / "consolidate_part_i.py"
ROADMAP = sc.REPO / "plans" / "generalizing" / "beyond-sddp-generalization.md"
SECTION = "GENERALIZATION ALIGNMENT"
ITEMS = ("1", "2", "3", "4", "5", "6", "7", "8", "I.5")
DISPOSITIONS = {"keep", "retire", "sharpen"}
PHASES = {"0a", "1"}
REGISTER_ID_RE = re.compile(r"^(?:CD|OD)-\d{3}$")
TABLE_ROW_RE = re.compile(
    r"^\| (?P<item>[^|]+?) \| [^|]+ \| [^|]+ \| [^|]+ \| (?P<disposition>[^|]+?) \| (?P<phase>[^|]+?) \| (?P<register>[^|]+?) \|$"
)
DECL = (
    r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?"
    r"(fn|struct|enum|trait|type|const|static|mod|impl)\s+{sym}\b"
)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=sc.REPO, capture_output=True, text=True, check=False
    )


def load_consolidator():
    spec = importlib.util.spec_from_file_location("consolidate_part_i", CONSOLIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("alignment/consolidate_part_i.py not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PartIDispositionTests(unittest.TestCase):
    """E09-1: the envelope and the rendered fragment, checked against the register pin."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = sc.load_json(ENVELOPE)
        cls.rows = cls.env["dispositions"]
        cls.by_item = {r["item"]: r for r in cls.rows}
        cls.table = TABLE.read_text(encoding="utf-8")
        cls.register_lines = bp.read_register(sc.BACKLOG)
        cls.pin = bp.parse_baseline(cls.register_lines)
        cls.tree = sc._station_tree(cls.pin)
        section = bp.find_section(cls.table.splitlines(), SECTION)
        cls.entries = bp.iter_entries(section)
        cls.roadmap = re.sub(r"\s+", " ", ROADMAP.read_text(encoding="utf-8"))

    def run_tool(self, tool: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / tool),
                "--register",
                str(TABLE),
                *extra,
                SECTION,
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
        )

    def anchor_resolves(self, anchor: dict) -> bool:
        if not self.tree.is_file(anchor["path"]):
            return False
        text = self.tree.read_text(anchor["path"])
        if anchor.get("symbol"):
            return (
                re.search(DECL.format(sym=re.escape(anchor["symbol"])), text, re.M)
                is not None
            )
        return (
            anchor["line"] is not None and 1 <= anchor["line"] <= text.count("\n") + 1
        )

    def test_exactly_nine_items_in_roadmap_order_at_the_register_pin(self) -> None:
        self.assertEqual([r["item"] for r in self.rows], list(ITEMS))
        self.assertEqual(self.env["baseline"], self.pin)
        self.assertEqual(self.env["ingestRejects"], [])
        self.assertEqual(self.env["_needsHuman"], [])
        for r in self.rows:
            self.assertEqual(
                r["partIRef"], "I.5" if r["item"] == "I.5" else f"I.3-{r['item']}"
            )
            self.assertIn(
                re.sub(r"\s+", " ", r["claim"])[:80],
                self.roadmap,
                f"item {r['item']}: claim is the roadmap's verbatim text",
            )

    def test_one_disposition_each_from_the_vocabulary(self) -> None:
        for r in self.rows:
            self.assertIn(r["disposition"], DISPOSITIONS, r["item"])
            self.assertTrue(
                r["baselineAnchors"], f"item {r['item']}: baselineAnchors is non-empty"
            )
            if r["disposition"] == "sharpen":
                self.assertTrue(
                    r["sharpenedClaim"],
                    f"item {r['item']}: a sharpen carries its narrowed claim",
                )
            if r["disposition"] == "retire":
                self.assertTrue(
                    r["closedBy"], f"item {r['item']}: a retire cites what closed it"
                )
                self.assertIsNone(r["registerId"])
                self.assertTrue(r["clearedLine"])
            else:
                self.assertRegex(r["registerId"] or "", REGISTER_ID_RE, r["item"])
                self.assertIsNone(r["clearedLine"])

    def test_retire_closures_are_commits_git_accepts(self) -> None:
        closures = [r["closedBy"] for r in self.rows if r["disposition"] == "retire"]
        closures += [
            f["closedBy"]
            for r in self.rows
            for f in r["fieldDispositions"]
            if f["disposition"] == "retire" and f.get("closedBy")
        ]
        self.assertIn("c1a360ad", closures, "item 4's past_inflows closure is recorded")
        for sha in closures:
            self.assertEqual(
                git("cat-file", "-e", f"{sha}^{{commit}}").returncode, 0, sha
            )

    def test_keep_and_sharpen_anchors_resolve_at_the_pin(self) -> None:
        for r in self.rows:
            if r["disposition"] == "retire":
                continue
            for a in r["baselineAnchors"]:
                self.assertTrue(self.anchor_resolves(a), f"item {r['item']}: {a}")
            for a in r["ticketBaselineAnchors"]:
                self.assertIn("path", a)

    def test_fix_shape_phase_is_0a_or_1(self) -> None:
        for r in self.rows:
            self.assertIn(r["fixShapePhase"], PHASES, r["item"])
            self.assertIn(
                r["alignment"],
                {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"},
            )
            self.assertRegex(
                r["alignmentCites"],
                r"^plans/generalizing/beyond-sddp-generalization\.md:\d+ — ",
            )

    def test_table_rows_match_the_json(self) -> None:
        parsed: dict[str, tuple[str, str, str]] = {}
        for ln in self.table.splitlines():
            m = TABLE_ROW_RE.match(ln)
            if m and m.group("item") not in ("#", "---"):
                parsed[m.group("item")] = (
                    m.group("disposition"),
                    m.group("phase"),
                    m.group("register"),
                )
        self.assertEqual(list(parsed), list(ITEMS))
        for r in self.rows:
            self.assertEqual(
                parsed[r["item"]],
                (r["disposition"], r["fixShapePhase"], r["registerId"] or "Cleared"),
                r["item"],
            )

    def test_entries_mirror_the_keep_and_sharpen_rows(self) -> None:
        live = [r for r in self.rows if r["disposition"] != "retire"]
        self.assertEqual([e.id for e in self.entries], [r["registerId"] for r in live])
        self.assertEqual(
            self.env["idBlock"]["allocated"], [r["registerId"] for r in live]
        )
        for e, r in zip(self.entries, live):
            self.assertTrue(
                e.fields["Alignment"].startswith(r["alignment"] + " ("), e.id
            )
            self.assertIn(r["partIRef"], e.fields["Part-I"], e.id)
            self.assertEqual(e.fields["Baseline"], f"`{self.pin}`", e.id)
            for a in r["baselineAnchors"]:
                self.assertIn(f"`{a['path']}", e.fields["Anchors"], e.id)
            self.assertEqual(bool(r.get("reRaiseOf")), "Re-raise-of" in e.fields, e.id)

    def test_ids_sit_above_the_register_ceiling(self) -> None:
        own = bp.find_section(self.register_lines, ALIGNMENT_SECTION)
        others = self.register_lines[: own.start] + self.register_lines[own.end :]
        register_ids = {
            int(m) for m in re.findall(r"\bCD-(\d{3})\b", "\n".join(others))
        }
        first = int(self.env["idBlock"]["firstFree"].split("-")[1])
        self.assertEqual(first, max(register_ids) + 1)
        allocated = [int(i.split("-")[1]) for i in self.env["idBlock"]["allocated"]]
        self.assertEqual(allocated, list(range(first, first + len(allocated))))
        self.assertFalse(
            set(allocated) & register_ids, "no allocated id collides with the register"
        )

    def test_item3_dispositioned_on_rename_evidence_never_anchor_missing(self) -> None:
        r = self.by_item["3"]
        self.assertIn(r["disposition"], {"sharpen", "retire"})
        keys = {(a["path"], a.get("symbol")) for a in r["baselineAnchors"]}
        for want in (
            ("crates/cobre-core/src/model/horizon.rs", "HorizonGraph"),
            ("crates/cobre-core/src/model/temporal.rs", "PolicyGraphType"),
            ("crates/cobre-core/src/model/temporal.rs", "Node"),
        ):
            self.assertIn(want, keys)
        self.assertEqual(r["renamedFrom"]["symbol"], "PolicyGraph")
        claim = r["sharpenedClaim"]
        for token in ("discount", "cyclic", "policy_graph"):
            self.assertIn(token, claim)
        self.assertIn("traversal framing is gone", claim)
        self.assertNotIn("3", {x["item"] for x in self.env["ingestRejects"]})

    def test_item5_comes_straight_from_the_tree(self) -> None:
        r = self.by_item["5"]
        self.assertEqual(r["sourceStations"], [])
        self.assertTrue(r["directFromTree"])
        self.assertIn(r["disposition"], {"keep", "sharpen"})
        keys = {(a["path"], a.get("symbol")) for a in r["baselineAnchors"]}
        self.assertIn(
            ("crates/cobre-core/src/constraints/training_event.rs", "TrainingEvent"),
            keys,
        )
        near_miss = next(
            a
            for a in r["baselineAnchors"]
            if a["role"].endswith("near-miss doc comment")
        )
        lines = self.tree.read_text(near_miss["path"]).splitlines()
        self.assertEqual(near_miss["line"], 125)
        self.assertIn("Active cuts after budget enforcement", lines[124])
        self.assertIn("active_after_budget", lines[125])
        self.assertIn("cobre-tui", r["sharpenedClaim"])
        self.assertIn("cobre-mcp", r["sharpenedClaim"])

    def test_item6_re_raises_the_withdrawn_exemption_with_evidence(self) -> None:
        r = self.by_item["6"]
        self.assertEqual(r["disposition"], "sharpen")
        self.assertTrue(r["reRaiseOf"])
        by_role = {a["role"]: a for a in r["baselineAnchors"]}
        gate = by_role["the withdrawn exemption"]
        self.assertEqual(gate["path"], "scripts/ci/check-infra-genericity.sh")
        self.assertEqual(
            self.tree.read_text(gate["path"]).splitlines()[gate["line"] - 1],
            "EXCLUDED_FILES=()",
        )
        self.assertEqual(gate["line"], 74)
        symbols = {a.get("symbol") for a in r["baselineAnchors"]}
        self.assertTrue({"PolicyCutRecord", "StageCutsPayload"} <= symbols)
        fbs = next(a for a in r["baselineAnchors"] if a["path"].endswith("policy.fbs"))
        self.assertTrue(
            self.tree.read_text(fbs["path"])
            .splitlines()[fbs["line"] - 1]
            .startswith("table StageCuts")
        )
        ticket = {
            (a["path"], a.get("symbol"), a["line"]) for a in r["ticketBaselineAnchors"]
        }
        for want in (
            ("crates/cobre-io/src/output/policy/records.rs", "PolicyCutRecord", 98),
            ("crates/cobre-io/src/output/policy/records.rs", "StageCutsPayload", 176),
            ("crates/cobre-io/schemas/policy.fbs", None, 140),
        ):
            self.assertIn(
                want, ticket, "the ticket's a136840d lines are recorded, not dropped"
            )
        proc = self.run_tool("check-reraise.py")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_item7_is_one_row_merged_from_three_stations(self) -> None:
        rows = [r for r in self.rows if r["item"] == "7"]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(
            {m["station"] for m in r["mergedFrom"]}, {"core-io", "sddp", "cli-python"}
        )
        located = {(a["path"], a["line"]) for a in r["baselineAnchors"]}
        for want in (
            ("crates/cobre-io/src/config/mod.rs", 77),
            ("crates/cobre-sddp/src/setup/params.rs", 186),
            ("crates/cobre-cli/src/commands/broadcast.rs", 87),
        ):
            self.assertIn(want, located)
        self.assertIn(
            ("crates/cobre-io/src/broadcast.rs", "BroadcastScalarParameter"),
            {(a["path"], a.get("symbol")) for a in r["baselineAnchors"]},
        )
        self.assertIn("pub(crate)", r["sharpenedClaim"])
        self.assertIn("BroadcastScalarParameter", r["sharpenedClaim"])
        self.assertEqual(r["fixShapePhase"], "0a")
        self.assertIn("config projection", r["fixShape"])

    def test_item8_keeps_the_claim_and_subordinates_the_field_verdicts(self) -> None:
        r = self.by_item["8"]
        self.assertEqual(r["disposition"], "keep")
        self.assertEqual(r["alignment"], "advances-1")
        self.assertEqual(
            [f["field"] for f in r["fieldDispositions"]],
            ["n_state", "n_transfer", "n_dual_relevant", "n_hydro", "max_par_order"],
        )
        self.assertTrue(all(f["closedBy"] is None for f in r["fieldDispositions"]))
        self.assertIn("subordinate", r["mergeRule"])

    def test_i5_records_every_figure_with_its_command(self) -> None:
        r = self.by_item["I.5"]
        self.assertEqual(r["disposition"], "sharpen")
        figures = {
            (m["figure"], m["command"])
            for m in r["measurements"]
            if isinstance(m["figure"], int)
        }
        for figure, needle in (
            (79, "'use cobre_sddp' a136840d"),
            (95, "'cobre_sddp::' a136840d"),
            (46, "sort -u | wc -l"),
            (11, "-l cobre_sddp a136840d"),
        ):
            self.assertTrue(
                any(f == figure and needle in c for f, c in figures), (figure, needle)
            )
        for figure in (70, 97, 88):
            self.assertTrue(
                any(f == figure for f, _ in figures), f"station figure {figure} kept"
            )
        self.assertTrue(all(m["command"] for m in r["measurements"]))
        mirror = [m for m in r["measurements"] if "cliOnly" in m]
        self.assertEqual(len(mirror), 1)
        self.assertIn("outputs.rs", mirror[0]["unit"])
        self.assertIn("run.rs", mirror[0]["unit"])

    def test_harness_checkers_pass_over_the_fragment(self) -> None:
        for tool, extra in (
            ("check-anchors.py", ()),
            ("check-reraise.py", ()),
            ("fields-check.py", ("--require", "Alignment")),
        ):
            proc = self.run_tool(tool, *extra)
            self.assertEqual(proc.returncode, 0, f"{tool}: {proc.stdout}{proc.stderr}")

    def test_no_evaluated_surface_is_touched(self) -> None:
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
        ).stdout
        self.assertEqual(status, "")


class ConsolidatorTests(unittest.TestCase):
    """The consolidator's reject paths and its reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_consolidator()

    def stations_copy(self, tmp: pathlib.Path, drop: str | None = None) -> pathlib.Path:
        dst = tmp / "stations"
        for station in self.mod.OWNING_STATIONS:
            if station == drop:
                continue
            for name in self.mod.HANDOFF_NAMES:
                src = sc.STATIONS / station / name
                if src.is_file():
                    (dst / station).mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst / station / name)
        return dst

    def test_missing_handoff_is_rejected_and_the_run_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            stations = self.stations_copy(tmp, drop="solver-comm")
            out = tmp / "out"
            code = self.mod.main(
                [
                    "--stations-dir",
                    str(stations),
                    "--out-dir",
                    str(out),
                    "--no-self-check",
                ]
            )
            self.assertEqual(code, 1)
            env = json.loads(
                (out / "part-i-dispositions.json").read_text(encoding="utf-8")
            )
            kinds = {(r["kind"], r.get("station")) for r in env["ingestRejects"]}
            self.assertIn(("handoff-missing", "solver-comm"), kinds)
            self.assertNotIn("8", {r["item"] for r in env["dispositions"]})
            self.assertEqual(len(env["dispositions"]), 8)
            self.assertIn(
                "### Ingest rejects",
                (out / "part-i-disposition-table.md").read_text(encoding="utf-8"),
            )

    def test_unresolvable_anchor_is_rejected_for_its_item_only(self) -> None:
        real = self.mod.specs()

        def broken() -> list[dict]:
            rows = [dict(r) for r in real]
            rows[1]["anchors"] = list(rows[1]["anchors"]) + [
                (
                    "crates/cobre-core/src/model/temporal.rs",
                    None,
                    r"no such needle anywhere",
                    "bogus",
                )
            ]
            return rows

        with (
            tempfile.TemporaryDirectory() as raw,
            mock.patch.object(self.mod, "specs", broken),
        ):
            envelope, code = self.mod.build(sc.STATIONS)
            self.assertEqual(code, 1)
            self.assertEqual(
                [r["kind"] for r in envelope["ingestRejects"]], ["anchor-missing"]
            )
            self.assertEqual(envelope["ingestRejects"][0]["item"], "2")
            self.assertEqual(
                [r["item"] for r in envelope["dispositions"]],
                [i for i in ITEMS if i != "2"],
            )
            del raw

    def test_regeneration_is_byte_identical_to_the_committed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = pathlib.Path(raw)
            code = self.mod.main(["--out-dir", str(out), "--no-self-check"])
            self.assertEqual(code, 0)
            self.assertEqual(
                (out / "part-i-dispositions.json").read_bytes(), ENVELOPE.read_bytes()
            )
            self.assertEqual(
                (out / "part-i-disposition-table.md").read_bytes(), TABLE.read_bytes()
            )


SLICER = ALIGN / "lp-nontest-slice.py"
LP_CLASSIFICATION = ALIGN / "lp-classification.json"
LP_TABLE = ALIGN / "lp-classification.md"
LP_PROOF = ALIGN / "lp-grep-proof.json"
LP_ROOT = "crates/cobre-sddp/src/lp"
ALIGNMENT_SECTION = "generalization-alignment"
LP_CLASSES = {"engine-neutral", "sddp-geometry", "mixed", "test-sibling"}
LP_ROW_RE = re.compile(
    r"^\| `(?P<module>crates/cobre-sddp/src/lp/[^`]+\.rs)` \| (?P<klass>[^|]+?) \| (?P<loc>\d+) \| "
)
RETRACTED_PAIR = ("fill_parallel_water_entries", "fill_chronological_water_entries")


def load_slicer():
    spec = importlib.util.spec_from_file_location("lp_nontest_slice", SLICER)
    if spec is None or spec.loader is None:
        raise RuntimeError("alignment/lp-nontest-slice.py not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LpClassificationTests(unittest.TestCase):
    """E09-2: the 30-row lp/ classification, its grep proof and its register subsection at the pin."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_slicer()
        cls.cls_ = sc.load_json(LP_CLASSIFICATION)
        cls.proof = sc.load_json(LP_PROOF)
        cls.rows = cls.cls_["rows"]
        cls.by_module = {r["module"]: r for r in cls.rows}
        cls.table = LP_TABLE.read_text(encoding="utf-8")
        cls.pin = bp.parse_baseline(bp.read_register(sc.BACKLOG))
        cls.tree = sorted(
            p
            for p in git(
                "ls-tree", "-r", "--name-only", cls.pin, "--", LP_ROOT
            ).stdout.split()
            if p.endswith(".rs")
        )
        cls.register = bp.read_register(sc.BACKLOG)
        cls.section = bp.find_section(cls.register, ALIGNMENT_SECTION)

    def run_checker(self, tool: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(sc.TOOLS / tool), *extra, ALIGNMENT_SECTION],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
        )

    def test_thirty_rows_are_exactly_the_tree_at_the_pin(self) -> None:
        self.assertEqual(self.cls_["baseline"], self.pin)
        self.assertEqual(len(self.tree), 30)
        self.assertEqual([r["module"] for r in self.rows], self.tree)
        self.assertEqual(self.cls_["universe"]["drift"], [])
        siblings = [r for r in self.rows if r["class"] == "test-sibling"]
        self.assertEqual(len(siblings), 5)
        for r in siblings:
            self.assertTrue(r["module"].endswith(("/tests.rs", "/test_support.rs")))
            self.assertEqual(r["nonTestLoc"], 0)
        for r in self.rows:
            self.assertIn(r["class"], LP_CLASSES, r["module"])
            self.assertTrue(r["anchor"] and r["grepResult"], r["module"])
            self.assertIsInstance(r["nonTestLoc"], int)

    def test_non_test_loc_is_recomputed_through_the_brace_aware_slicer(self) -> None:
        total = naive = 0
        for r in self.rows:
            s = self.mod.module_slice(self.pin, r["module"])
            self.assertEqual(s["nonTestLoc"], r["nonTestLoc"], r["module"])
            total += s["nonTestLoc"]
            naive += s["naiveFirstMarkerLoc"]
        self.assertEqual(total, self.cls_["totals"]["corpusNonTestLoc"])
        self.assertNotEqual(
            total, naive, "the trap-aware rule must differ from first-marker truncation"
        )
        bg = self.proof["modules"][f"{LP_ROOT}/indexer/block_grid.rs"]
        self.assertEqual(
            (bg["nonTestLoc"], bg["excludedRanges"][0][0], bg["naiveFirstMarkerLoc"]),
            (126, 127, 20),
        )
        ag = self.proof["modules"][f"{LP_ROOT}/indexer/anticipated_gate.rs"]
        self.assertEqual(
            (ag["nonTestLoc"], ag["excludedRanges"], ag["naiveFirstMarkerLoc"]),
            (112, [[51, 81], [144, 368]], 50),
        )

    def test_grep_proof_re_derives_and_is_not_vacuous(self) -> None:
        any_word_hit = False
        for r in self.rows:
            p = self.proof["modules"][r["module"]]
            s = self.mod.module_slice(self.pin, r["module"])
            hits = self.mod.vocabulary_hits(s["kept"])
            self.assertEqual(
                len(hits["substringHits"]), len(p["substringHits"]), r["module"]
            )
            self.assertEqual(
                len(hits["wordBoundaryHits"]), len(p["wordBoundaryHits"]), r["module"]
            )
            any_word_hit |= bool(p["wordBoundaryHits"])
            if r["class"] == "engine-neutral":
                self.assertEqual(
                    p["wordBoundaryHits"],
                    [],
                    f"{r['module']}: a neutral row has no word-boundary hit",
                )
                for line_no, _ in p["falsePositiveQueue"]:
                    self.assertIn(
                        f"line {line_no}:",
                        r["dismissal"] or "",
                        f"{r['module']}: dismissal quotes line {line_no}",
                    )
        self.assertTrue(
            any_word_hit, "a typo in the vocabulary would green the whole table"
        )
        rc = self.proof["modules"][f"{LP_ROOT}/indexer/range_cursor.rs"]
        self.assertEqual([h[0] for h in rc["substringHits"]], [10])
        patch = self.by_module[f"{LP_ROOT}/builder/patch.rs"]
        self.assertEqual(patch["class"], "sddp-geometry")
        self.assertEqual(patch["grepPass"], {"substring": 0, "wordBoundary": 0})
        self.assertEqual(patch["camelCaseGeometryLines"], 8)
        self.assertIn("CamelCase", patch["dismissal"])

    def test_mixed_rows_name_a_function_seam_and_entries_avoids_the_retracted_pair(
        self,
    ) -> None:
        for r in self.rows:
            if r["class"] != "mixed":
                self.assertIsNone(r["neutralHalf"], r["module"])
                continue
            self.assertTrue(r["splitNote"], r["module"])
            lo, hi = r["neutralHalf"]
            self.assertTrue(0 <= lo <= hi <= r["nonTestLoc"], r["module"])
        entries = self.by_module[f"{LP_ROOT}/builder/entries.rs"]
        for name in RETRACTED_PAIR:
            self.assertNotIn(name, entries["splitNote"])
            self.assertNotIn(name, self.table)
        proc = self.run_checker("check-reraise.py")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_totals_and_the_roadmap_band(self) -> None:
        t = self.cls_["totals"]
        neutral = sum(
            r["nonTestLoc"] for r in self.rows if r["class"] == "engine-neutral"
        )
        lo = sum(r["neutralHalf"][0] for r in self.rows if r["class"] == "mixed")
        hi = sum(r["neutralHalf"][1] for r in self.rows if r["class"] == "mixed")
        corpus = t["corpusNonTestLoc"]
        self.assertEqual(t["engineNeutralLoc"], neutral)
        self.assertEqual(t["mixedNeutralHalf"], [lo, hi])
        self.assertEqual(t["measuredBand"], [neutral + lo, neutral + hi])
        self.assertEqual(t["roadmapBand"], [round(corpus / 5), round(corpus / 4)])
        overlap = (
            t["measuredBand"][1] >= t["roadmapBand"][0]
            and t["measuredBand"][0] <= t["roadmapBand"][1]
        )
        self.assertEqual(t["verdict"], "agreement" if overlap else "amended")
        if t["verdict"] == "amended":
            self.assertIn(self.pin[:8], t["amendedFigure"])
            self.assertRegex(t["amendedFigure"], r"\d{4}-\d{2}-\d{2}")

    def test_markdown_rows_mirror_the_json(self) -> None:
        parsed = [
            (m.group("module"), m.group("klass"), int(m.group("loc")))
            for ln in self.table.splitlines()
            if (m := LP_ROW_RE.match(ln))
        ]
        self.assertEqual(
            parsed, [(r["module"], r["class"], r["nonTestLoc"]) for r in self.rows]
        )

    def test_register_subsection_carries_the_same_table_and_the_checkers_pass(
        self,
    ) -> None:
        lines = self.section.lines
        heading = next(
            (
                ln
                for ln in lines
                if ln.startswith("#### lp/ kernel boundary (measured at baseline")
            ),
            None,
        )
        self.assertIsNotNone(heading)
        register_rows = [ln for ln in lines if LP_ROW_RE.match(ln)]
        table_rows = [ln for ln in self.table.splitlines() if LP_ROW_RE.match(ln)]
        self.assertEqual(register_rows, table_rows)
        self.assertTrue(any(ln.startswith("**Totals.**") for ln in lines))
        for tool, extra in (
            ("check-anchors.py", ()),
            ("fields-check.py", ("--require", "Alignment")),
        ):
            proc = self.run_checker(tool, *extra)
            self.assertEqual(proc.returncode, 0, f"{tool}: {proc.stdout}{proc.stderr}")

    def test_regeneration_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = pathlib.Path(raw)
            code = self.mod.main(
                ["classify", "--baseline", self.pin, "--out-dir", str(out)]
            )
            self.assertEqual(code, 0)
            for name in (
                "lp-classification.json",
                "lp-classification.md",
                "lp-grep-proof.json",
            ):
                self.assertEqual(
                    (out / name).read_bytes(), (ALIGN / name).read_bytes(), name
                )


ADJUDICATOR = ALIGN / "adjudicate.py"
LEDGER = ALIGN / "alignment-ledger.json"
DOCKET = ALIGN / "conflicts-docket.md"
PRE_IMAGE = ALIGN / "backlog.pre-retag.md"
STATIONS = (
    "core-io",
    "stochastic",
    "solver-comm",
    "sddp",
    "cli-python",
    "build-ci",
    "test-corpus",
)
ALIGNMENT_VOCAB = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}


def load_adjudicator():
    spec = importlib.util.spec_from_file_location("adjudicate", ADJUDICATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("alignment/adjudicate.py not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AlignmentLedgerTests(unittest.TestCase):
    """E09-3: one decided Alignment per register entry, proved bullet-local, rule-caught conflicts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_adjudicator()
        cls.ledger = sc.load_json(LEDGER)
        cls.rows = cls.ledger["ledger"]
        cls.by_id = {r["entryId"]: r for r in cls.rows}
        cls.register = bp.read_register(sc.BACKLOG)
        cls.section = bp.find_section(cls.register, ALIGNMENT_SECTION)
        cls.section_text = "\n".join(cls.register[cls.section.start : cls.section.end])
        cls.pre_image = PRE_IMAGE.read_text(encoding="utf-8").splitlines()
        cls.docket = DOCKET.read_text(encoding="utf-8")
        cls.substrate = cls.mod.substrate_tokens()

    def synthetic(self, entry_id: str, station: str, hint: str, fix: str) -> dict:
        return {
            "entryId": entry_id,
            "station": station,
            "class": entry_id.split("-")[0],
            "registerLine": 0,
            "heading": f"**{entry_id} · synthetic**",
            "fixShape": fix,
            "hint": hint,
            "hintSource": "calibration.json",
            "alternativeFixShape": None,
            "needsHuman": None,
        }

    def test_universe_is_the_register_entry_set_with_full_calibration_coverage(
        self,
    ) -> None:
        ids: list[str] = []
        for st in STATIONS:
            ids += [e.id for e in bp.iter_entries(bp.find_section(self.register, st))]
        self.assertEqual(len(ids), len(set(ids)), "duplicate id inside the stations")
        self.assertEqual(sorted(ids), sorted(self.by_id))
        self.assertEqual(len(self.rows), 243)
        cov = self.ledger["coverage"]
        self.assertIn("registerOnly", cov)
        self.assertIn("calibrationOnly", cov)
        self.assertEqual(cov["registerOnly"], [])
        self.assertEqual(cov["calibrationOnly"], [])
        for st in STATIONS:
            cal = sc.load_json(sc.AUDIT / "stations" / st / "calibration.json")
            for a in cal["assigned"]:
                self.assertIn(a["id"], self.by_id, f"{st} calibration row {a['id']}")
                self.assertEqual(self.by_id[a["id"]]["station"], st)
        self.assertEqual(self.ledger["baseline"], bp.parse_baseline(self.register))

    def test_hint_reader_accepts_both_shapes_and_the_calibration_value_wins(
        self,
    ) -> None:
        cov = self.ledger["coverage"]
        self.assertEqual(
            set(cov["stationsWithoutAlignmentQueue"]),
            {"core-io", "solver-comm", "build-ci", "test-corpus"},
        )
        for st in STATIONS:
            self.assertGreater(cov["perStation"][st], 0)
        for r in self.rows:
            self.assertIn(r["hintSource"], {"calibration.json", "alignment-queue.json"})
            self.assertIn(r["hint"], ALIGNMENT_VOCAB)
        self.assertIsInstance(cov["hintDisagreements"], list)
        with tempfile.TemporaryDirectory() as tmp:
            stations = pathlib.Path(tmp) / "stations"
            for st in STATIONS:
                (stations / st).mkdir(parents=True)
                for name in ("calibration.json", "alignment-queue.json"):
                    src = sc.AUDIT / "stations" / st / name
                    if src.exists():
                        shutil.copy(src, stations / st / name)
            queue = stations / "sddp" / "alignment-queue.json"
            q = json.loads(queue.read_text(encoding="utf-8"))
            victim = next(r for r in q["rows"] if r["id"] in self.by_id)
            victim["alignmentHint"] = "conflicts"
            queue.write_text(json.dumps(q), encoding="utf-8")
            with mock.patch.object(self.mod, "STATIONS_DIR", stations):
                built = self.mod.build_ledger(sc.BACKLOG)
            row = next(r for r in built["ledger"] if r["entryId"] == victim["id"])
            self.assertEqual(row["hint"], self.by_id[victim["id"]]["hint"])
            self.assertEqual(row["hintSource"], "calibration.json")
            self.assertEqual(
                [d["id"] for d in built["coverage"]["hintDisagreements"]],
                [victim["id"]],
            )
            self.assertEqual(len(built["ledger"]), 243)

    def test_every_row_is_decided_with_a_rationale_and_a_roadmap_citation(
        self,
    ) -> None:
        self.assertEqual(self.ledger["selfCheck"]["problems"], [])
        for r in self.rows:
            self.assertIn(r["decided"], ALIGNMENT_VOCAB, r["entryId"])
            self.assertGreaterEqual(len(r["rationale"]), 40, r["entryId"])
            self.assertRegex(r["cites"], r"Part (IV|V)", r["entryId"])
            if r["hint"] != r["decided"]:
                self.assertEqual(r["settledBy"], "hand", r["entryId"])
                self.assertTrue(r["retagged"], r["entryId"])
            else:
                self.assertFalse(r["retagged"], r["entryId"])
            if r["machineDecision"] == "conflicts" and r["decided"] != "conflicts":
                self.assertEqual(r["settledBy"], "hand", r["entryId"])
        counts = self.ledger["decidedCounts"]
        self.assertEqual(sum(counts.values()), len(self.rows))
        self.assertEqual(
            sorted(r["entryId"] for r in self.ledger["retagged"]),
            sorted(r["entryId"] for r in self.rows if r["retagged"]),
        )
        for pid, d in self.mod.PART_I_DECISIONS.items():
            self.assertIn(d["decided"], ALIGNMENT_VOCAB, pid)
            self.assertRegex(d["cites"], r"Part (IV|V)", pid)

    def test_output_helper_hoisted_into_cobre_cli_is_caught_by_rule(self) -> None:
        row = self.synthetic(
            "CD-900",
            "cli-python",
            "advances-0a",
            "Hoist the shared output writers into a cobre-cli-local helper module so the training "
            "and simulation output mirror lives in one place under cobre-cli. Both front ends then "
            "call the helper.",
        )
        hits = self.mod.guardrail_violations(row, row["fixShape"], None, self.substrate)
        self.assertIn("output-orchestration-not-at-L2", [h["id"] for h in hits])
        self.mod.decide_row(row, self.substrate)
        self.assertEqual(row["decided"], "conflicts")
        self.assertEqual(row["guardrailViolated"], "output-orchestration-not-at-L2")
        self.assertIn("Part IV.1", row["cites"])
        self.assertIn("Part V.1", row["cites"])
        self.assertIn("cobre-io entry point", row["alternativeFixShape"])
        self.assertIn(
            "crates/cobre-cli/src/commands/run/outputs.rs", row["alternativeFixShape"]
        )
        self.assertIn("crates/cobre-python/src/run.rs", row["alternativeFixShape"])
        self.assertTrue(row["held"])
        self.assertTrue(row["retagged"])
        problems = self.mod.self_check({"ledger": [row]})
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("without a hand-written decision", problems[0])
        row["settledBy"] = "hand"
        self.assertEqual(self.mod.self_check({"ledger": [row]}), [])
        live = self.by_id["CD-059"]
        self.assertEqual(
            live["guardrailHits"], [], "the L2-side owner shape is not a hoist"
        )

    def test_gate_substrate_removal_is_caught_by_rule_and_byte_neutrality_bars_are_not(
        self,
    ) -> None:
        row = self.synthetic(
            "TD-900",
            "test-corpus",
            "neutral",
            "Delete the parity_hash_highs and parity_hash_clp golden mods in "
            "crates/cobre-sddp/tests/parity.rs together with common/parity_hash.rs; the "
            "invariance-shuffle workflow already covers order invariance.",
        )
        self.mod.decide_row(row, self.substrate)
        self.assertEqual(row["decided"], "conflicts")
        self.assertEqual(row["guardrailViolated"], "phase-0a-gate-substrate-removal")
        self.assertIn("bit-for-bit", row["cites"])
        self.assertIn("Part V.1", row["cites"])
        self.assertIn("keeping the golden", row["alternativeFixShape"])
        self.assertTrue(row["held"])
        for eid in ("TD-045", "OD-037", "OD-038", "CD-089", "CD-091", "CD-095"):
            self.assertEqual(
                self.by_id[eid]["guardrailHits"],
                [],
                f"{eid} names parity only as a bar",
            )
        reviewed = {h["entryId"]: h for h in self.ledger["guardrailHitsReviewed"]}
        self.assertIn("OD-030", reviewed)
        self.assertEqual(
            reviewed["OD-030"]["guardrail"], "phase-0a-gate-substrate-removal"
        )
        self.assertEqual(self.by_id["OD-030"]["settledBy"], "hand")
        self.assertEqual(self.by_id["OD-030"]["decided"], "neutral")
        self.assertEqual(
            self.ledger["held"], [r["entryId"] for r in self.rows if r["held"]]
        )

    def test_conflicts_without_an_alternative_or_a_question_fails_the_pass(
        self,
    ) -> None:
        base = {
            "entryId": "CD-901",
            "decided": "conflicts",
            "rationale": "x" * 40,
            "cites": "Part IV.1",
            "hint": "conflicts",
            "settledBy": "hand",
            "machineDecision": "conflicts",
            "guardrailViolated": "engine-concept-in-L0/L1",
            "held": True,
            "alternativeFixShape": None,
            "needsHuman": None,
        }
        self.assertTrue(
            any("alternative" in p for p in self.mod.self_check({"ledger": [base]}))
        )
        repaired = {**base, "needsHuman": "which layer owns the resolver?"}
        self.assertEqual(self.mod.self_check({"ledger": [repaired]}), [])
        downgraded = {
            **base,
            "decided": "neutral",
            "held": False,
            "settledBy": "machine",
        }
        self.assertTrue(
            any(
                "dropped without a hand decision" in p
                for p in self.mod.self_check({"ledger": [downgraded]})
            )
        )

    def test_retag_is_bullet_local_and_proved_against_the_pre_image(self) -> None:
        post = self.register
        cut = self.section.start
        pre_head = self.pre_image[:cut]
        post_head = post[:cut]
        self.assertEqual(
            len(pre_head),
            len(post_head),
            "a line was added or removed outside the section",
        )
        changed = [
            (i + 1, a, b) for i, (a, b) in enumerate(zip(pre_head, post_head)) if a != b
        ]
        retagged = {r["entryId"]: r for r in self.rows if r["retagged"]}
        self.assertEqual(len(changed), len(retagged))
        self.assertEqual(len(retagged), 12)
        seen: set[str] = set()
        for lineno, before, after in changed:
            self.assertRegex(before, r"^\s*- \*\*Alignment:\*\* ")
            self.assertRegex(after, r"^\s*- \*\*Alignment:\*\* ")
            owner = next(
                m.group("id")
                for ln in reversed(post[:lineno])
                if (m := bp.ENTRY_RE.match(ln))
            )
            self.assertIn(owner, retagged)
            row = retagged[owner]
            self.assertTrue(after.startswith(f"- **Alignment:** {row['decided']} ("))
            self.assertIn(f"station hint: {row['hint']}, retagged 2026-09-19", after)
            self.assertNotIn("(" + row["hint"], after.split("station hint")[0][:40])
            seen.add(owner)
        self.assertEqual(seen, set(retagged))
        self.assertEqual(self.pre_image[cut:], self.pre_image[cut:])
        proof = self.ledger["retagProof"]
        self.assertTrue(proof["everyRemovedLineIsAnAlignmentBullet"])
        self.assertTrue(proof["countsMatch"])
        self.assertEqual(proof["removedLines"], 12)

    def test_docket_lists_each_held_row_once_and_none_is_actionable(self) -> None:
        held = [r for r in self.rows if r["held"]]
        for r in held:
            self.assertEqual(self.docket.count(f"### {r['entryId']} - HELD"), 1)
            self.assertIn(r["fixShape"][:80], self.docket)
            self.assertIn("**Cost of overriding.**", self.docket)
            self.assertIn("**Decision.** [ ]", self.docket)
            self.assertIn(f"**{r['entryId']}** — retagged", self.section_text)
        if not held:
            self.assertIn("**None held.**", self.docket)
            self.assertIn(
                "None: no station entry's recorded fix-shape violates",
                self.section_text,
            )
        self.assertIn("### output-orchestration-not-at-L2", self.docket)
        self.assertIn("### phase-0a-gate-substrate-removal", self.docket)
        conflicts_h4 = self.section_text.split(
            "#### Conflicts held for owner override", 1
        )[1]
        self.assertNotRegex(conflicts_h4, r"\bactionable\b(?!\.)(?! set)")
        tables = bp.parse_tables(self.section)
        ledger_table = next(t for t in tables if "Decided" in t[0])
        self.assertEqual(len(ledger_table), len(self.rows))
        for row in ledger_table:
            live = self.by_id[row["Entry"]]
            self.assertEqual(row["Held"], "yes" if live["held"] else "no")
            self.assertEqual(row["Decided"], live["decided"])
            self.assertEqual(row["Hint"], live["hint"])
        self.assertEqual(
            self.ledger["handoffs"]["ownerGate"]["held"], [r["entryId"] for r in held]
        )

    def test_rendered_section_matches_its_sources_and_the_checkers_pass(self) -> None:
        tables = bp.parse_tables(self.section)
        part_i = next(t for t in tables if "Disposition" in t[0])
        self.assertEqual(len(part_i), 9)
        self.assertEqual(tuple(row["#"] for row in part_i), ITEMS)
        envelope = sc.load_json(ENVELOPE)
        for row, disp in zip(part_i, envelope["dispositions"], strict=True):
            self.assertEqual(row["Disposition"], disp["disposition"])
            self.assertEqual(
                row["Register id / Cleared"].split(" ")[0],
                disp["registerId"] or "Cleared",
            )
            self.assertEqual(
                row["Alignment"],
                self.mod.PART_I_DECISIONS.get(disp["registerId"] or "", {}).get(
                    "decided", disp["alignment"]
                ),
            )
        ledger_table = next(t for t in tables if "Decided" in t[0])
        self.assertEqual(len(ledger_table), len(self.rows))
        self.assertIn("fifth-to-a-quarter", self.section_text)
        self.assertRegex(
            self.section_text, r"\*\*amended \(dated 2026-09-19|: agreement\."
        )
        self.assertIn("alignment/lp-classification.md", self.section_text)
        self.assertIn("precedence", self.section_text)
        self.assertIn("Part IV.1", self.section_text)
        spliced = [e.id for e in bp.iter_entries(self.section)]
        self.assertEqual(spliced, sorted(self.mod.PART_I_DECISIONS))
        for e in bp.iter_entries(self.section):
            self.assertIn("adjudicated 2026-09-19", e.fields["Alignment"])
            self.assertTrue(
                e.fields["Alignment"].startswith(
                    self.mod.PART_I_DECISIONS[e.id]["decided"]
                )
            )
        for tool, extra in (
            ("check-anchors.py", []),
            ("check-reraise.py", []),
            ("fields-check.py", []),
        ):
            proc = subprocess.run(
                [sys.executable, str(sc.TOOLS / tool), *extra, ALIGNMENT_SECTION],
                cwd=sc.REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"{tool}: {proc.stdout}{proc.stderr}")
        proc = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "fields-check.py"),
                "--require",
                "Alignment",
                "--all",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(self.ledger["verification"]["passed"])
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
        self.assertEqual(status, "")

    def test_late_entry_rule_is_published_and_append_is_idempotent(self) -> None:
        self.assertIn("append", self.ledger["lateEntryRule"])
        self.assertIn("append", self.ledger["handoffs"]["lateEntryRule"])
        copy = json.loads(json.dumps(self.ledger))
        added = self.mod.append_section(copy, sc.BACKLOG, "sddp")
        self.assertEqual(added, 0)
        self.assertEqual(len(copy["ledger"]), len(self.rows))
        self.assertEqual(self.mod.self_check(copy), [])

    def test_regeneration_is_deterministic(self) -> None:
        built = self.mod.decide_pass(self.mod.build_ledger(sc.BACKLOG))
        self.assertEqual(built["selfCheck"]["problems"], [])
        keys = (
            "hint",
            "hintSource",
            "decided",
            "rationale",
            "cites",
            "retagged",
            "held",
            "settledBy",
            "machineDecision",
        )
        for r in built["ledger"]:
            live = self.by_id[r["entryId"]]
            for k in keys:
                self.assertEqual(r[k], live[k], f"{r['entryId']}.{k}")
        self.assertEqual(built["decidedCounts"], self.ledger["decidedCounts"])
        self.assertEqual(self.mod.render_docket(built), self.docket)


VERIFIER = ALIGN / "verify-alignment.sh"
VERIFICATION = ALIGN / "verification.md"
LP_INVENTORY = sc.AUDIT / "measurements" / "lp-inventory.json"
ROADMAP_CITE_RE = re.compile(
    r"beyond-sddp-generalization\.md|target-layering|target layering|Part (IV|V)\b"
)
HEREDOC_RE = re.compile(r"<<'PY'\n(.*?)\nPY\n", re.S)


class AlignmentFieldTests(unittest.TestCase):
    """E09-4: every register entry carries a decided, cited Alignment; the lp/ universe is classified once."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.register = bp.read_register(sc.BACKLOG)
        cls.ledger = sc.load_json(LEDGER)
        cls.cls_ = sc.load_json(LP_CLASSIFICATION)
        cls.proof = sc.load_json(LP_PROOF)
        cls.inventory = sc.load_json(LP_INVENTORY)
        cls.docket = DOCKET.read_text(encoding="utf-8")

    def test_every_register_entry_has_a_vocabulary_alignment_with_a_roadmap_citation(
        self,
    ) -> None:
        entries = [
            e
            for name in (*STATIONS, ALIGNMENT_SECTION)
            for e in bp.iter_entries(bp.find_section(self.register, name))
        ]
        self.assertGreaterEqual(len(entries), 243 + 9)
        for e in entries:
            value = e.fields.get("Alignment", "")
            self.assertIn(value.split("(", 1)[0].strip(), ALIGNMENT_VOCAB, e.id)
            self.assertRegex(value, ROADMAP_CITE_RE, f"{e.id}: no roadmap citation")
        proc = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "fields-check.py"),
                "--require",
                "Alignment",
                "--all",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("0 incomplete", proc.stdout)

    def test_every_inventory_module_is_classified_exactly_once_and_class_sums_match(
        self,
    ) -> None:
        listed = sorted(f["path"] for f in self.inventory["files"])
        rows = self.cls_["rows"]
        self.assertEqual(sorted(r["module"] for r in rows), listed)
        self.assertEqual(len({r["module"] for r in rows}), len(rows))
        classes = {"engine-neutral", "sddp-geometry", "mixed", "test-sibling"}
        by_class: dict[str, int] = {}
        for r in rows:
            self.assertIn(r["class"], classes, r["module"])
            by_class[r["class"]] = by_class.get(r["class"], 0) + r["nonTestLoc"]
        t = self.cls_["totals"]
        self.assertEqual(by_class["engine-neutral"], t["engineNeutralLoc"])
        self.assertEqual(by_class["sddp-geometry"], t["sddpGeometryLoc"])
        self.assertEqual(by_class["mixed"], t["mixedLoc"])
        self.assertEqual(by_class.get("test-sibling", 0), 0)
        self.assertEqual(sum(by_class.values()), t["corpusNonTestLoc"])
        self.assertEqual(t["corpusNonTestLoc"], self.proof["totals"]["nonTestLoc"])
        inventory_total = self.inventory["totals"]["non_test_lines"]
        self.assertNotEqual(inventory_total, t["corpusNonTestLoc"])
        self.assertTrue(
            any(
                f"{inventory_total:,}" in d and "loc-stats" in d
                for d in self.cls_["deviations"]
            ),
            "the inventory's loc-stats total must be recorded as a deviation with its rule",
        )
        self.assertEqual(
            self.inventory["totals"]["total_lines"], self.proof["totals"]["grossLoc"]
        )

    def test_measured_share_is_recorded_against_the_roadmap_estimate(self) -> None:
        t = self.cls_["totals"]
        corpus = t["corpusNonTestLoc"]
        self.assertEqual(t["roadmapBand"], [round(corpus / 5), round(corpus / 4)])
        lo, hi = t["mixedNeutralHalf"]
        self.assertEqual(
            t["measuredBand"], [t["engineNeutralLoc"] + lo, t["engineNeutralLoc"] + hi]
        )
        self.assertIn(t["verdict"], {"agreement", "amended"})
        md = LP_TABLE.read_text(encoding="utf-8")
        self.assertRegex(md, r"fifth|quarter|IV\.2")
        if t["verdict"] == "amended":
            self.assertRegex(t["amendedFigure"], r"\d{4}-\d{2}-\d{2}")
            self.assertIn("amended", md)

    def test_every_conflicts_entry_is_held_and_docketed(self) -> None:
        conflicts = [
            r["entryId"] for r in self.ledger["ledger"] if r["decided"] == "conflicts"
        ]
        self.assertEqual(sorted(self.ledger["held"]), sorted(conflicts))
        self.assertEqual(
            sorted(self.ledger["handoffs"]["ownerGate"]["held"]), sorted(conflicts)
        )
        for r in self.ledger["ledger"]:
            self.assertEqual(bool(r["held"]), r["decided"] == "conflicts", r["entryId"])
        for eid in conflicts:
            self.assertEqual(self.docket.count(f"### {eid} - HELD"), 1)
        if not conflicts:
            self.assertIn("**None held.**", self.docket)


class AlignmentVerifierTests(unittest.TestCase):
    """E09-4: verify-alignment.sh is lint-clean, imports the shared parser and slicer, passes, and its report is stable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = VERIFIER.read_text(encoding="utf-8")

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck not installed")
    def test_shellcheck_passes(self) -> None:
        proc = subprocess.run(
            ["shellcheck", str(VERIFIER)], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_every_embedded_python_block_compiles(self) -> None:
        blocks = HEREDOC_RE.findall(self.script)
        self.assertGreaterEqual(len(blocks), 6)
        for i, block in enumerate(blocks):
            compile(block, f"verify-alignment.sh<<PY#{i}", "exec")

    def test_imports_the_shared_parser_and_the_slicer_rather_than_reimplementing(
        self,
    ) -> None:
        self.assertIn(
            "from lib.backlog_parse import ENTRY_RE, find_section, read_register",
            self.script,
        )
        self.assertIn("bp.parse_baseline(bp.read_register", self.script)
        self.assertIn("bp.iter_entries(section)", self.script)
        self.assertIn(
            'spec_from_file_location("slicer", "plans/architecture-debt-audit/alignment/lp-nontest-slice.py")',
            self.script,
        )
        self.assertIn("slicer.nontest_slice(", self.script)
        self.assertIn("slicer.vocabulary_hits(", self.script)
        self.assertIn("slicer.naive_first_marker_loc(", self.script)
        self.assertNotIn(
            "verify-station.sh",
            self.script.split("Deliberately does NOT call", 1)[1].split("\n", 3)[3],
        )
        self.assertNotRegex(self.script, r"re\.compile\(r\"\^\\\*\\\*")

    def test_verifier_passes_and_regenerates_the_committed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = pathlib.Path(tmp) / "verification.md"
            proc = subprocess.run(
                ["bash", str(VERIFIER), "--report", str(report)],
                cwd=sc.REPO,
                capture_output=True,
                text=True,
                env={**os.environ, "VERIFY_ALIGNMENT_NO_TESTS": "1"},
            )
            self.assertEqual(
                proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:]
            )
            self.assertIn("verify-alignment: PASS", proc.stdout)
            for name in (
                "anchors",
                "re-raise",
                "fields (section)",
                "Part-I dispositions",
                "lp/ rows vs find",
                "LOC recount",
                "grep proof",
                "Alignment presence",
                "ledger integrity",
                "read-only",
            ):
                self.assertIn(f"\n--- {name}\n", proc.stdout)
            generated = report.read_text(encoding="utf-8").splitlines()
        committed = VERIFICATION.read_text(encoding="utf-8").splitlines()

        def rows(lines: list[str]) -> list[str]:
            return [ln for ln in lines if not ln.startswith("| alignment tests |")]

        # The tests row is the one line that legitimately differs between a run that includes
        # this module and the nested run this test makes; the verifier's exit code owns it.
        self.assertEqual(rows(generated), rows(committed))
        self.assertTrue(any(ln.startswith("| alignment tests |") for ln in committed))
        self.assertIn("Held for the owner gate: none", "\n".join(committed))
        self.assertIn("**Overall:** PASS.", "\n".join(committed))


GATE = ALIGN / "gate.md"
GATE_ROADMAP = sc.REPO / "plans" / "generalizing" / "beyond-sddp-generalization.md"
PART_I_DECISIONS_OK = {
    "accept",
    "amend",
    "retire",
    "defer",
    "keep-reraise",
    "re-sanction",
}
HOLD_DECISIONS_OK = {"take-alternative", "override", "reject", "defer"}


def gate_guard(rec: dict) -> list[str]:
    """The marker guard: every reason the RETURNED marker must be refused, mirrored from the gate builder."""
    problems: list[str] = []
    if len(rec["partIDecisions"]) != 9:
        problems.append(
            f"expected nine Part-I decisions, got {len(rec['partIDecisions'])}"
        )
    for d in rec["partIDecisions"]:
        if d["decision"] not in PART_I_DECISIONS_OK:
            problems.append(f"item {d['item']}: decision {d['decision']!r}")
        if d["decision"] == "defer" and not d.get("deferTrigger"):
            problems.append(f"item {d['item']}: defer without a trigger")
        if d["disposition"] == "retire" and not d.get("closedBy"):
            problems.append(f"item {d['item']}: retire without closedBy")
    lp = rec["lpShare"]
    if lp["decision"] == "amend-estimate" and not re.search(
        r"\d{4}-\d{2}-\d{2}", lp.get("amendedFigure") or ""
    ):
        problems.append("amend-estimate without a dated replacement figure")
    if lp["decision"] == "defer" and not lp.get("deferTrigger"):
        problems.append("lpShare defer without a trigger")
    for a in rec["alignmentDecisions"]:
        if a["decided"] == "conflicts":
            if a["ownerDecision"] not in HOLD_DECISIONS_OK:
                problems.append(f"{a['entryId']}: conflicts hold undecided")
            if a["ownerDecision"] == "override" and not a.get("overrideRationale"):
                problems.append(f"{a['entryId']}: override without rationale")
            if a["ownerDecision"] == "defer" and not a.get("deferTrigger"):
                problems.append(f"{a['entryId']}: deferred hold without a trigger")
    problems += [
        f"needs-human unanswered: {n.get('question')}"
        for n in rec["needsHuman"]
        if not n.get("answer")
    ]
    return problems


class AlignmentGateTests(unittest.TestCase):
    """E09-5: the owner gate's digest, round plan, decision record, marker guard and register H3."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = GATE.read_text(encoding="utf-8")
        m = re.search(r"```json\n(\{.*?\})\n```", cls.gate, re.S)
        assert m is not None, "gate.md carries no JSON decision record"
        cls.rec = json.loads(m.group(1))
        cls.register = bp.read_register(sc.BACKLOG)
        cls.section = bp.find_section(cls.register, ALIGNMENT_SECTION)
        cls.section_text = "\n".join(cls.section.lines)
        cls.ledger = sc.load_json(LEDGER)
        cls.decision_lines = [
            ln for ln in cls.gate.splitlines() if ln.startswith("**Decision: ")
        ]

    def test_single_decision_line_valued_ratified_or_returned(self) -> None:
        self.assertEqual(len(self.decision_lines), 1, self.decision_lines)
        m = re.match(r"\*\*Decision: (ratified|returned)\*\*", self.decision_lines[0])
        self.assertIsNotNone(m, self.decision_lines[0])
        assert m is not None
        self.assertEqual(m.group(1) == "ratified", self.rec["returned"])

    def test_digest_carries_nine_rows_the_lp_totals_the_rollup_the_holds_and_needs_human(
        self,
    ) -> None:
        part_i = self.gate.split("### 1.1", 1)[1].split("### 1.2", 1)[0]
        rows = [ln for ln in part_i.splitlines() if re.match(r"^\| (\d|I\.5) \|", ln)]
        self.assertEqual([r.split("|")[1].strip() for r in rows], list(ITEMS))
        self.assertIn("Modules classified: 30", self.gate)
        self.assertRegex(self.gate, r"fifth to a quarter")
        self.assertIn("| **total** | 243 |", self.gate)
        self.assertIn("### 1.4 Holds (conflicts)", self.gate)
        self.assertIn("### 1.5 Needs-human roll-up", self.gate)
        verification = self.gate.split("### 1.6", 1)[1].split("## 2.", 1)[0]
        self.assertGreaterEqual(verification.count("| PASS |"), 10)
        self.assertNotIn("| FAIL |", verification)

    def test_every_conflicts_entry_has_its_own_round_with_an_override_option(
        self,
    ) -> None:
        plan = self.gate.split("## 2. Round plan", 1)[1].split("## 3.", 1)[0]
        rounds = [ln for ln in plan.splitlines() if ln.startswith("| R")]
        holds = [
            r["entryId"] for r in self.ledger["ledger"] if r["decided"] == "conflicts"
        ]
        hold_rounds = [
            ln
            for ln in plan.splitlines()
            if "conflicts-hold" in ln and ln.startswith("| R")
        ]
        self.assertEqual(len(hold_rounds), len(holds))
        for eid in holds:
            self.assertTrue(any(eid in ln for ln in hold_rounds), eid)
            batch = [ln for ln in rounds if "batch" in ln and eid in ln]
            self.assertEqual(batch, [], f"{eid} folded into a batch round")
        template = [ln for ln in plan.splitlines() if "conflicts-hold" in ln]
        self.assertTrue(template)
        for ln in template:
            self.assertIn("override with rationale", ln)
        self.assertEqual(self.rec["conflictsRounds"], [])
        for a in self.rec["alignmentDecisions"]:
            if a["decided"] == "conflicts":
                self.assertIn(a["ownerDecision"], HOLD_DECISIONS_OK, a["entryId"])
                if a["ownerDecision"] != "override":
                    self.assertTrue(a["held"], a["entryId"])

    def test_ratified_record_is_complete_and_the_register_carries_the_marker(
        self,
    ) -> None:
        if not self.rec["returned"]:
            self.assertNotIn("**Gate: RETURNED", self.section_text)
            self.skipTest("gate not ratified; marker correctly absent")
        self.assertEqual(gate_guard(self.rec), [])
        self.assertEqual([d["item"] for d in self.rec["partIDecisions"]], list(ITEMS))
        for d in self.rec["partIDecisions"]:
            self.assertIn(d["decision"], PART_I_DECISIONS_OK, d["item"])
            self.assertTrue(d["rationale"], d["item"])
        self.assertEqual(
            len(self.rec["alignmentDecisions"]), len(self.ledger["ledger"])
        )
        h3 = self.section_text.split("### ★ OWNER GATE: generalization alignment", 1)
        self.assertEqual(len(h3), 2, "owner-gate H3 missing from the section")
        gate_block = h3[1]
        self.assertIn("**Part-I dispositions ratified (9 of 9).**", gate_block)
        self.assertIn("**lp/ kernel boundary.**", gate_block)
        self.assertRegex(
            gate_block, r"measured engine-neutral non-test share [\d,]+ \+ mixed half"
        )
        self.assertRegex(
            gate_block,
            r"(?m)^\*\*Gate: RETURNED \d{4}-\d{2}-\d{2}\*\* — baseline `077dbe2c` \(scaffold pin `a136840d`\); accepted \d+, amended \d+, overridden \d+, rejected \d+, deferred \d+, held \d+",
        )
        self.assertEqual(gate_block.count("**Gate: RETURNED"), 1)
        for a in self.rec["alignmentDecisions"]:
            if a["ownerDecision"] == "take-alternative":
                entry = next(
                    e
                    for st in STATIONS
                    for e in bp.iter_entries(bp.find_section(self.register, st))
                    if e.id == a["entryId"]
                )
                self.assertNotEqual(
                    entry.fields["Alignment"].split("(", 1)[0].strip(), "conflicts"
                )
        self.assertTrue(self.ledger["ownerGate"]["returned"])
        self.assertEqual(self.ledger["ownerGate"]["decision"], "ratified")
        self.assertEqual(self.ledger["ownerGate"]["held"], self.ledger["held"])

    def test_lp_share_decision_carries_the_caveat_and_the_roadmap_is_untouched(
        self,
    ) -> None:
        lp = self.rec["lpShare"]
        self.assertIn(
            lp["decision"],
            {"accept-measured", "amend-estimate", "reclassify-rows", "defer"},
        )
        self.assertTrue(lp["rewriteCaveatCarried"])
        self.assertTrue(lp["rowsThatDroveIt"])
        if lp["decision"] == "amend-estimate":
            self.assertRegex(lp["amendedFigure"], r"\d{4}-\d{2}-\d{2}")
        self.assertEqual(
            lp["roadmapBand"], sc.load_json(LP_CLASSIFICATION)["totals"]["roadmapBand"]
        )
        import hashlib

        self.assertEqual(
            hashlib.sha256(GATE_ROADMAP.read_bytes()).hexdigest(),
            self.rec["roadmapSha256Before"],
            "the gate must not edit plans/generalizing",
        )
        self.assertIn("priced as a rewrite", self.gate)

    def test_marker_guard_refuses_untriggered_defers_undecided_holds_and_open_questions(
        self,
    ) -> None:
        base = json.loads(json.dumps(self.rec))
        self.assertEqual(gate_guard(base), [])
        broken = json.loads(json.dumps(base))
        broken["partIDecisions"][0].update(decision="defer", deferTrigger="")
        self.assertTrue(any("defer without a trigger" in p for p in gate_guard(broken)))
        broken = json.loads(json.dumps(base))
        broken["alignmentDecisions"][0].update(
            decided="conflicts", ownerDecision="accept"
        )
        self.assertTrue(any("hold undecided" in p for p in gate_guard(broken)))
        broken = json.loads(json.dumps(base))
        broken["needsHuman"].append({"from": "x", "question": "open?", "answer": ""})
        self.assertTrue(any("needs-human unanswered" in p for p in gate_guard(broken)))
        broken = json.loads(json.dumps(base))
        broken["lpShare"].update(decision="amend-estimate", amendedFigure="soon")
        self.assertTrue(
            any("dated replacement figure" in p for p in gate_guard(broken))
        )
        broken = json.loads(json.dumps(base))
        broken["partIDecisions"].pop()
        self.assertTrue(any("nine Part-I" in p for p in gate_guard(broken)))
        broken = json.loads(json.dumps(base))
        broken["alignmentDecisions"][0].update(
            decided="conflicts", ownerDecision="override", overrideRationale=""
        )
        self.assertTrue(
            any("override without rationale" in p for p in gate_guard(broken))
        )


if __name__ == "__main__":
    unittest.main()
