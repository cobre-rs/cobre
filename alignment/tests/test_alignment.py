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
        register_ids = {
            int(m)
            for m in re.findall(r"\bCD-(\d{3})\b", "\n".join(self.register_lines))
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
                l
                for l in lines
                if l.startswith("#### lp/ kernel boundary (measured at baseline")
            ),
            None,
        )
        self.assertIsNotNone(heading)
        register_rows = [l for l in lines if LP_ROW_RE.match(l)]
        table_rows = [l for l in self.table.splitlines() if LP_ROW_RE.match(l)]
        self.assertEqual(register_rows, table_rows)
        self.assertTrue(any(l.startswith("**Totals.**") for l in lines))
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


if __name__ == "__main__":
    unittest.main()
