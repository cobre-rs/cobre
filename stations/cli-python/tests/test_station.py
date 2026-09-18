"""Executable proof of the cli-python station artifacts.

Every figure in inventory.json is re-measured here on the tree at the station baseline
(`git show <pin>:<path>` blobs through station_checks.Tree), never read back from the JSON;
prior-register.md is checked for the four owned ids, resolving anchors and the supersession block.
Later cli/python tickets append their stage classes to this module.
"""

from __future__ import annotations

import collections
import pathlib
import re
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

CLI = "crates/cobre-cli/src"
PY = "crates/cobre-python/src"
FACADE = "crates/cobre/src"
ROOTS = (CLI, PY, FACADE)
HEADER_BASELINE = re.compile(r"^Baseline:\s+([0-9a-f]{40})\b", re.M)
WRITE_CALL = re.compile(
    r"(?<![\w:])((?:cobre_io::|cobre_sddp::orchestration::)?write_[a-z_0-9]+)\s*\("
)
LOCAL_CLI = {
    "write_line",
    "write_training_outputs",
    "write_simulation_outputs",
    "write_sim_outputs_on_root",
}
LOCAL_PY = {"write_training_artifacts"} | {
    f"write_{s}_if_any"
    for s in (
        "fpha_hyperplanes",
        "evaporation_models",
        "generic_constraint_echo",
        "fixed_delivery",
        "fpha_deviation_points",
    )
}
PRIOR_IDS = ("CD-025", "CD-029", "CD-002", "CD-009")
ID_RE = re.compile(r"\b((?:CD|PD|OD|TD)-\d{3})\b")


def header_baseline() -> str:
    head = "\n".join(backlog_parse.read_register(sc.BACKLOG)[:40])
    hit = HEADER_BASELINE.search(head)
    assert hit, "BACKLOG.md header carries no `Baseline: <sha40>` line"
    return hit.group(1)


def substation_of(rel: str) -> str:
    if (
        rel.startswith(f"{CLI}/commands/run/")
        or rel == f"{CLI}/commands/broadcast.rs"
        or rel == f"{PY}/run.rs"
    ):
        return "S6a"
    if rel.startswith(f"{CLI}/"):
        return "S6b"
    return "S6c"


def writer_scan(tree: sc.Tree, paths: list[str]) -> dict[str, collections.Counter]:
    per_file: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for p in paths:
        for line in tree.read_text(p).splitlines():
            s = line.strip()
            if s.startswith("//") or s.startswith("use "):
                continue
            for m in WRITE_CALL.finditer(line):
                per_file[p][m.group(1).split("::")[-1]] += 1
    return per_file


def decl_line(text: str, pattern: str) -> int | None:
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            return i
    return None


def all_lines(text: str, pattern: str) -> list[int]:
    return [
        i for i, line in enumerate(text.splitlines(), 1) if re.search(pattern, line)
    ]


class InventoryTests(sc.StationCase):
    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.files = cls.inv["files"]
        cls.by_path = {f["path"]: f for f in cls.files}

    def test_envelope_and_baseline_come_from_the_register_header(self) -> None:
        for key in (
            "station",
            "baseline",
            "generatedAt",
            "crates",
            "totals",
            "files",
            "substationRollup",
            "coverage",
            "commandSurface",
            "sddpCoupling",
            "writerBoundary",
            "parityEnforcement",
            "testSurface",
            "priorAnchors",
            "supersessions",
            "commands",
            "ticketFigureDeviations",
        ):
            self.assertIn(key, self.inv)
        self.assertEqual(self.inv["station"], "cli-python")
        self.assertEqual(self.inv["baseline"], header_baseline())
        self.assertTrue(sc.is_register_pin(self.inv["baseline"]), sc.pin_history())
        self.assertEqual(
            [c["name"] for c in self.inv["crates"]],
            ["cobre-cli", "cobre-python", "cobre"],
        )

    def test_src_set_equals_the_tree_at_the_baseline(self) -> None:
        tree = self.tree()
        found = sorted(p for r in ROOTS for p in tree.rs_files(r))
        listed = [f["path"] for f in self.files]
        self.assertEqual(len(listed), len(set(listed)), "a path assigned twice")
        self.assertEqual(sorted(listed), found)
        self.assertEqual(len(found), 30)
        cov = self.inv["coverage"]
        self.assertEqual(cov["findCount"], len(found))
        self.assertEqual(cov["assignedCount"], len(listed))
        self.assertEqual(
            (cov["unassigned"], cov["doubleAssigned"], cov["phantom"]), ([], [], [])
        )
        for c in self.inv["crates"]:
            rows = [f for f in self.files if f["crate"] == c["name"]]
            self.assertEqual(c["srcFiles"], len(rows), c["name"])
            self.assertEqual(c["lines"], sum(f["lines"] for f in rows), c["name"])
            self.assertEqual(
                c["nonTestLines"], sum(f["nonTestLines"] for f in rows), c["name"]
            )
        self.assertEqual(self.inv["totals"]["files"], len(listed))
        self.assertEqual(
            self.inv["totals"]["lines"], sum(f["lines"] for f in self.files)
        )

    def test_every_file_reproduces_lines_non_test_substation_and_symbols(self) -> None:
        tree = self.tree()
        for f in self.files:
            text = tree.read_text(f["path"])
            self.assertEqual(
                f["lines"], tree.read_bytes(f["path"]).count(b"\n"), f["path"]
            )
            non_test, _ = sc.classify_lines(f["path"], text)
            self.assertEqual(f["nonTestLines"], non_test, f["path"])
            self.assertEqual(f["inlineTestLines"], f["lines"] - non_test, f["path"])
            self.assertEqual(f["inlineTestFns"], text.count("#[test]"), f["path"])
            self.assertEqual(f["substation"], substation_of(f["path"]), f["path"])
            self.assertEqual(f["crate"], f["path"].split("/")[1])
            if f["path"] != f"{FACADE}/lib.rs":
                self.assertTrue(f["topSymbols"], f["path"])
                first = f["topSymbols"][0]
                self.assertTrue(
                    sc.anchor_exists(f"`{f['path']}::{first['name']}`", tree),
                    f"{f['path']}::{first['name']}",
                )
        self.assertEqual(self.by_path[f"{FACADE}/lib.rs"]["lines"], 2)
        self.assertEqual(self.by_path[f"{FACADE}/lib.rs"]["topSymbols"], [])

    def test_partition_is_the_declared_three_sub_surfaces(self) -> None:
        roll = {r["id"]: r for r in self.inv["substationRollup"]}
        self.assertEqual(sorted(roll), ["S6a", "S6b", "S6c"])
        for r in roll.values():
            rows = [f for f in self.files if f["substation"] == r["id"]]
            self.assertEqual(r["files"], len(rows), r["id"])
            self.assertEqual(r["lines"], sum(f["lines"] for f in rows), r["id"])
            self.assertEqual(
                sorted(r["members"]), sorted(f["path"] for f in rows), r["id"]
            )
        s6a = set(roll["S6a"]["members"])
        self.assertEqual(len(s6a), 8)
        self.assertIn(f"{CLI}/commands/broadcast.rs", s6a)
        self.assertIn(f"{PY}/run.rs", s6a)
        self.assertEqual(
            len([p for p in s6a if p.startswith(f"{CLI}/commands/run/")]), 6
        )
        self.assertTrue(all(p.startswith(f"{CLI}/") for p in roll["S6b"]["members"]))
        self.assertTrue(
            all(p.startswith((f"{PY}/", f"{FACADE}/")) for p in roll["S6c"]["members"])
        )
        self.assertIn(f"{FACADE}/lib.rs", roll["S6c"]["members"])
        self.assertEqual(sum(r["files"] for r in roll.values()), 30)
        self.assertEqual(
            sum(r["lines"] for r in roll.values()), self.inv["totals"]["lines"]
        )

    def test_facade_and_workspace_facts_hold_at_the_baseline(self) -> None:
        tree = self.tree()
        cli, py, facade = self.inv["crates"]
        self.assertEqual(
            tree.read_text(f"{FACADE}/lib.rs").splitlines(), facade["libRs"]
        )
        self.assertIn("re-exports nothing", facade["libRs"][0])
        self.assertFalse(facade["dependenciesSection"])
        self.assertNotIn("[dependencies]", tree.read_text("crates/cobre/Cargo.toml"))
        self.assertEqual(facade["dependencies"], [])
        lo, hi = (int(x) for x in py["workspaceExcluded"]["lines"].split("-"))
        ws = tree.read_text("Cargo.toml").splitlines()
        self.assertTrue(ws[lo - 1].strip().startswith("exclude"))
        self.assertTrue(
            any("crates/cobre-python" in ws[i - 1] for i in range(lo, hi + 1))
        )
        toml = tree.read_text("crates/cobre-python/Cargo.toml").splitlines()
        self.assertIn("cdylib", toml[py["crateType"]["line"] - 1])

    def test_command_surface_reproduces(self) -> None:
        tree = self.tree()
        cs = self.inv["commandSurface"]
        main_rs = tree.read_text(f"{CLI}/main.rs")
        enum_line = decl_line(main_rs, r"^\s*enum Command\b")
        self.assertEqual(cs["enum"]["line"], enum_line)
        variants = []
        for line in main_rs.splitlines()[enum_line:]:
            if line.strip().startswith("}"):
                break
            m = re.match(r"^\s{4}([A-Z]\w*)\s*(\(|,|$)", line)
            if m:
                variants.append(m.group(1))
        self.assertEqual(cs["enum"]["variants"], variants)
        self.assertEqual(cs["enum"]["count"], len(variants))
        self.assertEqual(variants, ["Init", "Run", "Validate", "Schema", "Version"])
        self.assertEqual(
            cs["dispatch"]["line"],
            decl_line(main_rs, r"^\s*let result = match cli\.command"),
        )
        for v in variants:
            self.assertIn(f"Command::{v}", main_rs)
        self.assertFalse(tree.exists(f"{CLI}/commands/report.rs"))
        self.assertFalse(tree.exists(f"{CLI}/commands/summary.rs"))
        self.assertFalse(cs["broadcastModule"]["isSubcommand"])
        self.assertNotIn("Broadcast", variants)
        commands_mod = tree.read_text(f"{CLI}/commands/mod.rs").splitlines()
        self.assertEqual(
            commands_mod[cs["broadcastModule"]["declaredAt"]["line"] - 1].strip(),
            "pub(crate) mod broadcast;",
        )
        self.assertEqual(
            cs["broadcastModule"]["lines"],
            tree.read_bytes(f"{CLI}/commands/broadcast.rs").count(b"\n"),
        )
        sha = cs["removedSinceTicket"]["commit"].split()[0]
        rp = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{sha}^{{commit}}"],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rp.returncode, 0, sha)

    def test_sddp_coupling_reproduces(self) -> None:
        tree = self.tree()
        sc_ = self.inv["sddpCoupling"]
        per_file: dict[str, int] = {}
        use_lines = 0
        for p in tree.rs_files(CLI):
            t = tree.read_text(p)
            if "cobre_sddp::" in t:
                per_file[p] = t.count("cobre_sddp::")
            use_lines += len(all_lines(t, r"^use cobre_sddp"))
        self.assertEqual(sc_["occurrences"], sum(per_file.values()))
        self.assertEqual(sc_["filesWithOccurrences"], len(per_file))
        self.assertEqual(sc_["useLines"], use_lines)
        ranking = [(r["file"], r["occurrences"]) for r in sc_["perFileRanking"]]
        self.assertEqual(
            ranking, sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        self.assertEqual(ranking[0][0], f"{CLI}/commands/run/policy.rs")
        summary = tree.read_text(f"{CLI}/summary.rs").splitlines()
        self.assertTrue(
            summary[sc_["reExport"]["line"] - 1].startswith("pub use cobre_sddp::")
        )
        self.assertEqual(sc_["reExport"]["count"], len(sc_["reExport"]["items"]))
        self.assertGreaterEqual(sc_["reExport"]["count"], 6)
        self.assertFalse(sc_["reportRs"]["exists"])

    def test_writer_boundary_reproduces_and_the_sets_are_identical(self) -> None:
        tree = self.tree()
        wb = self.inv["writerBoundary"]
        run_files = sorted(tree.rs_files(f"{CLI}/commands/run"))
        cli = writer_scan(tree, run_files)
        cli_ext: set[str] = set()
        for r in wb["cli"]["perFile"]:
            c = cli.get(r["file"], collections.Counter())
            self.assertEqual(
                r["externalNames"],
                sorted(k for k in c if k not in LOCAL_CLI),
                r["file"],
            )
            self.assertEqual(r["externalCount"], len(r["externalNames"]))
            cli_ext |= set(r["externalNames"])
        py = writer_scan(tree, [f"{PY}/run.rs"])[f"{PY}/run.rs"]
        py_ext = {k for k in py if k not in LOCAL_PY}
        self.assertEqual(sorted(cli_ext), wb["cli"]["externalNames"])
        self.assertEqual(sorted(py_ext), wb["python"]["externalNames"])
        self.assertEqual(
            (wb["cli"]["externalCount"], wb["python"]["externalCount"]), (17, 17)
        )
        self.assertEqual((wb["cliOnly"], wb["pyOnly"]), ([], []))
        self.assertTrue(wb["identical"])
        per = {
            r["file"].rsplit("/", 1)[-1]: r["externalCount"]
            for r in wb["cli"]["perFile"]
        }
        self.assertEqual(
            (per["outputs.rs"], per["setup.rs"], per["simulation.rs"]), (13, 3, 1)
        )
        self.assertEqual(
            (per["mod.rs"], per["policy.rs"], per["training.rs"]), (0, 0, 0)
        )
        self.assertEqual(
            [o["name"] for o in wb["cli"]["localOrchestrators"]],
            [
                "write_training_outputs",
                "write_simulation_outputs",
                "write_sim_outputs_on_root",
            ],
        )
        for o in wb["cli"]["localOrchestrators"] + wb["python"]["localHelpers"]:
            text = tree.read_text(o["file"]).splitlines()
            self.assertRegex(text[o["line"] - 1], rf"fn {o['name']}\b", o["name"])
        self.assertEqual(len(wb["python"]["localHelpers"]), 6)
        self.assertEqual(wb["python"]["localHelpersSeen"], sorted(LOCAL_PY))
        raw = len(
            set(re.findall(r"write_[a-z_0-9]*\(", tree.read_text(f"{PY}/run.rs")))
        )
        self.assertEqual(wb["python"]["rawDistinctNames"], raw)
        self.assertEqual(raw, 23)

    def test_parity_enforcement_records_measured_coverage_per_layer(self) -> None:
        tree = self.tree()
        p = self.inv["parityEnforcement"]
        l1 = p["layer1_source"]
        self.assertEqual(l1["exit"], 0)
        self.assertIn("write functions in both paths", l1["stdout"])
        shared = sorted(set(l1["scriptSeesCli"]) & set(l1["scriptSeesPython"]))
        self.assertEqual(l1["sharedCount"], len(shared))
        self.assertGreaterEqual(l1["sharedCount"], l1["minSharedFloor"])
        cov = l1["coverageOfMeasuredWriters"]
        self.assertEqual(cov["measuredExternalNames"], 17)
        self.assertEqual(cov["coveredCount"] + len(cov["notCoveredByName"]), 17)
        self.assertEqual(cov["notCoveredByName"], ["write_scenario"])
        self.assertIn("SimulationParquetWriter", l1["scriptSeesCli"])
        self.assertEqual(
            l1["normalise"].get("write_checkpoint"), "write_policy_checkpoint"
        )
        script = tree.read_text("scripts/ci/check_python_parity.py")
        self.assertIn("def parse_imports", script)
        self.assertIn("--min-shared", script)
        for layer in p["layer2_runtime_python"]:
            self.assertTrue(tree.is_file(layer["file"]), layer["file"])
            self.assertEqual(
                layer["tests"],
                len(re.findall(r"^def test_", tree.read_text(layer["file"]), re.M)),
            )
        l3 = p["layer3_runtime_workspace"]
        self.assertEqual(
            l3["tests"], re.findall(r"^fn (\w+)\(", tree.read_text(l3["file"]), re.M)
        )
        ci = tree.read_text(".github/workflows/ci.yml").splitlines()
        for ref in p["ciReachability"]["layer2"]:
            path, line = ref.rsplit(":", 1)
            self.assertIn(
                "pytest crates/cobre-python/tests/",
                tree.read_text(path).splitlines()[int(line) - 1],
                ref,
            )
        m = re.search(
            r"\.github/workflows/ci\.yml:(\d+)", p["ciReachability"]["layer1_and_3"]
        )
        self.assertIsNotNone(m)
        assert m is not None
        self.assertIn("check_python_parity.py", ci[int(m.group(1)) - 1])
        claude = tree.read_text("CLAUDE.md").splitlines()
        self.assertIn("**Python parity**", claude[p["hardRule"]["line"] - 1])

    def test_test_surface_reproduces(self) -> None:
        tree = self.tree()
        ts = self.inv["testSurface"]
        bins = {b["path"]: b for b in ts["cobreCli"]["integrationBinaries"]}
        self.assertEqual(sorted(bins), sorted(tree.rs_files("crates/cobre-cli/tests")))
        self.assertEqual(len(bins), 14)
        for path, b in bins.items():
            self.assertEqual(b["testFns"], tree.read_text(path).count("#[test]"), path)
            self.assertEqual(b["lines"], tree.read_bytes(path).count(b"\n"), path)
        self.assertEqual(
            ts["cobreCli"]["binaryTestFns"], sum(b["testFns"] for b in bins.values())
        )
        inline = {
            f["path"][len(CLI) + 1 :]: f["inlineTestFns"]
            for f in self.files
            if f["crate"] == "cobre-cli" and f["inlineTestFns"]
        }
        self.assertEqual(
            ts["cobreCli"]["inlineTestFnsPerFile"],
            dict(sorted(inline.items(), key=lambda kv: (-kv[1], kv[0]))),
        )
        self.assertEqual(
            ts["cobreCli"]["totalTestFns"],
            ts["cobreCli"]["binaryTestFns"] + ts["cobreCli"]["inlineTestFns"],
        )
        py_files = sorted(
            p for p in tree.ls_files("crates/cobre-python/tests") if p.endswith(".py")
        )
        self.assertEqual(
            [t["path"] for t in ts["cobrePython"]["pytestFiles"]], py_files
        )
        for t in ts["cobrePython"]["pytestFiles"]:
            self.assertEqual(
                t["testFns"],
                len(re.findall(r"^def test_", tree.read_text(t["path"]), re.M)),
                t["path"],
            )
            self.assertEqual(
                t["lines"], tree.read_bytes(t["path"]).count(b"\n"), t["path"]
            )
        self.assertEqual(
            ts["cobrePython"]["testFunctions"],
            sum(t["testFns"] for t in ts["cobrePython"]["pytestFiles"]),
        )
        rust = {
            f["path"][len(PY) + 1 :]: f["inlineTestFns"]
            for f in self.files
            if f["crate"] == "cobre-python" and f["inlineTestFns"]
        }
        self.assertEqual(
            ts["cobrePython"]["rustTestFnsPerFile"],
            dict(sorted(rust.items(), key=lambda kv: (-kv[1], kv[0]))),
        )
        self.assertEqual(ts["cobrePython"]["rustTestFns"], sum(rust.values()))

    def test_ci_visibility_is_evidenced_on_the_baseline_workflow(self) -> None:
        tree = self.tree()
        ci = self.inv["testSurface"]["ciVisibility"]
        yml = tree.read_text(".github/workflows/ci.yml").splitlines()
        self.assertIn(
            "Run Rust tests for the bindings crate",
            yml[ci["rustTestsStep"]["stepNameLine"] - 1],
        )
        for n in ci["rustTestsStep"]["commandLines"]:
            self.assertIn(
                "cargo test --manifest-path crates/cobre-python/Cargo.toml", yml[n - 1]
            )
        self.assertIn("compiled and run in CI", ci["verdictAtBaseline"])
        self.assertIn("superseded", ci["verdictAtBaseline"])
        for doc in ci["staleDocs"]:
            self.assertIn(
                doc["heading"],
                tree.read_text(doc["file"]).splitlines()[doc["line"] - 1],
                doc["file"],
            )

    def test_supersessions_carry_old_new_and_command(self) -> None:
        tree = self.tree()
        sup = {s["fact"]: s for s in self.inv["supersessions"]}
        for s in sup.values():
            for key in ("fact", "inherited", "inheritedFrom", "baseline", "command"):
                self.assertIn(key, s, s["fact"])
            self.assertNotEqual(s["inherited"], s["baseline"], s["fact"])
        fc = sup["`StudyParams::from_config` non-test call sites"]
        self.assertEqual((fc["inherited"], fc["baseline"]), (1, 5))
        for site in fc["sites"]:
            self.assertIn(
                "StudyParams::from_config",
                tree.read_text(site["file"]).splitlines()[site["line"] - 1],
                site,
            )
        self.assertEqual(
            {s["file"] for s in fc["sites"]},
            {
                "crates/cobre-sddp/src/setup/mod.rs",
                f"{CLI}/commands/validate.rs",
                f"{CLI}/commands/broadcast.rs",
                f"{PY}/io.rs",
                f"{PY}/run.rs",
            },
        )
        names = sup["distinct `write_*(` names in cobre-python/src/run.rs"]
        self.assertEqual((names["inherited"], names["baseline"]), (26, 23))
        prod = sup["cobre-cli production (non-test) lines"]
        self.assertEqual(
            prod["inherited"], {"register 2026-08-22": 4522, "ticket @a136840d": 5419}
        )
        self.assertEqual(prod["baseline"], self.inv["crates"][0]["nonTestLines"])
        self.assertEqual(sup["Command enum variants"]["baseline"], 5)
        self.assertEqual(
            sup["parity gate shared writer names"]["baseline"],
            self.inv["parityEnforcement"]["layer1_source"]["sharedCount"],
        )

    def test_ticket_figure_deviations_are_real(self) -> None:
        dev = self.inv["ticketFigureDeviations"]["changed"]
        self.assertGreater(len(dev), 0)
        for d in dev:
            self.assertNotEqual(d["ticket"], d["measured"], d["figure"])
        figures = {d["figure"]: d for d in dev}
        self.assertEqual(
            (
                figures["crates.cobre-cli.srcFiles"]["ticket"],
                figures["crates.cobre-cli.srcFiles"]["measured"],
            ),
            (20, 18),
        )
        self.assertEqual(
            (figures["total.files"]["ticket"], figures["total.files"]["measured"]),
            (32, 30),
        )
        self.assertEqual(
            (
                figures["commandSurface.variants"]["ticket"],
                figures["commandSurface.variants"]["measured"],
            ),
            (7, 5),
        )
        self.assertEqual(figures["parityEnforcement.layer1.sharedCount"]["ticket"], 4)


class PriorRegisterTests(sc.StationCase):
    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (cls.station_dir() / "prior-register.md").read_text(encoding="utf-8")
        cls.register = "\n".join(backlog_parse.read_register(sc.BACKLOG))

    def test_owned_ids_table_has_the_four_rows_pending(self) -> None:
        table = self.text.split("## Owned prior IDs", 1)[1].split("\n## ", 1)[0]
        for rid in PRIOR_IDS:
            self.assertIsNotNone(
                re.search(rf"^\| {rid} \| \d \| .*\| _pending_ \|$", table, re.M), rid
            )
        rows = [ln for ln in table.splitlines() if ln.startswith("| CD-")]
        self.assertEqual(len(rows), 4)
        self.assertIn("→", table)
        self.assertIn("cobre-io (L2)", table)

    def test_every_cited_id_exists_in_the_register(self) -> None:
        for rid in sorted(set(ID_RE.findall(self.text))):
            self.assertIn(f"**{rid} · ", self.register, rid)

    def test_every_anchor_resolves_at_the_baseline(self) -> None:
        tree = self.tree()
        anchors = sc.anchors_in(self.text)
        self.assertGreaterEqual(len(anchors), 12)
        for a in anchors:
            self.assertTrue(sc.anchor_exists(a, tree), a)
        self.assertIsNone(
            re.search(r"`crates/[^`]+\.rs:\d+`", self.text), "a line-only code anchor"
        )

    def test_check_anchors_exits_zero_over_every_section(self) -> None:
        base = self.baseline()
        titles = [
            ln[3:].strip() for ln in self.text.splitlines() if ln.startswith("## ")
        ]
        self.assertGreaterEqual(len(titles), 5)
        for title in titles:
            self.assertEqual(
                sc.run_checker(
                    "check-anchors.py",
                    title,
                    "--register",
                    str(self.artifact("prior-register.md")),
                    "--baseline",
                    base,
                ),
                0,
                title,
            )

    def test_destination_rule_normative_and_supersession_blocks(self) -> None:
        for heading in (
            "## Destination rule",
            "## Normative — not candidates",
            "## Supersessions",
            "## Do not re-raise",
            "## Re-derive",
        ):
            self.assertIn(heading, self.text)
        rule = self.text.split("## Destination rule", 1)[1].split("\n## ", 1)[0]
        self.assertIn("cobre-io (L2)", rule)
        self.assertIn("`conflicts`", rule)
        norm = self.text.split("## Normative — not candidates", 1)[1].split("\n## ", 1)[
            0
        ]
        for needle in (
            "Python-parity hard rule",
            "Reserved-seams check",
            "no entry for the facade crate",
            "Stale mirror entry",
            "Part-I I.5",
        ):
            self.assertIn(needle, norm)
        sup = self.text.split("## Supersessions", 1)[1].split("\n## ", 1)[0]
        rows = [
            ln
            for ln in sup.splitlines()
            if ln.startswith("| ")
            and not ln.startswith("| Fact")
            and not ln.startswith("| --")
        ]
        self.assertGreaterEqual(len(rows), 7)
        for needle in ("from_config", "| 1 | 5 ", "| 26 | 23 |", "4522", "5419"):
            self.assertIn(needle, sup)


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_file_under_an_evaluated_surface_is_modified(self) -> None:
        out = subprocess.run(
            [
                "git",
                "diff",
                "--stat",
                "HEAD",
                "--",
                "crates",
                "docs",
                "schemas",
                "scripts",
                ".github",
                "Cargo.toml",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
