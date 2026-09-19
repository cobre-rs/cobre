"""Station tests for build-ci: the enforcement-surface inventory and the gate-wiring census.

Every fact is re-derived from the baseline blobs (`git show <pin>:<path>`), never from the worktree;
BACKLOG.md anchors are HEAD lines of the live register. Later build/CI tickets append their stage
classes here.
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

STUBS = ("cobre-mcp", "cobre-tui", "cobre-flow", "cobre-uc", "cobre-emt")
JOB_IDS = (
    "check",
    "test",
    "clippy",
    "clp",
    "fmt",
    "quality-scripts",
    "schemas",
    "docs-examples",
    "docs",
    "security",
    "deny",
    "license-notices",
    "coverage",
    "python",
)
JOB_RE = re.compile(r"^  ([a-z0-9-]+):$")
CLASSES = {"blocking", "advisory-by-design", "unwired"}
NEGATIVE_GREP = (
    'for f in scripts/ci/*.sh scripts/ci/*.py scripts/ci/lib/*.sh; do n=$(basename "$f"); '
    'grep -rq "$n" .github/workflows/ scripts/pre-commit || echo "UNWIRED: $n"; done'
)
ID_RE = re.compile(r"\b((?:CD|PD|OD|TD)-\d{3})\b")


def walk_anchors(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("path"), str) and ("line" in obj or "symbol" in obj):
            yield obj
        for v in obj.values():
            yield from walk_anchors(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_anchors(v)


class InventoryTests(sc.StationCase):
    SLUG = "build-ci"
    SECTION_TITLE = "build-ci"

    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.census = sc.load_json(cls.station_dir() / "gate-census.json")
        cls.prior = (cls.station_dir() / "prior-register.md").read_text(
            encoding="utf-8"
        )
        cls.t = cls.tree()
        cls.files = cls.t.files()

    def lines(self, path: str) -> list[str]:
        return self.t.read_text(path).splitlines()

    def test_envelope_and_baseline_come_from_the_register_header(self) -> None:
        for key in (
            "station",
            "baseline",
            "registerHead",
            "generatedAt",
            "anchorRule",
            "workspace",
            "ci",
            "gates",
            "schemas",
            "slurm",
            "designDocs",
            "figureCorrections",
            "ticketFigureDeviations",
            "commands",
        ):
            self.assertIn(key, self.inv)
        self.assertEqual(self.inv["station"], "build-ci")
        self.assertTrue(sc.is_register_pin(self.inv["baseline"]), sc.pin_history())
        self.assertEqual(self.inv["baseline"], self.baseline())
        self.assertEqual(self.census["baseline"], self.inv["baseline"])
        self.assertNotIn(
            "files", self.inv
        )  # no .rs census: this station has no crate surface

    def test_workspace_members_stubs_exclusion_build_scripts_and_submodules(
        self,
    ) -> None:
        cargo = self.lines("Cargo.toml")
        members = []
        inside = False
        for ln in cargo:
            s = ln.strip()
            if s.startswith("members = ["):
                inside = True
            elif inside and s == "]":
                break
            elif inside:
                members.append(s.strip('",').split("/")[1])
        ws = self.inv["workspace"]
        self.assertEqual([m["name"] for m in ws["members"]], members)
        self.assertEqual(ws["memberCount"], 13)
        self.assertEqual(tuple(ws["stubs"]), STUBS)
        self.assertEqual({m["name"] for m in ws["members"] if m["stub"]}, set(STUBS))
        for m in ws["members"]:
            self.assertIn(m["path"], cargo[m["manifestLine"] - 1])
        for stub in ws["stubFacts"]:
            self.assertIn("Reserved crate", stub["description"]["text"])
            self.assertFalse(stub["featuresTable"], stub["name"])
            self.assertLessEqual(
                sum(s["lines"] for s in stub["sources"]), 40, stub["name"]
            )
        (ex,) = ws["excluded"]
        self.assertEqual(ex["name"], "cobre-python")
        self.assertIn('"crates/cobre-python"', cargo[ex["anchor"]["line"] - 1])
        self.assertIn("maturin", ex["reason"])
        self.assertIn("cargo test --workspace", ex["reason"])
        self.assertEqual(
            sorted(b["path"] for b in ws["buildScripts"]),
            sorted(
                f for f in self.files if f.endswith("/build.rs") and "/vendor/" not in f
            ),
        )
        for b in ws["buildScripts"]:
            self.assertEqual(b["symbol"], "main")
            self.assertTrue(sc.symbol_resolves(b["path"], "main", self.t), b["path"])
        gitlinks = subprocess.run(
            [
                "git",
                "ls-tree",
                self.baseline(),
                "crates/cobre-solver/vendor/",
                "crates/cobre-sddp/vendor/",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        commits = {
            ln.split("\t")[1]: ln.split()[2] for ln in gitlinks if " commit " in ln
        }
        self.assertEqual({s["path"]: s["commit"] for s in ws["submodules"]}, commits)
        self.assertEqual(
            sorted(s["tag"] for s in ws["submodules"]),
            ["2020.2", "releases/1.17.11", "releases/2.11.13", "v1.13.1"],
        )
        self.assertEqual(
            sorted(s["crate"] for s in ws["features"]["slowTestsDeclarations"]),
            ["cobre-cli", "cobre-io", "cobre-sddp", "cobre-stochastic"],
        )
        for s in ws["features"]["slowTestsDeclarations"]:
            self.assertEqual(
                self.lines(s["anchor"]["path"])[s["anchor"]["line"] - 1].strip(),
                "slow-tests = []",
            )

    def test_ci_section_matches_the_workflows_at_the_baseline(self) -> None:
        ci = self.inv["ci"]
        tree_wf = sorted(f for f in self.files if f.startswith(".github/workflows/"))
        self.assertEqual([w["path"] for w in ci["workflows"]], tree_wf)
        self.assertEqual(ci["workflowCount"], 8)
        for w in ci["workflows"]:
            self.assertEqual(w["lines"], len(self.lines(w["path"])), w["path"])
        self.assertEqual(
            ci["totalWorkflowLines"], sum(w["lines"] for w in ci["workflows"])
        )
        yml = self.lines(".github/workflows/ci.yml")
        jobs_start = yml.index("jobs:") + 1
        found = [
            (JOB_RE.match(ln).group(1), i)
            for i, ln in enumerate(yml, 1)
            if i > jobs_start and JOB_RE.match(ln)
        ]
        self.assertEqual([j["id"] for j in ci["ciJobs"]], list(JOB_IDS))
        self.assertEqual([(j["id"], j["anchor"]["line"]) for j in ci["ciJobs"]], found)
        trig = ci["triggers"]
        self.assertEqual(
            (trig["push"], trig["pull_request"]), (["main", "develop"], ["main"])
        )
        self.assertEqual(yml[trig["anchor"]["line"] - 1].strip(), "on:")
        self.assertEqual(yml[4].strip(), "branches: [main, develop]")
        self.assertEqual(yml[6].strip(), "branches: [main]")
        nsf = ci["env"]["NON_SOLVER_FEATURES"]
        self.assertIn(nsf["value"], yml[nsf["anchor"]["line"] - 1])
        self.assertIn("slow-tests", nsf["value"].split())
        counts = {
            w["path"]: self.t.read_text(w["path"]).count("Build MPICH from source")
            for w in ci["workflows"]
        }
        self.assertEqual(
            {p: c for p, c in counts.items() if c}, ci["mpichFromSourceBlocks"]
        )
        self.assertEqual(ci["mpichFromSourceTotal"], 10)
        self.assertEqual(ci["githubTopLevel"], ["workflows"])
        self.assertIsNone(ci["compositeActionsDir"])

    def test_gate_corpus_is_decomposed_and_the_artifact_and_allowlist_are_not_gates(
        self,
    ) -> None:
        g = self.inv["gates"]
        tree = sorted(f for f in self.files if f.startswith("scripts/ci/"))
        executables = [f for f in tree if f.endswith((".sh", ".py"))]
        self.assertEqual(sorted(f["path"] for f in g["files"]), sorted(executables))
        self.assertEqual(
            (g["shCount"], g["pyCount"], g["libHelpers"]),
            (14, 2, ["scripts/ci/lib/comment_scan.sh"]),
        )
        self.assertEqual(
            (g["topLevelExecutables"], g["totalExecutables"], g["treeEntries"]),
            (16, 17, len(tree)),
        )
        for f in g["files"]:
            self.assertEqual(f["lines"], len(self.lines(f["path"])), f["path"])
        al = g["allowlist"]
        body = self.lines(al["path"])
        self.assertEqual(al["lines"], len(body))
        self.assertEqual(
            al["activeEntries"],
            sum(1 for ln in body if ln.strip() and not ln.strip().startswith("#")),
        )
        self.assertEqual(al["activeEntries"], 0)
        self.assertFalse(any("__pycache__" in f for f in self.files))
        gi = g["pycache"]["gitignoreAnchor"]
        self.assertIn("__pycache__", self.lines(gi["path"])[gi["line"] - 1])

    def test_gate_census_recomputes_from_the_wiring_sites(self) -> None:
        rows = {r["script"]: r for r in self.census["rows"]}
        executables = sorted(
            f
            for f in self.files
            if f.startswith("scripts/ci/") and f.endswith((".sh", ".py"))
        )
        self.assertEqual(sorted(rows), executables)
        sites = [f for f in self.files if f.startswith(".github/workflows/")] + [
            "scripts/pre-commit"
        ]
        self.assertEqual(sorted(self.census["wiringSites"]), sorted(sites))
        self.assertEqual(self.census["negativeGrepCommand"], NEGATIVE_GREP)
        recomputed = {}
        for path in executables:
            base = path.rsplit("/", 1)[1]
            hits = [
                (s, i, ln)
                for s in sites
                for i, ln in enumerate(self.lines(s), 1)
                if base in ln
            ]
            runs = [
                (s, i, ln)
                for s, i, ln in hits
                if "run:" in ln or s == "scripts/pre-commit"
            ]
            wf = [(s, i, ln) for s, i, ln in runs if s.startswith(".github/")]
            if not runs:
                recomputed[path] = "unwired"
                continue
            if wf:
                s, i, _ = wf[0]
                wl = self.lines(s)
                name_idx = next(
                    j
                    for j in range(i - 1, -1, -1)
                    if wl[j].strip().startswith("- name:")
                )
                advisory = any("ADVISORY" in wl[j] for j in range(name_idx + 1, i - 1))
                recomputed[path] = "advisory-by-design" if advisory else "blocking"
            else:
                recomputed[path] = "blocking"
        self.assertEqual({p: r["class"] for p, r in rows.items()}, recomputed)
        counts = collections.Counter(r["class"] for r in rows.values())
        self.assertEqual(
            dict(counts), {"blocking": 13, "advisory-by-design": 3, "unwired": 1}
        )
        self.assertEqual(self.census["counts"], dict(counts))
        self.assertEqual(
            self.census["expected"], {"unwired": 1, "advisory": 3, "blocking": 13}
        )
        for r in rows.values():
            self.assertIn(r["class"], CLASSES, r["script"])
            self.assertTrue(r["evidenceCommand"], r["script"])
            if r["class"] != "unwired":
                self.assertTrue(r["workflowJob"] or r["alsoWiredIn"], r["script"])
            if r["class"] == "blocking" and r["workflow"]:
                self.assertTrue(r["workflowJob"] and r["step"], r["script"])
                self.assertIn(
                    r["script"].rsplit("/", 1)[1],
                    self.lines(r["workflow"])[r["anchor"]["line"] - 1],
                )
            if r["class"] == "advisory-by-design":
                ev = r["advisoryEvidence"]
                self.assertIn(
                    "ADVISORY",
                    self.lines(ev["stepCommentAnchor"]["path"])[
                        ev["stepCommentAnchor"]["line"] - 1
                    ],
                )
                script_lines = self.lines(r["script"])
                exits = [
                    i for i, ln in enumerate(script_lines, 1) if ln.strip() == "exit 0"
                ]
                self.assertEqual(
                    ev["terminalExit0Anchor"], {"path": r["script"], "line": exits[-1]}
                )
        infra = rows["scripts/ci/check-infra-genericity.sh"]
        self.assertEqual(
            (infra["workflow"], infra["workflowJob"], infra["step"]),
            (".github/workflows/ci.yml", "quality-scripts", "Infra genericity gate"),
        )
        self.assertEqual(
            sorted(p for p, r in rows.items() if r["alsoWiredIn"]),
            ["scripts/ci/check-no-plan-leaks.sh", "scripts/ci/check_python_parity.py"],
        )
        for r in rows.values():
            for w in r["alsoWiredIn"]:
                self.assertEqual(w["path"], "scripts/pre-commit")
                self.assertIn(
                    r["script"].rsplit("/", 1)[1],
                    self.lines("scripts/pre-commit")[w["line"] - 1],
                )
        (unwired,) = [r for r in rows.values() if r["class"] == "unwired"]
        self.assertEqual(unwired["script"], "scripts/ci/check-comment-bloat.sh")
        inv = unwired["transitiveInvocation"]
        self.assertEqual(
            (inv["path"], inv["invokerClass"]),
            ("scripts/ci/quality-report.sh", "advisory-by-design"),
        )
        self.assertIn(
            "check-comment-bloat.sh", self.lines(inv["path"])[inv["line"] - 1]
        )
        self.assertIn(NEGATIVE_GREP, unwired["evidenceCommand"])
        self.assertEqual(
            {
                r["script"].rsplit("/", 1)[1]
                for r in rows.values()
                if r["class"] == "advisory-by-design"
            },
            {
                "check-comment-line-refs.sh",
                "check-comment-banners.sh",
                "quality-report.sh",
            },
        )
        self.assertEqual(len(rows["scripts/ci/lib/comment_scan.sh"]["sourcedBy"]), 4)

    def test_schemas_slurm_and_design_docs_exist_at_the_baseline(self) -> None:
        sch = self.inv["schemas"]
        self.assertEqual(
            sorted(sch["jsonFiles"]),
            sorted(
                f
                for f in self.files
                if f.startswith("schemas/") and f.endswith(".json")
            ),
        )
        self.assertEqual(sch["jsonCount"], 18)
        self.assertNotIn("schemas/policy.fbs", self.files)
        self.assertFalse(sch["rootPolicyFbsExists"])
        self.assertIn(sch["flatbuffers"], self.files)
        sl = self.inv["slurm"]
        self.assertEqual(
            sl["files"], sorted(f for f in self.files if f.startswith("tests/slurm/"))
        )
        self.assertIn(sl["driver"], self.files)
        dd = self.inv["designDocs"]
        self.assertEqual(
            sorted(dd["docs"]),
            sorted(
                f
                for f in self.files
                if f.startswith("docs/design/") and f != dd["index"]
            ),
        )
        self.assertEqual(dd["docCount"], 9)
        readme = self.lines(dd["index"])
        for value, line in dd["statusVocabularyAnchor"]["perValue"].items():
            self.assertTrue(readme[line - 1].startswith(f"- **{value}**"), value)
        self.assertEqual(
            readme[dd["maintenanceConvention"]["anchor"]["line"] - 1].strip(),
            "## Maintenance convention",
        )
        for row in dd["statusTable"]:
            self.assertIn(row["doc"], readme[row["line"] - 1])
            self.assertIn(row["status"], readme[row["line"] - 1])
            self.assertEqual(
                row["inVocabulary"], row["status"] in dd["statusVocabulary"]
            )
        self.assertEqual([r["line"] for r in dd["outOfVocabularyRows"]], [24, 26, 27])
        indexed = {r["doc"] for r in dd["statusTable"]}
        not_indexed = [d for d in dd["docs"] if d.rsplit("/", 1)[1] not in indexed]
        self.assertEqual([d["path"] for d in dd["docsNotInIndex"]], not_indexed)
        self.assertEqual(not_indexed, ["docs/design/post-horizon-input-unification.md"])

    def test_every_anchor_resolves_at_the_baseline_with_symbols_only_on_build_rs(
        self,
    ) -> None:
        for name, doc in (
            ("inventory.json", self.inv),
            ("gate-census.json", self.census),
        ):
            for a in walk_anchors(doc):
                path = a["path"]
                if path.startswith("plans/"):
                    continue
                self.assertIn(path, self.files, f"{name}: {path}")
                if "symbol" in a:
                    self.assertTrue(
                        path.endswith("build.rs"), f"{name}: symbol anchor on {path}"
                    )
                    self.assertTrue(
                        sc.symbol_resolves(path, a["symbol"], self.t),
                        f"{name}: {path}::{a['symbol']}",
                    )
                if "line" in a:
                    self.assertTrue(
                        1 <= a["line"] <= len(self.lines(path)),
                        f"{name}: {path}:{a['line']}",
                    )

    def test_prior_register_ids_resolve_and_the_drift_carries_both_anchors(
        self,
    ) -> None:
        register = backlog_parse.read_register(sc.BACKLOG)
        entry_lines = {}
        for i, ln in enumerate(register, 1):
            m = backlog_parse.ENTRY_RE.match(ln)
            if m:
                entry_lines.setdefault(m.group("id"), i)
        ids = set(ID_RE.findall(self.prior))
        self.assertTrue(
            {"CD-061", "CD-025", "OD-009", "CD-008", "PD-001", "PD-004"} <= ids
        )
        for id_ in ids:
            self.assertIn(id_, entry_lines, id_)
        for m in re.finditer(r"\| (CD-\d{3}|OD-\d{3}) \| L(\d+)", self.prior):
            self.assertEqual(entry_lines[m.group(1)], int(m.group(2)), m.group(0))
        self.assertIn("scripts/ci/check-infra-genericity.sh:79", self.prior)
        self.assertGreaterEqual(
            len(self.lines("scripts/ci/check-infra-genericity.sh")), 79
        )
        for line in (2013, 3182):
            self.assertIn(f"L{line}", self.prior)
            self.assertIn("schemas/policy.fbs", register[line - 1])
        mirror = "docs/design/reserved-seams-and-deferred-debt.md"
        self.assertIn(f"{mirror}:614", self.prior)
        self.assertIn("schemas/policy.fbs", self.lines(mirror)[613])
        self.assertIn("crates/cobre-io/schemas/policy.fbs", self.prior)
        self.assertIn("Do-not-touch", self.prior)
        self.assertIn("recorded, not raised", self.prior)
        self.assertNotIn(
            "**Alignment:**", self.prior
        )  # no disposition, no finding, no fix here


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_file_is_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])
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
