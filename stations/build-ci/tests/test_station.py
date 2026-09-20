"""Station tests for build-ci: the enforcement-surface inventory and the gate-wiring census.

Every fact is re-derived from the baseline blobs (`git show <pin>:<path>`), never from the worktree;
BACKLOG.md anchors are HEAD lines of the live register. Later build/CI tickets append their stage
classes here.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import shutil
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


LENSES = ("architecture", "over-engineering", "drift", "performance")
BUILD_RS = {
    "crates/cobre-solver/build.rs",
    "crates/cobre-sddp/build.rs",
    "crates/cobre-cli/build.rs",
}
STRENGTH = {"blocking", "advisory-by-design", "unwired", "transitively-advisory", "n-a"}
SWEEP_RE = re.compile(
    r"^(\.github/|scripts/|schemas/|crates/cobre-io/schemas/|Cargo\.toml$|crates/[^/]+/Cargo\.toml$"
    r"|crates/[^/]+/build\.rs$|tests/slurm/|docs/design/|ARCHITECTURE\.md$|CLAUDE\.md$)"
)
CRATE_SRC_RE = re.compile(r"^crates/[^/]+/(src|tests)/")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TIMING_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|µs|us|ns|s|sec|secs|seconds|min|mins|minutes?|h|hours?)\b",
    re.I,
)
DIFF_RE = re.compile(r"^\s*[-+]{3} |^\s*@@ |```", re.M)
ALIGN_VOCAB = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
PROMPT_ANCHOR_RE = re.compile(
    r"`((?:[\w.-]+/)+[\w.-]+|Cargo\.toml|ARCHITECTURE\.md|CLAUDE\.md):(\d+)"
)
MPICH_BUILD_LINES = (49, 87, 130, 180, 316, 377, 468, 522)


class CandidateEnvelopeTests(sc.StationCase):
    SLUG = "build-ci"
    SECTION_TITLE = "build-ci"

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = {
            lens: sc.load_json(cls.station_dir() / f"candidates-{lens}.json")
            for lens in LENSES
        }
        cls.census = sc.load_json(cls.station_dir() / "gate-census.json")
        cls.prompt = (cls.station_dir() / "attacker-prompt.md").read_text(
            encoding="utf-8"
        )
        cls.log = (cls.station_dir() / "attacker-log.md").read_text(encoding="utf-8")
        cls.t = cls.tree()
        cls.files = set(cls.t.files())

    def line_resolves(self, path: str, line) -> bool:
        return (
            path in self.files
            and isinstance(line, int)
            and 1 <= line <= len(self.t.read_text(path).splitlines())
        )

    def candidates(self, lens: str) -> list[dict]:
        return self.env[lens]["candidates"]

    def test_every_candidates_file_validates_through_the_shared_tool(self) -> None:
        for lens in LENSES:
            r = subprocess.run(
                [
                    sys.executable,
                    str(sc.TOOLS / "validate-envelope.py"),
                    "--role",
                    "attacker",
                    "--station",
                    "build-ci",
                    str(self.station_dir() / f"candidates-{lens}.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, f"{lens}: {r.stderr}")

    def test_header_names_the_station_the_pin_and_the_lens(self) -> None:
        for lens, env in self.env.items():
            self.assertEqual(
                (env["station"], env["stationLabel"], env["subStation"], env["lens"]),
                ("build-ci", "build-ci-docs", "build-ci", lens),
            )
            self.assertEqual(env["baseline"], self.baseline())
            self.assertIn(lens, LENSES)
            self.assertTrue(
                env["gatedAt"] and env["worker"].startswith("adversarial-attacker")
            )
            self.assertTrue(env["candidates"] or env["cleanVerdict"], lens)
            self.assertTrue(env["positives"], lens)

    def test_anchor_grammar_scope_and_resolution(self) -> None:
        for lens, env in self.env.items():
            for c in env["candidates"]:
                self.assertTrue(c["anchors"], c["title"])
                for a in c["anchors"]:
                    path = a["path"]
                    self.assertTrue(
                        SWEEP_RE.match(path), f"{lens}: {path} outside the sweep list"
                    )
                    self.assertIsNone(CRATE_SRC_RE.match(path), f"{lens}: {path}")
                    if "symbol" in a:
                        self.assertIn(
                            path, BUILD_RS, f"{lens}: symbol anchor on {path}"
                        )
                        self.assertTrue(IDENT_RE.match(a["symbol"]), a["symbol"])
                        self.assertTrue(
                            sc.symbol_resolves(path, a["symbol"], self.t),
                            f"{path}::{a['symbol']}",
                        )
                    else:
                        self.assertTrue(
                            self.line_resolves(path, a["line"]),
                            f"{lens}: {path}:{a['line']}",
                        )
                for side in ("docClaim", "treeFact"):
                    s = c.get(side)
                    if s:
                        self.assertTrue(
                            self.line_resolves(s["path"], s["line"]),
                            f"{lens}: {side} {s['path']}:{s['line']}",
                        )
                ts = c.get("twoSite")
                if ts:
                    for k in ("prose", "script", "step"):
                        self.assertTrue(
                            self.line_resolves(ts[k]["path"], ts[k]["line"]),
                            f"{lens}: twoSite.{k}",
                        )

    def test_enforcement_strength_agrees_with_the_census(self) -> None:
        rows = {r["script"]: r for r in self.census["rows"]}
        bloat = rows["scripts/ci/check-comment-bloat.sh"]
        self.assertEqual(
            (
                bloat["transitiveInvocation"]["path"],
                bloat["transitiveInvocation"]["line"],
            ),
            ("scripts/ci/quality-report.sh", 132),
        )
        self.assertIn("transitively-advisory", self.log)
        self.assertIn("scripts/ci/quality-report.sh:132", self.log)
        for lens, env in self.env.items():
            for c in env["candidates"]:
                strength = c["enforcementStrength"]
                self.assertIn(strength, STRENGTH, c["title"])
                self.assertNotEqual(strength, "unwired", c["title"])
                script = c.get("script")
                if script:
                    self.assertNotEqual(strength, "n-a", c["title"])
                    row = rows[script]
                    expected = row["class"]
                    if expected == "unwired" and row.get("transitiveInvocation"):
                        expected = "transitively-advisory"
                    self.assertEqual(strength, expected, f"{lens}: {script}")
                if strength == "advisory-by-design":
                    blob = json.dumps(c)
                    self.assertTrue("ADVISORY" in blob or "exit 0" in blob, c["title"])
        advisory = {
            r["script"].rsplit("/", 1)[1]
            for r in self.census["rows"]
            if r["class"] == "advisory-by-design"
        }
        self.assertEqual(
            advisory,
            {
                "check-comment-line-refs.sh",
                "check-comment-banners.sh",
                "quality-report.sh",
            },
        )
        for name in advisory:
            self.assertFalse(
                any(
                    (c.get("script") or "").endswith(name)
                    for env in self.env.values()
                    for c in env["candidates"]
                ),
                f"{name} may appear only in positives",
            )

    def test_architecture_two_site_and_pattern_probes(self) -> None:
        cands = self.candidates("architecture")
        scan = [c for c in cands if "SCAN_DIRS" in json.dumps(c) and c.get("twoSite")]
        self.assertEqual(len(scan), 1)
        ts = scan[0]["twoSite"]
        self.assertEqual(ts["prose"]["path"], "CLAUDE.md")
        self.assertIn(ts["prose"]["line"], range(39, 42))
        self.assertEqual(ts["script"]["path"], "scripts/ci/check-infra-genericity.sh")
        self.assertIn(ts["script"]["line"], range(62, 69))
        self.assertEqual(
            (ts["step"]["path"], ts["step"]["line"]), (".github/workflows/ci.yml", 262)
        )
        self.assertEqual(scan[0]["alignmentHint"], "advances-1")
        self.assertIn("IV.1", scan[0]["alignmentCitation"])
        blob = json.dumps(scan[0])
        self.assertTrue(
            "CLAUDE.md:39" in blob and "cobre-model" in blob and "cobre-network" in blob
        )
        self.assertIn("framework", scan[0]["fixShape"])
        pattern = [
            c
            for c in cands
            if any(
                a.get("line") == 79 and a["path"].endswith("check-infra-genericity.sh")
                for a in c["anchors"]
            )
        ]
        self.assertEqual(len(pattern), 1)
        blob = json.dumps(pattern[0])
        for token in (
            "cuts_active",
            "cutting_edge",
            "cut_nz_per_col",
            "137",
            "138",
            "141",
            "188",
        ):
            self.assertIn(token, blob)
        cited = {x["path"] for x in pattern[0].get("citedContext") or []}
        self.assertIn("crates/cobre-solver/src/freeze.rs", cited)
        self.assertFalse(any("freeze.rs" in a["path"] for a in pattern[0]["anchors"]))
        self.assertEqual(self.env["architecture"]["handoffs"], [])
        self.assertFalse(any(c.get("partIRef") == "I.3-6" for c in cands))

    def test_over_engineering_mpich_slow_tests_allowlist_and_umbrella(self) -> None:
        env = self.env["over-engineering"]
        text = json.dumps(env)
        self.assertNotIn("umbrella", text)
        self.assertNotIn("crates/cobre/", text)
        mpich = [
            c
            for c in env["candidates"]
            if (".github/workflows/ci.yml", 49)
            in {(a["path"], a.get("line")) for a in c["anchors"]}
        ]
        self.assertEqual(len(mpich), 1)
        anchors = {(a["path"], a.get("line")) for a in mpich[0]["anchors"]}
        for line in MPICH_BUILD_LINES:
            self.assertIn((".github/workflows/ci.yml", line), anchors)
        self.assertIn((".github/workflows/mpi-slurm.yml", 63), anchors)
        self.assertIn((".github/workflows/release-mpi.yml", 75), anchors)
        self.assertIn("variant", json.dumps(mpich[0]).lower())
        self.assertIn("matrix", mpich[0]["fixShape"])
        self.assertNotEqual(mpich[0]["alignmentHint"], "conflicts")
        yml = self.t.read_text(".github/workflows/ci.yml").splitlines()
        self.assertEqual(
            [i for i, ln in enumerate(yml, 1) if "Build MPICH from source" in ln],
            list(MPICH_BUILD_LINES),
        )
        slow = [
            c
            for c in env["candidates"]
            if any(
                a["path"]
                in ("crates/cobre-cli/Cargo.toml", "crates/cobre-io/Cargo.toml")
                for a in c["anchors"]
            )
        ]
        self.assertEqual(
            sorted(a["path"] for c in slow for a in c["anchors"][:1]),
            ["crates/cobre-cli/Cargo.toml", "crates/cobre-io/Cargo.toml"],
        )
        for c in slow:
            self.assertIn("reserved for workspace consistency", json.dumps(c))
            self.assertTrue(c["reservedSeamsCheck"]["checked"])
        for c in env["candidates"]:
            self.assertTrue(c.get("reservedSeamsCheck", {}).get("checked"), c["title"])
        allow = [
            p
            for p in env["positives"]
            if "allow-rationale-allowlist.txt" in p["subject"]
        ]
        self.assertEqual(len(allow), 1)
        self.assertTrue(
            ":1" in allow[0]["sanctionedBy"] and ":15" in allow[0]["sanctionedBy"]
        )
        stubs = [
            p
            for p in env["positives"]
            if p["subject"].startswith("The five reserved stub members")
        ]
        self.assertEqual(len(stubs), 1)
        self.assertIn("ARCHITECTURE.md:108-125", stubs[0]["sanctionedBy"])

    def test_drift_candidates_are_paired_and_the_readme_and_part_i_probes_land(
        self,
    ) -> None:
        env = self.env["drift"]
        for c in env["candidates"]:
            for side in ("docClaim", "treeFact"):
                self.assertTrue(
                    c.get(side) and c[side]["text"].strip(), f"{c['title']}: {side}"
                )
        readme = [
            c
            for c in env["candidates"]
            if {26, 27}
            <= {
                a.get("line")
                for a in c["anchors"]
                if a["path"] == "docs/design/README.md"
            }
        ]
        self.assertEqual(len(readme), 1)
        self.assertEqual(readme[0]["docClaim"]["path"], "docs/design/README.md")
        blob = json.dumps(readme[0])
        for name in (
            "anticipated-fixed-post-horizon-commitments.md",
            "external-scenarios-are-authoritative.md",
        ):
            self.assertIn(name, blob)
        self.assertIn("Implemented (retained pending fold into the live spec)", blob)
        item6 = [c for c in env["candidates"] if c.get("partIRef") == "I.3-6"]
        self.assertEqual(len(item6), 1)
        self.assertEqual(
            (item6[0]["treeFact"]["path"], item6[0]["treeFact"]["line"]),
            ("scripts/ci/check-infra-genericity.sh", 74),
        )
        self.assertEqual(
            self.t.read_text("scripts/ci/check-infra-genericity.sh")
            .splitlines()[73]
            .strip(),
            "EXCLUDED_FILES=()",
        )
        (handoff,) = env["handoffs"]
        self.assertEqual(
            (handoff["partIRef"], handoff["stationVerdict"]), ("I.3-6", "claim-stale")
        )
        self.assertNotIn("disposition", handoff)
        self.assertNotIn("alignment", {k.lower() for k in handoff})
        self.assertEqual({e["line"] for e in handoff["baselineEvidence"]}, {70, 74})
        for e in handoff["baselineEvidence"]:
            self.assertTrue(self.line_resolves(e["path"], e["line"]))

    def test_performance_entries_are_unmeasured_pointers(self) -> None:
        env = self.env["performance"]
        self.assertTrue(env["candidates"])
        for c in env["candidates"]:
            self.assertEqual(c["measured"], "UNMEASURED", c["title"])
            self.assertTrue(str(c.get("mechanism", "")).strip(), c["title"])
            blob = " ".join(
                [
                    c["title"],
                    c["mechanism"],
                    c["fixShape"],
                    c["evidence"].get("reading", ""),
                ]
            )
            self.assertIsNone(
                TIMING_RE.search(blob), f"{c['title']}: {TIMING_RE.search(blob)}"
            )
            self.assertNotIn("PD-", blob)
            self.assertNotIn("cobre_reduzido_2", blob)
            self.assertNotRegex(blob, r"\d+\s*(?:x|×)\s+(?:faster|slower)|\d+\s*%")
        yml = self.t.read_text(".github/workflows/ci.yml").splitlines()
        ceilings = [i for i, ln in enumerate(yml, 1) if "timeout-minutes" in ln]
        self.assertEqual(len(ceilings), 14)
        self.assertTrue(env["coverage"]["P1 CI wall-time / cache"])

    def test_fix_shapes_are_prose(self) -> None:
        for lens, env in self.env.items():
            for c in env["candidates"]:
                fix = c["fixShape"]
                self.assertGreaterEqual(len(fix), 20)
                self.assertIsNone(DIFF_RE.search(fix), f"{lens}: {c['title']}")
                self.assertNotIn("\n+", fix)
                self.assertNotIn("\n-", fix)

    def test_prompt_freezes_the_e7_contract(self) -> None:
        lens_blocks = re.findall(r"^## Lens: (\S+)", self.prompt, re.M)
        self.assertEqual(
            lens_blocks, ["architecture", "over-engineering", "drift", "performance"]
        )
        for heading, body in zip(
            lens_blocks, re.split(r"^## Lens: .*$", self.prompt, flags=re.M)[1:]
        ):
            self.assertGreaterEqual(len(re.findall(r"\*\*P\d", body)), 2, heading)
        for token in (
            "ANCHOR FORM",
            "ENFORCEMENT STRENGTH",
            "TWO-SITE RULE",
            "PULL, DON'T PUSH",
            "Read-only",
            "One JSON object",
            '"enforcementStrength"',
            '"docClaim"',
            '"treeFact"',
            '"measured"',
            '{ "path": ".github/workflows/ci.yml", "line": 49 }',
            "drift-unpaired",
        ):
            self.assertIn(token, self.prompt, token)
        self.assertIn(self.baseline(), self.prompt)
        checked = 0
        for path, line in PROMPT_ANCHOR_RE.findall(self.prompt):
            if path.startswith("plans/") or path not in self.files:
                continue  # informal cites (`ci.yml:262`) and the superseded ticket column are not anchors
            checked += 1
            self.assertTrue(
                self.line_resolves(path, int(line)), f"prompt anchor {path}:{line}"
            )
        self.assertGreater(checked, 40)

    def test_attacker_log_records_the_dispatch_and_the_gate(self) -> None:
        self.assertIn(self.baseline(), self.log)
        rows = [
            ln
            for ln in self.log.splitlines()
            if ln.startswith("| `adversarial-attacker`")
        ]
        self.assertEqual(len(rows), 4)
        for lens in LENSES:
            self.assertEqual(sum(f"| {lens} |" in r for r in rows), 1, lens)
        for heading in (
            "## Dispatch table",
            "## Rejections",
            "## Census join",
            "## Gaps",
            "## Read-only verification",
            "## Needs-human items surfaced for the gate",
        ):
            self.assertIn(heading, self.log)
        self.assertIn("`drift`", self.log)
        self.assertIn("ARCHITECTURE.md`, `CLAUDE.md`", self.log)
        self.assertIn("valid: performance", self.log)


INGEST_STATES = {
    "accepted",
    "rejected-anchor",
    "rejected-sanctioned-advisory",
    "rejected-re-raise",
    "rejected-defender",
    "merged",
    "held-conflicts",
    "unresolved",
}
INGEST_DISPOSITIONS = {
    "defended",
    "dup-of",
    "anchor-missing",
    "cleared-sanctioned",
    "re-raise-rejected",
    "conflicts-held",
    "returned-for-sharpening",
}
ENFORCEMENT = {
    "blocking",
    "advisory-by-design",
    "unwired",
    "transitively-advisory",
    "not-a-gate",
}
HANDOFF_FORBIDDEN = {"disposition", "severity", "id", "alignment", "recommendation"}
PROBE_TITLE = "INGEST ANCHOR PROBE — build-ci (2026-09, baseline)"


class IngestTests(sc.StationCase):
    SLUG = "build-ci"
    SECTION_TITLE = "build-ci"

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = sc.load_json(cls.station_dir() / "verdicts.json")
        cls.verdicts = cls.doc["verdicts"]
        cls.env = {
            lens: sc.load_json(cls.station_dir() / f"candidates-{lens}.json")
            for lens in LENSES
        }
        cls.log = (cls.station_dir() / "ingest-log.md").read_text(encoding="utf-8")
        cls.probe = (cls.station_dir() / "anchor-probe.md").read_text(encoding="utf-8")
        cls.brief = (cls.station_dir() / "defender-prompt.md").read_text(
            encoding="utf-8"
        )
        cls.handoff = sc.load_json(cls.station_dir() / "partI-handoff.json")
        cls.census = sc.load_json(cls.station_dir() / "gate-census.json")
        cls.t = cls.tree()
        cls.files = set(cls.t.files())

    def received(self) -> dict[str, dict]:
        out = {}
        for lens, env in self.env.items():
            for i, c in enumerate(env["candidates"]):
                out[f"{lens}-{i:02d}"] = c
        return out

    def line_resolves(self, path: str, line) -> bool:
        return (
            path in self.files
            and isinstance(line, int)
            and 1 <= line <= len(self.t.read_text(path).splitlines())
        )

    def test_every_candidate_has_exactly_one_verdict_entry(self) -> None:
        received = self.received()
        self.assertEqual(set(self.verdicts), set(received))
        self.assertEqual(self.doc["baseline"], self.baseline())
        self.assertEqual(self.doc["counts"]["received"], len(received))
        for ref, e in self.verdicts.items():
            self.assertEqual(e["candidateRef"], ref)
            self.assertEqual(e["title"], received[ref]["title"])
            self.assertIn(e["disposition"], INGEST_DISPOSITIONS, ref)
            self.assertIn(e["state"], INGEST_STATES, ref)
            self.assertIn(
                e["verdict"], (None, "confirmed", "dismissed", "unresolved"), ref
            )

    def test_states_follow_disposition_verdict_and_basis(self) -> None:
        for ref, e in self.verdicts.items():
            d, v, st = e["disposition"], e["verdict"], e["state"]
            if d == "dup-of":
                self.assertEqual(st, "merged", ref)
                self.assertIn(e["mergedInto"], self.verdicts, ref)
                self.assertIn(
                    ref, self.verdicts[e["mergedInto"]]["mergedFrom"] or [], ref
                )
                self.assertIsNone(v)
            elif d == "anchor-missing":
                self.assertEqual(st, "rejected-anchor", ref)
            elif d == "cleared-sanctioned":
                self.assertEqual(st, "rejected-sanctioned-advisory", ref)
            elif d == "defended":
                self.assertIn(v, ("confirmed", "dismissed", "unresolved"), ref)
                if v == "confirmed":
                    self.assertEqual(st, "accepted", ref)
                    self.assertTrue(e["survivingClaim"], ref)
                    self.assertNotEqual(
                        e["survivingClaim"].strip(), e["title"].strip(), ref
                    )
                    self.assertNotIn(e["title"], e["survivingClaim"], ref)
                    self.assertIsNone(e.get("dismissalBasis"), ref)
                elif v == "dismissed":
                    self.assertIn(
                        st, ("rejected-defender", "rejected-sanctioned-advisory"), ref
                    )
                    self.assertIn(
                        e["dismissalBasis"], self.doc["dismissalBasisVocabulary"], ref
                    )
                    self.assertTrue(e["basisCitation"], ref)
                    if e["dismissalBasis"] == "sanctioned-advisory":
                        self.assertEqual(st, "rejected-sanctioned-advisory", ref)
                        self.assertIn(
                            e["sanctionedBy"], self.doc["sanctionedByClosedSet"], ref
                        )
                    else:
                        self.assertIsNone(e.get("sanctionedBy"), ref)
                else:
                    self.assertEqual(st, "unresolved", ref)
                    self.assertTrue(e["needsHuman"], ref)
            if st == "rejected-sanctioned-advisory":
                cite = e.get("sanctionedBy") or "; ".join(
                    e["sanctionedAdvisoryScreen"]["touches"]
                )
                self.assertTrue(cite, ref)

    def test_accepted_anchors_resolve_and_symbols_sit_on_build_rs(self) -> None:
        for ref, e in self.verdicts.items():
            for a in e["anchors"]:
                self.assertTrue(SWEEP_RE.match(a["path"]), f"{ref}: {a['path']}")
                if "symbol" in a:
                    self.assertIn(a["path"], BUILD_RS, ref)
                    self.assertTrue(
                        sc.symbol_resolves(a["path"], a["symbol"], self.t), ref
                    )
                else:
                    self.assertTrue(
                        self.line_resolves(a["path"], a["line"]), f"{ref}: {a}"
                    )
        blocks = re.findall(r"^\*\*CD-9\d\d · probe · (\S+)\*\*", self.probe, re.M)
        self.assertEqual(sorted(blocks), sorted(self.verdicts))
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
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertEqual(json.loads(r.stdout)["failures"], [])

    def test_defended_verdicts_carry_strength_measurement_and_tags(self) -> None:
        rows = {r["script"]: r for r in self.census["rows"]}
        for ref, e in self.verdicts.items():
            if e["disposition"] != "defended" or e["verdict"] == "unresolved":
                continue
            self.assertIn(e["enforcementStrength"], ENFORCEMENT, ref)
            self.assertNotEqual(e["enforcementStrength"], "unwired", ref)
            cand = self.received()[ref]
            script = cand.get("script")
            if script:
                expected = rows[script]["class"]
                if expected == "unwired":
                    expected = "transitively-advisory"
                self.assertEqual(e["enforcementStrength"], expected, ref)
            self.assertEqual(
                e["measurement"],
                "UNMEASURED" if e["lens"] == "performance" else "n/a",
                ref,
            )
            if e["lens"] == "performance":
                blob = (e["argument"] or "") + " " + (e.get("survivingClaim") or "")
                self.assertIsNone(TIMING_RE.search(blob), ref)
                self.assertNotIn("PD-", blob, ref)
            self.assertIn(e["alignmentHint"], ALIGN_VOCAB, ref)
            self.assertEqual(e["conflicts"], e["alignmentHint"] == "conflicts", ref)
            for tag in ("partIRef", "reRaiseOf"):
                self.assertEqual(e.get(tag), cand.get(tag), f"{ref}: {tag}")
            self.assertFalse(DIFF_RE.search(e["argument"] or ""), ref)
        sharpens = [
            e for e in self.verdicts.values() if e.get("priorRelation") == "sharpens"
        ]
        self.assertTrue(sharpens)
        for e in sharpens:
            self.assertEqual(e["priorId"], "CD-061")
            self.assertEqual(e["reRaiseOf"], "CD-061")

    def test_part_i_handoff_is_evidence_without_a_disposition(self) -> None:
        h = self.handoff
        self.assertEqual(h["partIRef"], "I.3-6")
        self.assertEqual(h["stationVerdict"], "claim-stale")
        self.assertEqual(h["baseline"], self.baseline())
        self.assertFalse(HANDOFF_FORBIDDEN & {k.lower() for k in h}, sorted(h))
        self.assertTrue(h["baselineEvidence"])
        lines = {e["line"] for e in h["baselineEvidence"]}
        self.assertTrue({74, 70} <= lines, lines)
        for e in h["baselineEvidence"]:
            self.assertEqual(e["path"], "scripts/ci/check-infra-genericity.sh")
            self.assertTrue(self.line_resolves(e["path"], e["line"]), e)
        script = self.t.read_text("scripts/ci/check-infra-genericity.sh").splitlines()
        self.assertEqual(script[73].strip(), "EXCLUDED_FILES=()")
        self.assertIn("retired", script[71])
        self.assertIn("formerly sat here", script[39])
        drift = [e for e in self.verdicts.values() if e.get("partIRef") == "I.3-6"]
        self.assertEqual([e["lens"] for e in drift], ["drift"])
        self.assertEqual(len(self.env["drift"]["handoffs"]), 1)

    def test_defender_brief_fixes_the_station_obligations(self) -> None:
        for s in self.doc["sanctionedByClosedSet"]:
            self.assertIn(s, self.brief, s)
        for token in (
            "blocking | advisory-by-design | unwired | transitively-advisory",
            "not-a-gate",
            "UNMEASURED",
            "strictly narrower",
            "cobre-model",
            "cobre-network",
            "quality-report.sh:132",
            self.baseline(),
        ):
            self.assertIn(token, self.brief, token)
        for basis in self.doc["dismissalBasisVocabulary"]:
            self.assertIn(basis, self.brief, basis)

    def test_ingest_log_has_one_roster_row_per_candidate_and_the_pin(self) -> None:
        self.assertIn(self.baseline(), self.log)
        roster = self.log.split("## Per-candidate roster", 1)[1]
        rows = [
            ln
            for ln in roster.splitlines()
            if re.match(
                r"^\| (architecture|over-engineering|drift|performance)-\d\d \|", ln
            )
        ]
        self.assertEqual(
            sorted(r.split("|")[1].strip() for r in rows), sorted(self.verdicts)
        )
        for heading in (
            "## Candidate census",
            "### Anchor rejections",
            "## Cleared (sanctioned)",
            "## Dup-of merges",
            "## Pull-don't-push and two-site screen",
            "## Part-I item-6 evidence package",
            "## Defender summary",
            "## Reconciliation",
        ):
            self.assertIn(heading, self.log, heading)
        self.assertIn("quality-report.sh:132", self.log)
        self.assertIn("transitively-advisory", self.log)
        self.assertIn("mpi-slurm.yml", self.log)
        self.assertIn("release-mpi.yml", self.log)


SECTION_TITLE = "STATION 7 — build/CI/scripts/schemas/docs (2026-09)"
FINDING_ID_RE = re.compile(r"^(CD|PD|OD|TD)-\d{3}$")
HEADING_ID_RE = re.compile(r"^\*\*((?:CD|PD|OD|TD)-\d{3}) · Sev ([ABC]) ·", re.M)
RANK = {"A": 3, "B": 2, "C": 1}
WIRING_VOCAB = {"blocking", "advisory-by-design", "unwired", "transitively-advisory"}


class CalibrationTests(sc.StationCase):
    SLUG = "build-ci"
    SECTION_TITLE = SECTION_TITLE

    @classmethod
    def setUpClass(cls) -> None:
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.td = sc.load_json(cls.station_dir() / "td-queue.json")
        cls.census = sc.load_json(cls.station_dir() / "gate-census.json")
        cls.verdicts = sc.load_json(cls.station_dir() / "verdicts.json")["verdicts"]
        cls.register = sc.BACKLOG.read_text(encoding="utf-8")
        start = cls.register.index("\n### " + SECTION_TITLE + "\n")
        end = cls.register.find("\n## ", start + 1)
        cls.section = cls.register[start:end]
        cls.t = cls.tree()

    def run_tool(self, tool: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(sc.TOOLS / tool), *extra, SECTION_TITLE],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
        )

    def entry_block(self, fid: str) -> str:
        start = self.section.index(f"\n**{fid} · Sev ")
        rest = self.section[start + 1 :]
        m = re.search(
            r"\n(?:\*\*(?:CD|PD|OD|TD)-\d{3} · Sev |\*\*Informational · |#### )", rest
        )
        return rest[: m.start()] if m else rest

    def test_ids_are_well_formed_contiguous_and_unique_across_the_register(
        self,
    ) -> None:
        rows = self.cal["assigned"]
        self.assertTrue(rows)
        by_class: dict[str, list[int]] = {}
        for r in rows:
            self.assertRegex(r["id"], FINDING_ID_RE)
            self.assertIn(
                r["class"], ("CD", "OD")
            )  # no PD (informational station), no TD (epic 8)
            by_class.setdefault(r["class"], []).append(int(r["id"].split("-")[1]))
        for cls_, nums in by_class.items():
            floor = self.cal["idFloorsSeen"][cls_]
            self.assertEqual(nums, list(range(floor, floor + len(nums))), cls_)
        headings = HEADING_ID_RE.findall(self.register)
        ids = [h[0] for h in headings]
        self.assertEqual(
            len(ids), len(set(ids)), "duplicate finding heading in the register"
        )
        section_ids = [h[0] for h in HEADING_ID_RE.findall(self.section)]
        self.assertEqual(sorted(section_ids), sorted(r["id"] for r in rows))
        prior = self.register[: self.register.index("\n### " + SECTION_TITLE + "\n")]
        for r in rows:
            self.assertNotRegex(prior, rf"^\*\*{r['id']} ·", r["id"])
        refs = [r["candidateRef"] for r in rows]
        self.assertEqual(
            refs,
            sorted(
                refs,
                key=lambda x: (
                    ["architecture", "drift", "over-engineering", "performance"].index(
                        x.rsplit("-", 1)[0]
                    ),
                    x,
                ),
            ),
        )

    def test_every_confirmed_verdict_is_minted_or_informational_and_nothing_else_is(
        self,
    ) -> None:
        confirmed = {
            ref for ref, e in self.verdicts.items() if e["verdict"] == "confirmed"
        }
        minted = {r["candidateRef"] for r in self.cal["assigned"]}
        info = {r["candidateRef"] for r in self.cal["informational"]}
        self.assertEqual(minted | info, confirmed)
        self.assertFalse(minted & info)
        self.assertTrue(all(r.startswith("performance-") for r in info))
        self.assertFalse(any(r.startswith("performance-") for r in minted))
        for r in self.cal["informational"]:
            self.assertEqual(r["status"], "UNMEASURED")
        self.assertNotRegex(
            self.section, r"^\*\*PD-\d{3}", "PD entry in an informational-only station"
        )
        cleared = {c["candidateRef"] for c in self.cal["cleared"]}
        self.assertEqual(
            cleared,
            {ref for ref, e in self.verdicts.items() if e["verdict"] == "dismissed"},
        )
        self.assertEqual(
            {d["candidateRef"] for d in self.cal["dupOf"]},
            {ref for ref, e in self.verdicts.items() if e["disposition"] == "dup-of"},
        )

    def test_severity_calibration_downgrades_are_auditable(self) -> None:
        for r in self.cal["assigned"]:
            self.assertIn(r["severity"], ("A", "B", "C"), r["id"])
            self.assertIn(r["reviewerRating"], ("A", "B", "C"), r["id"])
            self.assertLessEqual(
                RANK[r["severity"]], RANK[r["reviewerRating"]], r["id"]
            )
            if RANK[r["severity"]] < RANK[r["reviewerRating"]]:
                self.assertTrue(r["downgradeReason"], r["id"])
                self.assertIn(
                    f"- **Reviewer rating:** {r['reviewerRating']} — downgraded to {r['severity']}",
                    self.section,
                    r["id"],
                )
            else:
                self.assertIsNone(r["downgradeReason"], r["id"])
            self.assertTrue(r["calibrationBasis"], r["id"])
            self.assertIn(r["effort"], ("S", "M", "L"))
            self.assertIn(r["confidence"], ("high", "med", "low"))
        self.assertEqual(
            self.cal["counts"]["downgrades"],
            sum(1 for r in self.cal["assigned"] if r["downgradeReason"]),
        )
        self.assertEqual(self.cal["counts"]["upgrades"], 0)

    def test_alignment_carries_a_roadmap_citation_and_conflicts_are_held_with_alternatives(
        self,
    ) -> None:
        for r in self.cal["assigned"]:
            self.assertIn(r["alignmentHint"], ALIGN_VOCAB, r["id"])
            self.assertTrue(r["alignmentCites"], r["id"])
            self.assertEqual(r["conflicts"], r["alignmentHint"] == "conflicts", r["id"])
            if r["conflicts"]:
                self.assertTrue(r.get("alternative"), r["id"])
            block = self.entry_block(r["id"])
            line = re.search(r"^- \*\*Alignment:\*\* (.+)$", block, re.M)
            self.assertIsNotNone(line, r["id"])
            retag = re.search(
                r"; station hint: (\S+), retagged \d{4}-\d{2}-\d{2} by alignment/alignment-ledger\.json\)$",
                line.group(1),
            )
            if retag:
                self.assertEqual(retag.group(1), r["alignmentHint"], r["id"])
                self.assertIn(line.group(1).split(" (", 1)[0], ALIGN_VOCAB, r["id"])
            else:
                self.assertTrue(
                    line.group(1).startswith(r["alignmentHint"] + " ("), r["id"]
                )
            self.assertTrue(
                "Part " in line.group(1) or "advances no phase" in line.group(1),
                r["id"],
            )
        self.assertEqual(self.cal["counts"]["conflicts"], 0)
        mpich = next(
            r
            for r in self.cal["assigned"]
            if r["candidateRef"] == "over-engineering-00"
        )
        self.assertEqual(mpich["alignmentHint"], "neutral")
        self.assertIn("one-consumer objection waived", mpich["waiver"])
        self.assertEqual(
            self.cal["pullDontPush"]["mpichConsumers"],
            {"ci.yml": 8, "mpi-slurm.yml": 1, "release-mpi.yml": 1},
        )
        self.assertIn(
            "**Pull-don't-push:** one-consumer objection waived", self.section
        )

    def test_two_site_rule_names_three_site_kinds_on_every_hard_rule_fix_shape(
        self,
    ) -> None:
        hard = [r for r in self.cal["assigned"] if r["hardRule"]]
        self.assertEqual(
            sorted(r["hardRule"] for r in hard), ["infra-genericity", "unsafe-code"]
        )
        for r in hard:
            kinds = {s["kind"] for s in r["sites"]}
            self.assertEqual(kinds, {"rule", "script", "ci_step"}, r["id"])
            for s in r["sites"]:
                self.assertTrue(self.line_ok(s["path"], s["line"]), s)
                if s["kind"] == "ci_step":
                    self.assertTrue(s["job"] and s["step"], s)
            self.assertIn(f"- **Two-site rule ({r['hardRule']}):**", self.section)
        gen = next(r for r in hard if r["hardRule"] == "infra-genericity")
        self.assertEqual(gen["severity"], "B")
        self.assertEqual(gen["category"], "leaky-boundary")
        self.assertIn("CD-010", gen["calibrationBasis"])
        self.assertTrue(
            any(
                a["path"] == "scripts/ci/check-infra-genericity.sh"
                and a.get("line") == 62
                for a in gen["anchors"]
            )
        )
        sites = {s["kind"]: s for s in gen["sites"]}
        self.assertEqual(
            (sites["rule"]["path"], sites["rule"]["line"]), ("CLAUDE.md", 39)
        )
        self.assertEqual(
            (sites["script"]["path"], sites["script"]["line"]),
            ("scripts/ci/check-infra-genericity.sh", 62),
        )
        self.assertEqual(
            (
                sites["ci_step"]["path"],
                sites["ci_step"]["line"],
                sites["ci_step"]["job"],
                sites["ci_step"]["step"],
            ),
            (
                ".github/workflows/ci.yml",
                261,
                "quality-scripts",
                "Infra genericity gate",
            ),
        )
        self.assertEqual(gen["alignmentHint"], "advances-1")
        self.assertIn("IV.1", gen["alignmentCites"])
        self.assertIn(
            "framework", gen["fixShape"]
        )  # amends the existing gate; a new framework is disclaimed

    def line_ok(self, path: str, line: int) -> bool:
        files = set(self.t.files())
        return path in files and 1 <= line <= len(self.t.read_text(path).splitlines())

    def test_gate_rows_carry_the_census_enforcement_and_the_census_table_renders(
        self,
    ) -> None:
        rows = {r["script"]: r for r in self.census["rows"]}
        for r in self.cal["assigned"]:
            if r["isGate"]:
                self.assertIn(r["enforcement"], WIRING_VOCAB, r["id"])
                if r["script"]:
                    expected = rows[r["script"]]["class"]
                    if expected == "unwired" and rows[r["script"]].get(
                        "transitiveInvocation"
                    ):
                        expected = "transitively-advisory"
                    self.assertEqual(r["enforcement"], expected, r["id"])
                self.assertIn(
                    f"- **Enforcement:** {r['enforcement']}",
                    self.entry_block(r["id"]),
                    r["id"],
                )
            else:
                self.assertIsNone(r["enforcement"], r["id"])
        table = self.section.split("#### Gate-wiring census", 1)[1].split("\n#### ", 1)[
            0
        ]
        trows = [ln for ln in table.splitlines() if ln.startswith("| `")]
        self.assertEqual(len(trows), 17)
        for ln in trows:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            wiring = cells[2].strip("*")
            self.assertIn(wiring, WIRING_VOCAB, ln)
            if wiring != "transitively-advisory":
                self.assertNotEqual(cells[3], "-", ln)
                self.assertNotEqual(cells[4], "-", ln)
        bloat = next(ln for ln in trows if "check-comment-bloat.sh" in ln)
        self.assertIn("**transitively-advisory**", bloat)
        self.assertIn("negative grep", bloat)
        self.assertIn("scripts/ci/quality-report.sh:132", bloat)
        self.assertEqual(sum(1 for ln in trows if "transitively-advisory" in ln), 1)

    def test_drift_entries_name_the_document_and_the_contradicting_tree_fact(
        self,
    ) -> None:
        drift = [r for r in self.cal["assigned"] if r["lens"] == "drift"]
        self.assertTrue(drift)
        for r in drift:
            self.assertTrue(r["docClaim"] and r["treeFact"], r["id"])
            for side in ("docClaim", "treeFact"):
                self.assertTrue(
                    self.line_ok(r[side]["path"], r[side]["line"]), (r["id"], side)
                )
            self.assertIn(
                f"doc claim `{r['docClaim']['path']}:{r['docClaim']['line']}`",
                self.section,
                r["id"],
            )
            self.assertIn(
                f"tree fact `{r['treeFact']['path']}:{r['treeFact']['line']}`",
                self.section,
                r["id"],
            )
        item6 = next(r for r in drift if r["partIRef"] == "I.3-6")
        self.assertEqual(
            (item6["treeFact"]["path"], item6["treeFact"]["line"]),
            ("scripts/ci/check-infra-genericity.sh", 74),
        )
        block = self.section.split("#### Part-I cross-references", 1)[1].split(
            "\n#### ", 1
        )[0]
        for token in (
            "claim-stale",
            "check-infra-genericity.sh:74",
            ":38-44",
            "Epic 9",
            "NO disposition",
        ):
            self.assertIn(token, block, token)
        self.assertNotIn("Alignment:** advances", block)

    def test_td_queue_is_an_explicit_empty_envelope_naming_epic_8(self) -> None:
        self.assertEqual(self.td["rows"], [])
        self.assertIn("epic 8", self.td["emptyReason"])
        self.assertIn("epic 8", self.td["owner"])
        self.assertEqual(self.td["baseline"], self.baseline())
        self.assertIn("testing-architecture.md", self.td["yardstick"])

    def test_the_three_register_checkers_pass_over_the_station_section(self) -> None:
        for tool, extra in (
            ("check-anchors.py", ()),
            ("check-reraise.py", ()),
            ("fields-check.py", ()),
            ("fields-check.py", ("--require", "Alignment")),
        ):
            r = self.run_tool(tool, *extra)
            self.assertEqual(r.returncode, 0, f"{tool} {extra}: {r.stdout}{r.stderr}")
        self.assertIn(self.baseline(), self.section)
        gate_block = self.section.split("#### Owner gate — decisions", 1)[1].split(
            "\n#### ", 1
        )[0]
        self.assertTrue(
            "_(pending — filled by the gate ticket)_" in gate_block
            or re.search(
                r"^\*\*Gate: RETURNED \d{4}-\d{2}-\d{2}\*\*", gate_block, re.M
            ),
            "the owner-gate block is neither pending nor returned",
        )
        for heading in (
            "#### Architecture (CD)",
            "#### Over-engineering",
            "#### Drift",
            "#### Performance (informational",
            "#### Positives",
            "#### ↩︎ Cleared",
            "#### Owner gate — decisions",
        ):
            self.assertIn(heading, self.section, heading)
        self.assertEqual(
            self.section.count("**Informational · UNMEASURED ·"),
            len(self.cal["informational"]),
        )


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(sc.TOOLS))
import census_checks  # noqa: E402
import station_verify  # noqa: E402

STATION_CHECKS = (
    "check-anchors",
    "check-reraise",
    "fields-check",
    "fields-check-alignment",
    "census",
    "oracle-genericity",
    "oracle-doc-paths",
    "oracle-doc-voice",
    "premises",
    "read-only",
)
GATE_GLOBS = ("scripts/ci/*.sh", "scripts/ci/*.py", "scripts/ci/lib/*.sh")
GUARDED = ("crates", "docs/design", "schemas", "scripts", ".github", "Cargo.toml")
JOBS_BLOCK_RE = re.compile(r"^jobs:\n(.*?)(?=^[a-z]|\Z)", re.S | re.M)


class SectionVerifyTests(sc.StationCase):
    """Executable proof of the station verification (E07-5).

    verify-station.sh runs this module, so nothing here may invoke it (recursion); the station
    driver verify-census.sh is run once with --no-shared for the same reason. Every census figure
    is recounted from the baseline tree and compared with inventory.json, gate-census.json and the
    rendered section; the failure paths are proven on tampered copies through census_checks.
    """

    SLUG = "build-ci"
    SECTION_TITLE = SECTION_TITLE

    @classmethod
    def setUpClass(cls) -> None:
        cls.driver = cls.station_dir() / "verify-census.sh"
        cls.driver_run = cls.run_driver(SECTION_TITLE, "--no-shared")
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.census = sc.load_json(cls.station_dir() / "gate-census.json")
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.lines = backlog_parse.read_register(sc.BACKLOG)
        cls.heading = census_checks.resolve_heading(cls.lines, SECTION_TITLE)
        cls.section = census_checks.section_body(cls.lines, cls.heading[0])
        cls.t = cls.tree()
        cls.measured = cls.recount(cls.t)

    @classmethod
    def run_driver(cls, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(cls.driver), *args],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )

    @classmethod
    def recount(cls, t: sc.Tree) -> dict:
        """The census re-measured from the baseline tree, in census_checks' measured shape."""
        workflows = t.ls_files(".github/workflows/*.yml")
        ci = t.read_text(".github/workflows/ci.yml")
        block = JOBS_BLOCK_RE.search(ci).group(1)
        wiring = "\n".join(t.read_text(w) for w in workflows) + t.read_text(
            "scripts/pre-commit"
        )
        gate_files = t.ls_files(*GATE_GLOBS)
        unwired = [
            g.rsplit("/", 1)[-1]
            for g in gate_files
            if g.rsplit("/", 1)[-1] not in wiring
        ]
        transitive = [
            f"{path}:{n}"
            for path in t.ls_files("scripts/ci/*.sh")
            if path != "scripts/ci/check-comment-bloat.sh"
            for n, line in enumerate(t.read_text(path).splitlines(), 1)
            if "check-comment-bloat.sh" in line
        ]
        return {
            "mpichCi": ci.count("Build MPICH from source"),
            "mpichTotal": sum(
                t.read_text(w).count("Build MPICH from source") for w in workflows
            ),
            "schemas": len(t.ls_files("schemas/*.json")),
            "fbs": t.is_file("crates/cobre-io/schemas/policy.fbs"),
            "rootFbs": t.is_file("schemas/policy.fbs"),
            "buildRs": len(
                [p for p in t.ls_files("crates/*/build.rs") if "/vendor/" not in p]
            ),
            "workflows": len(workflows),
            "actionsDir": t.is_dir(".github/actions"),
            "jobsRaw": len(re.findall(r"^  [a-z0-9-]+:$", ci, re.M)),
            "jobIds": re.findall(r"^  ([a-z0-9_-]+):$", block, re.M),
            "unwired": unwired,
            "transitive": transitive,
            "gateFiles": gate_files,
        }

    def test_checkers_exit_zero_over_the_station_title(self) -> None:
        for tool in ("check-anchors.py", "check-reraise.py", "fields-check.py"):
            self.assertEqual(sc.run_checker(tool, SECTION_TITLE), 0, tool)
        self.assertEqual(
            sc.run_checker("fields-check.py", SECTION_TITLE, "--require", "Alignment"),
            0,
        )

    def test_heading_resolves_exactly_once(self) -> None:
        self.assertEqual(len(self.heading), 1)
        self.assertEqual(self.lines[self.heading[0] - 1], f"### {SECTION_TITLE}")
        scaffold = [
            n
            for n, raw in enumerate(self.lines, 1)
            if raw.endswith("— build-ci") and raw.startswith("## ★ QUALITY EVALUATION")
        ]
        self.assertEqual(len(scaffold), 1)
        self.assertLess(scaffold[0], self.heading[0])

    def test_zero_or_many_headings_exit_two_listing_candidates(self) -> None:
        run = self.run_driver("STATION 7", "--no-shared")
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("0 headings match 'STATION 7'", run.stderr)
        self.assertIn(f"### {SECTION_TITLE}", run.stderr)
        dup = [*self.lines, f"### {SECTION_TITLE}"]
        self.assertEqual(len(census_checks.resolve_heading(dup, SECTION_TITLE)), 2)
        self.assertEqual(len(census_checks.heading_candidates(dup)), 3)
        self.assertEqual(
            census_checks.main(["heading", str(sc.BACKLOG), "★ STATION 7"]), 2
        )

    def test_no_tracked_evaluated_surface_is_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])
        diff = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", *GUARDED],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(diff, "")

    def test_census_in_inventory_equals_a_fresh_recount(self) -> None:
        m = self.measured
        self.assertEqual(m["jobIds"], list(JOB_IDS))
        self.assertEqual(m["jobsRaw"], len(JOB_IDS) + 1)
        self.assertEqual(m["unwired"], ["check-comment-bloat.sh"])
        self.assertEqual(m["transitive"], ["scripts/ci/quality-report.sh:132"])
        self.assertEqual((m["mpichCi"], m["mpichTotal"]), (8, 10))
        self.assertEqual((m["schemas"], m["fbs"], m["rootFbs"]), (18, True, False))
        self.assertEqual((m["buildRs"], m["workflows"], m["actionsDir"]), (3, 8, False))
        self.assertEqual(len(m["gateFiles"]), 17)
        self.assertEqual(census_checks.inventory_findings(self.inv, m), [])
        self.assertEqual(census_checks.census_completeness(self.census, m), [])
        self.assertEqual(census_checks.tree_findings(m), [])
        self.assertEqual(
            census_checks.section_figure_findings(self.section, self.inv, m), []
        )

    def test_shared_verifier_accepts_the_gate_census_shape(self) -> None:
        listed = station_verify.reconstruct_listed(self.inv)
        roots = station_verify.crate_src_roots(self.inv)
        suffixes = station_verify.census_suffixes(self.inv)
        self.assertEqual((roots, suffixes), (["scripts/ci"], (".sh", ".py")))
        tree = station_verify.baseline_rs_files(
            sc.REPO, self.inv["baseline"], roots, suffixes
        )
        self.assertEqual(sorted(tree), self.measured["gateFiles"])
        self.assertEqual(station_verify.inventory_diff(listed, tree), ([], []))
        self.assertEqual(
            station_verify.census_suffixes({"crate": "cobre-core"}), (".rs",)
        )
        tampered = json.loads(json.dumps(self.inv))
        tampered["gates"]["files"].append(
            {"path": "scripts/ci/gone.sh", "kind": "sh", "lines": 1}
        )
        absent, unlisted = station_verify.inventory_diff(
            station_verify.reconstruct_listed(tampered), tree
        )
        self.assertEqual((absent, unlisted), (["scripts/ci/gone.sh"], []))
        with self.assertRaises(KeyError):
            station_verify.reconstruct_listed({"baseline": self.inv["baseline"]})

    def test_tampered_census_and_inventory_fail_naming_recorded_versus_measured(
        self,
    ) -> None:
        m = self.measured
        short = json.loads(json.dumps(self.census))
        short["rows"] = [
            r
            for r in short["rows"]
            if not r["script"].endswith("check-no-plan-leaks.sh")
        ]
        bad = census_checks.census_completeness(short, m)
        self.assertTrue(
            any("census rows 16 != measured gate files 17" in b for b in bad), bad
        )
        self.assertTrue(
            any("only-on-disk ['scripts/ci/check-no-plan-leaks.sh']" in b for b in bad),
            bad,
        )
        unclassified = json.loads(json.dumps(self.census))
        unclassified["rows"][0]["class"] = "wired"
        self.assertTrue(
            any(
                "outside the vocabulary" in b
                for b in census_checks.census_completeness(unclassified, m)
            )
        )
        headless = json.loads(json.dumps(self.census))
        row = next(r for r in headless["rows"] if r["class"] == "blocking")
        row["workflowJob"] = None
        self.assertTrue(
            any(
                "without workflow/job/step" in b
                for b in census_checks.census_completeness(headless, m)
            )
        )
        no_invoker = json.loads(json.dumps(self.census))
        next(r for r in no_invoker["rows"] if r["class"] == "unwired")[
            "transitiveInvocation"
        ] = None
        self.assertTrue(
            any(
                "without its invoking gate anchored" in b
                for b in census_checks.census_completeness(no_invoker, m)
            )
        )
        drifted = json.loads(json.dumps(self.inv))
        drifted["ci"]["ciJobCount"] = 15
        self.assertEqual(
            census_checks.inventory_findings(drifted, m),
            ["inventory.ci.ciJobCount recorded 15 but re-measured 14"],
        )
        raw = dict(m, jobsRaw=14)
        self.assertTrue(
            any("raw job-id grep = 14" in b for b in census_checks.tree_findings(raw))
        )

    def test_section_quoting_fifteen_jobs_or_a_copied_figure_fails(self) -> None:
        m = self.measured
        fifteen = self.section.replace(
            "8 workflows, 14 `ci.yml` jobs", "8 workflows, 15 `ci.yml` jobs"
        )
        bad = census_checks.section_figure_findings(fifteen, self.inv, m)
        self.assertTrue(any("(8, 15) but re-measured (8, 14)" in b for b in bad), bad)
        self.assertTrue(
            any("'15 `ci.yml` jobs'" in b and "raw grep's 15" in b for b in bad), bad
        )
        copied = self.section.replace("census (17 gate files", "census (16 gate files")
        self.assertTrue(
            any(
                "census heading (16,) but re-measured (17,)" in b
                for b in census_checks.section_figure_findings(copied, self.inv, m)
            )
        )
        stale = self.section.replace("(18 JSON exports)", "(19 JSON exports)")
        self.assertTrue(
            any(
                "schema exports (19,)" in b
                for b in census_checks.section_figure_findings(stale, self.inv, m)
            )
        )

    def test_body_assertions_hold_and_fail_on_tampered_entries(self) -> None:
        self.assertEqual(census_checks.body_findings(self.section, self.cal), [])
        entries = census_checks.entries_of(self.section)
        self.assertEqual(sorted(entries), sorted(r["id"] for r in self.cal["assigned"]))
        for block in entries.values():
            self.assertEqual(len(re.findall(r"^- \*\*Alignment:\*\*", block, re.M)), 1)
        downgraded = [r["id"] for r in self.cal["assigned"] if r.get("downgradeReason")]
        self.assertEqual(len(downgraded), 5)
        no_align = self.section.replace(
            "- **Alignment:** neutral", "- **Alignment:** undecided", 1
        )
        self.assertTrue(
            any(
                "Alignment values ['undecided']" in b
                for b in census_checks.body_findings(no_align, self.cal)
            )
        )
        dropped = re.sub(
            r"^- \*\*Reviewer rating:\*\* A.*\n", "", self.section, count=1, flags=re.M
        )
        bad = census_checks.body_findings(dropped, self.cal)
        self.assertTrue(
            any("calibration downgrade from A not recorded" in b for b in bad), bad
        )
        described = (
            "**CD-999 · Sev B · duplication · effort S · confidence high**\n"
            "Calibration downgraded from the reviewer's A.\n\n- **Alignment:** neutral\n"
        )
        self.assertEqual(
            census_checks.body_findings(described, {"assigned": []}),
            ["CD-999: a downgrade is described without the reviewer's original rating"],
        )
        one_site = self.section.replace("rule = `CLAUDE.md:39`", "rule = (none)", 1)
        self.assertTrue(
            any(
                "does not name rule + script + ci_step" in b
                for b in census_checks.body_findings(one_site, self.cal)
            )
        )
        self.assertIn(
            "zero entries", census_checks.body_findings("#### empty\n", self.cal)[0]
        )

    def test_verify_census_no_shared_exits_zero_with_every_check_passing(self) -> None:
        out = self.driver_run.stdout
        self.assertEqual(self.driver_run.returncode, 0, out + self.driver_run.stderr)
        self.assertIn("## Station-specific checks — build-ci", out)
        self.assertIsNotNone(
            re.search(r"^\| verify-station \| .* \| - \| SKIP", out, re.M)
        )
        for check in STATION_CHECKS:
            self.assertIsNotNone(
                re.search(rf"^\| {check} \| .* \| 0 \| PASS \|$", out, re.M), check
            )
        self.assertNotIn("| FAIL |", out)
        self.assertIn(
            "14 ci.yml jobs (the raw `grep -nE '^  [a-z0-9-]+:$'` census command matches 15",
            out,
        )
        self.assertIn("17 gate files", out)
        self.assertIn(
            "8x `Build MPICH from source` in ci.yml (10 across the workflows", out
        )
        self.assertIn("18 `schemas/*.json` + `crates/cobre-io/schemas/policy.fbs`", out)
        self.assertIn("3 build.rs · 8 workflows", out)

    def test_driver_is_shell_clean_and_writes_only_beside_itself(self) -> None:
        self.assertEqual(
            subprocess.run(["bash", "-n", str(self.driver)], check=False).returncode, 0
        )
        if shutil.which("shellcheck"):
            run = subprocess.run(
                ["shellcheck", str(self.driver)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout)
        text = self.driver.read_text(encoding="utf-8")
        redirects = re.findall(r">\s*\"?\$\{?([A-Z_]+)\}?(?=[/\"\s])", text)
        self.assertTrue(redirects)
        self.assertEqual(set(redirects), {"SCRATCH", "BLOCK"})
        for measured in (
            "M_MPICH_CI",
            "M_GATEFILES",
            "M_SCHEMAS",
            "M_BUILDRS",
            "M_WORKFLOWS",
        ):
            self.assertIn(f'{measured}="', text, measured)


GATE_DATE = "2026-09-19"
GATE_HEADINGS = (
    "### 1.1 The genericity two-site entry",
    "### 1.2 The PATTERN vocabulary limitation",
    "### 1.3 Census-derived entries",
    "### 1.4 Over-engineering entries",
    "### 1.5 Drift entries",
    "### 1.6 Holds",
    "### 1.7 Cleared and sanctioned — presented read-only, no decision taken",
    "### 1.8 Queues handed on",
    "### 1.9 Worker needs-human items",
    "## 2. Round plan",
    "## 3. Decision record",
    "### 3.1 The two-site entry",
    "### 3.2 The PATTERN vocabulary limitation",
    "### 3.3 Census, over-engineering and drift entries",
    "### 3.4 Holds",
    "### 3.5 Worker needs-human answers",
    "### 3.6 Presented, no decision taken",
    "### 3.7 Informational",
    "## 4. Handoffs after the gate",
)
DECISION_VERBS = {
    "accept",
    "amend-scope",
    "downgrade",
    "reject",
    "defer",
    "override-conflicts",
}
TWO_SITES = (
    "CLAUDE.md",
    "scripts/ci/check-infra-genericity.sh",
    ".github/workflows/ci.yml",
)
ENTRY_ROUNDS = ("R1-two-site", "R3-census", "R4-over-eng", "R5-drift")
SANCTIONED_GATES = (
    "check-comment-line-refs.sh",
    "check-comment-banners.sh",
    "check-comment-bloat.sh",
    "quality-report.sh",
)
GATE_TIMING_RE = re.compile(
    r"(?<![\w.\-/:])\d+(?:\.\d+)?\s?(?:ms|µs|us|ns|secs?|seconds?|minutes?|hours?)\b"
)


def table_rows(block: str, first_cell: str) -> list[list[str]]:
    return [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in block.splitlines()
        if ln.startswith(first_cell)
    ]


class GateTests(sc.StationCase):
    """The owner gate (E07-6): gate.md, decisions.json, the BACKLOG owner-gate block, the
    entry-level owner-decision bullets, the handoff sync and the marker.

    The gate ticket runs this class after recording the owner decision; the class asserts the
    recorded state and never asks a question or writes a file.
    """

    SLUG = "build-ci"
    SECTION_TITLE = SECTION_TITLE

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = cls.station_dir().joinpath("gate.md").read_text(encoding="utf-8")
        cls.rec = sc.load_json(cls.station_dir() / "decisions.json")
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.census = sc.load_json(cls.station_dir() / "gate-census.json")
        cls.handoff = sc.load_json(cls.station_dir() / "partI-handoff.json")
        cls.register = "\n".join(backlog_parse.read_register(sc.BACKLOG))
        cls.section = cls.register.split("### " + SECTION_TITLE, 1)[1].split(
            "\n## ", 1
        )[0]
        cls.owner_gate = cls.section.split("#### Owner gate — decisions", 1)[1].split(
            "\n#### ", 1
        )[0]
        cls.cleared = cls.section.split("#### ↩︎ Cleared", 1)[1].split("\n#### ", 1)[0]
        cls.decisions = cls.rec["decisions"]
        cls.by_id = {r["id"]: r for r in cls.cal["assigned"]}

    def block(self, entry_id: str) -> str:
        return self.section.split(f"**{entry_id} · ", 1)[1].split("\n**", 1)[0]

    def test_gate_md_carries_one_decision_line_and_the_two_site_entry_leads(
        self,
    ) -> None:
        lines = [ln for ln in self.gate.splitlines() if ln.startswith("**Decision: ")]
        self.assertEqual(len(lines), 1)
        self.assertIsNotNone(
            re.match(r"^\*\*Decision: (ratified|returned)\*\*", lines[0])
        )
        positions = [self.gate.index(h) for h in GATE_HEADINGS]
        self.assertEqual(positions, sorted(positions), "gate.md sections out of order")
        lead = self.gate.split(GATE_HEADINGS[0], 1)[1].split("\n### ", 1)[0]
        rows = [r for r in table_rows(lead, "| ") if re.match(r"^\d+$", r[0])]
        self.assertEqual([r[1] for r in rows], ["CD-099"])
        for site in TWO_SITES:
            self.assertIn(site, rows[0][7])
        self.assertEqual(rows[0][3], "blocking")
        pattern = self.gate.split(GATE_HEADINGS[1], 1)[1].split("\n### ", 1)[0]
        self.assertIn("architecture-01", pattern)
        self.assertIn("scripts/ci/check-infra-genericity.sh:15", pattern)
        for heading in GATE_HEADINGS[2:5]:
            body = self.gate.split(heading, 1)[1].split("\n### ", 1)[0]
            for r in table_rows(body, "| "):
                if not re.match(r"^(CD|OD)-\d{3}$", r[1]):
                    continue
                self.assertIn(r[3], {*WIRING_VOCAB, "n/a"}, r[1])
                if self.by_id[r[1]].get("isGate"):
                    self.assertIn(r[3], WIRING_VOCAB, r[1])
                self.assertIn(r[4], ALIGN_VOCAB, r[1])
        ids_in_digest = [
            r[1]
            for h in GATE_HEADINGS[2:5]
            for r in table_rows(self.gate.split(h, 1)[1].split("\n### ", 1)[0], "| ")
            if re.match(r"^(CD|OD)-\d{3}$", r[1])
        ]
        self.assertEqual(sorted(ids_in_digest + ["CD-099"]), sorted(self.by_id))
        self.assertEqual(ids_in_digest[:4], sorted(ids_in_digest[:4]))
        self.assertEqual([m.group(0) for m in GATE_TIMING_RE.finditer(self.gate)], [])

    def test_round_plan_order_options_and_omissions(self) -> None:
        plan = self.gate.split("## 2. Round plan", 1)[1].split("\n## ", 1)[0]
        rows = [r for r in table_rows(plan, "| R") if re.match(r"^R\d", r[0])]
        ids = [r[0] for r in rows]
        self.assertEqual(
            ids[:5],
            ["R1-two-site", "R2-vocabulary", "R3-census", "R4-over-eng", "R5-drift"],
        )
        self.assertEqual(ids[5:25], [f"R6-nh-{n}" for n in range(1, 21)])
        self.assertEqual(ids[-1], "R7-handoffs")
        self.assertNotIn("conflicts-hold", [r[1] for r in rows])
        for r in rows:
            self.assertLessEqual(len(r[3].split("·")), 4, r[0])
        self.assertIn("presented read-only, disposition: epic 9", plan)
        for gate in SANCTIONED_GATES:
            self.assertIn(gate, plan)
        self.assertIn("cobre-{mcp,tui,flow,uc,emt}", plan)
        self.assertIn("allow-rationale-allowlist.txt", plan)
        self.assertIn("CI wall time is informational only", plan)
        self.assertIn("conflicts holds: none at this station", plan)
        self.assertEqual(len(self.rec["noRoundFor"]), 7)

    def test_decisions_cover_every_calibrated_entry(self) -> None:
        self.assertEqual({d["id"] for d in self.decisions}, set(self.by_id))
        for d in self.decisions:
            self.assertIn(d["decision"], DECISION_VERBS, d["id"])
            self.assertTrue(d["rationale"].strip(), d["id"])
            self.assertIn(d["round"], ENTRY_ROUNDS, d["id"])
            self.assertEqual(
                d["wiring"], self.by_id[d["id"]].get("enforcement") or "n/a", d["id"]
            )
            if self.by_id[d["id"]].get("isGate"):
                self.assertIn(d["wiring"], WIRING_VOCAB, d["id"])
            self.assertIn(d["alignment"], ALIGN_VOCAB, d["id"])
            if d["decision"] == "downgrade":
                self.assertLess(
                    RANK[d["newSeverity"]], RANK[d["reviewerSeverity"]], d["id"]
                )
            if d["decision"] == "defer":
                self.assertTrue(
                    d["deferTrigger"], f"{d['id']}: defer without a trigger"
                )
            if d["decision"] == "reject":
                self.assertTrue(d["clearedMoved"], d["id"])
            if d["decision"] == "override-conflicts":
                self.assertTrue(d["overrideRationale"], d["id"])
            if d["hardRule"]:
                self.assertEqual(len(d["sites"]), 3, d["id"])
        lead = next(d for d in self.decisions if d["id"] == "CD-099")
        self.assertEqual(lead["round"], "R1-two-site")
        for site in TWO_SITES:
            self.assertTrue(any(site in s for s in lead["sites"]), site)
        self.assertEqual(lead["alignment"], "advances-1")
        self.assertEqual(self.rec["vocabulary"]["id"], "architecture-01")
        self.assertEqual(self.rec["vocabulary"]["reRaiseOf"], "CD-061")
        self.assertFalse(self.rec["vocabulary"]["clearedMoved"])
        self.assertEqual(self.rec["censusRow"]["id"], "check-comment-bloat.sh")
        self.assertEqual(self.rec["censusRow"]["wiring"], "transitively-advisory")
        self.assertEqual(self.rec["holds"], [])
        answered = [
            n for n in self.rec["needsHuman"] if str(n.get("answer", "")).strip()
        ]
        self.assertEqual(len(answered), len(self.rec["needsHuman"]))
        self.assertEqual(len(answered), 20)
        counts = self.rec["counts"]
        verbs = [d["decision"] for d in self.decisions]
        for key, verb in (
            ("accepted", "accept"),
            ("downgraded", "downgrade"),
            ("rejected", "reject"),
            ("deferred", "defer"),
            ("overridden", "override-conflicts"),
        ):
            self.assertEqual(counts[key], verbs.count(verb), key)
        self.assertEqual(counts["entries"], len(self.decisions))
        self.assertEqual(counts["needsHumanAnswered"], 20)
        self.assertEqual(counts["held"], 0)
        self.assertTrue(self.rec["returned"])
        self.assertEqual(self.rec["decision"], "ratified")
        self.assertEqual(self.rec["handoffConfirmation"]["decision"], "confirm")
        self.assertEqual(
            self.rec["presentedNoDecision"]["partI"]["disposition"], "epic 9"
        )

    def test_backlog_carries_the_ratified_line_the_table_and_the_marker(self) -> None:
        counts = self.rec["counts"]
        head = self.section.split("\n**Station.**", 1)[0]
        self.assertNotIn("_(owner gate pending", head)
        self.assertNotIn("_(pending — filled by the gate ticket)_", self.owner_gate)
        if self.rec["decision"] == "ratified":
            self.assertIn(
                f"**Ratified {GATE_DATE}** — owner gate; baseline `{self.baseline()[:8]}`",
                head,
            )
        else:
            self.assertNotIn("**Ratified ", head)
        marker = re.search(
            rf"^\*\*Gate: RETURNED {GATE_DATE}\*\* — baseline `{self.baseline()[:8]}`; "
            r"accepted (\d+), amended (\d+), downgraded (\d+), rejected (\d+), deferred (\d+), overridden (\d+), held (\d+)",
            self.owner_gate,
            re.M,
        )
        self.assertIsNotNone(marker, "Gate: RETURNED marker missing or malformed")
        assert marker is not None
        self.assertEqual(
            [int(marker.group(i)) for i in range(1, 8)],
            [
                counts[k]
                for k in (
                    "accepted",
                    "amended",
                    "downgraded",
                    "rejected",
                    "deferred",
                    "overridden",
                    "held",
                )
            ],
        )
        for d in self.decisions:
            self.assertEqual(
                self.owner_gate.count(f"| {d['id']} | {d['decision']} |"), 1, d["id"]
            )
            row = next(
                ln
                for ln in self.owner_gate.splitlines()
                if ln.startswith(f"| {d['id']} | ")
            )
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if RANK[d["newSeverity"]] < RANK[d["reviewerSeverity"]]:
                self.assertEqual(
                    cells[2],
                    f"{d['newSeverity']} (reviewer: {d['reviewerSeverity']})",
                    d["id"],
                )
            else:
                self.assertEqual(cells[2], d["newSeverity"], d["id"])
            self.assertEqual(cells[3], d["wiring"], d["id"])
            if d["decision"] == "defer":
                self.assertRegex(cells[6], r"^trigger: \S", d["id"])
            if d["hardRule"]:
                self.assertTrue(cells[6].startswith("sites: "), d["id"])
        self.assertIn("**Cleared by this gate (do not re-raise):**", self.owner_gate)
        for d in self.decisions:
            if d["decision"] == "reject":
                self.assertIn(f"**↩︎ CLEARED — {d['id']}", self.cleared, d["id"])
                self.assertNotIn(f"**{d['id']} · Sev", self.section)
        self.assertIn("disposition: epic 9", self.owner_gate)
        self.assertIn("advisory-by-design", self.owner_gate)
        self.assertIn("no PD id", self.owner_gate)
        self.assertNotIn("conflicts (held)", self.section)
        self.assertIsNone(re.search(r"^\*\*PD-\d{3}", self.section, re.M))
        self.assertNotIn("E10 queue", self.section)
        self.assertNotIn("perf-queue.json row", self.section)
        self.assertEqual(sc.run_checker("check-reraise.py", SECTION_TITLE), 0)

    def test_owner_decisions_sit_beneath_the_alignment_field(self) -> None:
        with_directions = [d for d in self.decisions if d["directions"]]
        self.assertGreaterEqual(len(with_directions), 12)
        for d in with_directions:
            block = self.block(d["id"])
            align_at = block.index("- **Alignment:**")
            for direction in d["directions"]:
                line = f"- **Owner decision ({GATE_DATE}, {direction['round']}):** {direction['choice']}"
                self.assertIn(line, block, d["id"])
                self.assertGreater(block.index(line), align_at, d["id"])
            if d["decision"] == "defer":
                self.assertIn(
                    f"- **Owner decision ({GATE_DATE}, {d['round']}):** defer — trigger: ",
                    block,
                )
        for info in self.rec["informational"]:
            head = f"**Informational · UNMEASURED · {info['candidateRef']}**"
            block = self.section.split(head, 1)[1].split("\n**", 1)[0]
            for direction in info["directions"]:
                self.assertIn(
                    f"- **Owner decision ({GATE_DATE}, {direction['round']}):** {direction['answer']}",
                    block,
                )
        queued_out = self.section.split("#### Queued out", 1)[1].split("\n#### ", 1)[0]
        self.assertEqual(queued_out.count(f"Owner gate {GATE_DATE}"), 2)

    def test_handoff_synced_and_item_6_untouched(self) -> None:
        two_site = self.handoff["twoSiteEdit"]
        self.assertEqual(two_site["id"], "CD-099")
        self.assertEqual(two_site["decision"], "accept")
        self.assertEqual(two_site["alignment"], "advances-1")
        for site in TWO_SITES:
            self.assertTrue(any(site in s for s in two_site["sites"]), site)
        self.assertNotIn("disposition", self.handoff)
        self.assertEqual(self.handoff["stationVerdict"], "claim-stale")
        self.assertTrue(self.handoff["baselineEvidence"])
        self.assertEqual(self.handoff["ownerGate"]["disposition"], "epic 9")
        self.assertIn(
            "scripts/ci/check-infra-genericity.sh:74",
            "".join(
                f"{b['path']}:{b['line']}" for b in self.handoff["baselineEvidence"]
            ),
        )
        read_only = self.gate.split("### 3.6 Presented, no decision taken", 1)[1].split(
            "\n### ", 1
        )[0]
        self.assertIn("I.3-6", read_only)
        self.assertIn("disposition: epic 9", read_only)
        self.assertIn("scripts/ci/check-infra-genericity.sh:74", read_only)

    def test_sanctioned_items_carry_no_id_and_no_question(self) -> None:
        dismissed = {c["candidateRef"] for c in self.cal["cleared"]}
        informational = {i["candidateRef"] for i in self.cal["informational"]}
        decided_refs = {d["candidateRef"] for d in self.decisions}
        self.assertFalse(decided_refs & dismissed)
        self.assertFalse(decided_refs & informational)
        for gate in SANCTIONED_GATES:
            self.assertNotIn(
                gate,
                [
                    r.get("script", "") or ""
                    for r in self.cal["assigned"]
                    if r.get("hardRule")
                ],
            )
        for subject in self.rec["presentedNoDecision"]["sanctioned"]:
            self.assertIn(subject, self.gate)
        plan_rounds = set(self.rec["rounds"])
        self.assertNotIn("R6-hold", " ".join(plan_rounds))
        self.assertEqual(len([r for r in plan_rounds if r.startswith("R6-nh-")]), 20)


if __name__ == "__main__":
    unittest.main()
