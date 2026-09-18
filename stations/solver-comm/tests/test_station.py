"""cobre-solver + cobre-comm station tests: inventory census, prior register, handoffs.

Run from the repository root:
    python3 -m unittest plans/architecture-debt-audit/stations/solver-comm/tests/test_station.py
Every figure is re-derived from the tree the station evaluated (inventory.json's
`baseline`, read through `station_checks.Tree`), never from the worktree. Later
solver+comm tickets append their own stage classes here; the station verification runs
the module.
"""

from __future__ import annotations

import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

SRC_ROOTS = ("crates/cobre-solver/src", "crates/cobre-comm/src")
CRATES = ("cobre-solver", "cobre-comm")
TOP_SYMBOL = re.compile(
    r"^(pub(\([a-z]+\))? )?(unsafe )?(fn|struct|enum|trait|type|const|static|impl|mod) "
)
UNSAFE_LINE = re.compile(r"\bunsafe\b")
ANCHOR_TOKEN = re.compile(
    r"`(?P<path>(?:crates|scripts|docs|plans|schemas|examples|tests|\.github|\.claude)/[\w./-]+?"
    r"\.(?:rs|toml|md|json|sh|py|yml|yaml))"
    r"(?:::(?P<sym>[A-Za-z_]\w*)|:(?P<line>\d+)(?:-(?P<line_to>\d+))?)?`"
)
PRIOR_SECTIONS = (
    "## Ingest filters (sanctioned)",
    "## Dup-of handoffs",
    "## Intended behaviour, never a finding",
    "## Seeded candidates",
    "## Do not re-raise",
)
TYPES_RS_SYMBOLS = {
    "StageTemplate",
    "RowBatch",
    "Basis",
    "LpSolution",
    "SolutionView",
    "SolverStatistics",
    "SolverError",
}
TRAITS_RS_SYMBOLS = {
    "CommData",
    "Communicator",
    "LocalCommunicator",
    "LocalCommKind",
    "SharedRegion",
    "SharedMemoryProvider",
    "TopologyProvider",
}


def top_symbol_names(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        m = TOP_SYMBOL.match(line)
        if m and m.group(4) != "impl":
            ident = re.match(r"[A-Za-z_]\w*", line[m.end() :])
            if ident:
                names.append(ident.group(0))
    return names


def unsafe_sites(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if UNSAFE_LINE.search(line) and not line.strip().startswith("//")
    )


class InventoryTests(sc.StationCase):
    SLUG = "solver-comm"
    SECTION_TITLE = "solver-comm"

    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.by_path = {f["path"]: f for f in cls.inv["files"]}

    def test_inventory_parses_and_carries_the_envelope(self) -> None:
        for key in (
            "station",
            "baseline",
            "producedAt",
            "commands",
            "crates",
            "files",
            "testSurface",
            "reconciliation",
        ):
            self.assertIn(key, self.inv)
        self.assertEqual(self.inv["station"], "solver-comm")
        for cmd in (
            "setEquality",
            "linesRaw",
            "linesNonTest",
            "topSymbols",
            "testSurface",
        ):
            self.assertIn(cmd, self.inv["commands"])
        self.assertEqual(set(self.inv["crates"]), set(CRATES))

    def test_baseline_is_a_register_pin_and_the_handoffs_agree(self) -> None:
        base = self.inv["baseline"]
        self.assertTrue(
            sc.is_register_pin(base),
            f"{base[:12]} is neither the register pin nor a superseded one: {sc.pin_history()}",
        )
        handoffs = sc.load_json(self.artifact("handoffs.json"))
        self.assertEqual(handoffs["baseline"], base)
        self.assertIn(
            f"baseline `{base[:8]}`",
            self.artifact("prior-register.md").read_text(encoding="utf-8"),
        )

    def test_file_set_equals_the_tree_at_the_baseline(self) -> None:
        tree = self.tree()
        listed = sorted(self.by_path)
        actual = sorted(p for root in SRC_ROOTS for p in tree.rs_files(root))
        self.assertEqual(listed, actual)
        self.assertEqual(len(actual), 30)
        self.assertEqual(sum(p.startswith("crates/cobre-solver/") for p in actual), 23)
        self.assertEqual(sum(p.startswith("crates/cobre-comm/") for p in actual), 7)

    def test_every_file_reproduces_its_line_split_and_symbols_at_the_baseline(
        self,
    ) -> None:
        tree = self.tree()
        for f in self.inv["files"]:
            with self.subTest(path=f["path"]):
                text = tree.read_text(f["path"])
                raw = tree.read_bytes(f["path"]).count(b"\n")
                phys, _ = sc.classify_lines(f["path"], text)
                self.assertEqual(f["linesRaw"], raw)
                self.assertEqual(f["linesNonTest"], phys)
                self.assertEqual(f["linesRaw"], f["linesNonTest"] + f["linesInSrcTest"])
                self.assertEqual(
                    [s["name"] for s in f["topSymbols"] if s["kind"] != "impl"],
                    top_symbol_names(text),
                )
                self.assertEqual(f["unsafeSites"], unsafe_sites(text))
                base = f["path"].rsplit("/", 1)[-1]
                if base in ("tests.rs", "test_support.rs"):
                    self.assertEqual(f["kind"], "sibling-test-module")
                    self.assertEqual(f["linesNonTest"], 0)
                else:
                    self.assertEqual(f["kind"], "source")
                    self.assertTrue(
                        f["topSymbols"], "a source file with no top-level item"
                    )

    def test_named_symbols_are_present(self) -> None:
        types_rs = {
            s["name"]
            for s in self.by_path["crates/cobre-solver/src/types.rs"]["topSymbols"]
        }
        traits_rs = {
            s["name"]
            for s in self.by_path["crates/cobre-comm/src/traits.rs"]["topSymbols"]
        }
        self.assertTrue(TYPES_RS_SYMBOLS <= types_rs, TYPES_RS_SYMBOLS - types_rs)
        self.assertTrue(TRAITS_RS_SYMBOLS <= traits_rs, TRAITS_RS_SYMBOLS - traits_rs)

    def test_unsafe_boundary_flag_matches_the_unsafe_sites(self) -> None:
        for f in self.inv["files"]:
            if f["kind"] != "source":
                continue
            with self.subTest(path=f["path"]):
                self.assertEqual("unsafe-boundary" in f["flags"], f["unsafeSites"] > 0)

    def test_part_i_item_8_surfaces_are_flagged(self) -> None:
        flagged = {
            f["path"]
            for f in self.inv["files"]
            if any(x.startswith("partI-item-8") for x in f["flags"])
        }
        self.assertEqual(
            flagged,
            {
                "crates/cobre-solver/src/types.rs",
                "crates/cobre-solver/src/freeze.rs",
                "crates/cobre-solver/src/trait_def.rs",
                "crates/cobre-solver/src/backends/profiled.rs",
            },
        )
        self.assertIn(
            "partI-item-8-declaration",
            self.by_path["crates/cobre-solver/src/types.rs"]["flags"],
        )

    def test_crate_totals_reconcile_with_loc_stats(self) -> None:
        tree = self.tree()
        for crate in CRATES:
            with self.subTest(crate=crate):
                rows = [f for f in self.inv["files"] if f["crate"] == crate]
                meta = self.inv["crates"][crate]
                self.assertEqual(meta["srcFiles"], len(rows))
                self.assertEqual(meta["linesRaw"], sum(f["linesRaw"] for f in rows))
                self.assertEqual(
                    meta["linesNonTest"], sum(f["linesNonTest"] for f in rows)
                )
                self.assertEqual(
                    meta["linesInSrcTest"], sum(f["linesInSrcTest"] for f in rows)
                )
                self.assertEqual(
                    meta["linesRaw"], meta["linesNonTest"] + meta["linesInSrcTest"]
                )
                self.assertEqual(meta["locStats"], sc.loc_stats(crate, tree))
        solver = self.inv["crates"]["cobre-solver"]
        extras = sum(
            tree.read_bytes(p).count(b"\n")
            for p in (
                "crates/cobre-solver/build.rs",
                "crates/cobre-solver/examples/audit_mm_dispatch.rs",
            )
        )
        self.assertEqual(
            solver["locStats"]["prod_all"], solver["linesNonTest"] + extras
        )
        self.assertIn("build.rs", self.inv["reconciliation"]["cobre-solver"])
        comm = self.inv["crates"]["cobre-comm"]
        self.assertEqual(comm["locStats"]["prod_all"], comm["linesNonTest"])

    def test_test_surface_reconciles_and_states_the_denominator(self) -> None:
        tree = self.tree()
        for crate in CRATES:
            with self.subTest(crate=crate):
                ts = self.inv["testSurface"][crate]
                bins = sorted(
                    p
                    for p in tree.ls_files(f"crates/{crate}/tests")
                    if p.endswith(".rs") and p.count("/") == 3
                )
                self.assertEqual(
                    sorted(b["path"] for b in ts["integrationBinaries"]), bins
                )
                for b in ts["integrationBinaries"]:
                    self.assertEqual(
                        b["lines"], tree.read_bytes(b["path"]).count(b"\n")
                    )
                self.assertEqual(
                    ts["integrationLines"],
                    sum(b["lines"] for b in ts["integrationBinaries"]),
                )
                sib = sum(
                    f["linesRaw"]
                    for f in self.inv["files"]
                    if f["crate"] == crate and f["kind"] == "sibling-test-module"
                )
                inline = sum(
                    f["linesInSrcTest"]
                    for f in self.inv["files"]
                    if f["crate"] == crate and f["kind"] == "source"
                )
                self.assertEqual(ts["siblingTestLines"], sib)
                self.assertEqual(ts["inlineTestLines"], inline)
                self.assertEqual(
                    ts["testLinesTotal"], ts["integrationLines"] + sib + inline
                )
                self.assertEqual(
                    ts["testLinesTotal"],
                    self.inv["crates"][crate]["locStats"]["test_all"],
                )
                self.assertEqual(
                    ts["nonTestSrcLines"], self.inv["crates"][crate]["linesNonTest"]
                )
                self.assertIn(str(ts["testLinesTotal"]), ts["denominator"])
                self.assertIn(str(ts["nonTestSrcLines"]), ts["denominator"])
        solver = self.inv["testSurface"]["cobre-solver"]
        self.assertEqual(len(solver["integrationBinaries"]), 8)
        self.assertEqual(
            len(self.inv["testSurface"]["cobre-comm"]["integrationBinaries"]), 2
        )
        self.assertEqual(
            {m["path"] for m in solver["siblingTestModules"]},
            {
                "crates/cobre-solver/src/backends/clp/tests.rs",
                "crates/cobre-solver/src/backends/highs/tests.rs",
            },
        )
        self.assertGreater(
            solver["testLinesTotal"], solver["nonTestSrcLines"], "the denominator claim"
        )
        hoist = solver["hoistTarget"]
        self.assertTrue(
            tree.read_text(hoist["path"])
            .splitlines()[hoist["line"] - 1]
            .startswith("pub mod test_support")
        )
        self.assertIn(
            "test-support",
            tree.read_text(hoist["path"]).splitlines()[hoist["gateLine"] - 1],
        )
        rationale = solver["localFixtureRationale"]
        quoted = " ".join(
            line.lstrip("/! ").strip()
            for line in tree.read_text(rationale["path"]).splitlines()[
                rationale["lines"][0] - 1 : rationale["lines"][1]
            ]
        )
        self.assertEqual(rationale["quote"], quoted)

    def test_perf_helpers_carry_their_measured_consumers(self) -> None:
        tree = self.tree()
        helpers = self.by_path["crates/cobre-comm/src/lib.rs"]["perfHelpers"]
        self.assertEqual(
            {s["name"] for s in helpers["symbols"]},
            {"per_rank_counts", "prefix_displs"},
        )
        consumers = {(c["path"], c["line"]) for c in helpers["consumers"]}
        expected = set()
        for f in tree.rs_files("crates/cobre-sddp/src"):
            for i, line in enumerate(tree.read_text(f).splitlines(), 1):
                if (
                    re.search(r"\b(per_rank_counts|prefix_displs)\b", line)
                    and "use " not in line
                ):
                    expected.add((f, i))
        self.assertEqual(consumers, expected)
        self.assertTrue(
            consumers, "the perf lens needs at least one exercising call site"
        )

    def test_reserved_seam_anchors_resolve(self) -> None:
        tree = self.tree()
        for f in self.inv["files"]:
            for a in f.get("reservedSeamAnchors", []):
                with self.subTest(path=f["path"], symbol=a["symbol"]):
                    line = tree.read_text(f["path"]).splitlines()[a["line"] - 1]
                    self.assertRegex(line, rf"\b{re.escape(a['symbol'])}\b")


class PriorRegisterTests(sc.StationCase):
    SLUG = "solver-comm"
    SECTION_TITLE = "solver-comm"

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = cls.artifact_text("prior-register.md")
        cls.register = backlog_parse.read_register(sc.BACKLOG)

    @classmethod
    def artifact_text(cls, name: str) -> str:
        return (cls.station_dir() / name).read_text(encoding="utf-8")

    def test_four_sections_plus_the_do_not_reraise_list(self) -> None:
        positions = [self.text.find(h) for h in PRIOR_SECTIONS]
        self.assertTrue(
            all(p >= 0 for p in positions), dict(zip(PRIOR_SECTIONS, positions))
        )
        self.assertEqual(positions, sorted(positions), "sections out of order")

    def test_every_entry_has_a_reraisekey(self) -> None:
        entries = re.findall(r"^### .+$", self.text, re.M)
        self.assertEqual(len(entries), 4)
        self.assertEqual(len(re.findall(r"^- \*\*reraiseKey\*\*:", self.text, re.M)), 4)

    def test_cited_register_ids_exist_with_the_stated_status(self) -> None:
        ids = backlog_parse.register_findings(self.register)
        self.assertIn("OD-001", ids)
        self.assertIn("CD-019", ids)
        joined = "\n".join(self.register)
        self.assertIn("OD-001 → KEEP-RESERVED", joined)

    def test_every_anchor_resolves_at_the_station_baseline(self) -> None:
        tree = self.tree()
        register_pin = backlog_parse.parse_baseline(self.register)
        for m in ANCHOR_TOKEN.finditer(self.text):
            path, sym, line = m.group("path"), m.group("sym"), m.group("line")
            # the register itself is a branch file: resolve it on the worktree, at the cited line
            source = (
                sc.WORKTREE
                if path.startswith("plans/") or path.startswith(".claude/")
                else tree
            )
            with self.subTest(anchor=m.group(0)):
                self.assertTrue(source.is_file(path), f"missing {path}")
                if line is not None:
                    lines = source.read_text(path).splitlines()
                    self.assertLessEqual(int(line), len(lines))
                    if m.group("line_to"):
                        self.assertLessEqual(int(m.group("line_to")), len(lines))
                if sym is not None:
                    self.assertTrue(
                        sc.anchor_exists(f"`{path}::{sym}`", source), f"{path}::{sym}"
                    )
        self.assertIn(register_pin[:8], self.text)

    def test_shared_memory_filter_names_the_sanctioned_symbols(self) -> None:
        block = self.text.split("## Ingest filters (sanctioned)")[1].split(
            "## Dup-of handoffs"
        )[0]
        for sym in (
            "SharedMemoryProvider",
            "SharedRegion",
            "LocalCommunicator",
            "LocalCommKind",
            "HeapRegion",
            "split_local",
        ):
            self.assertIn(sym, block)
        self.assertIn("docs/design/reserved-seams-and-deferred-debt.md:79", block)
        self.assertIn("OD-001", block)
        self.assertIn("Cleared as `sanctioned`", block)
        self.assertIn("No OD id is assigned", block)

    def test_cut_sync_dup_of_anchors_resolve_in_sddp_and_nowhere_in_the_station(
        self,
    ) -> None:
        tree = self.tree()
        text = tree.read_text("crates/cobre-sddp/src/cut/cut_sync.rs").splitlines()
        for sym, line in (
            ("sync_cuts", 243),
            ("pack_local_records", 400),
            ("sync_packed_records", 495),
            ("sync_level_records", 581),
        ):
            self.assertIn(f"pub fn {sym}", text[line - 1], f"{sym} moved off :{line}")
            self.assertIn(
                f"`:{line}`" if sym != "sync_cuts" else f"cut_sync.rs:{line}", self.text
            )
        for root in SRC_ROOTS:
            for f in tree.rs_files(root):
                self.assertNotRegex(
                    tree.read_text(f),
                    r"pub fn (sync_cuts|pack_local_records|sync_packed_records)\b",
                    f,
                )

    def test_basis_asymmetry_cites_both_backends_and_forbids_equalizing(self) -> None:
        block = self.text.split("## Intended behaviour, never a finding")[1].split(
            "## Seeded candidates"
        )[0]
        tree = self.tree()
        self.assertIn(
            "SolverError::BasisInconsistent",
            tree.read_text(
                "crates/cobre-solver/src/backends/highs/interface.rs"
            ).splitlines()[486],
        )
        clp = tree.read_text(
            "crates/cobre-solver/src/backends/clp/solver.rs"
        ).splitlines()
        self.assertIn("silently accept an inconsistent offered basis", clp[213])
        self.assertIn("Clp_dual", clp[214])
        trait = tree.read_text("crates/cobre-solver/src/trait_def.rs").splitlines()
        self.assertIn("BasisInconsistent", trait[104])
        self.assertIn("BasisInconsistent", trait[135])
        for token in (
            "interface.rs:487",
            "solver.rs:213",
            ":214-216",
            "trait_def.rs:105",
            ":136",
            "must never be proposed",
        ):
            self.assertIn(token, block)
        conformance = tree.read_text("crates/cobre-solver/tests/conformance.rs")
        for name in (
            "test_solver_highs_solve_rejects_inconsistent_basis_status_combination",
            "test_solver_clp_solve_accepts_inconsistent_basis_status_combination_silently",
        ):
            self.assertIn(name, conformance)
            self.assertIn(name, block)

    def test_seeded_lint_tables_quote_the_cargo_constraint_and_the_drift(self) -> None:
        block = self.text.split("## Seeded candidates")[1].split("## Do not re-raise")[
            0
        ]
        tree = self.tree()
        self.assertIn("does not permit combining `lints.workspace = true`", block)
        for path, rust_line, clippy_line in (
            ("crates/cobre-solver/Cargo.toml", 37, 41),
            ("crates/cobre-comm/Cargo.toml", 35, 39),
            ("crates/cobre-python/Cargo.toml", 46, 50),
        ):
            lines = tree.read_text(path).splitlines()
            self.assertEqual(lines[rust_line - 1], "[lints.rust]", path)
            self.assertEqual(lines[clippy_line - 1], "[lints.clippy]", path)
        root = tree.read_text("Cargo.toml").splitlines()
        self.assertEqual(root[34], "[workspace.lints.rust]")
        self.assertEqual(root[64], "[workspace.lints.clippy]")
        python = tree.read_text("crates/cobre-python/Cargo.toml")
        self.assertNotIn("too_many_arguments", python)
        self.assertIn(
            "too_many_arguments", tree.read_text("crates/cobre-solver/Cargo.toml")
        )
        self.assertIn("5 of the 6 clippy entries", " ".join(block.split()))


class HandoffTests(sc.StationCase):
    SLUG = "solver-comm"
    SECTION_TITLE = "solver-comm"

    @classmethod
    def setUpClass(cls) -> None:
        cls.handoffs = sc.load_json(cls.station_dir() / "handoffs.json")

    def test_e5_block_shape_and_anchors(self) -> None:
        e5 = self.handoffs["E5"]
        for key in (
            "dupOf",
            "mirrorAnchor",
            "registerId",
            "anchors",
            "idAssigned",
            "note",
        ):
            self.assertIn(key, e5)
        self.assertIsNone(e5["idAssigned"])
        self.assertEqual(e5["registerId"], "CD-019")
        tree = self.tree()
        for a in e5["anchors"]:
            with self.subTest(symbol=a["symbol"]):
                self.assertTrue(a["path"].startswith("crates/cobre-sddp/"))
                self.assertIn(
                    f"pub fn {a['symbol']}",
                    tree.read_text(a["path"]).splitlines()[a["line"] - 1],
                )
        mirror_path, mirror_line = e5["mirrorAnchor"].rsplit(":", 1)
        self.assertIn(
            "Superseded cut-sync public methods",
            tree.read_text(mirror_path).splitlines()[int(mirror_line) - 1],
        )

    def test_reserved_slots_are_explicit_nulls_with_an_owner(self) -> None:
        for key in ("E7", "E9", "E10"):
            with self.subTest(slot=key):
                self.assertIsNone(self.handoffs[key]["block"])
                self.assertTrue(self.handoffs[key]["filledBy"])


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_file_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])


if __name__ == "__main__":
    unittest.main()
