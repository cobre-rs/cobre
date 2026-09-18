"""cobre-solver + cobre-comm station tests: inventory census, prior register, handoffs.

Run from the repository root:
    python3 -m unittest plans/architecture-debt-audit/stations/solver-comm/tests/test_station.py
Every figure is re-derived from the tree the station evaluated (inventory.json's
`baseline`, read through `station_checks.Tree`), never from the worktree. Later
solver+comm tickets append their own stage classes here; the station verification runs
the module.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
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

    def test_every_slot_names_its_owner_and_unfilled_slots_are_explicit_nulls(
        self,
    ) -> None:
        for key in ("E7", "E9", "E10"):
            with self.subTest(slot=key):
                self.assertIn("block", self.handoffs[key])
                self.assertTrue(self.handoffs[key]["filledBy"])
        e10 = self.handoffs["E10"]["block"]
        self.assertIsInstance(e10, list, "E10 is filled by the calibrate ticket")
        self.assertTrue(e10)
        for row in e10:
            self.assertIn(row["layout"], ("4t", "2x2"), row["entryId"])
            self.assertEqual(row["status"], "UNMEASURED")
        self.assertIsNotNone(
            self.handoffs["E9"]["block"], "E9 is filled by the Part-I item 8 ticket"
        )
        self.assertIsNotNone(
            self.handoffs["E7"]["block"], "E7 is filled by the Part-I item 8 ticket"
        )


class PartIHandoffTests(sc.StationCase):
    """Part-I item 8 re-verified at the station baseline (partI-handoff.json)."""

    SLUG = "solver-comm"
    SECTION_TITLE = "solver-comm"
    FIELDS = ("n_state", "n_transfer", "n_dual_relevant", "n_hydro", "max_par_order")

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = sc.load_json(cls.station_dir() / "partI-handoff.json")
        cls.handoffs = sc.load_json(cls.station_dir() / "handoffs.json")

    def test_exactly_item_8_with_a_valid_claim_disposition(self) -> None:
        self.assertEqual(self.env["partIRef"], "I.3-8")
        self.assertEqual([d["partIRef"] for d in self.env["dispositions"]], ["I.3-8"])
        item = self.env["dispositions"][0]
        self.assertIn(item["disposition"], {"keep", "retire", "sharpen"})
        self.assertEqual(self.env["baseline"], self.baseline())
        self.assertEqual(self.env["station"], "solver-comm")

    def test_a_kept_or_sharpened_claim_anchors_the_definition_and_names_the_five_fields(
        self,
    ) -> None:
        item = self.env["dispositions"][0]
        tree = self.tree()
        if item["disposition"] == "retire":
            sha = item["retiredAt"]
            self.assertEqual(
                sc._git("cat-file", "-e", f"{sha}^{{commit}}").returncode, 0, sha
            )
            return
        a = item["baselineAnchor"]
        self.assertTrue(sc.anchor_exists(f"`{a['path']}::{a['symbol']}`", tree), a)
        self.assertIn(
            "pub struct StageTemplate",
            tree.read_text(a["path"]).splitlines()[a["line"] - 1],
        )
        self.assertEqual(tuple(item["fieldsReverified"]), self.FIELDS)
        self.assertIn("Epic 9", item["alignmentDestination"])

    def test_five_field_dispositions_resolve_with_their_doc_vocabulary_at_the_baseline(
        self,
    ) -> None:
        tree = self.tree()
        rows = self.env["perFieldDisposition"]
        self.assertEqual([r["field"] for r in rows], list(self.FIELDS))
        lines = tree.read_text("crates/cobre-solver/src/types.rs").splitlines()
        for r in rows:
            with self.subTest(field=r["field"]):
                path, ln = r["anchor"].rsplit(":", 1)
                decl = lines[int(ln) - 1]
                self.assertRegex(decl, rf"^\s*pub {r['field']}: usize,")
                self.assertIn(r["disposition"], {"keep", "retire", "sharpen"})
                doc_start = int(ln) - 1
                while doc_start > 0 and lines[doc_start - 1].strip().startswith("///"):
                    doc_start -= 1
                doc = " ".join(
                    x.strip().lstrip("/").strip()
                    for x in lines[doc_start : int(ln) - 1]
                )
                for token in r["docVocabulary"]:
                    self.assertIn(
                        token, doc, f"{r['field']}: {token!r} not in its own doc block"
                    )
                self.assertTrue(
                    r["dispositionReason"]
                    and r["ownerAfterShed"]
                    and r["sharpenVariantRejected"]
                )
                self.assertEqual(
                    r["reservedSeamsCheck"], {"mirrorHits": 0, "backlogHits": 0}
                )

    def test_propagation_is_one_production_site_and_two_test_fixtures(self) -> None:
        tree = self.tree()
        sites = self.env["propagationSites"]
        scopes = {s["path"]: s["scope"] for s in sites}
        self.assertEqual(
            scopes,
            {
                "crates/cobre-solver/src/freeze.rs": "production",
                "crates/cobre-solver/src/trait_def.rs": "test",
                "crates/cobre-solver/src/backends/profiled.rs": "test",
            },
        )
        for s in sites:
            with self.subTest(path=s["path"]):
                lines = tree.read_text(s["path"]).splitlines()
                lo, hi = (int(x) for x in s["lines"].split("-"))
                span = lines[lo - 1 : hi]
                self.assertEqual(len(span), 5)
                for field, line in zip(self.FIELDS, span, strict=True):
                    self.assertIn(field, line)
                gate = next(
                    i + 1
                    for i, x in enumerate(lines)
                    if re.match(r"^#\[cfg\((all\()?test", x)
                )
                if s["scope"] == "production":
                    self.assertLess(hi, gate)
                    self.assertIsNone(s["gatedBy"])
                else:
                    self.assertGreater(lo, gate)
                    self.assertIn(f"{s['path']}:{gate}", s["gatedBy"])
        self.assertIn("ONE site", self.env["propagationSummary"])

    def test_write_only_evidence_names_writer_copy_forward_and_test_readers(
        self,
    ) -> None:
        tree = self.tree()
        ev = self.env["writeOnlyEvidence"]
        self.assertEqual(ev["productionReads"], 0)
        writer = tree.read_text(
            "crates/cobre-sddp/src/lp/builder/template.rs"
        ).splitlines()
        self.assertIn("n_transfer = ctx.n_hydros * ctx.max_par_order", writer[444])
        for field, ln in zip(self.FIELDS, range(459, 464), strict=True):
            self.assertIn(field, writer[ln - 1])
        freeze = tree.read_text("crates/cobre-solver/src/freeze.rs").splitlines()
        for field, ln in zip(self.FIELDS, range(158, 163), strict=True):
            self.assertRegex(freeze[ln - 1], rf"out\.{field} = base\.{field};")
        self.assertGreaterEqual(len(ev["testReaders"]), 2)
        for reader in ev["testReaders"]:
            path, span = reader.split(" ")[0].rsplit(":", 1)
            lo = int(span.split("-")[0])
            with self.subTest(reader=reader):
                self.assertTrue(tree.is_file(path), path)
                self.assertRegex(
                    tree.read_text(path).splitlines()[lo - 1],
                    r"n_(state|transfer|dual_relevant|hydro)|max_par_order",
                )
        # the geometry's owner one layer up
        ss = tree.read_text(
            "crates/cobre-sddp/src/lp/indexer/state_space.rs"
        ).splitlines()
        self.assertIn("pub n_state: usize", ss[96])
        self.assertIn("pub hydro_count: usize", ss[99])
        self.assertIn("pub max_par_order: usize", ss[103])
        layout = tree.read_text(
            "crates/cobre-sddp/src/lp/builder/layout.rs"
        ).splitlines()
        self.assertIn("n_dual_relevant: usize", layout[508])
        self.assertIn("let n_dual_relevant = 0_usize;", layout[1354])

    def test_no_production_stage_template_read_exists_at_the_baseline(self) -> None:
        """Every production `.field` access of the five names is on a non-StageTemplate receiver."""
        tree = self.tree()
        rx = re.compile(
            r"([A-Za-z_][A-Za-z0-9_]*)(?:\([^)]*\))?\.(n_state|n_transfer|n_dual_relevant|n_hydro|max_par_order)\b"
        )
        template_receivers: set[tuple[str, int, str]] = set()
        for path in tree.files():
            if not (path.startswith("crates/") and path.endswith(".rs")):
                continue
            if (
                "/tests/" in path
                or "/benches/" in path
                or path.endswith(("tests.rs", "test_support.rs"))
            ):
                continue
            lines = tree.read_text(path).splitlines()
            gate = next(
                (
                    i + 1
                    for i, x in enumerate(lines)
                    if re.match(r"^#\[cfg\((all\()?test", x)
                ),
                len(lines) + 1,
            )
            for i, line in enumerate(lines[: gate - 1], 1):
                for m in rx.finditer(line):
                    recv = m.group(1)
                    # a receiver is a StageTemplate when its declaration in the file says so
                    if re.search(
                        rf"\b{re.escape(recv)}\s*:\s*&?(mut\s+)?StageTemplate\b",
                        "\n".join(lines),
                    ):
                        template_receivers.add((path, i, recv))
        self.assertEqual(
            {p for p, _, _ in template_receivers},
            {"crates/cobre-solver/src/freeze.rs"},
            "a production StageTemplate read appeared; a field read in production flips its disposition to keep",
        )
        self.assertEqual({ln for _, ln, _ in template_receivers}, set(range(158, 163)))

    def test_reserved_seams_check_is_recorded_and_the_mirror_has_no_hit(self) -> None:
        tree = self.tree()
        check = self.env["reservedSeamsCheck"]
        mirror = tree.read_text(backlog_parse.MIRROR)
        self.assertEqual(
            sum(bool(re.search(rf"\b{tok}\b", mirror)) for tok in check["tokens"]),
            check["mirrorHits"],
        )
        self.assertEqual(check["mirrorHits"], 0)
        self.assertEqual(check["backlogHits"], 0)
        register = "\n".join(backlog_parse.read_register(sc.BACKLOG))
        entries = [
            e
            for s in backlog_parse.all_evaluation_sections(
                backlog_parse.read_register(sc.BACKLOG)
            )
            for e in backlog_parse.iter_entries(s)
        ]
        hits = [
            e.id for e in entries if "StageTemplate" in "\n".join([e.heading, *e.body])
        ]
        # Before this station recorded its section no register entry covered StageTemplate (so
        # I.3-8 is a new finding, not a dup-of); afterwards the ONLY entries that mention it are
        # this station's own I.3-8 rows.
        own = {
            r["id"]
            for r in sc.load_json(self.artifact("calibration.json"))["assigned"]
            if r.get("partIRef") == "I.3-8"
        }
        self.assertEqual(
            set(hits),
            own,
            "a register entry outside this station covers StageTemplate: dup-of, not a new finding",
        )
        self.assertIn(
            "StageTemplate shed",
            register,
            "the milestone row the backlogTokenHits note classifies",
        )
        self.assertIn("NEW finding", check["verdict"])

    def test_fix_shape_is_phase_1_prose_tagged_advances_1(self) -> None:
        self.assertEqual(self.env["proposedAlignment"], "advances-1")
        fix = self.env["fixShape"]
        for token in (
            "col_starts",
            "row_scale",
            "freeze.rs:158-162",
            "state_space.rs:97,100,104",
            "layout.rs:509",
            "advances-1",
            "Rejected sharpen variant",
        ):
            self.assertIn(token, fix)
        self.assertTrue(
            all(
                self.env["l0PurityTest"][k]
                for k in (
                    "noEngineConceptPlacedInCobreSolver",
                    "noDependencyFromCobreSolverOntoAnEngineCrate",
                    "noOneConsumerAbstraction",
                )
            )
        )
        roadmap = pathlib.Path(
            sc.REPO / "plans/generalizing/beyond-sddp-generalization.md"
        )
        if roadmap.exists():
            lines = roadmap.read_text(encoding="utf-8").splitlines()
            self.assertIn("StageTemplate", lines[1470])
            self.assertIn("cobre-solver", lines[1371])
            self.assertIn("StageTemplate", lines[351])

    def test_genericity_gate_blind_spot_is_recorded_without_a_gate_edit(self) -> None:
        tree = self.tree()
        g = self.env["genericityGateBlindSpot"]
        freeze = tree.read_text("crates/cobre-solver/src/freeze.rs").splitlines()
        self.assertIn("cut_nz_per_col", freeze[21])
        for ln in g["productionUses"]:
            self.assertIn("cut_nz_per_col", freeze[ln - 1])
            self.assertLess(ln, g["cfgTestBoundary"])
        self.assertTrue(freeze[g["cfgTestBoundary"] - 1].startswith("#[cfg(test)]"))
        gate = tree.read_text(g["gateScript"]).splitlines()
        self.assertTrue(gate[g["patternLine"] - 1].startswith("PATTERN="))
        self.assertIn("\\bcut\\b", gate[g["patternLine"] - 1])
        self.assertIsNone(
            re.search(r"\bcut\b", "cut_nz_per_col"), "the evasion mechanism"
        )
        self.assertIs(g["changeProposed"], False)
        self.assertEqual(
            (sc.REPO / g["gateScript"]).read_text(encoding="utf-8"),
            tree.read_text(g["gateScript"]),
            "the gate script must be untouched in the working tree",
        )
        e7 = self.handoffs["E7"]["block"]
        for key in (
            "gateScript",
            "evadingIdentifier",
            "anchor",
            "patternEvaded",
            "gateExitAtBaseline",
            "changeProposed",
        ):
            self.assertIn(key, e7)
        self.assertEqual(e7["gateExitAtBaseline"], 0)
        self.assertIs(e7["changeProposed"], False)
        e9 = self.handoffs["E9"]["block"]
        self.assertEqual(e9["partIRef"], "I.3-8")
        self.assertEqual(e9["proposedAlignment"], "advances-1")
        self.assertEqual(set(e9["perFieldDisposition"]), set(self.FIELDS))


# ---------------------------------------------------------------------------
# Attacker fan-out: the four merged candidate envelopes and the station guards.
# ---------------------------------------------------------------------------

LENSES = ("architecture", "performance", "over-engineering", "test-bloat")
REF_PREFIX = {
    "architecture": "ARCH",
    "performance": "PERF",
    "over-engineering": "OE",
    "test-bloat": "TB",
}
IN_SCOPE = ("crates/cobre-solver/", "crates/cobre-comm/")
DUP_OF_PATH = "crates/cobre-sddp/src/cut/cut_sync.rs"
LAYOUTS = {"4t", "2x2"}
SHARED_MEMORY = (
    "SharedMemoryProvider",
    "SharedRegion",
    "LocalCommunicator",
    "LocalCommKind",
    "HeapRegion",
    "split_local",
)
TIMING_NUMBER = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\b|\u00d7|%|(?:ms|\u00b5s|us|ns|s|sec|secs|seconds|minutes|min|speedup|faster|slower)\b)",
    re.I,
)
NEW_CRATE = re.compile(
    r"\bnew\b[^.]{0,60}\b(fixture|test[- ]support|helper)s?\b[^.]{0,40}\bcrate\b", re.I
)
DIFF_MARKER = re.compile(r"```|^\+\+\+ |^--- |^@@ ", re.M)


def guard_scope(cand: dict) -> str | None:
    """The anchor-scope guard's verdict: None (kept) or the drop code."""
    paths = [a["path"] for a in cand["anchors"]]
    if any(p.startswith(IN_SCOPE) for p in paths):
        return None
    if cand.get("dupOf") and paths and all(p == DUP_OF_PATH for p in paths):
        return None
    return "out-of-scope-anchor"


def guard_layout(cand: dict) -> str | None:
    """The performance layout guard's verdict: None (kept) or the drop code."""
    if cand.get("measurementLayout") not in LAYOUTS:
        return "layout-missing"
    blob = " ".join(
        [
            cand.get("title", ""),
            cand.get("mechanism", ""),
            cand.get("fixShape", ""),
            json.dumps(cand.get("evidence", {})),
        ]
    )
    return "timing-number" if TIMING_NUMBER.search(blob) else None


def guard_fixture(fix_shape: str) -> bool:
    """True when the test-bloat fixture guard would rewrite this fix-shape (a new fixture crate)."""
    return bool(NEW_CRATE.search(fix_shape))


class CandidateEnvelopeTests(sc.StationCase):
    SLUG = "solver-comm"
    SECTION_TITLE = "solver-comm"

    @classmethod
    def setUpClass(cls) -> None:
        cls.merged = {
            lens: sc.load_json(cls.station_dir() / f"candidates.{lens}.json")
            for lens in LENSES
        }
        cls.raw = {
            lens: sc.load_json(cls.station_dir() / "raw" / f"sc-{lens}.json")
            for lens in LENSES
        }
        cls.log = (cls.station_dir() / "attacker-log.md").read_text(encoding="utf-8")
        cls.prompt = (cls.station_dir() / "attacker-prompt.md").read_text(
            encoding="utf-8"
        )

    def candidates(self, lens: str) -> list[dict]:
        return self.merged[lens]["candidates"]

    def test_prompt_carries_the_four_lenses_the_rules_and_the_envelope(self) -> None:
        for heading in (
            "## Lens: architecture",
            "## Lens: performance",
            "## Lens: over-engineering",
            "## Lens: test-bloat",
        ):
            self.assertIn(heading, self.prompt)
        self.assertGreaterEqual(len(re.findall(r"\*\*P\d+ ", self.prompt)), 15)
        for key in (
            '"station"',
            '"baseline"',
            '"lens"',
            '"candidates"',
            '"positives"',
            '"_needsHuman"',
            '"measurementLayout"',
        ):
            self.assertIn(key, self.prompt)
        rules = re.findall(r"^(\d)\. \*\*([^*]+)\*\*", self.prompt, re.M)
        self.assertEqual([n for n, _ in rules], [str(i) for i in range(1, 9)])
        labels = " ".join(label for _, label in rules).lower()
        for needle in (
            "read-only",
            "one json object",
            "anchor scope",
            "reserved seams first",
            "layout, never a number",
            "prose",
        ):
            self.assertIn(needle, labels)
        for lens in LENSES:
            rendered = (self.station_dir() / "prompts" / f"sc-{lens}.md").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("{{", rendered)
            self.assertIn(f"Lens: `{lens}`", rendered)
            self.assertIn(self.baseline(), rendered)

    def test_every_merged_and_raw_envelope_validates(self) -> None:
        for lens in LENSES:
            for path in (
                self.station_dir() / f"candidates.{lens}.json",
                self.station_dir() / "raw" / f"sc-{lens}.json",
            ):
                with self.subTest(file=path.name):
                    proc = subprocess.run(
                        [
                            "python3",
                            str(sc.TOOLS / "validate-envelope.py"),
                            "--role",
                            "attacker",
                            "--station",
                            "solver-comm",
                            str(path),
                        ],
                        cwd=sc.REPO,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_envelope_lens_baseline_and_refs_are_consistent(self) -> None:
        for lens in LENSES:
            with self.subTest(lens=lens):
                doc = self.merged[lens]
                self.assertEqual(doc["lens"], lens)
                self.assertEqual(doc["baseline"], self.baseline())
                self.assertEqual(doc["station"], "solver-comm")
                self.assertEqual(doc["lenses"], {f"sc-{lens}": "merged"})
                refs = [c["candidateRef"] for c in doc["candidates"]]
                self.assertEqual(
                    refs,
                    [f"SC-{REF_PREFIX[lens]}-{i:03d}" for i in range(1, len(refs) + 1)],
                )
                keys = [
                    (sorted(a["path"] for a in c["anchors"])[0], c["title"])
                    for c in doc["candidates"]
                ]
                self.assertEqual(
                    keys,
                    sorted(keys),
                    "merged candidates are sorted by first anchor path then title",
                )
                self.assertTrue(all(c["lens"] == lens for c in doc["candidates"]))
                self.assertEqual(
                    len(doc["candidates"]) + len(doc["dropped"]),
                    len(self.raw[lens]["candidates"]),
                )
                self.assertEqual(doc["positives"], self.raw[lens]["positives"])

    def test_every_anchor_is_in_scope_or_the_single_dup_of_exception(self) -> None:
        for lens in LENSES:
            for c in self.candidates(lens):
                with self.subTest(ref=c["candidateRef"]):
                    self.assertIsNone(guard_scope(c), c["title"])
                    for a in c["anchors"]:
                        self.assertTrue(a.get("symbol") or a.get("line") is not None, a)
                        if not a["path"].startswith(IN_SCOPE):
                            self.assertEqual(a["path"], DUP_OF_PATH)
                            self.assertEqual(
                                c.get("dupOf"), "Superseded cut-sync public methods"
                            )

    def test_dup_of_handoff_candidates_carry_the_verified_cut_sync_anchors(
        self,
    ) -> None:
        dups = [c for lens in LENSES for c in self.candidates(lens) if c.get("dupOf")]
        self.assertTrue(dups, "at least one lens emitted the E5 dup-of handoff")
        tree = self.tree()
        text = tree.read_text(DUP_OF_PATH).splitlines()
        for c in dups:
            with self.subTest(ref=c["candidateRef"]):
                for a in c["anchors"]:
                    self.assertEqual(a["path"], DUP_OF_PATH)
                    if a.get("line"):
                        self.assertIn(a["symbol"], text[a["line"] - 1])

    def test_performance_candidates_carry_a_layout_call_sites_and_no_number(
        self,
    ) -> None:
        perf = self.candidates("performance")
        self.assertTrue(perf)
        for c in perf:
            with self.subTest(ref=c["candidateRef"]):
                self.assertIsNone(guard_layout(c), c["title"])
                self.assertTrue(c.get("mechanism", "").strip())
                self.assertTrue(
                    c.get("exercisingCallSites"),
                    "perf candidates list exercising call sites",
                )
                solver_side = all(
                    a["path"].startswith("crates/cobre-solver/") for a in c["anchors"]
                )
                self.assertEqual(c["measurementLayout"], "4t" if solver_side else "2x2")
        collective = [
            c
            for c in perf
            if any(a["path"].startswith("crates/cobre-comm/") for a in c["anchors"])
        ]
        self.assertTrue(
            collective,
            "the per_rank_counts/prefix_displs probe produced a collective candidate",
        )
        sites = " ".join(" ".join(c["exercisingCallSites"]) for c in collective)
        for site in (
            "cut_sync.rs:182",
            "stats_aggregation.rs:118",
            "rank_distribution.rs:61",
        ):
            self.assertIn(site, sites)

    def test_over_engineering_routes_the_shared_memory_seam_to_positives(self) -> None:
        for lens in LENSES:
            for c in self.candidates(lens):
                anchored = " ".join(
                    f"{a['path']} {a.get('symbol', '')}" for a in c["anchors"]
                )
                self.assertFalse(
                    any(s in anchored for s in SHARED_MEMORY),
                    f"{c['candidateRef']} anchors on a sanctioned shared-memory symbol",
                )
                self.assertNotRegex(
                    c["fixShape"],
                    r"(remove|delete|drop|retire)\b[^.]{0,80}\b(SharedMemoryProvider|SharedRegion|LocalCommunicator|LocalCommKind|HeapRegion|split_local)\b",
                    f"{c['candidateRef']} proposes removing the sanctioned seam",
                )
        oe = self.merged["over-engineering"]
        seam = [p for p in oe["positives"] if "SharedMemoryProvider" in p["subject"]]
        self.assertTrue(seam)
        self.assertIn("reserved-seams-and-deferred-debt.md:79", seam[0]["sanctionedBy"])
        self.assertTrue(
            any(
                "BasisInconsistent" in json.dumps(p) or "basis" in p["subject"].lower()
                for p in oe["positives"]
            )
        )

    def test_architecture_reverifies_item_8_and_records_the_gate_blind_spot(
        self,
    ) -> None:
        arch = self.candidates("architecture")
        item8 = [c for c in arch if c.get("partIRef") == "I.3-8"]
        self.assertEqual(len(item8), 1)
        c = item8[0]
        blob = json.dumps(c)
        self.assertEqual(c["alignmentHint"], "advances-1")
        types_lines = sorted(
            a["line"]
            for a in c["anchors"]
            if a["path"] == "crates/cobre-solver/src/types.rs" and a.get("line")
        )
        for ln in (270, 278, 287, 290, 297):
            self.assertIn(ln, types_lines)
        for field in (
            "n_state",
            "n_transfer",
            "n_dual_relevant",
            "n_hydro",
            "max_par_order",
        ):
            self.assertIn(field, blob)
        for site in ("158", "380", "252"):
            self.assertIn(
                site,
                c["fixShape"] + json.dumps(c["evidence"]),
                f"propagation site line {site}",
            )
        self.assertTrue(
            "StateSpace" in blob and "StageRowLayout" in blob,
            "the owner after the shed is named",
        )
        cut = [c for c in arch if "cut_nz_per_col" in c["title"]]
        self.assertEqual(len(cut), 1)
        blob = json.dumps(cut[0])
        for ln in ("137", "138", "141", "188"):
            self.assertIn(ln, blob)
        self.assertIn("word character", blob)
        self.assertIn("E7", blob)
        self.assertNotRegex(
            cut[0]["fixShape"].lower(),
            r"(edit|change|patch|widen|fix)\s+(the\s+)?(gate|pattern|script)",
        )

    def test_test_bloat_hoists_fixtures_into_test_support_and_protects_the_probes(
        self,
    ) -> None:
        tb = self.merged["test-bloat"]
        fixture = [
            c
            for c in tb["candidates"]
            if "make_fixture_stage_template" in json.dumps(c)
        ]
        self.assertTrue(fixture)
        main = max(fixture, key=lambda c: len(c["anchors"]))
        anchors = {
            (a["path"].rsplit("/", 1)[-1], a.get("line")) for a in main["anchors"]
        }
        self.assertIn(("conformance.rs", 34), anchors)
        self.assertIn(("clp_determinism.rs", 33), anchors)
        blob = json.dumps(main)
        self.assertIn("sentinel_inf_row_probe", blob)
        self.assertIn("profile_retry_composition", blob)
        self.assertIn("test_support", main["fixShape"])
        self.assertIn("164", main["fixShape"])
        for c in tb["candidates"]:
            self.assertFalse(
                guard_fixture(c["fixShape"]),
                f"{c['candidateRef']} proposes a new fixture crate",
            )
        subjects = " ".join(p["subject"] for p in tb["positives"])
        self.assertIn("_q1_sign_convention_probe.rs", subjects)
        self.assertIn("_clp_sign_convention_probe.rs", subjects)
        for c in tb["candidates"]:
            self.assertFalse(
                any("_sign_convention_probe" in a["path"] for a in c["anchors"]),
                c["candidateRef"],
            )

    def test_no_fix_shape_is_a_diff_or_a_code_block(self) -> None:
        for lens in LENSES:
            for c in self.candidates(lens):
                self.assertIsNone(DIFF_MARKER.search(c["fixShape"]), c["candidateRef"])

    def test_attacker_log_records_one_dispatch_per_lens_and_the_gaps_section(
        self,
    ) -> None:
        for lens in LENSES:
            self.assertRegex(
                self.log,
                re.compile(
                    rf"^\| sc-{re.escape(lens)} \| {re.escape(lens)} \|.*\| valid", re.M
                ),
            )
        rows = re.findall(r"^\| sc-[a-z-]+ \| [a-z-]+ \| ", self.log, re.M)
        self.assertEqual(len(rows), 4)
        self.assertIn("## Gaps carried forward", self.log)
        self.assertIn("check-anchors.py", self.log)
        self.assertIn("## Prior-register screen", self.log)

    # --- the guards must bite on tampered input, since the real run tripped none of them ---

    def test_scope_guard_drops_out_of_scope_unless_dup_of(self) -> None:
        sddp_only = {
            "title": "x",
            "anchors": [{"path": "crates/cobre-sddp/src/cut/cut_sync.rs", "line": 243}],
        }
        self.assertEqual(guard_scope(sddp_only), "out-of-scope-anchor")
        self.assertIsNone(
            guard_scope({**sddp_only, "dupOf": "Superseded cut-sync public methods"})
        )
        other_sddp = {
            "title": "x",
            "dupOf": "Superseded cut-sync public methods",
            "anchors": [
                {"path": "crates/cobre-sddp/src/lp/builder/template.rs", "line": 445}
            ],
        }
        self.assertEqual(
            guard_scope(other_sddp),
            "out-of-scope-anchor",
            "dupOf licenses only cut_sync.rs",
        )
        self.assertIsNone(
            guard_scope(
                {
                    "title": "x",
                    "anchors": [{"path": "crates/cobre-comm/src/lib.rs", "line": 83}],
                }
            )
        )

    def test_layout_guard_drops_missing_layout_and_timing_numbers(self) -> None:
        base = {
            "title": "t",
            "mechanism": "m",
            "fixShape": "prose",
            "evidence": {"command": "grep"},
            "anchors": [],
        }
        self.assertEqual(guard_layout(base), "layout-missing")
        self.assertEqual(
            guard_layout({**base, "measurementLayout": "8t"}), "layout-missing"
        )
        self.assertIsNone(guard_layout({**base, "measurementLayout": "4t"}))
        self.assertEqual(
            guard_layout(
                {**base, "measurementLayout": "4t", "fixShape": "saves 12 ms per solve"}
            ),
            "timing-number",
        )
        self.assertEqual(
            guard_layout(
                {**base, "measurementLayout": "2x2", "mechanism": "a 3x speedup"}
            ),
            "timing-number",
        )
        self.assertIsNone(
            guard_layout(
                {
                    **base,
                    "measurementLayout": "2x2",
                    "evidence": {"command": "grep", "output": "vec![0usize; n]"},
                }
            ),
            "a usize literal is not a timing",
        )

    def test_fixture_guard_rewrites_only_a_new_crate_proposal(self) -> None:
        self.assertTrue(
            guard_fixture(
                "Create a new cobre-solver-fixtures crate shared by the binaries."
            )
        )
        self.assertFalse(
            guard_fixture(
                "Add a fixtures submodule inside the shipped test_support module."
            )
        )


INGEST_DISPOSITIONS = {
    "defended",
    "anchor-missing",
    "out-of-station",
    "sanctioned",
    "dup-of",
    "re-raise",
    "informational",
}
# The ticket's test vocabulary (accepted / rejected-anchor / rejected-reserved-seam /
# rejected-defender / merged) maps onto disposition + defender verdict; the map is the
# single place the two vocabularies meet.
INGEST_STATE = {
    ("defended", "confirmed"): "accepted",
    ("defended", "dismissed"): "rejected-defender",
    ("anchor-missing", None): "rejected-anchor",
    ("out-of-station", None): "rejected-anchor",
    ("sanctioned", None): "rejected-reserved-seam",
    ("dup-of", None): "merged",
    ("informational", None): "accepted",
}
MIRROR_ENTRIES = {
    "Shared-memory communicator trait hierarchy",
    "Superseded cut-sync public methods",
}
ALIGN = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
CUT_SYNC_SYMBOLS = ("sync_cuts", "pack_local_records", "sync_packed_records")
ASYMMETRY = re.compile(
    r"BasisInconsistent|isBasisConsistent|basis[- ]validation|"
    r"(?:loud|silent)\w*\W.{0,60}\bbasis|\bbasis\b.{0,80}(?:loud|silent)",
    re.I | re.S,
)


def ingest_ref(attacker_ref: str, lens: str) -> str:
    return f"{lens}-{int(attacker_ref.rsplit('-', 1)[1]):02d}"


def normalised(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


class IngestTests(sc.StationCase):
    """verdicts.json + ingest-log.md + anchor-probe.md + the E5 dup-of handoff records."""

    SLUG = "solver-comm"

    @classmethod
    def setUpClass(cls) -> None:
        cls.verdicts = sc.load_json(cls.station_dir() / "verdicts.json")
        cls.handoffs = sc.load_json(cls.station_dir() / "handoffs.json")
        cls.log = (
            cls.station_dir().joinpath("ingest-log.md").read_text(encoding="utf-8")
        )
        cls.probe = (
            cls.station_dir().joinpath("anchor-probe.md").read_text(encoding="utf-8")
        )
        cls.candidates: dict[str, dict] = {}
        for lens in LENSES:
            merged = sc.load_json(cls.station_dir() / f"candidates.{lens}.json")
            for cand in merged["candidates"]:
                cls.candidates[ingest_ref(cand["candidateRef"], lens)] = {
                    **cand,
                    "lens": lens,
                }

    def entries(self) -> dict[str, dict]:
        return self.verdicts["verdicts"]

    def state(self, entry: dict) -> str:
        verdict = (entry.get("defender") or {}).get("verdict")
        return INGEST_STATE[(entry["disposition"], verdict)]

    def test_one_verdict_per_candidate_and_attacker_ref_round_trips(self) -> None:
        self.assertEqual(set(self.entries()), set(self.candidates))
        for ref, entry in self.entries().items():
            self.assertEqual(entry["candidateRef"], ref)
            self.assertEqual(entry["attackerRef"], self.candidates[ref]["candidateRef"])
            self.assertEqual(entry["lens"], self.candidates[ref]["lens"])
        self.assertEqual(self.verdicts["counts"]["received"], len(self.candidates))
        self.assertEqual(self.verdicts["baseline"], self.baseline())

    def test_every_verdict_maps_onto_the_ticket_state_vocabulary(self) -> None:
        for ref, entry in self.entries().items():
            self.assertIn(entry["disposition"], INGEST_DISPOSITIONS, ref)
            state = self.state(entry)
            self.assertIn(
                state,
                {
                    "accepted",
                    "rejected-anchor",
                    "rejected-reserved-seam",
                    "rejected-defender",
                    "merged",
                },
                ref,
            )
            if entry["disposition"] == "defended":
                self.assertIn(entry["defender"]["verdict"], ("confirmed", "dismissed"))
            else:
                self.assertNotIn(
                    "defender", entry, f"{ref}: no defender off the defended route"
                )

    def test_confirmed_carries_a_strictly_narrower_surviving_claim(self) -> None:
        for ref, entry in self.entries().items():
            defender = entry.get("defender")
            if defender and defender["verdict"] == "confirmed":
                claim = normalised(defender.get("survivingClaim", ""))
                title = normalised(self.candidates[ref]["title"])
                self.assertTrue(claim, f"{ref}: confirmed without survivingClaim")
                self.assertNotEqual(
                    claim, title, f"{ref}: survivingClaim restates the title"
                )
                self.assertNotIn(
                    title, claim, f"{ref}: survivingClaim contains the title"
                )
                self.assertNotIn("sanctionedBy", defender, ref)

    def test_every_defender_argument_is_reasoning_with_an_alignment_hint(self) -> None:
        for ref, entry in self.entries().items():
            defender = entry.get("defender")
            if defender:
                self.assertGreaterEqual(len(defender["argument"].strip()), 120, ref)
                self.assertIn(defender["alignmentHint"], ALIGN, ref)
                self.assertEqual(entry["alignmentHint"], defender["alignmentHint"], ref)

    def test_sanctioned_dismissals_cite_one_of_the_two_mirror_entries(self) -> None:
        mirror = self.tree().read_text(backlog_parse.MIRROR)
        for entry_name in MIRROR_ENTRIES:
            self.assertIn(f"### {entry_name}", mirror)
        for ref, entry in self.entries().items():
            cite = (entry.get("defender") or {}).get("sanctionedBy") or entry.get(
                "sanctionedBy"
            )
            if entry["disposition"] in ("sanctioned", "dup-of") or cite is not None:
                self.assertIn(cite, MIRROR_ENTRIES, f"{ref}: sanctionedBy {cite!r}")

    def test_basis_asymmetry_dismissals_name_intended_behaviour(self) -> None:
        for ref, entry in self.entries().items():
            defender = entry.get("defender")
            if (
                defender
                and defender["verdict"] == "dismissed"
                and ASYMMETRY.search(defender["argument"])
            ):
                self.assertTrue(
                    defender.get("intendedBehaviour", "").strip(),
                    f"{ref}: dismissal rests on the basis-validation asymmetry without intendedBehaviour",
                )
                self.assertRegex(defender["intendedBehaviour"], r"(?i)intended")

    def test_shared_memory_hierarchy_never_reaches_a_defender(self) -> None:
        # The ratified seam is an ingest filter: it may appear only in positives, never as a
        # candidate, so no verdict names one of its six symbols as a removal target.
        for ref, entry in self.entries().items():
            if entry["disposition"] == "sanctioned":
                self.assertEqual(
                    entry["sanctionedBy"],
                    "Shared-memory communicator trait hierarchy",
                    ref,
                )
                self.assertNotIn("defender", entry, ref)
            else:
                anchors = {a.get("symbol") for a in entry["anchors"]}
                self.assertFalse(
                    anchors & set(SHARED_MEMORY), f"{ref}: anchors the reserved seam"
                )
        cleared = self.log.split("## Cleared (sanctioned)", 1)[1].split("\n## ", 1)[0]
        self.assertIn("Shared-memory communicator trait hierarchy", cleared)

    def test_cut_sync_candidates_merge_into_the_e5_handoff_without_an_id(self) -> None:
        tree = self.tree()
        source = tree.read_text(DUP_OF_PATH)
        for symbol in (*CUT_SYNC_SYMBOLS, "sync_level_records"):
            self.assertRegex(source, rf"(?m)^\s*pub fn {symbol}\b")
        merged = {
            ref for ref, e in self.entries().items() if e["disposition"] == "dup-of"
        }
        self.assertTrue(merged)
        records = {r["candidateRef"]: r for r in self.handoffs["records"]}
        self.assertEqual(set(records), merged)
        self.assertEqual(set(self.handoffs["E5"]["ingestRecords"]), merged)
        for ref in merged:
            entry, record = self.entries()[ref], records[ref]
            self.assertEqual(entry["dupOf"], "Superseded cut-sync public methods")
            self.assertEqual(entry["registerId"], "CD-019")
            self.assertIsNone(entry["assignedId"])
            self.assertEqual(record["toEpic"], "E5")
            self.assertEqual(record["kind"], "dup-of")
            self.assertIsNone(record["assignedId"])
            self.assertEqual(record["attackerRef"], entry["attackerRef"])
            self.assertEqual(record["supersededBy"]["symbol"], "sync_level_records")
            self.assertTrue(
                {a["symbol"] for a in record["anchors"]} & set(CUT_SYNC_SYMBOLS)
            )
            self.assertTrue(
                all(a["path"] == DUP_OF_PATH for a in entry["anchors"]), ref
            )
        self.assertIsNone(self.handoffs["E5"]["idAssigned"])

    def test_informational_capability_trait_is_recorded_neutral_with_iii6(self) -> None:
        info = [
            e for e in self.entries().values() if e["disposition"] == "informational"
        ]
        self.assertEqual(len(info), 1)
        entry = info[0]
        self.assertEqual(entry["attackerRef"], "SC-ARCH-009")
        self.assertIn("III.6", entry["roadmapRef"])
        self.assertEqual(entry["alignmentHint"], "neutral")
        self.assertTrue(
            {"crates/cobre-solver/src/trait_def.rs", "crates/cobre-solver/src/lib.rs"}
            <= {a["path"] for a in entry["anchors"]}
        )
        held = self.log.split("## Held as conflicts (L0 purity test)", 1)[1].split(
            "\n## ", 1
        )[0]
        self.assertIn("one consumer", held)
        self.assertRegex(self.log, r"## Informational \(recorded, no severity\)")

    def test_part_i_refs_survive_ingest(self) -> None:
        for ref, entry in self.entries().items():
            cand_ref = self.candidates[ref].get("partIRef")
            self.assertEqual(entry.get("partIRef"), cand_ref, ref)
            defender = entry.get("defender")
            if cand_ref and defender:
                self.assertEqual(entry["partIRef"], "I.3-8")

    def test_perf_verdicts_keep_their_layout_and_carry_no_number(self) -> None:
        for ref, entry in self.entries().items():
            if entry["lens"] == "performance":
                self.assertIn(entry["measurementLayout"], LAYOUTS, ref)
                self.assertEqual(
                    entry["measurementLayout"],
                    self.candidates[ref]["measurementLayout"],
                )
                self.assertIn(entry["proposedSeverity"], ("A", "B", "C"), ref)
                defender = entry.get("defender") or {}
                blob = " ".join(
                    str(defender.get(k, "")) for k in ("argument", "survivingClaim")
                )
                self.assertIsNone(
                    TIMING_NUMBER.search(blob), f"{ref}: number in verdict"
                )

    def test_accepted_anchors_resolve_at_the_station_baseline(self) -> None:
        tree = self.tree()
        for ref, entry in self.entries().items():
            if self.state(entry) in ("accepted", "merged"):
                for anchor in entry["anchors"]:
                    self.assertTrue(tree.is_file(anchor["path"]), f"{ref}: {anchor}")
                    if anchor.get("line") is not None:
                        lines = tree.read_text(anchor["path"]).count("\n") + 1
                        self.assertLessEqual(anchor["line"], lines, f"{ref}: {anchor}")
                    elif anchor.get("symbol"):
                        self.assertTrue(
                            sc.symbol_resolves(anchor["path"], anchor["symbol"], tree),
                            f"{ref}: {anchor}",
                        )

    def test_anchor_probe_has_one_block_per_candidate_and_checker_passes(self) -> None:
        for ref, cand in self.candidates.items():
            self.assertEqual(
                self.probe.count(f"### {ref} · {cand['candidateRef']} · "), 1, ref
            )
        for anchor in ANCHOR_TOKEN.finditer(self.probe):
            self.assertTrue(
                anchor.group("path").startswith(IN_SCOPE)
                or anchor.group("path") == DUP_OF_PATH,
                anchor.group(0),
            )
        result = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "check-anchors.py"),
                "INGEST ANCHOR PROBE — solver-comm (2026-09, baseline)",
                "--register",
                str(self.artifact("anchor-probe.md")),
                "--baseline",
                self.baseline(),
                "--allow-drift",
            ],
            capture_output=True,
            text=True,
            cwd=sc.REPO,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"checked \d+ anchors, 0 failing")

    def test_ingest_log_has_one_roster_row_per_candidate_and_names_the_baseline(
        self,
    ) -> None:
        self.assertIn(f"Baseline: `{self.baseline()}`", self.log)
        roster = self.log.split("## Per-candidate roster", 1)[1]
        for ref, entry in self.entries().items():
            self.assertEqual(
                roster.count(f"| {ref} | {entry['attackerRef']} |"),
                1,
                f"{ref}: expected exactly one roster row",
            )
        for lens in LENSES:
            self.assertRegex(
                self.log, rf"\| {lens} \| candidates\.{lens}\.json \| \d+ \|"
            )

    def test_counts_block_sums_to_the_received_total(self) -> None:
        counts = self.verdicts["counts"]
        entries = list(self.entries().values())
        by_disposition = {
            "anchorRejected": "anchor-missing",
            "outOfStation": "out-of-station",
            "sanctionedCleared": "sanctioned",
            "dupOf": "dup-of",
            "reRaiseRejected": "re-raise",
            "informational": "informational",
            "defended": "defended",
        }
        for key, disposition in by_disposition.items():
            self.assertEqual(
                counts[key],
                sum(1 for e in entries if e["disposition"] == disposition),
                key,
            )
        self.assertEqual(
            sum(counts[k] for k in by_disposition),
            counts["received"],
            "dispositions != received",
        )
        verdicts = [e["defender"]["verdict"] for e in entries if "defender" in e]
        self.assertEqual(counts["confirmed"], verdicts.count("confirmed"))
        self.assertEqual(counts["dismissed"], verdicts.count("dismissed"))
        self.assertEqual(counts["confirmed"] + counts["dismissed"], counts["defended"])
        self.assertEqual(
            counts["needsHuman"],
            sum(1 for e in entries if (e.get("defender") or {}).get("_needsHuman")),
        )


ID_RE = re.compile(r"^(CD|PD|OD|TD)-(\d{3})$")
ID_FLOORS = {"CD": 40, "PD": 6, "OD": 10, "TD": 1}
SEVERITIES = {"A", "B", "B (A-risk)", "C"}
ENTRY_HEADING = re.compile(r"^\*\*((?:CD|PD|OD|TD)-\d{3}) · Sev ", re.M)
STATION_HEADING = "### Station 4 — cobre-solver + cobre-comm (2026-09, baseline "
SECTION_SKELETON = (
    "#### Architecture",
    "#### Performance",
    "#### Over-engineering",
    "#### Test bloat",
    "#### Informational (recorded, no severity, no id)",
    "#### Positives (recorded so the report is not a defect-only list)",
    "#### ↩︎ Cleared (dismissed — do not re-raise)",
    "#### Prior-register dispositions",
    "#### Part-I cross-references",
    "#### Handoff queues",
    "#### Owner gate — decisions",
)
PRECEDENTS = {
    "architecture-06": "CD-010",
    "architecture-08": "CD-011",
    "architecture-01": "CD-009",
}


class CalibrationTests(sc.StationCase):
    """E04-5: id assignment, house calibration, alignment, the rendered section and the queues.

    perf-queue is Sev-A/B-gated and carries the station's layout contract (4t = solver-side / FFI
    boundary, 2x2 = collective); td-queue carries every TD with its yardstick section; the E9/E7/E5/
    E10 handoff blocks live in handoffs.json. The Part-I item-8 entry calibrates to B (A-risk) on
    CD-010, the gate blind spot to C on CD-011 and the lint tables to C on CD-009.
    """

    SLUG = "solver-comm"

    @classmethod
    def setUpClass(cls) -> None:
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.assigned = cls.cal["assigned"]
        cls.by_ref = {r["candidateRef"]: r for r in cls.assigned}
        cls.verdicts = sc.load_json(cls.station_dir() / "verdicts.json")
        cls.perf = sc.load_json(cls.station_dir() / "perf-queue.json")
        cls.td = sc.load_json(cls.station_dir() / "td-queue.json")
        cls.handoffs = sc.load_json(cls.station_dir() / "handoffs.json")
        cls.register = "\n".join(backlog_parse.read_register(sc.BACKLOG))
        scaffold = "## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — solver-comm"
        cls.section = cls.register.split(scaffold, 1)[1].split("\n## ", 1)[0]
        cls.prior = cls.register.replace(cls.section, "")

    def test_envelope_and_baseline(self) -> None:
        self.assertEqual(self.cal["station"], "solver-comm")
        self.assertEqual(self.cal["baseline"], self.baseline())
        self.assertIn(STATION_HEADING + self.baseline()[:8] + ")", self.section)
        self.assertEqual(self.cal["conflicts"], [])

    def test_one_assigned_row_per_confirmed_verdict_and_none_otherwise(self) -> None:
        confirmed = {
            ref
            for ref, e in self.verdicts["verdicts"].items()
            if e["disposition"] == "defended"
            and e["defender"]["verdict"] == "confirmed"
        }
        self.assertEqual(set(self.by_ref), confirmed)
        for row in self.assigned:
            claim, title = normalised(row["survivingClaim"]), normalised(row["title"])
            self.assertTrue(claim and claim != title and title not in claim, row["id"])
        self.assertEqual(
            {d["candidateRef"] for d in self.cal["dupOf"]},
            {"architecture-02", "over-engineering-02"},
        )
        self.assertTrue(all(d["assignedId"] is None for d in self.cal["dupOf"]))
        self.assertEqual(
            [c["candidateRef"] for c in self.cal["cleared"]], ["performance-01"]
        )

    def test_ids_well_formed_in_range_and_class_matches_lens(self) -> None:
        for row in self.assigned:
            m = ID_RE.match(row["id"])
            self.assertIsNotNone(m, row["id"])
            assert m is not None
            cls, num = m.group(1), int(m.group(2))
            self.assertEqual(cls, row["class"])
            self.assertEqual(
                cls,
                {
                    "architecture": "CD",
                    "performance": "PD",
                    "over-engineering": "OD",
                    "test-bloat": "TD",
                }[row["lens"]],
            )
            self.assertGreaterEqual(num, ID_FLOORS[cls])

    def test_ids_contiguous_from_the_runtime_floor_per_class(self) -> None:
        floors = self.cal["idFloorsSeen"]
        for cls in ID_FLOORS:
            nums = sorted(
                int(r["id"].split("-")[1]) for r in self.assigned if r["class"] == cls
            )
            self.assertEqual(
                nums, list(range(floors[cls], floors[cls] + len(nums))), cls
            )
            prior_max = max(
                [int(n) for n in re.findall(rf"\b{cls}-(\d{{3}})\b", self.prior)]
                + [ID_FLOORS[cls] - 1]
            )
            self.assertEqual(
                floors[cls],
                prior_max + 1,
                f"{cls}: floor is not the register's next free",
            )

    def test_ids_unique_across_the_whole_register(self) -> None:
        headings = ENTRY_HEADING.findall(self.register)
        dup = {h for h in headings if headings.count(h) > 1}
        self.assertEqual(dup, set(), f"duplicate entry headings: {sorted(dup)}")
        for row in self.assigned:
            self.assertIsNone(
                re.search(rf"\b{row['id']}\b", self.prior),
                f"{row['id']} pre-exists in the register",
            )
            self.assertEqual(
                self.section.count(f"**{row['id']} · Sev {row['severity']} · "),
                1,
                row["id"],
            )

    def test_severity_alignment_and_the_three_precedents(self) -> None:
        for row in self.assigned:
            self.assertIn(row["severity"], SEVERITIES, row["id"])
            self.assertIn(row["alignmentHint"], ALIGN, row["id"])
            self.assertIn(row["effort"], ("S", "M", "L"))
            self.assertIn(row["confidence"], ("high", "med", "low"))
        self.assertEqual(self.by_ref["architecture-06"]["severity"], "B (A-risk)")
        self.assertEqual(self.by_ref["architecture-06"]["alignmentHint"], "advances-1")
        self.assertEqual(self.by_ref["architecture-08"]["severity"], "C")
        self.assertEqual(self.by_ref["architecture-01"]["severity"], "C")
        self.assertEqual(self.by_ref["architecture-01"]["category"], "duplication")
        for ref, precedent in PRECEDENTS.items():
            self.assertIn(precedent, self.by_ref[ref]["calibrationBasis"], ref)
            block = self.section.split(f"**{self.by_ref[ref]['id']} · ", 1)[1].split(
                "\n**", 1
            )[0]
            self.assertIn(precedent, block)
        for site in ("freeze.rs", "trait_def.rs", "backends/profiled.rs"):
            self.assertIn(
                site,
                self.section.split("#### Part-I cross-references", 1)[1].split(
                    "\n#### ", 1
                )[0],
            )

    def test_downgrades_record_the_reviewer_rating(self) -> None:
        rank = {"A": 3, "B (A-risk)": 2.5, "B": 2, "C": 1}
        for row in self.assigned:
            downgraded = rank[row["severity"]] < rank[row["reviewerRating"]]
            block = self.section.split(f"**{row['id']} · ", 1)[1].split("\n**", 1)[0]
            if downgraded:
                self.assertTrue(row["downgradeReason"], row["id"])
                self.assertIn(
                    f"- **Reviewer rating:** {row['reviewerRating']} — recalibrated to {row['severity']}",
                    block,
                )
            else:
                self.assertIsNone(row["downgradeReason"], row["id"])
                self.assertNotIn("**Reviewer rating:**", block)
        self.assertEqual(self.by_ref["architecture-01"]["reviewerRating"], "B")

    def test_fields_check_exits_zero_over_the_section(self) -> None:
        for tool in ("fields-check.py", "check-anchors.py", "check-reraise.py"):
            result = subprocess.run(
                [sys.executable, str(sc.TOOLS / tool), "solver-comm"],
                capture_output=True,
                text=True,
                cwd=sc.REPO,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, f"{tool}: {result.stdout}{result.stderr}"
            )

    def test_every_entry_carries_alignment_with_roadmap_citation_and_the_baseline(
        self,
    ) -> None:
        for row in self.assigned:
            block = self.section.split(f"**{row['id']} · ", 1)[1].split("\n**", 1)[0]
            m = re.search(
                r"^- \*\*Alignment:\*\* (\S+) \(provisional; Epic 9 adjudicates .*beyond-sddp-generalization\.md",
                block,
                re.M,
            )
            self.assertIsNotNone(m, row["id"])
            assert m is not None
            self.assertEqual(m.group(1), row["alignmentHint"])
            self.assertIn(f"- **Baseline:** `{self.baseline()}`", block)
            self.assertIn(
                "plans/generalizing/beyond-sddp-generalization.md",
                row["alignmentCites"] + " " + block,
            )

    def test_l0_purity_held_no_row_and_informational_is_neutral_with_iii6(self) -> None:
        self.assertFalse(
            [r for r in self.assigned if r["alignmentHint"] == "conflicts"]
        )
        self.assertTrue(all(r["actionable"] for r in self.assigned))
        info = self.cal["informational"]
        self.assertEqual([i["attackerRef"] for i in info], ["SC-ARCH-009"])
        self.assertEqual(info[0]["alignmentHint"], "neutral")
        self.assertIn("III.6", info[0]["roadmapRef"])
        self.assertEqual(
            info[0]["heldVariant"]["conditions"], ["one-consumer-abstraction"]
        )
        self.assertTrue(info[0]["heldVariant"]["alternative"])
        block = self.section.split("#### Informational", 1)[1].split("\n#### ", 1)[0]
        self.assertIn("III.6", block)
        self.assertIn("one-consumer-abstraction", block)
        self.assertRegex(block, r"\*\*Alignment:\*\* neutral")

    def test_test_bloat_fix_shapes_hoist_into_test_support_and_probes_are_positives(
        self,
    ) -> None:
        self.assertNotRegex(
            self.section, r"(?i)\bnew (?:fixture|test-support|helper)s? crate\b"
        )
        tb03 = self.section.split(f"**{self.by_ref['test-bloat-03']['id']} · ", 1)[
            1
        ].split("\n**", 1)[0]
        self.assertIn("test_support", tb03)
        self.assertIn("crates/cobre-solver/src/lib.rs:164", tb03)
        positives = self.section.split("#### Positives", 1)[1].split("\n#### ", 1)[0]
        for probe in ("_q1_sign_convention_probe.rs", "_clp_sign_convention_probe.rs"):
            self.assertIn(probe, positives)
            self.assertNotRegex(
                self.section,
                rf"^\*\*TD-\d{{3}}[^\n]*\n[^\n]*{re.escape(probe)}",
                msg=probe,
            )

    def test_perf_queue_only_sev_ab_pd_with_layout_call_sites_and_no_number(
        self,
    ) -> None:
        ab = {
            r["id"]
            for r in self.assigned
            if r["class"] == "PD" and r["severity"][0] in "AB"
        }
        self.assertEqual({q["id"] for q in self.perf["queue"]}, ab)
        self.assertTrue(ab)
        for q in self.perf["queue"]:
            self.assertIn(q["layout"], LAYOUTS, q["id"])
            self.assertEqual(
                q["layout"], self.by_ref[q["candidateRef"]]["measurementLayout"]
            )
            self.assertEqual(q["status"], "UNMEASURED")
            self.assertIsNone(q["measured"])
            self.assertTrue(q["exercisingCallSites"], q["id"])
            self.assertTrue(q["byteNeutralFixShape"])
            self.assertIsNone(
                TIMING_NUMBER.search(q["claim"] + " " + q["byteNeutralFixShape"]),
                q["id"],
            )
            # station contract: solver-side / FFI-boundary claims measure at 4t, collectives at 2x2
            self.assertEqual(
                q["layout"], "2x2" if "cobre-comm" in q["anchor"] else "4t", q["id"]
            )
            block = self.section.split(f"**{q['id']} · ", 1)[1].split("\n**", 1)[0]
            self.assertIn(
                f"- **Measurement:** UNMEASURED — layout `{q['layout']}`", block
            )
            self.assertIn("- **Queued to:** performance-sweep", block)
        self.assertEqual(len(self.handoffs["E10"]["block"]), len(self.perf["queue"]))

    def test_td_queue_carries_every_td_with_its_yardstick(self) -> None:
        tds = {r["id"] for r in self.assigned if r["class"] == "TD"}
        self.assertEqual({q["id"] for q in self.td["queue"]}, tds)
        self.assertEqual(self.td["targetStation"], "test-corpus (Epic 8)")
        for q in self.td["queue"]:
            self.assertRegex(q["yardstick"], r"§\d")
            self.assertTrue(q["anchors"] and q["binaries"], q["id"])
            block = self.section.split(f"**{q['id']} · ", 1)[1].split("\n**", 1)[0]
            self.assertIn("- **Yardstick:** docs/design/testing-architecture.md", block)
            self.assertIn("test-corpus", block)

    def test_handoff_blocks_e9_e7_e5_e10(self) -> None:
        e9 = self.handoffs["E9"]["block"]
        self.assertEqual(e9["idAssigned"], self.by_ref["architecture-06"]["id"])
        self.assertEqual(
            set(e9["perFieldDisposition"]),
            {"n_state", "n_transfer", "n_dual_relevant", "n_hydro", "max_par_order"},
        )
        self.assertEqual(set(e9["perFieldDisposition"].values()), {"retire"})
        self.assertEqual(len(e9["propagationSites"]), 3)
        self.assertEqual(e9["proposedAlignment"], "advances-1")
        self.assertEqual(
            {x["id"] for x in e9["entries"]},
            {
                self.by_ref[k]["id"]
                for k in ("architecture-06", "test-bloat-03", "test-bloat-06")
            },
        )
        e7 = self.handoffs["E7"]["block"]
        self.assertEqual(e7["idAssigned"], self.by_ref["architecture-08"]["id"])
        self.assertEqual(e7["evadingIdentifier"], "cut_nz_per_col")
        self.assertEqual(e7["patternEvaded"], r"\bcut\b")
        self.assertFalse(e7["changeProposed"])
        self.assertIsNone(self.handoffs["E5"]["idAssigned"])
        self.assertTrue(all(r["assignedId"] is None for r in self.handoffs["records"]))
        for q in self.handoffs["E10"]["block"]:
            self.assertIn(q["layout"], LAYOUTS)
            self.assertEqual(q["status"], "UNMEASURED")
        handoffs = self.section.split("#### Handoff queues", 1)[1].split("\n#### ", 1)[
            0
        ]
        for tag in ("E9", "E7", "E5", "E10"):
            self.assertIn(f"**{tag} — ", handoffs)

    def test_section_skeleton_cleared_and_owner_gate_empty(self) -> None:
        pos = [self.section.find(h) for h in SECTION_SKELETON]
        self.assertTrue(
            all(p >= 0 for p in pos),
            [h for h, p in zip(SECTION_SKELETON, pos) if p < 0],
        )
        self.assertEqual(pos, sorted(pos), "lens and report headings out of order")
        cleared = self.section.split("#### ↩︎ Cleared", 1)[1].split("\n#### ", 1)[0]
        self.assertIn("Shared-memory communicator trait hierarchy", cleared)
        self.assertIn("Superseded cut-sync public methods", cleared)
        self.assertIn("performance-01", cleared)
        for sym in SHARED_MEMORY:
            self.assertNotRegex(
                self.section,
                rf"^\*\*(?:CD|PD|OD|TD)-\d{{3}}[^\n]*\n[^\n]*\b{sym}\b",
                msg=sym,
            )
        for sym in CUT_SYNC_SYMBOLS:
            self.assertNotRegex(
                self.section,
                rf"^\*\*(?:CD|PD|OD|TD)-\d{{3}}[^\n]*\n[^\n]*\b{sym}\b",
                msg=sym,
            )
        gate = self.section.split("#### Owner gate — decisions", 1)[1]
        self.assertIn("_(filled by the gate ticket)_", gate)
        self.assertNotIn("**Decision:", gate)


GENERIC_CHECKS = (
    "check-anchors",
    "check-reraise",
    "fields-check",
    "register",
    "inventory-set-equality",
    "infra-genericity",
    "read-only-workspace",
)
STATION_CHECKS = (
    "sanctioned-polarity",
    "cut-sync-anchors",
    "handoff-shape",
    "blind-spot",
    "read-only-workspace",
)


class SectionVerifyTests(sc.StationCase):
    """Executable proof of the station verification (E04-6).

    verify-station.sh runs this module, so nothing here may invoke it (recursion); the shared
    verifier's seven checks are re-asserted directly and its rendered verification.md is read.
    The station-specific verifier is invoked with --no-append so the test never rewrites the
    committed report.
    """

    SLUG = "solver-comm"

    @classmethod
    def setUpClass(cls) -> None:
        report = cls.station_dir() / "verification.md"
        if not report.exists():
            # verify-station.sh runs this module BEFORE rendering the report, so the very
            # first run has nothing to read; the next run (on the committed report) asserts it.
            raise unittest.SkipTest("verification.md not rendered yet (bootstrap run)")
        cls.report = report.read_text(encoding="utf-8")

    def test_three_harness_checkers_exit_zero_over_the_station_slug(self) -> None:
        for tool in ("check-anchors.py", "check-reraise.py", "fields-check.py"):
            result = subprocess.run(
                [sys.executable, str(sc.TOOLS / tool), self.SLUG],
                capture_output=True,
                text=True,
                cwd=sc.REPO,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, f"{tool}: {result.stdout}{result.stderr}"
            )

    def test_no_tracked_file_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])

    def test_inventory_module_set_equals_the_tree_at_the_baseline(self) -> None:
        listed = {
            f["path"] for f in sc.load_json(self.artifact("inventory.json"))["files"]
        }
        tree = self.tree()
        actual = {p for root in SRC_ROOTS for p in tree.rs_files(root)}
        self.assertEqual(
            sorted(listed - actual), [], "listed but absent at the baseline"
        )
        self.assertEqual(
            sorted(actual - listed), [], "present at the baseline but unlisted"
        )
        self.assertEqual(len(actual), 30)

    def test_verify_handoffs_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, str(self.artifact("verify-handoffs.py")), "--no-append"],
            capture_output=True,
            text=True,
            cwd=sc.REPO,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS (5/5 station-specific checks)", result.stdout)
        for check in STATION_CHECKS:
            self.assertRegex(result.stdout, rf"\| {check} \| .* \| 0 \| PASS \|")

    def test_verification_report_carries_both_tables_all_passing(self) -> None:
        self.assertIn(f"Station baseline: `{self.baseline()}`", self.report)
        for i, check in enumerate(GENERIC_CHECKS, 1):
            self.assertRegex(
                self.report, rf"\| {i} \| {check} \| `[^`]+` \| 0 \| PASS \|"
            )
        # The "Test suite: …" line is written AFTER this module runs and reflects this very
        # run, so asserting it here would be circular; only the check rows are asserted.
        self.assertIn("## Station-specific checks — solver-comm", self.report)
        for check in STATION_CHECKS:
            self.assertRegex(self.report, rf"\| {check} \| .* \| 0 \| PASS \|")
        self.assertNotIn("| FAIL |", self.report)
        self.assertIn("carried-in:", self.report)

    def test_blind_spot_evidence_pair_holds(self) -> None:
        gate = subprocess.run(
            ["bash", "scripts/ci/check-infra-genericity.sh"],
            capture_output=True,
            text=True,
            cwd=sc.REPO,
            check=False,
        )
        self.assertEqual(
            gate.returncode,
            0,
            "the genericity gate now reports a violation; the finding's premise broke",
        )
        freeze = self.tree().read_text("crates/cobre-solver/src/freeze.rs").splitlines()
        cfg_test = next(
            i + 1 for i, text in enumerate(freeze) if text.strip() == "#[cfg(test)]"
        )
        hits = [
            i + 1
            for i, text in enumerate(freeze)
            if "cut_nz_per_col" in text and i + 1 < cfg_test
        ]
        self.assertEqual(cfg_test, 244)
        self.assertEqual(
            hits,
            [22, 137, 138, 141, 188],
            "cut_nz_per_col production hits moved or vanished",
        )


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_file_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])


if __name__ == "__main__":
    unittest.main()
