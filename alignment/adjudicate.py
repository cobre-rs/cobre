#!/usr/bin/env python3
"""Adjudicate the Alignment field of every register entry against the generalization roadmap.

The universe is the register itself: every `**<ID> ·` entry in the seven dated station
sections of BACKLOG.md, one ledger row each. Hints come from both legal shapes — a
station's alignment-queue.json where it emitted one, and calibration.json's assigned
rows (alignmentHint, alignmentCites, conflicts.alternative) for every station; the
calibration value wins when the two disagree and the disagreement is logged. Decisions
are mechanical where they can be: the Part IV.1 guardrails run first (a violation is
`conflicts` and is held), then the Phase 0a / 0b / 1 deliverable tables; a row whose
station hint and machine verdict disagree is settled by a hand-written decision in
HAND_DECISIONS, never by a generated sentence. The retag rewrites only the Alignment
bullet of the rows whose decided value differs from the hint and proves it against a
pre-image snapshot; the rendered section carries the Part-I table, the lp/ share, the
ledger table and the conflicts block.

Subcommands: build | decide | docket | retag | render | verify | all [--register PATH]
             append --section NAME   (late-entry rule: adjudicate a new section's entries
                                      with the same rubric and add them to the ledger)
Exit: 0 clean, 1 a self-check failed, 2 register or a station artifact unreadable.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve()
ALIGN = HERE.parent
AUDIT = HERE.parents[1]
TOOLS = AUDIT / "tools"
sys.path.insert(0, str(TOOLS))

from lib import backlog_parse as bp  # noqa: E402

ROOT = AUDIT.parents[1]
REGISTER = AUDIT / "BACKLOG.md"
STATIONS_DIR = AUDIT / "stations"
ROADMAP_REL = "plans/generalizing/beyond-sddp-generalization.md"
LEDGER = ALIGN / "alignment-ledger.json"
DOCKET = ALIGN / "conflicts-docket.md"
PRE_IMAGE = ALIGN / "backlog.pre-retag.md"
PART_I = ALIGN / "part-i-dispositions.json"
PART_I_FRAGMENT = ALIGN / "part-i-disposition-table.md"
LP_CLASSIFICATION = ALIGN / "lp-classification.json"
GATE_SUBSTRATE = STATIONS_DIR / "test-corpus" / "gate-substrate.json"
STATIONS = (
    "core-io",
    "stochastic",
    "solver-comm",
    "sddp",
    "cli-python",
    "build-ci",
    "test-corpus",
)
SECTION_TAIL = "generalization-alignment"
VOCAB = ("advances-0a", "advances-0b", "advances-1", "neutral", "conflicts")
TODAY = "2026-09-19"
LEDGER_REF = "alignment/alignment-ledger.json"
EXIT_OK, EXIT_FAIL, EXIT_UNREADABLE = 0, 1, 2

# ---------------------------------------------------------------- guardrail layer (Part IV.1)
# The L0/L1 purity test reuses the crate list the CI genericity gate enforces
# (scripts/ci/check-infra-genericity.sh: cobre-core, cobre-io, cobre-solver, cobre-stochastic,
# cobre-comm). Part IV.1 says that list is AMENDED for cobre-model / cobre-network, never
# assumed to extend, so the future L1 crates are named here as destinations to test, not as
# crates the gate already scans.
L0 = ("cobre-core", "cobre-solver", "cobre-comm")
L1 = ("cobre-model", "cobre-stochastic", "cobre-network")
L2 = ("cobre-io",)
ENGINES = ("cobre-sddp", "cobre-direct", "cobre-study")
ENTRY_POINTS = ("cobre-cli", "cobre-python")
GENERICITY_GATE_CRATES = (
    "cobre-core",
    "cobre-io",
    "cobre-solver",
    "cobre-stochastic",
    "cobre-comm",
)
CRATE_RE = re.compile(
    r"cobre[-_](core|solver|comm|model|stochastic|network|io|sddp|direct|study|cli|python)"
)
PARADIGM_RE = re.compile(
    r"(?<![-_])\b(?:sddp|benders|cut[- ]pool|cost-to-go|state[- ]space|policy graph|cvar|risk measure|"
    r"forward pass|backward pass|cut selection|cut projection)\b",
    re.I,
)
PLACEMENT_RE = re.compile(
    r"\b(move|relocate|re-home|rehome|add|introduce|put|land|hoist|promote|belong|home)",
    re.I,
)
SENTENCE_RE = re.compile(r"(?<=[.;])\s+")
SUBSTRATE_RE = re.compile(
    r"\b(parity_hash\w*|parity\.rs|parity_baselines\w*|invariance-shuffle(?:\.yml)?|mpi_wire(?:\.rs)?)\b",
    re.I,
)
DELETION_RE = re.compile(r"\b(delet\w*|remov\w*|drop\w*|retir\w*|prun\w*)\b", re.I)
GUARDRAILS = {
    "engine-concept-in-L0/L1": "Part IV.1 — L0/L1 name no engine or problem",
    "engine-to-engine-dependency": "Part IV.1 — base engines depend on the shared kernels, never on each other",
    "Engine-enum-below-L4": "Part IV.1 / Part IV.4 — the Engine enum lives only at L4",
    "one-consumer-abstraction": "Part V.0 — pull, don't push: no abstraction with a single consumer",
    "output-orchestration-not-at-L2": "Part IV.1 (L2 cobre-io owns shared output orchestration) / Part V.1 (Phase 0a deliverable)",
    "phase-0a-gate-substrate-removal": "Part V.1 — the Phase 0a gate: SDDP output bit-for-bit unchanged from the pinned baseline",
}
# The two conflict classes this evaluation expects, each with the roadmap-consistent
# alternative a rule-caught row is given when its station left `alternative` null, and the
# cost the owner pays to override it. Written once here; the docket renders from it.
CLASS_TEMPLATES: dict[str, dict[str, str]] = {
    "output-orchestration-not-at-L2": {
        "shape": "A fix-shape hoisting shared writers into a cobre-cli-local helper (the Wave-5 CD-025/CD-029 restatement).",
        "alternative": "One cobre-io entry point taking the resolved output set, called by both "
        "`crates/cobre-cli/src/commands/run/outputs.rs` and `crates/cobre-python/src/run.rs`; the CLI keeps only argument "
        "resolution and path assembly — the Phase-0a output-orchestration deliverable the hint was reaching for.",
        "overrideCost": "A helper in cobre-cli is unreachable from cobre-python, so the hand-mirrored outputs.rs-to-run.rs pair "
        "the fix claims to remove is re-created at L4 and Python parity stays a hand-kept rule.",
    },
    "phase-0a-gate-substrate-removal": {
        "shape": "A test-corpus TD entry deleting a golden or determinism harness listed in stations/test-corpus/gate-substrate.json "
        "(the `parity_hash_highs` / `parity_hash_clp` mods of `crates/cobre-sddp/tests/parity.rs`, "
        "`crates/cobre-sddp/tests/common/parity_hash.rs`, `.github/workflows/invariance-shuffle.yml`).",
        "alternative": "Retire the duplicated harness while keeping the golden: the parity_hash_* mods, common/parity_hash.rs and "
        "invariance-shuffle.yml stay; only the second copy of whatever they duplicate goes.",
        "overrideCost": "The Phase-0a gate states SDDP output bit-for-bit unchanged from the pinned baseline, and the substrate is how "
        "that is proved; deleting it leaves the gate unprovable.",
    },
}


def crates_named(text: str) -> set[str]:
    return {m.group(0).replace("_", "-").lower() for m in CRATE_RE.finditer(text)}


def guardrail_violations(
    row: dict[str, Any],
    fix_shape: str,
    consumers: int | None = None,
    substrate: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Named guardrail hits over a recorded Fix-shape bullet; each names the sentence that fired."""
    hits: list[dict[str, str]] = []
    sentences = SENTENCE_RE.split(fix_shape)
    homes = crates_named(fix_shape)
    for s in sentences:
        if (
            any(c in crates_named(s) for c in L0 + L1)
            and PARADIGM_RE.search(s)
            and PLACEMENT_RE.search(s)
        ):
            hits.append({"id": "engine-concept-in-L0/L1", "sentence": s.strip()})
            break
    if sum(1 for e in ENGINES if e in homes) > 1 and re.search(
        r"depend|import|call", fix_shape, re.I
    ):
        hits.append({"id": "engine-to-engine-dependency", "sentence": fix_shape[:200]})
    if re.search(
        r"\bEngine\b[^.]{0,40}\benum\b|\benum\b[^.]{0,40}\bEngine\b", fix_shape
    ) and not any(c in homes for c in ENTRY_POINTS):
        hits.append({"id": "Engine-enum-below-L4", "sentence": fix_shape[:200]})
    if consumers is not None and consumers < 2 and row.get("class") != "OD":
        hits.append(
            {
                "id": "one-consumer-abstraction",
                "sentence": f"{consumers} consumer(s) recorded",
            }
        )
    for s in sentences:
        named = crates_named(s)
        if (
            re.search(r"\boutput", s, re.I)
            and re.search(r"orchestrat|writer|mirror", s, re.I)
            and any(c in named for c in ENTRY_POINTS)
            and "cobre-io" not in named
            and re.search(r"hoist|move|helper|local|own", s, re.I)
        ):
            hits.append({"id": "output-orchestration-not-at-L2", "sentence": s.strip()})
            break
    tokens = SUBSTRATE_RE
    for s in sentences:
        if (
            tokens.search(s) or any(t and t.lower() in s.lower() for t in substrate)
        ) and DELETION_RE.search(s):
            hits.append(
                {"id": "phase-0a-gate-substrate-removal", "sentence": s.strip()}
            )
            break
    return hits


# ---------------------------------------------------------------- phase layer (Part V.1, V.2, IV.5)
# Keys are matched against the entry's recorded Fix-shape bullet; the first match wins and
# writes both `decided` and `cites`. A fix-shape that IS the deliverable matches; one that
# merely mentions its objects does not, which is why the patterns pair the object with the
# deliverable's action.
PHASE_TABLES: dict[str, dict[str, tuple[str, str, str]]] = {
    "advances-0a": {
        "engine-seam": (
            r"\bEngine\b[^.]{0,80}\b(enum|dispatch)|dispatch\w* on (the|an) engine|engine[- ]seam|StudySetup-typed|(run-|phase[- ])plan[^.]{0,60}(engine|L4)",
            "Engine enum + dispatch at L4 replacing the cobre_sddp::StudySetup-typed run pipeline",
            "Part IV.4; Part V.1 (Phase 0a deliverables)",
        ),
        "study-config": (
            r"admission gate|Layer-2 config gate|study block|study/config|(single|one) config projection|solver-profile scoping|per-engine (section|solver)",
            "study/config block + admission gate: per-engine section admissibility, per-engine solver-profile scoping",
            "Part V.1 (Phase 0a deliverables)",
        ),
        "output-orchestration": (
            r"(shared )?output[- ]orchestration|one cobre-io entry point[^.]{0,80}(outputs\.rs|run\.rs|both front ends)|write_results[^.]{0,40}orchestrat|family names it writes as public data",
            "shared output-orchestration entry point in cobre-io, both cobre-cli and cobre-python wired through it",
            "Part IV.1 (L2 cobre-io); Part V.1 (Phase 0a deliverables)",
        ),
        "mpi-d14": (
            r"\bD14\b|rank-0-executes|ranks_participated",
            "rank-0-executes semantics; non-roots skip setup, ranks_participated is 1",
            "Part IV.4 (D14); Part V.1",
        ),
        "engine-tagged-setup": (
            r"engine-tagged|skip\w* (the )?stochastic reconstruction",
            "setup stages tagged by engine so a non-consuming engine skips stochastic reconstruction",
            "Part IV.4; Part V.1",
        ),
    },
    "advances-0b": {
        "model-carve": (
            r"cobre-model|BuildProblem|ProblemTemplate|VarDomain|kernel carve|carve[^.]{0,40}lp/|0b carve|carve-out",
            "carve cobre-model (BuildProblem, ProblemTemplate, VarDomain) out of the engine-neutral share of lp/",
            "Part IV.2; Part V.1 (the D12 split: Phase 0b)",
        ),
    },
    "advances-1": {
        "uncertainty-store": (
            r"Switchable|uncertainty store|(leave|off|out of) (`?System`?|`?Stage`?)[^.]{0,60}(cobre-stochastic|store)|uncertainty representation[^.]{0,40}cobre-stochastic|persistent store handle",
            "InflowModel/LoadModel/NcsModel/CorrelationModel and the external-scenario tables leave System for the cobre-stochastic store",
            "Part IV.5; Part V.2",
        ),
        "stage-config": (
            r"StageRiskConfig|ScenarioSourceConfig|inflow_lags[^.]{0,40}leave|per-node config",
            "StageRiskConfig/ScenarioSourceConfig/inflow_lags leave Stage for an SDDP per-node config",
            "Part IV.5; Part V.2",
        ),
        "training-event": (
            r"training_event\.rs[^.]{0,80}(move|leave|telemetry)|telemetry module",
            "constraints/training_event.rs leaves cobre-core for a telemetry module",
            "Part IV.5",
        ),
        "stage-template": (
            r"(shed|delete|drop|remove)[^.]{0,80}\b(n_state|n_transfer|n_dual_relevant|n_hydro|max_par_order)\b|\b(n_state|n_transfer|n_dual_relevant|n_hydro|max_par_order)\b[^.]{0,80}(shed|delete|drop|remove)|five fields from the L0 container|five shed fields|cost of the shed",
            "StageTemplate sheds n_state/n_transfer/n_dual_relevant/n_hydro/max_par_order",
            "Part IV.5",
        ),
        "checkpoint": (
            r"engine-neutral policy artifact|value-function[- ]artifact|generaliz\w+[^.]{0,60}checkpoint|checkpoint[^.]{0,60}generaliz",
            "policy checkpoint generalized off literal cut records",
            "Part IV.1 (cobre-io row); Part V.2",
        ),
        "case-v2": (
            r"case format v2|case v2|v1 (compat )?shim",
            "case format v2 with a bit-for-bit v1 shim",
            "Part V.2",
        ),
    },
}
NEUTRAL_CITES = "Part IV.1 (layering unaffected); Part V.1/V.2 (no phase deliverable)"
NEUTRAL_RATIONALE = (
    "a real finding whose fix-shape neither advances nor obstructs a phase deliverable; "
    "schedulable on its own merits"
)


def machine_decision(fix_shape: str) -> tuple[str, str | None, str, str]:
    """(decided, key, rationale, cites) from the phase tables; neutral when nothing IS a deliverable."""
    for phase, table in PHASE_TABLES.items():
        for key, (pattern, deliverable, cites) in table.items():
            if re.search(pattern, fix_shape, re.I):
                return (
                    phase,
                    key,
                    f"fix-shape IS the {key} deliverable: {deliverable}",
                    cites,
                )
    return "neutral", None, NEUTRAL_RATIONALE, NEUTRAL_CITES


# ---------------------------------------------------------------- hand decisions
# Every row whose station hint and machine verdict disagree, and every hinted-advances row,
# is settled here by reading the entry. `decided` is final; `rationale` is the written reason;
# `cites` names the Part IV / Part V basis. A row absent from this table keeps its machine
# verdict only when that verdict equals the station hint.
HAND_DECISIONS: dict[str, dict[str, Any]] = {
    # ---- confirmed hints (decided == hint), with the citation the station bullet lacked or abbreviated
    "CD-044": {
        "decided": "advances-1",
        "rationale": "StageLagTransition is uncertainty representation (the PAR lag transition) homed in cobre-core with zero L0 consumers; relocating it to cobre-stochastic is the Part IV.5 move of the uncertainty representation off cobre-core, executed for one type",
        "cites": "Part IV.5 (the uncertainty representation leaves cobre-core for cobre-stochastic); Part V.2",
    },
    "CD-051": {
        "decided": "advances-0a",
        "rationale": "resolving both scenario sources once inside the Layer-2 config gate and reporting into the validation context is the admission-gate deliverable: admission decided where the config is admitted, not by whoever reads it",
        "cites": "Part V.1 (Phase 0a: study/config block + admission gate); Part IV.1 (L2 cobre-io)",
    },
    "CD-059": {
        "decided": "advances-0a",
        "rationale": "the cobre-io-side owner shape for the write_results contract is the shared output-orchestration entry point itself: either write_results orchestrates the full artifact set or stops claiming to, so the CLI/Python hand-mirror has one L2 owner",
        "cites": "Part V.1 (Phase 0a: shared output-orchestration entry point in cobre-io); Part IV.1 (L2)",
    },
    "CD-061": {
        "decided": "advances-0a",
        "rationale": "the recorded Milestone-0a shape — cobre-io keeps the write mechanics while the engine supplies the row type and column list — is the results seam's contract for the output-orchestration entry point, and it refuses the one-consumer generic schema trait",
        "cites": "Part V.1 (Phase 0a: shared output orchestration, III.7 results seam); Part IV.1 (L2); Part V.0 (pull, don't push)",
    },
    "CD-065": {
        "decided": "advances-1",
        "rationale": "the split between a persistent store handle and a transient per-run source selection is the store/RunParams boundary the Phase-1 Switchable<T> uncertainty store formalizes; the seam is drawn where the store will be cut",
        "cites": "Part V.2 (Phase 1: the cobre-stochastic uncertainty store); Part IV.5",
    },
    "CD-070": {
        "decided": "advances-1",
        "rationale": "the fix-shape is the Phase-1 relocation itself: the external-scenario tables leave System for the cobre-stochastic store and the External-ingestion adaptation moves with them, with a bounded interim extraction that introduces no engine noun and no L1-to-engine edge",
        "cites": "Part IV.5 (external-scenario tables leave System); Part V.2 (Phase 1 store)",
    },
    "CD-071": {
        "decided": "advances-1",
        "rationale": "a Phase-1 destination note on the struct that becomes the store: precomputed components as payload, entity order and dimensions as addressing, seeds to the RunParams half; nothing executed here, the carve target is named",
        "cites": "Part V.2 (Phase 1: Switchable<T> store); Part IV.5",
    },
    "CD-079": {
        "decided": "advances-1",
        "rationale": "deleting n_state/n_transfer/n_dual_relevant/n_hydro/max_par_order from the L0 container is the StageTemplate shed Part IV.5 names verbatim; the fixture census sizes the same deliverable",
        "cites": "Part IV.5 (StageTemplate sheds the five fields, :1471); Part IV.1 (cobre-solver purified + capabilities, :1372)",
    },
    "TD-040": {
        "decided": "advances-1",
        "rationale": "the fixture-contract test re-encodes the five shed fields, so retiring or relocating it is collateral executed with the StageTemplate shed and sized into the same Phase-1 estimate",
        "cites": "Part IV.5 (StageTemplate shed collateral)",
    },
    "CD-088": {
        "decided": "advances-0a",
        "rationale": "the sddp owner gate directed the write-payload projection to land with Phase 0a's shared output orchestration in cobre-io rather than as an intra-engine move, so the fix-shape is absorbed by that deliverable",
        "cites": "Part V.1 (Phase 0a: shared output orchestration in cobre-io); Part IV.1 (L2)",
    },
    "CD-089": {
        "decided": "advances-0a",
        "rationale": "one phase-plan owner at L4 consumed by both front ends, with the Engine enum and dispatch staying in cobre-cli/cobre-python, is the engine-seam deliverable's phase plan",
        "cites": "Part IV.4 (the Engine enum lives only at L4); Part V.1 (Phase 0a: Engine enum + dispatch)",
    },
    "CD-091": {
        "decided": "advances-0a",
        "rationale": "a LoadError kind map owned by cobre-io and called by both front ends is shared front-end contract data at L2 for the validate path the Phase-0a seam must cover; the owner gate confirmed L2 as the home",
        "cites": "Part V.1 (Phase 0a: the seam covers the full command surface, validate included); Part IV.1 (L2 owns shared front-end contract data)",
    },
    "CD-095": {
        "decided": "advances-0a",
        "rationale": "cobre-io exposing the family names it writes as public data, with both readers iterating the declaration, is the shared results seam's read side at L2 — the output-orchestration deliverable's contract, not a hand-kept copy",
        "cites": "Part V.1 (Phase 0a: shared output orchestration, III.7); Part IV.1 (L2)",
    },
    # ---- retags: decided != hint
    "CD-045": {
        "decided": "neutral",
        "rationale": "sorting or validating the seven model tables in the L0 builder hardens an invariant the tables already carry; it moves nothing off System and is not a Phase-1 deliverable — the invariant travels with the tables when they relocate",
        "cites": "Part IV.1 (layering unaffected); Part IV.5 (the tables relocate later; this is not that move)",
    },
    "OD-016": {
        "decided": "neutral",
        "rationale": "deleting a dead aggregate loader and its second assembly path in cobre-io is speculative-generality removal; it is not case format v2 nor any other Phase-1 deliverable",
        "cites": "Part IV.1 (layering unaffected); Part V.2 (no phase deliverable)",
    },
    "OD-020": {
        "decided": "neutral",
        "rationale": "dropping an unread `&Config` parameter from write_dictionaries is signature hygiene inside cobre-io; it neither creates nor advances the shared output-orchestration entry point",
        "cites": "Part IV.1 (layering unaffected); Part V.1 (no phase deliverable)",
    },
    "TD-001": {
        "decided": "neutral",
        "rationale": "deleting tautological derived-trait and field-echo tests on training_event.rs is test hygiene; the Phase-1 deliverable is the module's move out of cobre-core, which this neither performs nor prepares",
        "cites": "Part IV.1 (layering unaffected); Part IV.5 (the training_event move is a different change)",
    },
    "CD-082": {
        "decided": "neutral",
        "rationale": "making the scalar-parameter table a constructor input of StudySetup and failing loud on a miss is engine-internal construction hardening at L3; the Phase-0a admission gate is the L2/L4 study-config deliverable, which this helps but is not",
        "cites": "Part IV.1 (L3 engine internals); Part V.1 (the admission gate is a study/config deliverable, not a StudySetup invariant)",
    },
    "TD-045": {
        "decided": "neutral",
        "rationale": "replacing eleven test-fixture clones with the shared test_support constructors is coverage-neutral consolidation; it does not shed a StageTemplate field or relocate anything the Phase-1 purification names",
        "cites": "Part IV.1 (layering unaffected); Part IV.5 (no shed executed)",
    },
    "CD-085": {
        "decided": "neutral",
        "rationale": "adding two typed commitment-hold resolvers beside their siblings completes state_space.rs's accessor family; the lp/ classification puts state_space.rs in sddp-geometry, so the accessor does not move with the cobre-model carve and the 0b hint does not attach",
        "cites": "Part IV.2 (the carve takes only the engine-neutral substrate; state_space.rs is geometry); Part IV.1 (L3)",
    },
    "OD-037": {
        "decided": "neutral",
        "rationale": "deleting the dead Col and Row newtypes removes two consumer-less items from the carve's inventory; it is speculative-generality removal, not the cobre-model carve",
        "cites": "Part IV.1 (layering unaffected); Part IV.2 (the carve is a different change)",
    },
    "OD-038": {
        "decided": "neutral",
        "rationale": "deleting the dead FphaRowRange type and narrowing two module docs is speculative-generality removal inside the engine; the cobre-model carve is not advanced by it",
        "cites": "Part IV.1 (layering unaffected); Part IV.2 (the carve is a different change)",
    },
    "CD-087": {
        "decided": "neutral",
        "rationale": "retiring the crate-root module aliases as an internal import path is import hygiene that de-risks the carve by naming the owning cluster; it moves no item and is not the carve",
        "cites": "Part IV.1 (layering unaffected); Part IV.2 (the carve is a different change)",
    },
    "CD-092": {
        "decided": "neutral",
        "rationale": "hoisting the describe_prep_error join into cobre-sddp's validate_phases.rs moves engine-specific rendering into the engine; consistent with the layering, but not a Phase-0a seam deliverable",
        "cites": "Part IV.1 (L3 owns its rendering; layering unaffected); Part V.1 (no phase deliverable)",
    },
    "CD-099": {
        "decided": "advances-0b",
        "rationale": "making the genericity crate list single-owned is Part IV.1's own action item for the moment an L1 crate lands; cobre-model lands in Phase 0b under the D12 split, so the amendment belongs to 0b, not to the Phase-1 data-model break the station tagged",
        "cites": "Part IV.1 (the closed five-crate enumeration is amended, not assumed to extend, :1303); Part V.1 (Phase 0b carves cobre-model)",
    },
    # ---- rule hits reviewed and settled
    "OD-030": {
        "decided": "neutral",
        "rationale": "the substrate predicate fired because the fix-shape removes a now-dead SweepDirection argument at one call site inside cobre-sddp/tests/parity.rs; the golden parity_hash_* roster, its hashes and the harness are untouched, so the Phase-0a gate substrate is not removed",
        "cites": "Part IV.1 (layering unaffected); Part V.1 (gate substrate intact)",
    },
    "TD-028": {
        "decided": "neutral",
        "rationale": "hoisting Stage and ScenarioSourceConfig test builders into cobre-core's test-support surface is fixture consolidation; the Phase-1 deliverable moves ScenarioSourceConfig off Stage, which a test builder for it does not do",
        "cites": "Part IV.1 (layering unaffected); Part IV.5 (no relocation executed)",
    },
    "CD-086": {
        "decided": "neutral",
        "rationale": "rejecting the enumerated-plus-DCS combination in the study admission gate adds one SDDP-internal rule to the gate; the Phase-0a deliverable is the gate's per-engine admissibility and solver-profile scoping, which this uses but does not build, and road (b) is a driver-side fold",
        "cites": "Part IV.1 (L3 engine internals); Part V.1 (the admission gate's 0a deliverable is per-engine admissibility)",
    },
}

# The Part-I entries the consolidation ticket minted carry their own provisional alignment;
# this pass adjudicates them with the same rubric and writes the decided form when the
# fragment is spliced into the section.
PART_I_DECISIONS: dict[str, dict[str, str]] = {
    "CD-119": {
        "decided": "advances-1",
        "rationale": "the eight stochastic-input fields leaving System for the cobre-stochastic store is the Part IV.5 deliverable verbatim",
        "cites": "Part IV.5; Part V.2",
    },
    "CD-120": {
        "decided": "advances-1",
        "rationale": "StageRiskConfig / ScenarioSourceConfig / inflow_lags leaving Stage is the Part IV.5 deliverable verbatim",
        "cites": "Part IV.5; Part V.2",
    },
    "CD-121": {
        "decided": "neutral",
        "rationale": "HorizonGraph stays in the purified core per Part IV.1; the residual policy vocabulary is settled with the horizon model and no roadmap row names that rename",
        "cites": "Part IV.1 (cobre-core purified keeps HorizonGraph)",
    },
    "CD-122": {
        "decided": "advances-1",
        "rationale": "the anticipated-commitment and defluence seeds following the commitment and routing state is the Part IV.5 InitialConditions deliverable",
        "cites": "Part IV.5; Part V.2",
    },
    "CD-123": {
        "decided": "advances-1",
        "rationale": "training_event.rs moving to a telemetry module the verticals emit into is the Part IV.5 deliverable verbatim",
        "cites": "Part IV.5",
    },
    "CD-124": {
        "decided": "advances-1",
        "rationale": "generalizing the persisted artifact behind an engine-neutral policy artifact is the Part IV.1 cobre-io row's checkpoint deliverable",
        "cites": "Part IV.1 (cobre-io row); Part V.2",
    },
    "CD-125": {
        "decided": "advances-0a",
        "rationale": "one config projection behind the study admission gate, both front ends wired through it, is the Phase-0a study/config deliverable",
        "cites": "Part V.1 (Phase 0a: study/config block + admission gate)",
    },
    "CD-126": {
        "decided": "advances-1",
        "rationale": "StageTemplate shedding the five fields to the kernel/engine side is the Part IV.5 deliverable verbatim",
        "cites": "Part IV.5; Part IV.1 (cobre-solver purified + capabilities)",
    },
    "CD-127": {
        "decided": "advances-0a",
        "rationale": "the Engine enum and dispatch in cobre-cli / cobre-python plus the shared output-orchestration entry point are the Phase-0a deliverables",
        "cites": "Part IV.4; Part V.1 (Phase 0a)",
    },
}


# ---------------------------------------------------------------- universe and hints
def read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"unreadable: {path} ({exc.strerror})") from exc


def register_lines(register: pathlib.Path) -> list[str]:
    return bp.read_register(register)


def station_entries(
    lines: list[str], stations: tuple[str, ...] = STATIONS
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for st in stations:
        section = bp.find_section(lines, st)
        for entry in bp.iter_entries(section):
            if entry.id in seen:
                raise SystemExit(
                    f"duplicate entry id {entry.id}: {seen[entry.id]} and {st}"
                )
            seen[entry.id] = st
            alignment = entry.fields.get("Alignment", "")
            rows.append(
                {
                    "entryId": entry.id,
                    "station": st,
                    "class": entry.id.split("-")[0],
                    "registerLine": entry.lineno,
                    "heading": entry.heading,
                    "fixShape": entry.fields.get("Fix-shape", ""),
                    "registerAlignment": alignment,
                    "registerAlignmentValue": alignment.split("(", 1)[0]
                    .strip()
                    .strip("`"),
                }
            )
    return rows


def queue_hints(path: pathlib.Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Hints from an alignment-queue.json in either of its two shapes: `alignment[]`
    ({id, hint, cites, alternative}) or `rows[]` ({id, alignmentHint, kind, gateDecision})."""
    q = read_json(path)
    out: dict[str, dict[str, Any]] = {}
    if isinstance(q.get("alignment"), list):
        for a in q["alignment"]:
            if a.get("id"):
                out[a["id"]] = {
                    "hint": a.get("hint"),
                    "cites": a.get("cites"),
                    "alternative": a.get("alternative"),
                    "kind": "new-entry",
                }
        return out, "alignment[]"
    for r in q.get("rows", []):
        out[r["id"]] = {
            "hint": r.get("alignmentHint"),
            "cites": None,
            "alternative": None,
            "kind": r.get("kind"),
            "gateDecision": r.get("gateDecision"),
        }
    return out, "rows[]"


def calibration_hints(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for a in read_json(path).get("assigned", []):
        conflicts = a.get("conflicts")
        alt = (
            conflicts.get("alternative")
            if isinstance(conflicts, dict)
            else a.get("alternative")
        )
        out[a["id"]] = {
            "hint": a.get("alignmentHint"),
            "cites": a.get("alignmentCites"),
            "alternative": alt,
            "fixShapeKind": a.get("fixShapeKind"),
        }
    return out


def build_ledger(register: pathlib.Path) -> dict[str, Any]:
    lines = register_lines(register)
    baseline = bp.parse_baseline(lines)
    rows = station_entries(lines)
    by_id = {r["entryId"]: r for r in rows}
    hints: dict[str, tuple[dict[str, Any], str]] = {}
    disagreements: list[dict[str, Any]] = []
    queue_only: list[dict[str, Any]] = []
    calibration_ids: set[str] = set()
    without_queue: list[str] = []
    queue_shapes: dict[str, str] = {}
    for st in STATIONS:
        queue_path = STATIONS_DIR / st / "alignment-queue.json"
        from_queue: dict[str, dict[str, Any]] = {}
        if queue_path.exists():
            from_queue, shape = queue_hints(queue_path)
            queue_shapes[st] = shape
        else:
            without_queue.append(st)
        for eid, cal in calibration_hints(
            STATIONS_DIR / st / "calibration.json"
        ).items():
            calibration_ids.add(eid)
            if eid in from_queue and from_queue[eid]["hint"] != cal["hint"]:
                disagreements.append(
                    {
                        "id": eid,
                        "station": st,
                        "queue": from_queue[eid]["hint"],
                        "calibration": cal["hint"],
                    }
                )
            hints[eid] = (cal, "calibration.json")
        for eid, q in from_queue.items():
            if eid in hints:
                continue
            if eid in by_id:
                hints[eid] = (q, "alignment-queue.json")
            else:
                queue_only.append(
                    {
                        "id": eid,
                        "station": st,
                        "hint": q["hint"],
                        "kind": q.get("kind"),
                        "gateDecision": q.get("gateDecision"),
                    }
                )
    for eid, (h, source) in hints.items():
        if eid in by_id:
            by_id[eid].update(
                hint=h["hint"],
                hintSource=source,
                hintCites=h.get("cites"),
                alternativeFixShape=h.get("alternative"),
                decided=None,
                rationale=None,
                cites=None,
                guardrailViolated=None,
                held=False,
                retagged=False,
                needsHuman=None,
            )
    for r in rows:
        r.setdefault("hint", None)
        r.setdefault("hintSource", None)
        r.setdefault("hintCites", None)
        r.setdefault("alternativeFixShape", None)
        for k in ("decided", "rationale", "cites", "guardrailViolated", "needsHuman"):
            r.setdefault(k, None)
        r.setdefault("held", False)
        r.setdefault("retagged", False)
    coverage = {
        "registerEntryIds": sorted(by_id),
        "registerOnly": sorted(set(by_id) - set(hints)),
        "calibrationOnly": sorted(calibration_ids - set(by_id)),
        "queueOnly": queue_only,
        "hintDisagreements": disagreements,
        "stationsWithoutAlignmentQueue": without_queue,
        "queueShapes": queue_shapes,
        "perStation": {
            st: sum(1 for r in rows if r["station"] == st) for st in STATIONS
        },
    }
    return {
        "artifact": "alignment-ledger",
        "baseline": baseline,
        "adjudicatedAt": TODAY,
        "vocabulary": list(VOCAB),
        "authority": f"{ROADMAP_REL} — Part IV.1 (L0-L4 layering + the three invariants), Part IV.4 (engine-selection seam), Part IV.5 (what leaves cobre-core), Part V.1 (phase 0a/0b deliverables + gate), Part V.2 (phase 1 + the numerically-frozen rule)",
        "precedence": "where this ledger and a station entry's Alignment bullet disagree, the decided value governs; the bullet is retagged in place and the superseded hint is preserved inline",
        "rubric": "guardrails first (a violation is `conflicts` and is held), then the Phase 0a / 0b / 1 deliverable tables (a fix-shape that IS a deliverable earns the tag; one that merely helps does not), else neutral; a row whose hint and machine verdict disagree is settled by a hand-written decision in adjudicate.py HAND_DECISIONS",
        "coverage": coverage,
        "ledger": sorted(rows, key=lambda r: r["registerLine"]),
    }


# ---------------------------------------------------------------- decide pass
def substrate_tokens() -> tuple[str, ...]:
    g = read_json(GATE_SUBSTRATE)
    tokens: set[str] = set()
    for b in g.get("goldenRoster", {}).get("backends", []):
        tokens.add(b["module"])
    for m in g.get("determinismGates", {}).get("modules", []):
        tokens.add(m.get("name", ""))
    tokens.update({"parity_hash", "invariance-shuffle", "parity_baselines"})
    return tuple(t for t in tokens if t)


def decide_row(row: dict[str, Any], substrate: tuple[str, ...]) -> dict[str, Any]:
    fix = row["fixShape"]
    violations = guardrail_violations(row, fix, None, substrate)
    machine, key, rationale, cites = machine_decision(fix)
    if violations:
        machine, key = "conflicts", violations[0]["id"]
        rationale = f"guardrail {violations[0]['id']} fired on: {violations[0]['sentence'][:200]}"
        cites = GUARDRAILS[violations[0]["id"]]
    row["machineDecision"] = machine
    row["machineKey"] = key
    row["guardrailHits"] = violations
    hand = HAND_DECISIONS.get(row["entryId"])
    if hand:
        row.update(
            decided=hand["decided"],
            rationale=hand["rationale"],
            cites=hand["cites"],
            settledBy="hand",
        )
        if hand["decided"] == "conflicts":
            row.update(
                guardrailViolated=hand.get("guardrailViolated"),
                alternativeFixShape=hand.get("alternativeFixShape"),
                needsHuman=hand.get("needsHuman"),
                held=True,
            )
        else:
            row.update(guardrailViolated=None, held=False)
    else:
        row.update(
            decided=machine, rationale=rationale, cites=cites, settledBy="machine"
        )
        if machine == "conflicts":
            template = CLASS_TEMPLATES.get(key or "", {})
            row.update(
                guardrailViolated=key,
                held=True,
                alternativeFixShape=row.get("alternativeFixShape")
                or template.get("alternative"),
                overrideCost=template.get("overrideCost"),
            )
    row["retagged"] = bool(row.get("hint")) and row["hint"] != row["decided"]
    return row


def self_check(ledger: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for r in ledger["ledger"]:
        eid = r["entryId"]
        if r["decided"] not in VOCAB:
            problems.append(f"{eid}: undecided ({r['decided']!r})")
        if r["decided"] == "hint":
            problems.append(f"{eid}: decided is the literal string 'hint'")
        if len(r.get("rationale") or "") < 40:
            problems.append(f"{eid}: rationale shorter than 40 characters")
        c = r.get("cites") or ""
        if "Part IV" not in c and "Part V" not in c:
            problems.append(f"{eid}: no Part IV / Part V citation")
        if r.get("hint") and r["hint"] != r["decided"] and r.get("settledBy") != "hand":
            problems.append(
                f"{eid}: hint {r['hint']} != decided {r['decided']} without a hand-written decision"
            )
        if (
            r.get("machineDecision") == "conflicts"
            and r.get("settledBy") != "hand"
            and r["decided"] != "conflicts"
        ):
            problems.append(f"{eid}: guardrail hit dropped without a hand decision")
        if r["decided"] == "conflicts":
            if not (r.get("alternativeFixShape") or r.get("needsHuman")):
                problems.append(
                    f"{eid}: conflicts without an alternative or a needsHuman marker"
                )
            if not r.get("guardrailViolated"):
                problems.append(f"{eid}: conflicts without a named guardrail")
            if not r.get("held"):
                problems.append(f"{eid}: conflicts not held")
        elif r.get("held"):
            problems.append(f"{eid}: held but not conflicts")
    return problems


def decide_pass(ledger: dict[str, Any]) -> dict[str, Any]:
    substrate = substrate_tokens()
    for row in ledger["ledger"]:
        decide_row(row, substrate)
    ledger["partIEntries"] = []
    for pid, d in PART_I_DECISIONS.items():
        ledger["partIEntries"].append(
            {
                "entryId": pid,
                **d,
                "settledBy": "hand",
                "source": "alignment/part-i-dispositions.json",
            }
        )
    problems = self_check(ledger)
    ledger["selfCheck"] = {"problems": problems, "passed": not problems}
    ledger["held"] = [r["entryId"] for r in ledger["ledger"] if r["held"]]
    ledger["needsHuman"] = [
        {"entryId": r["entryId"], "question": r["needsHuman"]}
        for r in ledger["ledger"]
        if r.get("needsHuman")
    ]
    ledger["retagged"] = [
        {"entryId": r["entryId"], "hint": r["hint"], "decided": r["decided"]}
        for r in ledger["ledger"]
        if r["retagged"]
    ]
    ledger["decidedCounts"] = {
        v: sum(1 for r in ledger["ledger"] if r["decided"] == v) for v in VOCAB
    }
    ledger["guardrailHitsReviewed"] = [
        {
            "entryId": r["entryId"],
            "guardrail": h["id"],
            "sentence": h["sentence"][:200],
            "decided": r["decided"],
        }
        for r in ledger["ledger"]
        for h in r.get("guardrailHits", [])
    ]
    ledger["lateEntryRule"] = (
        "entries recorded after this pass (the perf sweep's measurement blocks and the reconciliation epic's "
        "Wave 4-7 dispositions) are adjudicated by re-running `alignment/adjudicate.py append --section <name>` "
        "over the new section; the rubric is the script, not this one-time reading"
    )
    return ledger


# ---------------------------------------------------------------- conflicts docket
def render_docket(ledger: dict[str, Any]) -> str:
    held = [r for r in ledger["ledger"] if r["held"]]
    out = [
        "# Conflicts docket — alignment adjudication (owner gate input)",
        "",
        f"Adjudicated {ledger['adjudicatedAt']} at the register pin `{ledger['baseline'][:8]}` over {len(ledger['ledger'])} station entries. "
        "A `conflicts` decision names the Part IV.1 guardrail it violates and carries a roadmap-consistent alternative "
        "(or the owner scoping question); a held row enters no actionable set until the owner overrides or accepts.",
        "",
        "## Guardrails (the rule, not the reading)",
        "",
    ]
    out += [f"- **{k}** — {v}" for k, v in GUARDRAILS.items()]
    out += ["", "## Held rows", ""]
    if not held:
        out += [
            "**None held.** No station entry's recorded fix-shape violates a guardrail: none places a paradigm concept in an L0/L1 crate, "
            "none introduces an engine-to-engine edge, none puts the Engine enum below L4, none hoists shared output orchestration into "
            "cobre-cli (the Wave-5 CD-025/CD-029 restatements were closed at the cli-python owner gate in favour of the L2 entry point, "
            "and no live entry restates them), and none deletes Phase-0a gate substrate (the E08 fix-shape refusals kept the golden "
            "parity_hash_* roster, common/parity_hash.rs and invariance-shuffle.yml out of every accepted fix-shape).",
            "",
            "Rule hits reviewed and dismissed by hand (recorded so the predicates are shown to be live):",
            "",
        ]
        for h in ledger.get("guardrailHitsReviewed", []):
            out.append(
                f"- {h['entryId']} — `{h['guardrail']}` fired on “{h['sentence'][:160]}” → decided {h['decided']}."
            )
        out += [
            "",
            "The two expected conflict classes, written out so an override request has a template:",
            "",
        ]
        for cls, t in CLASS_TEMPLATES.items():
            out += [
                f"### {cls} (no live instance)",
                "",
                f"**Shape.** {t['shape']}",
                f"**Roadmap-consistent alternative.** {t['alternative']}",
                f"**Cost of overriding.** {t['overrideCost']}",
                "",
            ]
        return "\n".join(out) + "\n"
    for r in held:
        out += [
            f"### {r['entryId']} - HELD for owner override - retagged {r['hint']} -> conflicts",
            "",
            f"**Entry.** {r['heading']}",
            f"**Guardrail violated.** {r['guardrailViolated']} - {ROADMAP_REL} {GUARDRAILS.get(r['guardrailViolated'], '')}.",
            f"**Station's fix-shape.** {r['fixShape']}",
            f"**Roadmap-consistent alternative.** {r.get('alternativeFixShape') or '(owner scoping question: ' + str(r.get('needsHuman')) + ')'}",
            f"**Cost of overriding.** {r.get('overrideCost', 'the layering invariant named above is breached for this entry; the owner buys the station shape and loses the guardrail')}",
            "**Decision.** [ ] accept alternative   [ ] override, rationale: ______   [ ] defer to <phase>",
            "",
        ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- retag (bullet-local) and its proof
ALIGN_BULLET_RE = re.compile(
    r"^(?P<indent>\s*)- \*\*Alignment:\*\*\s*(?P<value>[^\s(]+)\s*(?P<rest>.*)$"
)


def retag_bullet(line: str, row: dict[str, Any]) -> str:
    m = ALIGN_BULLET_RE.match(line)
    if not m:
        raise SystemExit(
            f"{row['entryId']}: Alignment bullet not in the expected form: {line!r}"
        )
    return (
        f"{m.group('indent')}- **Alignment:** {row['decided']} ({row['cites']}; station hint: {row['hint']}, "
        f"retagged {TODAY} by {LEDGER_REF})"
    )


def retag_register(ledger: dict[str, Any], register: pathlib.Path) -> dict[str, Any]:
    lines = register.read_text(encoding="utf-8").splitlines()
    PRE_IMAGE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    changed: list[dict[str, Any]] = []
    for r in ledger["ledger"]:
        if not r["retagged"]:
            continue
        start = r["registerLine"] - 1
        assert lines[start].startswith(f"**{r['entryId']}"), (
            r["entryId"],
            lines[start][:40],
        )
        idx = next(
            (
                i
                for i in range(start + 1, len(lines))
                if ALIGN_BULLET_RE.match(lines[i]) and not bp.ENTRY_RE.match(lines[i])
            ),
            None,
        )
        stop = next(
            (
                i
                for i in range(start + 1, len(lines))
                if bp.ENTRY_RE.match(lines[i]) or bp.SECTION_RE.match(lines[i])
            ),
            len(lines),
        )
        if idx is None or idx >= stop:
            raise SystemExit(f"{r['entryId']}: no Alignment bullet inside the entry")
        before = lines[idx]
        lines[idx] = retag_bullet(before, r)
        changed.append(
            {
                "entryId": r["entryId"],
                "line": idx + 1,
                "before": before,
                "after": lines[idx],
            }
        )
    register.write_text("\n".join(lines) + "\n", encoding="utf-8")
    proof = retag_proof(ledger, register)
    ledger["retagProof"] = proof
    ledger["retagChanges"] = changed
    return ledger


def retag_proof(
    ledger: dict[str, Any], register: pathlib.Path, exclude_section: str | None = None
) -> dict[str, Any]:
    """Every line the retag changed outside the alignment section is an Alignment bullet, and their count is the retagged count."""
    pre = PRE_IMAGE.read_text(encoding="utf-8").splitlines()
    post = register.read_text(encoding="utf-8").splitlines()
    excluded: set[int] = set()
    if exclude_section:
        try:
            sec = bp.find_section(post, exclude_section)
            excluded = set(range(sec.start, sec.end))
        except bp.SectionNotFound:
            pass
    removed = [
        ln
        for ln in difflib.unified_diff(pre, post, n=0, lineterm="")
        if ln.startswith("-") and not ln.startswith("---")
    ]
    added_lines: list[str] = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(
        None, pre, post, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        for j in range(j1, j2):
            if j not in excluded:
                added_lines.append(post[j])
    non_alignment = [ln for ln in removed if not ALIGN_BULLET_RE.match(ln[1:])]
    non_alignment_added = [ln for ln in added_lines if not ALIGN_BULLET_RE.match(ln)]
    want = sum(1 for r in ledger["ledger"] if r["retagged"])
    return {
        "preImage": str(PRE_IMAGE.relative_to(ROOT)),
        "removedLines": len(removed),
        "retaggedRows": want,
        "everyRemovedLineIsAnAlignmentBullet": not non_alignment,
        "everyAddedLineOutsideTheSectionIsAnAlignmentBullet": not non_alignment_added,
        "countsMatch": len(removed) == want,
        "offenders": non_alignment[:5] + non_alignment_added[:5],
    }


# ---------------------------------------------------------------- section render
def part_i_table(part_i: dict[str, Any]) -> list[str]:
    out = [
        "| # | Claim | Disposition | Baseline anchor | Register id / Cleared | Phase | Alignment |",
        "|---|-------|-------------|-----------------|-----------------------|-------|-----------|",
    ]
    for r in part_i["dispositions"]:
        a = r["baselineAnchors"][0]
        anchor = (
            f"`{a['path']}::{a['symbol']}`"
            if a.get("symbol")
            else f"`{a['path']}:{a['line']}`"
        )
        reg = (
            r["registerId"] or f"Cleared ({r.get('clearedLine') or r.get('closedBy')})"
        )
        decided = PART_I_DECISIONS.get(r["registerId"] or "", {}).get(
            "decided", r["alignment"]
        )
        out.append(
            f"| {r['item']} | {r['title']} | {r['disposition']} | {anchor} | {reg} | {r['fixShapePhase']} | {decided} |"
        )
    return out


def spliced_part_i_entries() -> list[str]:
    text = PART_I_FRAGMENT.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(text) if ln.startswith("### Entries"))
    body = text[start + 1 :]
    out: list[str] = []
    current: str | None = None
    for line in body:
        m = bp.ENTRY_RE.match(line)
        if m:
            current = m.group("id")
        am = ALIGN_BULLET_RE.match(line)
        if am and current in PART_I_DECISIONS:
            d = PART_I_DECISIONS[current]
            line = f"- **Alignment:** {d['decided']} ({d['cites']}; adjudicated {TODAY} by {LEDGER_REF}: {d['rationale']})"
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return out


def lp_share_line() -> list[str]:
    c = read_json(LP_CLASSIFICATION)
    t = c["totals"]
    verdict = (
        "agreement"
        if t["verdict"] == "agreement"
        else f"**amended (dated {TODAY}, baseline `{c['baseline'][:8]}`)** — {t['amendedFigure']}"
    )
    return [
        f"30 modules classified in alignment/lp-classification.md (rows in alignment/lp-classification.json): "
        f"engine-neutral non-test LOC {t['engineNeutralLoc']:,} plus the mixed modules' neutral half {t['mixedNeutralHalf'][0]:,}-{t['mixedNeutralHalf'][1]:,} "
        f"= {t['measuredBand'][0]:,}-{t['measuredBand'][1]:,} of {t['corpusNonTestLoc']:,} ({round(100 * t['measuredShare'][0])}-{round(100 * t['measuredShare'][1])}%), "
        f"against Part IV.2's fifth-to-a-quarter ({t['roadmapBand'][0]:,}-{t['roadmapBand'][1]:,} over the measured corpus; the ticket's ~2.3k-2.9k of ~11.7k): {verdict}. "
        "Extraction stays priced as a rewrite (Part IV.2). The measured table follows.",
    ]


def ledger_table(ledger: dict[str, Any]) -> list[str]:
    out = [
        "| Entry | Station | Class | Hint | Decided | Basis (Part IV/V) | Held |",
        "|-------|---------|-------|------|---------|-------------------|------|",
    ]
    for r in ledger["ledger"]:
        basis = (r["cites"] or "").replace("|", "\\|")
        out.append(
            f"| {r['entryId']} | {r['station']} | {r['class']} | {r['hint'] or '—'} | {r['decided']} | {basis} | {'yes' if r['held'] else 'no'} |"
        )
    return out


def conflicts_block(ledger: dict[str, Any]) -> list[str]:
    held = [r for r in ledger["ledger"] if r["held"]]
    out = ["#### Conflicts held for owner override", ""]
    if not held:
        out.append(
            f"None: no station entry's recorded fix-shape violates a Part IV.1 guardrail at `{ledger['baseline'][:8]}` "
            f"({len(ledger.get('guardrailHitsReviewed', []))} rule hit reviewed and dismissed by hand; full docket and the two expected "
            "conflict-class templates in alignment/conflicts-docket.md). No held row is actionable."
        )
        return out
    for r in held:
        out.append(
            f"- **{r['entryId']}** — retagged {r['hint']} → conflicts; guardrail `{r['guardrailViolated']}`; alternative: {r.get('alternativeFixShape') or 'owner scoping question: ' + str(r.get('needsHuman'))}. Held; not actionable. Docket: alignment/conflicts-docket.md."
        )
    return out


def render_section(ledger: dict[str, Any], register: pathlib.Path) -> None:
    lines = register.read_text(encoding="utf-8").splitlines()
    sec = bp.find_section(lines, SECTION_TAIL)
    body = lines[sec.start + 1 : sec.end]
    lp_start = next(
        (
            i
            for i, ln in enumerate(body)
            if ln.startswith("#### lp/ kernel boundary (measured at baseline")
        ),
        None,
    )
    if lp_start is None:
        raise SystemExit(
            "the lp/ kernel boundary H4 (E09-2) is missing from the section"
        )
    lp_end = next(
        (i for i in range(lp_start + 1, len(body)) if bp.SECTION_RE.match(body[i])),
        len(body),
    )
    lp_block = body[lp_start:lp_end]
    while lp_block and not lp_block[-1].strip():
        lp_block.pop()
    part_i = read_json(PART_I)
    c = ledger["decidedCounts"]
    new_body = [
        "",
        f"Adjudicated {ledger['adjudicatedAt']} at the register pin `{ledger['baseline'][:8]}` against {ROADMAP_REL} Part IV.1 (L0-L4 and its three "
        "invariants), Part IV.4, Part IV.5 and Part V.1/V.2. Stations proposed hints; the decided value here is the register's single authority "
        "(precedence: where a station bullet and this ledger disagree, the bullet is retagged in place and the superseded hint is preserved inline as "
        f"`Alignment: <decided> (<cite>; station hint: <hint>, retagged {TODAY} by {LEDGER_REF})`). Rubric: guardrails first, then the phase "
        "deliverable tables, else neutral; a fix-shape that IS a deliverable earns the tag, one that merely helps does not. "
        f"{len(ledger['ledger'])} entries adjudicated: {c['advances-0a']} advances-0a, {c['advances-0b']} advances-0b, {c['advances-1']} advances-1, "
        f"{c['neutral']} neutral, {c['conflicts']} conflicts; {len(ledger['retagged'])} retagged; {len(ledger['held'])} held. "
        "Machine-readable ledger: alignment/alignment-ledger.json; late entries are adjudicated by `adjudicate.py append`.",
        "",
        "### Part-I re-verification (nine items)",
        "",
        *part_i_table(part_i),
        "",
        "### Part-I entries (keep and sharpen rows)",
        "",
        *spliced_part_i_entries(),
        "",
        "### lp/ kernel boundary (measured share)",
        "",
        *lp_share_line(),
        "",
        *lp_block,
        "",
        "### Alignment ledger",
        "",
        *ledger_table(ledger),
        "",
        *conflicts_block(ledger),
        "",
    ]
    lines = lines[: sec.start + 1] + new_body + lines[sec.end :]
    register.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


# ---------------------------------------------------------------- verify
def run_tool(tool: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOLS / tool), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def verify(ledger: dict[str, Any], register: pathlib.Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for tool, args in (
        ("check-anchors.py", ["--register", str(register), SECTION_TAIL]),
        ("check-reraise.py", ["--register", str(register), SECTION_TAIL]),
        ("fields-check.py", ["--register", str(register), SECTION_TAIL]),
        (
            "fields-check.py --all",
            ["--register", str(register), "--require", "Alignment", "--all"],
        ),
    ):
        proc = run_tool(tool.split()[0], *args)
        results[tool] = {
            "exit": proc.returncode,
            "tail": (proc.stdout + proc.stderr).strip().splitlines()[-1:],
        }
    lines = register.read_text(encoding="utf-8").splitlines()
    tables = bp.parse_tables(bp.find_section(lines, SECTION_TAIL))
    rendered = [t for t in tables if t and "Decided" in t[0]]
    part_i = [t for t in tables if t and "Disposition" in t[0]]
    results["ledgerTableRows"] = len(rendered[0]) if rendered else 0
    results["partITableRows"] = len(part_i[0]) if part_i else 0
    results["ledgerRowsMatch"] = bool(rendered) and len(rendered[0]) == len(
        ledger["ledger"]
    )
    results["partIRowsAreNine"] = bool(part_i) and len(part_i[0]) == 9
    status = subprocess.run(
        [
            "git",
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
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    results["evaluatedSurfacesClean"] = status == ""
    results["passed"] = (
        all(
            v["exit"] == 0
            for v in results.values()
            if isinstance(v, dict) and "exit" in v
        )
        and results["ledgerRowsMatch"]
        and results["partIRowsAreNine"]
        and results["evaluatedSurfacesClean"]
    )
    return results


# ---------------------------------------------------------------- late entries
def append_section(
    ledger: dict[str, Any], register: pathlib.Path, section_name: str
) -> int:
    lines = register_lines(register)
    known = {r["entryId"] for r in ledger["ledger"]}
    new_rows = [
        r for r in station_entries(lines, (section_name,)) if r["entryId"] not in known
    ]
    substrate = substrate_tokens()
    for r in new_rows:
        r.update(
            hint=r["registerAlignmentValue"] or None,
            hintSource="register bullet",
            hintCites=None,
            alternativeFixShape=None,
            held=False,
            retagged=False,
            needsHuman=None,
        )
        decide_row(r, substrate)
        r["appendedAt"] = TODAY
    ledger["ledger"] += new_rows
    ledger["ledger"].sort(key=lambda r: r["registerLine"])
    return len(new_rows)


# ---------------------------------------------------------------- main
def save(ledger: dict[str, Any]) -> None:
    LEDGER.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "command",
        choices=(
            "build",
            "decide",
            "docket",
            "retag",
            "render",
            "verify",
            "all",
            "append",
        ),
    )
    ap.add_argument("--register", type=pathlib.Path, default=REGISTER)
    ap.add_argument(
        "--section",
        help="append: the register section whose new entries are adjudicated",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="all/retag: restore the register from the pre-image snapshot first, so the pass re-runs from the same pre-image",
    )
    args = ap.parse_args(argv)
    register = args.register
    if args.reset and args.command in ("all", "retag"):
        if not PRE_IMAGE.exists():
            print("--reset without a pre-image snapshot", file=sys.stderr)
            return EXIT_FAIL
        register.write_text(PRE_IMAGE.read_text(encoding="utf-8"), encoding="utf-8")
        PRE_IMAGE.unlink()
    if args.command in ("build", "all"):
        ledger = build_ledger(register)
        save(ledger)
        cov = ledger["coverage"]
        print(
            f"{len(ledger['ledger'])} entries, {len(cov['registerOnly'])} register-only, {len(cov['calibrationOnly'])} calibration-only, "
            f"{len(cov['queueOnly'])} queue-only (prior dispositions), {len(cov['hintDisagreements'])} hint disagreements"
        )
        if cov["registerOnly"] or cov["calibrationOnly"]:
            print(
                f"coverage gap: register-only {cov['registerOnly']} calibration-only {cov['calibrationOnly']}",
                file=sys.stderr,
            )
            return EXIT_FAIL
        if args.command == "build":
            return EXIT_OK
    ledger = read_json(LEDGER)
    if args.command in ("decide", "all"):
        ledger = decide_pass(ledger)
        save(ledger)
        for p in ledger["selfCheck"]["problems"]:
            print(f"self-check FAIL: {p}", file=sys.stderr)
        print(
            f"decided {ledger['decidedCounts']}; retagged {len(ledger['retagged'])}; held {len(ledger['held'])}"
        )
        for r in ledger["retagged"]:
            print(f"  {r['entryId']:8} {r['hint']:12} -> {r['decided']}")
        if ledger["selfCheck"]["problems"]:
            return EXIT_FAIL
        if args.command == "decide":
            return EXIT_OK
    if args.command in ("docket", "all"):
        DOCKET.write_text(render_docket(ledger), encoding="utf-8")
        print(f"docket: {len(ledger['held'])} held rows")
        if args.command == "docket":
            return EXIT_OK
    if args.command in ("retag", "all"):
        if PRE_IMAGE.exists() and args.command == "retag":
            print("pre-image already exists; refusing to retag twice", file=sys.stderr)
            return EXIT_FAIL
        ledger = retag_register(ledger, register)
        save(ledger)
        p = ledger["retagProof"]
        print(
            f"retag: {p['removedLines']} bullets rewritten (ledger says {p['retaggedRows']}); bullet-local {p['everyRemovedLineIsAnAlignmentBullet']}"
        )
        if not (p["everyRemovedLineIsAnAlignmentBullet"] and p["countsMatch"]):
            print(f"retag proof FAILED: {p}", file=sys.stderr)
            return EXIT_FAIL
        if args.command == "retag":
            return EXIT_OK
    if args.command in ("render", "all"):
        render_section(ledger, register)
        print("section rendered under the scaffold's generalization-alignment heading")
        if args.command == "render":
            return EXIT_OK
    if args.command in ("verify", "all"):
        results = verify(ledger, register)
        ledger["verification"] = results
        ledger["handoffs"] = {
            "ownerGate": {
                "held": ledger["held"],
                "needsHuman": ledger["needsHuman"],
                "docket": str(DOCKET.relative_to(ROOT)),
                "retagged": len(ledger["retagged"]),
            },
            "lateEntryRule": ledger["lateEntryRule"],
        }
        save(ledger)
        print(
            json.dumps(
                {k: v for k, v in results.items() if k != "passed"}, ensure_ascii=False
            )
        )
        print(f"verify: {'PASS' if results['passed'] else 'FAIL'}")
        return EXIT_OK if results["passed"] else EXIT_FAIL
    if args.command == "append":
        if not args.section:
            ap.error("append needs --section")
        n = append_section(ledger, register, args.section)
        ledger["selfCheck"] = {
            "problems": self_check(ledger),
            "passed": not self_check(ledger),
        }
        save(ledger)
        print(
            f"appended {n} new entries from {args.section!r}; self-check {'ok' if ledger['selfCheck']['passed'] else 'FAIL'}"
        )
        return EXIT_OK if ledger["selfCheck"]["passed"] else EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
