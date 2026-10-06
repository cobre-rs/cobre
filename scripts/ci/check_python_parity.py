#!/usr/bin/env python3
"""Check that CLI and Python bindings write the same output files.

Parses both the CLI `run` module (`crates/cobre-cli/src/`, a directory) and
`crates/cobre-python/src/` for calls to writers from `cobre_io` and
`cobre_sddp::policy::orchestration`. Resolves bare imported calls back to their
canonical names through `use` statements, then compares the two sets.

Also checks that `write_success_marker` is the last write of each phase writer
in `PHASE_WRITERS`, and of every other function in either tree that calls it.
A crate-local helper that writes counts as a write.

Usage:
    python3 scripts/ci/check_python_parity.py              # default: --max 0, --min-shared 21
    python3 scripts/ci/check_python_parity.py --max 0 --min-shared 21
    python3 scripts/ci/check_python_parity.py --min-shared 22  # floor breach check

Exit code 0 if parity holds (mismatches <= --max, shared >= --min-shared) and no
write follows the marker, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Normalisation map: different names that map to the same logical write.
NORMALISE: dict[str, str] = {
    "write_checkpoint": "write_policy_checkpoint",
    "io_write_policy_checkpoint": "write_policy_checkpoint",
}

MARKER_WRITER = "write_success_marker"

# The phase writers: each must call MARKER_WRITER directly exactly once.
PHASE_WRITERS: tuple[tuple[str, str], ...] = (
    ("crates/cobre-cli/src/commands/run/outputs.rs", "write_training_outputs"),
    ("crates/cobre-cli/src/commands/run/outputs.rs", "write_simulation_outputs"),
    ("crates/cobre-python/src/run.rs", "write_training_outputs"),
    ("crates/cobre-python/src/run.rs", "run_simulation_phase_py"),
)


def parse_imports(text: str) -> dict[str, str]:
    """Parse `use` statements from cobre_io/cobre_sddp::policy::orchestration into a local_name -> canonical_name map.

    Handles single-line, multi-line brace groups, nested {}, and `as` aliases.
    """
    import_map: dict[str, str] = {}

    # Join continuation lines into whole statements by accumulating until `;`.
    current = ""

    for line in text.splitlines():
        stripped = line.strip()
        # Skip comments and attributes.
        if stripped.startswith("//") or stripped.startswith("#["):
            continue

        current += " " + line
        if ";" in current:
            # Process complete statement.
            statement = current[: current.index(";") + 1]
            current = ""

            # Check if it's a relevant import.
            if not (
                "use cobre_io::" in statement
                or "use cobre_sddp::orchestration::" in statement
                or "use cobre_sddp::policy::orchestration::" in statement
            ):
                continue

            # Parse the import.
            _parse_import_statement(statement, import_map)

    return import_map


def _parse_import_statement(statement: str, import_map: dict[str, str]) -> None:
    """Parse a single import statement and update import_map."""
    statement = statement.strip()

    # Handle `as` aliases.
    if " as " in statement:
        # Extract the original name and the alias.
        match = re.search(r"use\s+([\w:]+)\s+as\s+(\w+)\s*;", statement)
        if match:
            full_path = match.group(1)
            alias = match.group(2)
            canonical = full_path.split("::")[-1]
            import_map[alias] = canonical
            return

    # Handle brace groups.
    if "{" in statement:
        # Extract base path and items.
        match = re.match(r".*use\s+([\w:]+)::\{([^}]+)\}", statement)
        if match:
            items_str = match.group(2)

            for item in items_str.split(","):
                item = item.strip()
                if not item:
                    continue

                # Handle nested paths.
                if "::" in item:
                    # Nested item like `output::write_foo`.
                    parts = item.split("::")
                    canonical = parts[-1]
                    import_map[canonical] = canonical
                else:
                    # Simple item.
                    import_map[item] = item
    else:
        # Single-item import.
        match = re.match(r".*use\s+([\w:]+)::(\w+)\s*;", statement)
        if match:
            canonical = match.group(2)
            import_map[canonical] = canonical


def _extract_from_text(text: str, names: set[str], import_map: dict[str, str]) -> None:
    """Collect write function names from one Rust source body into ``names``."""
    for line in text.splitlines():
        stripped = line.strip()
        # Skip comments, use/import lines, and attributes.
        if (
            stripped.startswith("//")
            or stripped.startswith("use ")
            or stripped.startswith("#[")
        ):
            continue

        # Match fully-qualified calls.
        for match in re.finditer(
            r"(?:cobre_io|cobre_sddp::(?:policy::)?orchestration)::([\w:]+::)*(write_\w+|export_\w+)\s*\(",
            line,
        ):
            name = match.group(2)
            if name:
                name = NORMALISE.get(name, name)
                names.add(name)

        # Match bare calls and resolve through import map.
        # Matches write_foo(, export_foo(, FooWriter(, and FooWriter::
        for match in re.finditer(
            r"\b(write_\w+|export_\w+|\w+Writer)(?:\s*\(|::)", line
        ):
            name = match.group(1)
            if name in import_map:
                canonical = import_map[name]
                canonical = NORMALISE.get(canonical, canonical)
                names.add(canonical)


def extract_write_functions(path: Path) -> set[str]:
    """Extract the set of write function names from a Rust source location.

    Accepts either a single ``.rs`` file or a directory module: when ``path``
    is a directory, every ``.rs`` file beneath it is scanned recursively and the
    extracted names are unioned. This keeps the full CLI write surface in scope
    even when the writes are spread across submodules of a directory module.
    """
    names: set[str] = set()

    if path.is_dir():
        for rs_file in sorted(path.rglob("*.rs")):
            text = rs_file.read_text(errors="replace")
            import_map = parse_imports(text)
            _extract_from_text(text, names, import_map)
        return names

    if not path.exists():
        print(f"WARNING: {path} does not exist", file=sys.stderr)
        return set()

    text = path.read_text(errors="replace")
    import_map = parse_imports(text)
    _extract_from_text(text, names, import_map)
    return names


def strip_comments_and_strings(text: str) -> str:
    """Blank out comments and string and char literals, keeping newlines and offsets.

    A `'` opens a char literal only when an escape, or one character and a
    closing `'`, follows it, so lifetimes (`'a`) are kept.
    """
    pieces: list[str] = []
    copied = pos = 0
    while match := re.compile(
        r"""//|/\*|(?<!\w)[bc]?r(?P<hashes>#*)"|(?<!\w)[bc]"|["']"""
    ).search(text, pos):
        start = match.start()
        token = match.group()
        if token == "//":
            end = text.find("\n", start)
        elif token == "/*":
            depth, end = 0, -1
            for delimiter in re.compile(r"/\*|\*/").finditer(text, start):
                depth += 1 if delimiter.group() == "/*" else -1
                if depth == 0:
                    end = delimiter.end()
                    break
        elif match.group("hashes") is not None:
            closing = '"' + match.group("hashes")
            close = text.find(closing, match.end())
            end = close + len(closing) if close != -1 else -1
        elif token.endswith('"'):
            literal = re.compile(r'(?:[^"\\]|\\.)*"', re.S).match(text, match.end())
            end = literal.end() if literal else -1
        else:
            literal = re.compile(r"'(?:\\.[^']*|[^\\'\n])'").match(text, start)
            if literal is None:
                pos = start + 1
                continue
            end = literal.end()
        if end == -1:
            end = len(text)
        pieces.append(text[copied:start])
        pieces.append(re.sub(r"[^\n]", " ", text[start:end]))
        copied = pos = end
    pieces.append(text[copied:])
    return "".join(pieces)


def function_spans(code: str) -> list[tuple[str, int, int]]:
    """Return `(name, body_start, body_end)` for each `fn` with a body in stripped code.

    `body_start` is the offset of the body's `{` and `body_end` the offset just
    past its matching `}`. A signature that reaches a `;` outside any `()` or
    `[]` before its `{` is a declaration without a body and is skipped.
    """
    spans: list[tuple[str, int, int]] = []
    for fn in re.finditer(r"\bfn\s+(\w+)", code):
        nesting, body_start = 0, -1
        for token in re.compile(r"[()\[\]{;]").finditer(code, fn.end()):
            char = token.group()
            if char in "([":
                nesting += 1
            elif char in ")]":
                nesting -= 1
            elif nesting == 0:
                if char == "{":
                    body_start = token.start()
                break
        if body_start == -1:
            continue
        depth, body_end = 0, len(code)
        for brace in re.compile(r"[{}]").finditer(code, body_start):
            depth += 1 if brace.group() == "{" else -1
            if depth == 0:
                body_end = brace.end()
                break
        spans.append((fn.group(1), body_start, body_end))
    return spans


def direct_write_calls(body: str, import_map: dict[str, str]) -> list[tuple[int, str]]:
    """Return `(offset, name)` for each writer call in stripped `body`, in source order.

    Covers `write_*` / `export_*` free-function and path calls, `*Writer::`
    constructor calls, and bare calls to names `import_map` resolves to
    `cobre_io` or `cobre_sddp::policy::orchestration`. Method calls and `fn`
    definitions are not calls.
    """
    calls: dict[int, str] = {}
    for match in re.finditer(
        r"(?<![.\w])(?<!fn\s)((?:\w+::)*)(write_\w+|export_\w+)\s*\(", body
    ):
        name = match.group(2)
        calls[match.start(2)] = name if match.group(1) else import_map.get(name, name)
    for match in re.finditer(r"(?<![.\w])(\w+Writer)::(\w+)\s*\(", body):
        calls[match.start(1)] = f"{match.group(1)}::{match.group(2)}"
    for match in re.finditer(r"(?<![.\w:])(?<!fn\s)(\w+)\s*\(", body):
        if match.group(1) in import_map:
            calls.setdefault(match.start(1), import_map[match.group(1)])
    return sorted(calls.items())


def _function_calls(
    code: str, import_map: dict[str, str]
) -> list[tuple[str, list[tuple[int, str]], list[tuple[int, str]]]]:
    """Return `(name, direct_write_calls, local_calls)` per function in stripped code.

    Local calls are bare, `crate::`, `super::` and `self::` calls. Offsets are
    absolute, and a call inside a nested function belongs to the nested one.
    """
    spans = function_spans(code)
    functions = []
    for name, start, end in spans:
        nested = [(s, e) for _, s, e in spans if start < s and e <= end]
        body = code[start:end]
        direct = [
            (start + offset, callee)
            for offset, callee in direct_write_calls(body, import_map)
            if not any(s <= start + offset < e for s, e in nested)
        ]
        local = [
            (start + call.start(1), call.group(1))
            for call in re.finditer(
                r"(?<![.\w:])(?<!fn\s)(?:(?:crate|super|self)::(?:\w+::)*)?(\w+)\s*\(",
                body,
            )
            if not any(s <= start + call.start(1) < e for s, e in nested)
        ]
        functions.append((name, direct, local))
    return functions


def local_writers(files: dict[str, str]) -> set[str]:
    """Return the names of the functions in one crate tree that write.

    `files` maps every `.rs` file of the tree to its stripped source. A function
    writes when it makes a direct writer call, or a local call to a function
    that writes.
    """
    functions = [
        function
        for code in files.values()
        for function in _function_calls(code, parse_imports(code))
    ]
    writers = {name for name, direct, _ in functions if direct}
    while True:
        callers = {
            name
            for name, _, local in functions
            if any(callee in writers for _, callee in local)
        }
        if callers <= writers:
            return writers
        writers |= callers


def check_marker_order(root: Path) -> list[str]:
    """Return one `<file>::<function>: ...` failure per marker-order violation.

    Presence: every `PHASE_WRITERS` function exists and calls `MARKER_WRITER`
    directly exactly once. Order: no function in either crate tree that calls
    `MARKER_WRITER` directly makes a writer call, or a call to a local writer,
    after its first marker call.
    """
    scanned: dict[str, tuple[list, set[str]]] = {}
    for tree in ("crates/cobre-cli/src", "crates/cobre-python/src"):
        files: dict[str, str] = {}
        for path in sorted((root / tree).rglob("*.rs")):
            file = f"{tree}/{path.relative_to(root / tree).as_posix()}"
            files[file] = strip_comments_and_strings(path.read_text(errors="replace"))
        writers = local_writers(files)
        for file, code in files.items():
            scanned[file] = (_function_calls(code, parse_imports(code)), writers)

    failures: list[str] = []
    for file, function in PHASE_WRITERS:
        if file not in scanned:
            failures.append(f"{file}::{function}: file not found")
            continue
        bodies = [direct for name, direct, _ in scanned[file][0] if name == function]
        if not bodies:
            failures.append(f"{file}::{function}: function not found")
        for direct in bodies:
            count = sum(callee == MARKER_WRITER for _, callee in direct)
            if count != 1:
                failures.append(
                    f"{file}::{function}: calls `{MARKER_WRITER}` directly "
                    f"{count} time(s), expected exactly once"
                )

    for file, (functions, writers) in scanned.items():
        for name, direct, local in functions:
            markers = [offset for offset, callee in direct if callee == MARKER_WRITER]
            if not markers:
                continue
            later = sorted(
                (offset, callee)
                for offset, callee in direct + [c for c in local if c[1] in writers]
                if offset > markers[0] and callee != MARKER_WRITER
            )
            if later:
                failures.append(
                    f"{file}::{name}: `{later[0][1]}` is called after `{MARKER_WRITER}`"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Python parity for output writes."
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Maximum allowed mismatches (default: 0). Exit 1 if exceeded.",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=21,
        help="Minimum shared write functions (default: 21). Exit 1 if below this floor. The floor exists to catch a gate that stopped seeing call sites; raise it when writers are added, never lower it.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args()

    cli_path = args.root / "crates" / "cobre-cli" / "src"
    python_path = args.root / "crates" / "cobre-python" / "src"

    cli_writes = extract_write_functions(cli_path)
    python_writes = extract_write_functions(python_path)

    cli_only = sorted(cli_writes - python_writes)
    python_only = sorted(python_writes - cli_writes)
    mismatches = len(cli_only) + len(python_only)
    shared = sorted(cli_writes & python_writes)

    order_failures = check_marker_order(args.root)

    # Check floor before mismatch.
    set_failed = True
    if len(shared) < args.min_shared:
        print(
            f"FAIL: {len(shared)} write functions in both paths (floor: {args.min_shared}). "
            f"The gate stopped seeing call sites. Check that imports are resolved correctly."
        )
    elif mismatches > args.max:
        print(f"FAIL: {mismatches} parity mismatch(es) (max allowed: {args.max})")
        if cli_only:
            print()
            print("  In CLI but missing from Python:")
            for name in cli_only:
                print(f"    - {name}")
        if python_only:
            print()
            print("  In Python but missing from CLI:")
            for name in python_only:
                print(f"    - {name}")
        print()
        print("Fix: add the missing write call(s) to the other path.")
        print("CLI path:    crates/cobre-cli/src/")
        print("Python path: crates/cobre-python/src/")
    else:
        set_failed = False

    for failure in order_failures:
        print(f"FAIL: {failure}")

    if set_failed or order_failures:
        sys.exit(1)
    print(
        f"OK: {mismatches} parity mismatch(es) (max allowed: {args.max}). "
        f"{len(shared)} write functions in both paths. "
        f"`{MARKER_WRITER}` is the last write in each of {len(PHASE_WRITERS)} phase writers."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
