# FlatBuffers Schema for Policy Checkpoints

The binary files under a study's `policy/` directory are
[FlatBuffers](https://flatbuffers.dev/) buffers. Cobre's runtime writes
and reads them through a hand-rolled, allocation-free path in Rust, but
external consumers (Python, C++, TypeScript, Java, Go, …) can use the
canonical schema file shipped with the source tree to generate a typed
reader in any language `flatc` supports.

| File path                     | Root table                                        |
| ----------------------------- | ------------------------------------------------- |
| `policy/cuts/stage_NNN.bin`   | `StageCuts`                                       |
| `policy/basis/stage_NNN.bin`  | `StageBasis`                                      |
| `policy/states/stage_NNN.bin` | `StageStates` (only when `exports.states = true`) |

The schema lives at
[`crates/cobre-io/schemas/policy.fbs`](https://github.com/ONS-Brasil/cobre/blob/main/crates/cobre-io/schemas/policy.fbs)
under namespace `Cobre.IO.Policy`. It has no `file_identifier` and no
`root_type` — pass `--root-type` to `flatc` to select the entry point
for each file.

## Quick start: dumping a `.bin` to JSON

`flatc` ships a converter that turns any FlatBuffers buffer into JSON
when given the schema. This is the closest thing to a human-readable
view of a policy checkpoint:

```bash
flatc -t --strict-json --raw-binary \
    --root-type StageCuts \
    crates/cobre-io/schemas/policy.fbs \
    -- output/policy/cuts/stage_000.bin
# writes stage_000.json next to the .bin
```

For the basis or states files, swap the `--root-type` argument for
`StageBasis` or `StageStates`.

## Generating a typed reader

`flatc` emits idiomatic source code for any of its supported target
languages. Pick the one matching your toolchain.

### Python

```bash
flatc --python crates/cobre-io/schemas/policy.fbs
# emits Cobre/IO/Policy/{Cut,StageCuts,StageBasis,StageStates}.py
```

```python
from Cobre.IO.Policy.StageCuts import StageCuts

with open("output/policy/cuts/stage_000.bin", "rb") as f:
    buf = bytearray(f.read())

cuts = StageCuts.GetRootAs(buf, 0)
print("stage_id =", cuts.StageId())
for i in range(cuts.CutsLength()):
    cut = cuts.Cuts(i)
    print(cut.CutId(), cut.Intercept(), [cut.Coefficients(j) for j in range(cut.CoefficientsLength())])
```

> **Python users on the cobre PyO3 binding can skip `flatc` entirely**:
> `cobre.results.load_policy(output_dir)` returns a structured Python
> dict already. Use `flatc` only if you need partial reads on huge
> files or you are not using the Python wheel.

### C++

```bash
flatc --cpp crates/cobre-io/schemas/policy.fbs
# emits policy_generated.h
```

### TypeScript / JavaScript

```bash
flatc --ts crates/cobre-io/schemas/policy.fbs
# emits TypeScript modules under cobre/io/policy/
```

For other targets see `flatc --help`; the schema is plain enough to
work with all of them.

## Field-by-field reference

The authoritative description of every field lives in
[`policy.fbs`](https://github.com/ONS-Brasil/cobre/blob/main/crates/cobre-io/schemas/policy.fbs)
itself — every field carries an inline doc comment. The
[Output Format](./output-format.md) page has a tabular summary suitable
for reading on the web.

### Reserved slot: `Cut.domination_count`

Field id 4 of the `Cut` table (`domination_count`) is marked
`deprecated`. It was used by policy files written before the v0.5.0
release and is preserved in the schema only so that:

1. The vtable slot number is permanently burned and cannot be reused
   by a future field.
2. Pre-v0.5.0 policy files continue to deserialise via FlatBuffers'
   graceful-absence rule — the slot is read, ignored, and discarded.

Generated readers emit no accessor for it; generated writers cannot
emit it. The Cobre runtime's own writer never sets it.

## How drift is prevented

The schema is not consumed by Cobre's own build. Two independent
implementations describe the same wire format:

- The **schema file** `crates/cobre-io/schemas/policy.fbs`, with
  explicit `(id: N)` attributes on every field.
- The **hand-rolled writer/reader** in
  `crates/cobre-io/src/output/policy.rs`, which encodes vtable slots
  via the `*_FIELD_*: u16` constants. The slot offset is
  `(field_id + 2) * 2`.

A conformance test, `tests/flatbuffers_schema_conformance.rs` in
`cobre-io`, round-trips representative buffers in **both** directions:

- Hand-rolled writer → `flatc -t` → JSON: catches the writer emitting
  a slot the schema does not declare, or at the wrong offset.
- JSON → `flatc -b` → hand-rolled reader: catches the schema
  declaring a slot the reader expects at a different offset.

The test is gated behind the `flatc-conformance` cargo feature so that
the everyday `cargo test` does not depend on `flatc`. To run it:

```bash
cargo test -p cobre-io \
    --features flatc-conformance \
    --test flatbuffers_schema_conformance
```

If you change either the schema or the slot constants, run the
conformance test before merging. The CI workflow that has `flatc`
available runs it on every pull request that touches `policy.rs` or
the schema file.

## Versioning policy

FlatBuffers' graceful-absence rule lets us add new fields to any table
without breaking older readers, **as long as** new fields are appended
at the end with the next available `id`. This is the only schema
change that does not require an output-format version bump:

- **Adding a field at the next free id** → backward compatible.
  Old readers see the field as absent and use the FlatBuffers default
  (zero / empty vector). New readers see the value when the writer
  was new enough to emit it.
- **Removing a field** → mark it `deprecated`, never reuse the id.
  See `Cut.domination_count` for a worked example.
- **Changing a field's type** → breaking. Bumps the major output
  format version.
- **Renaming a field** → breaking for `flatc`-generated code (the
  accessor name changes). Avoid; if necessary, treat as a major bump.
- **Reordering fields** → harmless if `(id: N)` attributes stay put.
  The wire layout is determined by the ids, not by source order.
