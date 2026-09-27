# The stage-LP builder contract

**Status:** Live spec (normative). This page states what the stage-LP builder in
`cobre-sddp` is for and what every change to it must satisfy. Where the code falls
short of it, the gap is a defect to fix, not a reason to change the contract.

The builder is the code that turns a study into the linear programs the SDDP
algorithm solves: `lp::builder` builds them, `lp::indexer` holds the address and
state-layout types, and `setup` resolves the inputs they are built from. It exists
to serve the algorithm, and four jobs define it. Every piece of builder code should
serve at least one of them; code that serves none is a candidate for deletion.

## The four jobs

### 1. Build

Turn resolved study input into one LP template per stage: columns, rows,
coefficients, bounds and costs (`StageTemplates`). This runs once, at setup.

The builder is a transform of input that is already resolved. It reads typed owners
built in `setup` — for example the block clock (`BlockClock`), the time value and
delivery calendar (`TimeValue`), the anticipated-plant set (`AnticipatedPlants`) and
the state layout (`StateSpace`) — rather than resolving raw input itself.

- **Must not:** re-derive a fact that a resolved owner already holds, or resolve
  input inside the builder. Resolution belongs in `setup`.

### 2. Update

Patch the parts of a template that change between solves, without allocating.
Which rows and columns a patch touches is decided at build time; the per-solve work
only writes precomputed values into precomputed positions (`PatchBuffer`). The
per-solve edits are:

- the column bounds that pin the incoming state (`fill_col_state_patches`);
- the right-hand sides of the inflow-definition rows (`fill_z_inflow_patches`) and of
  the load-balance rows (`fill_load_patches`);
- the column bounds of stochastic non-controllable sources (`apply_ncs_col_bounds`);
- the future-cost column θ, pinned to zero at the terminal stage;
- the cut rows appended to, or toggled on, a solver instance (owned by the cut layer,
  which reads the builder's row count, θ column and column scales).

The objective is set at build time and is not part of the per-solve update.

- **Must not:** allocate on the per-solve path, or decide at solve time which rows
  or columns to touch.

### 3. Address

Give the algorithm fast access to the parts of each LP it reads or writes. Two kinds
of consumer read the address facts, with different needs:

- **Per-solve training** (forward pass, backward pass, lower bound) reads the state
  layout (`StateSpace`), the cut projections (`CutStateProjection`), each template's
  row and column counts and scales, and the positions its patches write to. These
  reads must be contiguous ranges or precomputed index vectors.
- **Output decoding** (simulation extraction, policy export) reads the full per-stage
  address map (`StageGeometry`) and its inverse, which classifies a row or column
  back to its entity and block (`classify_incoming_column` for state columns).

Each row or column family has one address formula, owned in one place, with its block
stride taken from one source. Consumers call accessors instead of computing
`start + offset` by hand. A materialized copy of an address is allowed only when it
serves a named per-solve access pattern, is written from the single owner when the
templates are built, and is never recomputed independently.

A family's owner exposes its range. A consumer that reads the whole family, whether
by copying it, slicing it or checking a length against it, uses that range and
computes no address. An element accessor is added on the owner when a consumer
first needs individual elements, and every element read of that family then goes
through it. A family no consumer indexes element by element gets no accessor.

Every count that describes the study or the LP layout has one owner. That includes
entity counts, blocks per stage, stages, state dimensions, PAR order, ring depth and
buckets. Another struct may hold a copy only in one of three forms, and only when the
copy is built from the owner, so that it cannot disagree with it:

- an output record that reports the count;
- the shape of storage the struct itself owns;
- a copy on the per-solve path that serves a named access pattern.

The constructor of such a struct takes the owner, never a loose count, and no struct
literal sets the copy independently. Any other copy is a second home for the fact and
is deleted.

- **Must not:** derive the same address fact in two places, keep a copy of a count
  outside the three permitted forms, or let a consumer compute an entity's row or
  column by hand.

### 4. Keep the invariants the algorithm relies on

These are properties of the layout and coefficients that SDDP's correctness
depends on. Their full statements, with the regression tests that pin them, are in
`.claude/rules/sddp.md`. The builder owes them:

- **Cut sign and subgradients:** the sign convention and scaling of the state
  coefficients that cuts are built from ("Benders cut sign & subgradient
  extraction").
- **State pinning through column bounds:** every incoming state dimension is fixed
  by its column bounds, not by equality rows ("State pinning uses column bounds, not
  equality rows").
- **Stable slots:** an append-only cut pool and a layout that lets stored bases match
  by slot identity ("Cut pool is append-only; basis matches by slot identity"; bases
  are applied through `reconstruct_basis`).
- **One liveness rule for state dimensions:** the builder's own reachability decides
  which state dimensions are live (for example `for_each_live_commitment_slot` for
  anticipated commitments). The cut mask and every manifest of state slots read that
  same decision, never a second rule.
- **Determinism:** templates are bit-for-bit identical regardless of the order in
  which entities are declared, and identical across fresh runs.
- **Formulation contracts:** FPHA uses average storage, NCS availability is a
  dimensionless factor, and anticipated deliveries are discounted relative to their
  decision stage.
- **Layout conventions the runtime relies on without re-deriving them:**
  - state columns come first in every stage LP, and a state dimension's outgoing
    column index equals its state index (`assemble_outgoing_state` copies the
    leading block of the primal);
  - the inflow-definition rows occupy the same range at every stage;
  - a stage's base row count is fixed and cut rows are appended after it, the
    boundary that basis reconstruction (`ReconstructionTarget`) and cut-dual
    extraction read;
  - a column scale has one meaning: state pins and reduced costs are divided by it,
    and commitment columns are left unscaled;
  - state boxes are built from physical bounds before column scaling is applied;
  - θ is the same column at every stage, and its objective coefficient is the
    stage's one-step discount factor.

- **Must not:** change the code behind one of these without a test that fails when
  the invariant breaks. A byte-identical snapshot proves that a change moved nothing;
  it does not prove that the result is right.

## Questions every change must answer

Before adding or reshaping builder code, answer these in the change itself (its
description, its tests, or its review):

1. **Which job does it serve?** If it serves none, don't add it.
2. **Does it create a second home for a fact?** A new owner deletes the other
   derivations of its fact in the same change, or names the change that will.
3. **Is the input already resolved?** If the builder has to resolve something, move
   that resolution into `setup`.
4. **Does it run per solve?** Then it must not allocate, and its addresses come from
   positions computed at build time.
5. **Does it touch an invariant from job 4?** Name the test that fails if the
   invariant breaks. When a test pins a numeric result, prefer a value derived by
   hand or in closed form over one recorded from a run; a recorded value can encode a
   bug.
6. **Does a real consumer need it today?** Size it for what is known and keep it
   minimal for what is not: no abstraction, parameter or generality that no current
   consumer uses.

## Outside the builder

These consume the builder's outputs and are not part of this contract:

- solving the LPs (the `cobre-solver` backends);
- cut generation, cut selection and the forward and backward passes;
- scenario sampling and MPI work distribution;
- loading and validating input (`cobre-io`).
