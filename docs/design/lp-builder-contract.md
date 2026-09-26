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

Patch the parts of a template that change between solves, without allocating:
noise on right-hand sides (inflows, loads), the column bounds that pin the incoming
state, and the cut rows appended to a solver instance. Which rows and columns a
patch touches is decided at build time; the per-solve work only writes precomputed
values into precomputed positions (`PatchBuffer`).

- **Must not:** allocate on the per-solve path, or decide at solve time which rows
  or columns to touch.

### 3. Address

Give the algorithm fast access to the parts of each LP it reads or writes: to apply
patches, to read the outgoing state, to read duals and reduced costs for cut
coefficients, and to extract simulation output (`StageGeometry`, `StateSpace`,
`CutStateProjection`). Each row or column family has one address formula, owned in
one place, with its block stride taken from one source. Consumers call accessors
instead of computing `start + offset` by hand. Per-solve consumers get contiguous
ranges or precomputed index vectors.

- **Must not:** hold the same address fact in two structures, or let a consumer
  compute an entity's row or column by hand.

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
