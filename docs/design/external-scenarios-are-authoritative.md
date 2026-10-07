# External scenarios are authoritative

> **Status:** Implemented (retained pending fold into the live spec). The behavior
> described below ships; the current-behavior and design sections now both describe
> the live tree. Symbols are cited by name and file so the pointers survive
> line-number churn.

**Scope:** how an externally-supplied scenario file (`scheme == External`, per
class: inflow / load / NCS) relates to the seasonal-statistics files. Narrows the
role of the seasonal-stats files to _scenario generation_ and makes the external
scenario file the sole source of a class's realized values under the External
scheme — including the deterministic (σ = 0) case.

**Relationship to other docs:**

- `.claude/rules/sddp.md` owns the policy-load and stochastic correctness
  contracts; this proposal changes one `cobre-io` semantic validation rule (the
  external-library σ check) and its rationale, and adds no numerical contract.

---

## 1. Problem statement

A study that supplies its own scenarios sets a class's `SamplingScheme` to
`External` and provides `scenarios/external_<class>_scenarios.parquet`. From a
user's point of view the external file _is_ the data: editing it should change the
run, and nothing else should need to be edited in lock-step.

Two things break that expectation today:

1. **A deterministic external value is silently discarded, then loudly rejected.**
   For load and NCS the realized value is reconstructed as `μ + σ·η`, where the
   deviate `η` is obtained by standardizing the external value with the seasonal
   mean/std (`standardize_external_simple` in `cobre-stochastic`
   `sampling/external.rs`; reconstruction in `transform_load_noise` /
   `transform_ncs_noise`, `cobre-sddp` `stochastic/noise.rs`). When `σ = 0`, `η` is
   forced to `0`, so the realized value collapses to `μ` — the seasonal mean from
   `load_seasonal_stats.parquet` — and the external value is ignored. To keep that
   silent collapse from being a silent _lie_, rule 50 in
   `check_external_library_coherence` (`cobre-io`
   `validation/semantic/scenarios.rs`) rejects any `σ = 0` external value that
   disagrees with `μ`. The net effect: to change a deterministic external load the
   user must edit `load_seasonal_stats.parquet` (`mean_mw`), not the external file
   they were editing — and keep the two consistent or hit a `BusinessRuleViolation`.

2. **Deterministic external inflow is rejected outright.** The same rule 50 rejects
   _every_ `σ = 0` external inflow (the `name != "inflow"` guard), so a deterministic
   external inflow deck cannot load at all, even when its values are internally
   consistent.

The friction is worst for decks produced by an external converter, where all three
classes are emitted as `External` for a shared scenario axis: a deterministic class
then carries a redundant external file whose values can never take effect.

## 2. The principle

> **When a class's scheme is `External`, the external scenario file is the
> authoritative source of that class's realized values. The seasonal-statistics
> files (`inflow_seasonal_stats`, `load_seasonal_stats`, `non_controllable_stats`)
> and the PAR coefficients are _scenario-generation_ inputs — consulted for the
> generated schemes (`InSample` / `OutOfSample`), not required as a second copy of
> data the external file already carries.**

Concretely: under `External`, the mean/std a class needs are **properties of the
external samples**, derived per (entity, stage) over the scenario axis. The data
itself decides whether a class is deterministic — a constant column is σ = 0
(mean = the value); a varying column is σ > 0.

The one principled exception is the PAR _state dynamics_ of inflow (below): the
autoregressive coupling of inflow lags is part of the model, not a generation
detail, so for an AR(p > 0) inflow it stays owner-supplied.

## 3. Why the σ > 0 case already round-trips (and the σ = 0 case does not)

For any σ > 0, the standardize-then-reconstruct round trip is the identity:
`μ + σ·((v − μ)/σ) = v`, for load/NCS regardless of which μ/σ pair is used, as long
as the same pair standardizes and reconstructs. So **editing an external file
already works today whenever the class is stochastic** — the seasonal μ/σ only have
to be _some_ consistent pair, and the realized value is the external value either
way.

The information loss is confined to σ = 0: there is no deviate that encodes a
departure from the mean, so the reconstruction cannot recover anything but μ. The
fix therefore only has to make μ, in the σ = 0 case, equal the external value — which
it does automatically once μ is derived from the external samples (a constant column
has mean equal to its constant value).

## 4. Design

### 4.1 Load and NCS — derive (μ, σ) from the external samples

Under `External`, build the standardized library for load and NCS from moments
computed over the external scenario axis (`PrecomputedNormal` sourced from the
samples rather than from `system.load_models()` / `ncs_models()`). Reconstruction is
unchanged: the realized value is `(μ + σ·η).max(0)` for load and
`max_gen·clamp(μ + σ·η, 0, 1)` for NCS, which recovers the external value under either
moment source because the same pair standardizes and reconstructs. The σ = 0 case is
bit-exact (μ = the constant value, η = 0, no division); the σ > 0 case recovers the
external value **up to the round-off of the standardize/reconstruct round trip**
(`μ + σ·((v − μ)/σ)` is algebraically `v`, not IEEE-754-guaranteed bit-for-bit under a
different intermediate `(μ, σ)`). Only the never-observed internal `η` differs. This is
_result-neutral_, not proven bit-identical — see §6 for how the invariant is enforced.
The σ = 0 case now yields the external value instead of collapsing to a
separately-supplied μ.

Consequence: `load_seasonal_stats.parquet` and `non_controllable_stats.parquet`
become **optional** under the External scheme — a class supplied entirely through its
external file needs no seasonal-stats twin. `load_factors.json` /
`non_controllable_factors.json` (the block shape) are unaffected and still apply.

### 4.2 Inflow AR(0) — the load case wearing an inflow hat

An AR(0) inflow has no lag terms, so its deterministic base is exactly μ
(`PrecomputedPar::deterministic_base` reduces to the mean at order 0) and it is not a
state variable. Under `External`, treat it exactly as load: derive μ (and σ) from the
external samples so a deterministic (σ = 0) AR(0) external inflow loads and uses its
own values. σ > 0 AR(0) already round-trips.

### 4.3 Inflow AR(p > 0) — PAR stays the model, not generation

For an autoregressive inflow the lag coupling is intrinsic: the realized inflow feeds
the inflow-lag **state** (`shift_lag_state` in `cobre-sddp` `stochastic/noise.rs`),
and the PAR ψ coefficients price those lag-state dimensions into the Benders cuts. The
ψ structure and residual σ are model artifacts that cannot be recovered from realized
samples. So for AR(p > 0):

- σ > 0 is **unchanged** — the seasonal-stats + AR coefficients define the dynamics
  and the truncation floor; the external file supplies the realizations, which
  round-trip as today. This path is not touched (byte-neutral).
- σ = 0 is **kept rejected** (decision, §7): a deterministic AR(p > 0) inflow would
  require the external series to follow the exact deterministic PAR recursion at every
  stage — a whole-trajectory constraint of marginal value. The rejection stays; only
  its _message_ is corrected (see §5, F2): it is not that "η inversion is undefined,"
  it is that the value must equal the deterministic PAR output the loader cannot
  compute upstream.

### 4.4 What is unchanged

- The forward pass still replays the external η library (`ClassSampler::External`);
  the enumerated `nodes[]` dialect still pins external columns for all passes.
- The "External scheme with an empty external file is an error"
  check (`check_external_scheme_has_files`) stays — External still means "supply the
  file."
- Hot-path reconstruction for load/NCS is untouched; the only change is where the
  moments come from.

## 5. Validation changes (`cobre-io`)

- **Rule 50 rewrite.** The σ = 0 "must equal μ" branch is removed for load and NCS:
  under External, μ is _defined by_ the external file, so there is nothing to
  disagree with. The rule's remaining job is the σ = 0 inflow decision: accept AR(0),
  reject AR(p > 0) with an accurate message. Its regression tests
  (`rule_50_*`) are re-pointed accordingly.
- **Seasonal-stats optionality.** Under External, an absent seasonal-stats file for
  that class is no longer an implicit error; the dimensional coverage check remains
  conditional on the file being present.
- **F2 (message accuracy).** The catalogue and inline rationale stop asserting "PAR
  inversion is undefined at σ = 0" (it is defined — `solve_par_noise` returns a value
  or a reject sentinel); they state the real reason.
- **F4 (dead branch).** The unreachable negative-σ (`std = {s}`) format arm in rule
  50's fallthrough is removed; negative σ is already rejected at the schema layer, so
  only the "no matching stats entry" case remains live.

## 6. Byte-neutrality and determinism

- The golden parity hash (`compute_parity_hash`, `tests/common/parity_hash.rs`)
  digests **solver outputs only** — cut coefficients, storage, water values,
  spillage, thermal generation — never the internal η library.
- **No golden `parity_hash_*` (HiGHS or CLP) or `mpi_wire` baseline exercises an
  External load/NCS or External AR(0)-inflow deck**, so no baseline can move and none
  is re-generated. The guarantee is enforced as a **result-neutrality gate**: the
  before/after output hash diff on every existing deck must be empty; a moved hash is
  an escalation signal, never a re-baseline. (This is the honest form of the §4.1
  round-trip — result-neutral by coverage, not bit-identity by IEEE-754.)
- Moment derivation is a deterministic reduction over the scenario axis in canonical
  order, preserving the declaration-order-invariance and run-to-run reproducibility
  contracts.
- The AR(p > 0) σ > 0 inflow path is not modified, so its truncation/state behavior
  is untouched.

## 7. Decisions

- **AR(p > 0) deterministic external inflow: rejected, message corrected.** Not
  supported in this change; the whole-trajectory constraint is of marginal value.
- **Vehicle: design doc + progressive plan.** This proposal is the plan's spec.

## 8. Downstream (out of scope here)

An external converter that emits all three classes as `External` can, after this
change, treat each external scenario file as the editable source of truth and stop
emitting a redundant deterministic seasonal-stats twin. That is a change in the
converter's own repository, sequenced after this one; it needs no further cobre
change once the external file is authoritative.

## 9. Non-goals

- No change to the generated schemes (`InSample` / `OutOfSample`), which continue to
  read the seasonal-stats + PAR inputs as their generating distribution.
- No change to the PAR state/cut machinery for autoregressive inflow.
- No new file format and no new wire format; the external and seasonal-stats parquet
  schemas are unchanged (one becomes optional under External).
