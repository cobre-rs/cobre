# Deferred: Inflow Non-Negativity Truncation Methods

## Status

**Deferred to post-v0.1.0.** Only the penalty method is implemented in v0.1.0.

## Background

The SDDP literature describes three strategies for handling negative inflow
realisations from PAR(p) models:

1. **Penalty** (implemented in v0.1.0) — add slack variables to the water
   balance rows with a high objective cost. The LP solver absorbs negative
   inflow through virtual inflow at the penalty cost.

2. **Truncation** (deferred) — evaluate the AR model externally, compute the
   full inflow value, and if negative, modify the LP row bounds to force
   inflow to zero.

3. **Truncation with penalty** (deferred) — combine both: bound the slack
   variable by the maximum feasible infeasibility and apply a penalty cost.

## Why truncation is non-trivial in Cobre

Cobre's LP formulation encodes the AR dynamics implicitly in the water balance
constraint. The forward pass patches noise `η` into the LP row bounds via
`fill_forward_patches`, but the **full inflow value** `a_h` is never computed as
a scalar before the LP solve. The AR state transition is:

```
a_h = deterministic_base + Σ ψ_ℓ · lag_ℓ + σ_m · η
```

where `ψ_ℓ` are AR coefficients, `lag_ℓ` are previous-stage inflow values
(carried in the state vector), and `σ_m` is the seasonal standard deviation.

Truncation requires knowing `a_h` before LP patching:

- If `a_h < 0`, compute an adjusted noise or modified row bounds
- If `a_h >= 0`, no adjustment needed

This means the AR model must be evaluated **outside the LP**, using the
current state vector and noise realisation, before any LP patching occurs.

## Reference: SPTcpp implementation

The SPTcpp codebase (`~/git/SPTcpp`) implements all three methods:

### Pure truncation (`truncamento`)

In `C_ModeloOtimizacao.cpp` (lines 3557-3589):

```cpp
if (realizacao < valor_minimo_convexo) {
    realizacao_flex = valor_minimo_convexo - realizacao;
    novo_valor_sup = valor_viabilidade + realizacao_flex;
    // Pure truncation: fix variable bounds (LB = UB)
    novo_valor_inf = novo_valor_sup;
}
```

The inflow realisation is computed externally in `C_VariavelAleatoria.cpp` via
`gerarCenariosEspacoAmostral`. When the realisation is negative, the LP
variable bounds are set to a fixed value that forces inflow to zero.

### Truncation with penalty (`truncamento_penalizacao`)

Same bound computation, but the lower bound is left at zero (not fixed to
upper bound), creating slack freedom:

```cpp
// LB = 0.0, UB = valor_viabilidade + realizacao_flex
// Penalty cost applied to the AR dynamics slack variable
```

The penalty is applied via `setCofRestricao(varYP_FINF, equZP, -penalidade)`
in `C_ModeloOtimizacao_multiestagio_estocastico.cpp` (lines 771-778).

### Key difference from Cobre's penalty method

| Aspect            | Cobre v0.1.0 (penalty) | SPTcpp (truncation variants) |
| ----------------- | ---------------------- | ---------------------------- |
| Where slack lives | Water balance row      | AR dynamics row              |
| Slack bounds      | `[0, +∞)`              | `[0, max_infeasibility]`     |
| External AR eval  | Not needed             | Required                     |
| LP modification   | Adds slack columns     | Modifies variable bounds     |

## Reference: Paper

Oliveira et al. (2022), "Assessing Negative Inflow on Stochastic Dual Dynamic
Programming", _Energies_ 15(3):1115.

- Section 2.2: Penalty method (equations 18-26)
- Section 2.3: Truncation method (equations 27-35, corrected 36-44)
- Section 2.4: Combined truncation with penalty (equations 45-53)

Key equation for truncation (eq. 44):

```
l = max(max(-d₁/c₁, -d₂/c₂, ...) - x*, 0)
```

where `d_h` is the deterministic inflow component, `c_h` is the AR coefficient,
and `x*` is the optimal state.

## Implementation plan for post-v0.1.0

### Required changes

1. **External AR evaluation function** — Given the current state vector, noise
   realisation, AR coefficients, and seasonal parameters, compute the full
   inflow `a_h` for each hydro plant. This must be accessible from both the
   forward pass and the simulation pipeline.

2. **Truncation logic in forward pass** — Before `fill_forward_patches`, call
   the external AR evaluator. If any `a_h < 0`, compute adjusted noise or
   modified row bounds.

3. **Bounded slack for truncation-with-penalty** — The slack variable must be
   bounded by `max(-a_h, 0)`, not unbounded as in the pure penalty method.
   This requires per-scenario, per-stage bound updates.

4. **AR coefficient access** — The forward pass currently does not have access
   to the AR coefficients (they are baked into the LP template). A new data
   path from `StochasticContext` to the forward pass is needed.

### Estimated effort

Medium-high. The external AR evaluation is straightforward, but threading the
AR coefficients through the forward pass and ensuring the backward pass is
unaffected requires careful design.

### Backwards compatibility

The `InflowNonNegativityMethod` enum will gain `Truncation` and
`TruncationWithPenalty { cost }` variants. The config parser in `cobre-io`
will accept `"truncation"` and `"truncation_with_penalty"` method strings.
Existing `"penalty"` and `"none"` configs remain unchanged.
