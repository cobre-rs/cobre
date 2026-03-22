# PAR Model Fitting for Incremental Inflows: Investigation Report

## Executive Summary

Converting NEWAVE inflow data to cobre revealed a critical problem: cobre's AIC-based PAR model fitting produces **explosive AR coefficients** (up to |coeff| = 535) for incremental inflow series, causing wildly oscillating simulated inflows, massive nonnegativity slack usage, and unrealistic system behavior (spillage, excess energy, curtailment). This report details the investigation, root causes, and the approach needed in cobre to fix this.

## Problem Description

When running cobre with the converted NEWAVE case, the simulation results showed:

- **3.5M m3/s of inflow nonnegativity slack** per scenario (the penalty mechanism absorbing negative generated inflows)
- **83K m3/s of spillage** concentrated in a few hydros
- **80K MW of excess energy** and **312K MW of NCS curtailment**
- **97 out of 158 hydros** triggering the nonnegativity penalty
- **Zero inflow penalty cost** visible in the cost breakdown despite massive slack usage
- Spot prices near zero despite realistic load levels

## Root Cause: Explosive AR Coefficients

Cobre's stochastic fitting pipeline uses AIC (Akaike Information Criterion) to select PAR model orders, with a maximum order of 6. For the converted case, this produced:

| Max   | AR coefficient |     | Number of hydros |
| ----- | -------------- | --- | ---------------- |
| > 5   | 72 out of 154  |
| > 10  | 44             |
| > 20  | 24             |
| > 100 | 5              |

The worst case: **hydro 75 with |coeff| = 535.7**. An AR coefficient of 535 means a deviation of 1 m3/s from the mean in the previous month gets amplified to 535 m3/s in the next — an explosive, physically meaningless model.

### Illustrative Example: PIMENTAL (hydro 156)

PIMENTAL is a headwater plant with positive incremental inflows (mean ~2,424 m3/s, no negatives in 96 years of history). Despite this benign data:

- **Cobre fitted**: AR(6) for March with coefficients `[0.97, -0.48, 0.09, 0.002, 0.13, -0.14]`, AR(2) for August with coefficient `48.9`
- **NEWAVE fitted**: AR(0) for 11 out of 12 months (only April has AR(1) with phi=0.0)
- **In simulation**: Cobre generated average inflow of **-612 m3/s** (negative!) for a plant whose history has minimum inflow = 0, causing 78K m3/s of spillage

## NEWAVE's Approach: What parpvaz.dat Reveals

### Order Selection Statistics

Analyzing the parpvaz.dat output for the same case:

| Metric                              | NEWAVE                         | Cobre                                            |
| ----------------------------------- | ------------------------------ | ------------------------------------------------ |
| Non-zero order (month, plant) pairs | **135**                        | **1,810**                                        |
| Zero-order pairs                    | **1,893**                      | **218**                                          |
| Max order used                      | **4**                          | **6**                                            |
| Order distribution                  | {0: 1893, 1: 67, 2: 44, 4: 24} | {1: 235, 2: 191, 3: 266, 4: 234, 5: 295, 6: 589} |

NEWAVE assigns order 0 to **93.3%** of all (month, plant) pairs. Cobre assigns a non-zero order to **89.3%**. The approaches are almost exactly inverted.

### NEWAVE's Three-Stage Validation Pipeline

NEWAVE's VORDEM routine implements a rigorous validation pipeline that goes far beyond simple information criteria. By studying both the parpvaz.dat output and the Python reimplementation in `ft-newave-2021/testes/parpa/yulewalker.py`, we identified three stages:

#### Stage 1: FACP-Based Initial Order Selection

NEWAVE uses the **Partial Autocorrelation Function** (FACP), not AIC, to select initial orders. The FACP at lag k is defined as the last coefficient of the AR(k) model fitted via Yule-Walker equations. This isolates the marginal contribution of lag k after removing effects of lags 1 to k-1.

The selection criterion is a **significance test**: select the maximum lag k where |FACP(k)| exceeds the 95% confidence interval `1.96 / sqrt(n_samples)`. With 96 years of history, this threshold is approximately 0.20.

For incremental inflows (PAR-A model), the FACP computation uses **conditional partial autocorrelation** that conditions on the annual mean component. This accounts for the fact that incremental inflows have an annual structure that must be factored out before assessing lag-by-lag temporal dependence.

#### Stage 2: Contribution Analysis (the key innovation)

After fitting the AR model, NEWAVE computes **contributions** — a denormalized, recursively-composed measure of how each lag influences the current period. This is the critical step that prevents explosive models.

The contribution of lag k for month p is computed as:

```
fi(p, k) = phi(p, k) * std(Z_p) / std(Z_{p-k})
```

where `phi(p, k)` is the AR coefficient for month p at lag k, and `std(Z_p)` is the standard deviation of the inflow series in month p. This denormalization converts the dimensionless AR coefficient into a physically meaningful transfer factor in the same units as the ratio of standard deviations.

For the PAR-A model (incremental inflows), the annual component `psi` is distributed across lags:

```
fi_extended(p, k) = fi(p, k) + psi(p) / 12
```

The contributions are then **recursively composed** across months. A matrix `A` is built where:

```
A(0, j) = fi(p, j)                    [direct effects]
A(i, j) = A(i-1, 0) * fi(p-i, j) + A(i-1, j+1)   [recursive composition]
```

The contribution of lag k for month p is `A(k-1, 0)`. This captures not just the direct effect of lag k, but also the **indirect effects** that propagate through the periodic chain of months. A coefficient that looks small in isolation can produce a large cumulative contribution when amplified through the monthly chain.

#### Stage 3: Iterative Order Reduction

If **any contribution within the selected order is negative**, the model is rejected and the maximum allowed order for that month is reduced by 1. The entire process (FACP-based order selection, Yule-Walker estimation, contribution computation, negativity check) is then repeated with the reduced constraint.

This loop continues until no negative contributions remain. The process can reduce a month's order all the way to 0, meaning the generated inflow for that month is simply a draw from the seasonal distribution with no autoregressive dependence.

### Why Negative Contributions Are Rejected

The contribution of lag k represents the expected change in current inflow per unit change in the inflow k months ago, accounting for the full periodic cascade. A negative contribution means: "if more water flowed k months ago, we expect less water now." While this might occur statistically in noisy data, it violates the physical structure of the hydrological system:

1. **Hydrological persistence**: Wet periods tend to be followed by wet periods; dry by dry. Negative contributions contradict this basic property.
2. **Numerical stability**: Alternating signs in the recursive composition amplify small perturbations exponentially, leading to the explosive behavior we observed.
3. **Water balance coherence**: In the incremental inflow formulation, negative contributions can cause the stochastic model to generate inflows more negative than the historical minimum, creating fictitious water deficits.

## Why Cobre's AIC Approach Fails for Incremental Inflows

### The Fundamental Mismatch

AIC balances model fit against complexity (number of parameters). For incremental inflow series, which are the **difference of two highly correlated natural inflow series**, the resulting signal has:

- **Much higher coefficient of variation** (std/mean) than natural inflows
- **Weak or noisy autocorrelation structure** — the persistence signal is largely cancelled by the subtraction
- **Spurious partial autocorrelations** at higher lags that pass AIC's penalty but fail physical validation

AIC sees these spurious correlations as "information worth modeling" and selects high orders with large coefficients. The fitted model then generates noise that the historical data never exhibited.

### The Contribution Check Catches What AIC Misses

A coefficient of `phi = 48.9` at lag 2 for August (as cobre fitted for PIMENTAL) would pass AIC because it marginally improves the in-sample fit. But the contribution analysis reveals that this coefficient, when composed through the periodic chain, creates an explosive feedback loop. NEWAVE's contribution check catches this and reduces the order to 0.

## What Cobre Needs

### Core Requirement: Contribution-Based Order Validation

Cobre's PAR fitting pipeline must implement a contribution analysis and iterative order reduction step after the initial order selection (whether by AIC, FACP significance, or any other criterion). Specifically:

1. **Denormalize AR coefficients** into contributions using seasonal standard deviations
2. **Compose contributions recursively** through the periodic monthly chain
3. **Check for negative contributions** within the selected order
4. **Reduce the maximum allowed order** for months with negative contributions
5. **Repeat** until convergence (no negative contributions remain)

### Additional Considerations

- **For PAR-A (annual component)**: The contribution calculation must include the annual term `psi / 12` distributed across all lags, as in the Python reference implementation.
- **Order 0 must be supported**: A month with order 0 means no autoregressive component — the generated inflow is drawn from the seasonal distribution (mean + std \* noise) without temporal dependence. This is the correct behavior when no stable AR model can be fitted.
- **The FACP significance test is preferable to AIC**: NEWAVE's approach of testing |FACP(k)| against a confidence interval is more conservative and physically motivated than AIC. It directly tests whether each additional lag has a statistically significant marginal contribution, rather than optimizing a global information criterion.
- **Coefficient magnitude bounds**: Even without the full contribution analysis, capping |AR coefficients| at some reasonable threshold (e.g., 2.0 or 3.0) would prevent the worst explosive behavior as a safety net.

### Why Not Just Copy NEWAVE

While NEWAVE's approach works, it is overly conservative — assigning order 0 to 93% of (month, plant) pairs likely discards real temporal structure for many hydros. Cobre can improve on this by:

1. Using the contribution check as a **validation gate** rather than the sole selection criterion
2. Allowing AIC or FACP to propose candidate orders, then validating with contributions
3. If a candidate order fails, trying lower orders before falling back to 0
4. Providing diagnostic output (contribution values, rejected orders) for model quality assessment

This preserves the temporal correlation structure where it's genuinely present while preventing the explosive behavior that arises from overfitting.

## Action Items

1. **Implement contribution analysis** in cobre's PAR fitting module (the recursive composition algorithm is ~40 lines of code, well-defined mathematically)
2. **Add iterative order reduction** that reduces max_order for months with negative contributions and re-fits
3. **Support order 0** as a valid outcome (deterministic seasonal mean + noise)
4. **Consider replacing AIC with FACP significance** as the initial order selection criterion, using contribution analysis as the validation step
5. **Add diagnostic output** to the fitting report showing contribution values and any order reductions applied
