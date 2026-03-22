# PAR Fitting: PACF and Yule-Walker Divergence Analysis

## 1. Problem Statement

After implementing PACF-based order selection and contribution-based validation
(the `par-model-validation` plan), the convertido case shows a per-(hydro, season)
order distribution that is nearly the inverse of NEWAVE's:

| Metric                     | NEWAVE        | Cobre (PACF + contrib) |
| -------------------------- | ------------- | ---------------------- |
| Zero-order pairs           | 1,893 (93.3%) | 163 (8.6%)             |
| Non-zero-order pairs       | 135 (6.7%)    | 1,733 (91.4%)          |
| Max order used             | 4             | 6                      |
| Max coeff magnitude        | ~2            | 315                    |
| Hydros with coeff mag > 10 | 0             | 29                     |

The contribution validation catches oscillatory instability (negative
contributions) but does not prevent explosive amplification from coefficients
that remain large and positive. The root cause is not in the contribution check
itself but in the stages that precede it: the PACF computation and the
Yule-Walker system construction.

This report details three fundamental algorithmic differences between NEWAVE's
reference Python implementation (`ft-newave-2021/testes/parpa/yulewalker.py`)
and cobre's Rust implementation that explain the divergence.

---

## 2. Difference 1: PACF Computation Method

This is the single largest source of divergence.

### 2.1 NEWAVE: Periodic PACF via Progressive Matrix Solves

NEWAVE computes the periodic PACF by solving a fresh Yule-Walker system for
each candidate order. For month `p` at lag `k`:

1. Build a `(k+1) x (k+1)` matrix of periodic autocorrelations via
   `_matriz_extendida(p, k)`.
2. Solve the Yule-Walker system `R * phi = rho` via matrix inversion.
3. Extract the **last coefficient** `phi[k-1]` as `PACF(k)`.

```python
# yulewalker.py, YuleWalkerPAR.facp (lines 168-186)
def facp(self, p, maxlag, configs):
    self._atualiza_tabela_configs(configs)
    acps = []
    for o in range(1, maxlag):
        mat = self._matriz_extendida(p, o)
        acps.append(self._resolve_yw(mat, o)[o - 1])
    return np.array(acps)
```

Each call to `_matriz_extendida(p, o)` builds a matrix whose entries are
`M[i,j] = rho(p - i, j - i)`, where `rho(p, k)` is the periodic
autocorrelation that depends on the reference month `p - i`. This matrix is
symmetric but **not Toeplitz** because the autocorrelation function varies with
the reference period.

The key property: **PACF(k) is the last coefficient of an AR(k) model fitted to
the full periodic covariance structure.** Each order is an independent system
solve — order k does not reuse order k-1 results.

### 2.2 Cobre: Stationary PACF via Levinson-Durbin Reflection Coefficients

Cobre computes PACF differently:

1. Compute a single vector of autocorrelations `rho_m(1), ..., rho_m(max_order)`
   for the target season `m`.
2. Pass this vector to the Levinson-Durbin recursion.
3. Use the **reflection coefficients** (`parcor`) from L-D as PACF values.

```rust
// estimation.rs, estimate_ar_with_pacf (lines 709-730)
let autocorrelations = compute_autocorrelations(
    est.hydro_id, est.season_id, actual_order,
    n_seasons, pair_obs, &stats_map, &group_obs,
);
let ld = levinson_durbin(&autocorrelations, actual_order);
let pacf_result = select_order_pacf(&ld.parcor, n_obs, z_alpha);
```

The Levinson-Durbin recursion assumes that the input autocorrelation sequence
comes from a **stationary process**, i.e. that the covariance matrix is
Toeplitz: `R[i,j] = rho(|i - j|)`. Under this assumption, the reflection
coefficients equal the last coefficient of each progressive AR(k) fit.

### 2.3 Why This Matters

For a PAR (periodic AR) model, the covariance matrix is **not Toeplitz**. The
autocorrelation `rho(p, k)` depends on both the reference month `p` and the lag
`k`. When `p` changes (as you go down rows of the YW matrix), the
autocorrelation values change because different months have different seasonal
dynamics.

NEWAVE builds the correct periodic covariance matrix for each order:

```python
# yulewalker.py, _matriz_extendida (lines 114-132)
for i in range(ORDEM_MATRIZ):
    for j in range(i + 1, ORDEM_MATRIZ):
        m = p - i                    # reference month shifts per row
        lag = j - i
        col_p = self.periodos + m    # index into 24-month horizon
        mat[i, j] = self._autocorr(col_p, col_p - lag)
        mat[j, i] = mat[i, j]
```

Row `i` uses reference month `p - i`, so row 0 has `rho(p, *)`, row 1 has
`rho(p-1, *)`, etc. This captures the fact that the seasonal dynamics of the
lagged month differ from those of the target month.

Cobre's Levinson-Durbin implicitly assumes all rows use the same autocorrelation
function (Toeplitz property). For a model with 12 seasons where seasonal
standard deviations and autocorrelation structures differ markedly, this can
produce substantially different PACF values — and therefore different order
selections.

### 2.4 Concrete Consequence

When the Toeplitz approximation is poor (common for incremental inflow series
with heterogeneous seasonal dynamics), cobre's L-D parcor values tend to be
**larger** than NEWAVE's periodic PACF values. Larger PACF values pass the
`1.96/sqrt(N)` significance threshold more easily, leading cobre to accept
higher AR orders that NEWAVE would reject.

---

## 3. Difference 2: PAR-A Annual Component and Conditional PACF

### 3.1 NEWAVE: PAR-A Model with Annual Mean Conditioning

NEWAVE uses the PAR-A extension (Periodic AR with Annual component), which adds
the rolling 12-month mean as an extra predictor. The Yule-Walker matrix gains
an additional row and column:

```python
# yulewalker.py, YuleWalkerPARA._matriz_extendida (lines 434-460)
ORDEM_MATRIZ = lag + 2  # one extra for annual component
mat = np.ones((ORDEM_MATRIZ, ORDEM_MATRIZ))

# Standard autocorrelations in the top-left block
for i in range(ORDEM_MATRIZ - 1):
    for j in range(i + 1, ORDEM_MATRIZ - 1):
        mat[i, j] = self._autocov(col_p, col_p - lag_ij)
        mat[j, i] = mat[i, j]

# Annual component covariances in the last row/column
for i in range(ORDEM_MATRIZ - 1):
    mat[ORDEM_MATRIZ - 1, i] = self._cov_media(col_p, col_p - i)
    mat[i, ORDEM_MATRIZ - 1] = mat[ORDEM_MATRIZ - 1, i]
```

The solved system returns `lag + 1` coefficients: `[phi_1, ..., phi_lag, psi]`,
where `psi` is the annual mean coefficient. This absorbs low-frequency
persistence that would otherwise inflate the AR lag coefficients.

### 3.2 NEWAVE: Conditional PACF via Schur Complement

NEWAVE computes the PAR-A PACF using conditional partial correlations. For
lag `k`, it partitions the covariance matrix into blocks:

- `Sigma_11` (2x2): covariance between `z(t)` and `z(t-k)`
- `Sigma_22`: covariance among intermediate lags `z(t-1)..z(t-k+1)` AND the
  annual mean `A(t)`
- `Sigma_12`: cross-covariance

Then computes:

```python
# yulewalker.py, YuleWalkerPARA.facp (lines 497-558)
cond = sig11 - sig12 @ np.linalg.inv(sig22) @ sig12.T
phi = cond[0, 1] / np.sqrt(cond[0, 0] * cond[1, 1])
```

This is the **partial correlation conditional on both intermediate lags and the
annual mean**. By conditioning on `A(t)`, the PACF measures only the residual
temporal dependence after the annual pattern is removed. For incremental inflow
series that have a strong annual cycle, this conditioning drastically reduces the
apparent partial correlation at most lags.

### 3.3 Cobre: No Annual Component

Cobre implements a pure PAR model with no annual component. All temporal
persistence — including low-frequency annual patterns — must be captured
entirely by the AR lag coefficients. This has two consequences:

1. **Inflated coefficients**: The AR coefficients absorb the annual signal,
   producing magnitudes of 40-300 instead of 0.5-2.0.
2. **Higher selected orders**: Without conditioning on the annual mean, the PACF
   values are larger, so more lags pass the significance threshold.

### 3.4 Annual Component in Contributions

NEWAVE distributes the annual effect uniformly across all months in the
contribution recursion:

```python
# yulewalker.py, YuleWalkerPARA.contribuicoes (lines 590-653)
for j in range(max_lag):
    matriz_aux[0, j] = fis[p][j] + psis[p] / 12
```

Without this term, cobre's contribution analysis operates on the raw (inflated)
AR coefficients, which may produce large positive contributions that pass the
negativity check but represent physically unrealistic amplification factors.

---

## 4. Difference 3: Autocorrelation Computation Details

### 4.1 Normalization and Divisor

| Aspect               | NEWAVE                                   | Cobre                                   |
| -------------------- | ---------------------------------------- | --------------------------------------- |
| Signal normalization | Pre-normalize: `z = (x - mu) / sigma`    | Raw values with inline mean subtraction |
| Covariance divisor   | `1/N` (population)                       | `1/(N-1)` (Bessel correction)           |
| Cross-year lag       | Drop 1 observation when lag crosses year | No adjustment (align by min count)      |

### 4.2 NEWAVE's 24-Month Horizon

NEWAVE creates a 24-month signal array (`2 * periodos`) that duplicates the
12-month seasonal structure across two years. This enables proper handling of
lags that cross the year boundary:

```python
# When the lag wraps around a year boundary, drop one observation
if lag < self.periodos and p >= self.periodos:
    sinal_m = sinal_m[1:]
    sinal_lag = sinal_lag[:-1]
```

Cobre's cross-seasonal autocorrelation aligns observations by index position
(`min(N_current, N_lag)` pairs) without this year-boundary adjustment. With
N ~= 96, the effect is small (one observation difference) but compounds across
multiple lags and seasons.

### 4.3 Impact

The 1/N vs 1/(N-1) difference alone is ~1% for N=96 and unlikely to explain the
order selection divergence. The year-boundary adjustment is similarly minor. These
differences are secondary to the PACF method and annual component issues above.

---

## 5. Summary: Cascade of Effects

The three differences compound in a specific cascade:

```
No annual component (PAR vs PAR-A)
  -> Low-frequency persistence leaks into AR lags
  -> AR coefficients are inflated (|phi| >> 1)
  -> Autocorrelation sequence has spurious high-lag structure

Stationary PACF (L-D parcor vs periodic matrix solve)
  -> Toeplitz approximation overestimates PACF at higher lags
  -> More lags pass significance threshold
  -> Higher orders selected

High orders with inflated coefficients
  -> Contribution check catches negative contributions (oscillation)
  -> But does NOT catch large positive contributions (amplification)
  -> Explosive coefficients survive validation
```

---

## 6. Proposed Improvement Path

### 6.1 Phase 1: Proper Periodic Yule-Walker System (High Impact)

Replace cobre's approach of computing a single autocorrelation vector and
passing it to Levinson-Durbin with NEWAVE's approach of building and solving
a progressive series of periodic Yule-Walker matrices.

**For each candidate order `k` and season `m`:**

1. Build a `(k+1) x (k+1)` matrix where `M[i,j] = rho(m - i, j - i)`,
   using the periodic autocorrelation function that shifts the reference month
   per row.
2. Solve `R * phi = rho` via matrix inversion (the matrices are small — at most
   7x7 for max_order=6).
3. The last coefficient `phi[k-1]` is the periodic PACF at lag k.

This replaces `levinson_durbin` with explicit matrix solves for order selection.
The L-D recursion can still be used for the final coefficient estimation at the
selected order, but the PACF must come from the periodic matrix approach.

**Implementation scope:**

- New function `periodic_pacf(season, max_order, autocorr_fn)` in
  `cobre-stochastic/src/par/fitting.rs`
- New function `periodic_yw_matrix(season, order, autocorr_fn)` that builds the
  matrix with row-dependent reference months
- Update `estimate_ar_with_pacf` in `estimation.rs` to use `periodic_pacf`
  instead of `ld.parcor`
- The final AR coefficients at the selected order come from solving the YW
  system at that specific order (not from Levinson-Durbin)

**Expected impact:** The periodic PACF will produce smaller values at higher
lags for incremental inflow series, leading to more conservative order selection
(closer to NEWAVE's distribution).

### 6.2 Phase 2: PAR-A Annual Component (High Impact)

Add the annual mean component `psi` to the PAR model, matching NEWAVE's
PAR-A formulation.

**Sub-steps:**

a. **Rolling annual mean computation**: For each (hydro, season), compute the
rolling 12-month average `A(t)` from historical observations. Normalize it
separately (mean and std).

b. **Extended YW matrix**: Add one row and one column to the periodic YW matrix
for the covariance between `z(t-lag)` and `A(t)`. The matrix becomes
`(k+2) x (k+2)` for order k.

c. **Conditional PACF**: Compute PACF via the Schur complement (conditional
correlation), conditioning on both intermediate lags and the annual mean:
`PACF(k) = Sigma_cond[0,1] / sqrt(Sigma_cond[0,0] * Sigma_cond[1,1])`

d. **Coefficient estimation**: The solved YW system returns `[phi_1, ..., phi_k, psi]`.
Store `psi` alongside the AR coefficients in the inflow model.

e. **Integration into LP**: The `PrecomputedPar` structure and LP builder need
to account for the annual component in the deterministic base and noise
computation.

f. **Contribution analysis**: Add `psi/12` to each lag's contribution in the
recursive composition, matching NEWAVE's formula.

**Expected impact:** The annual conditioning will absorb low-frequency
persistence, dramatically reducing AR coefficient magnitudes and causing
most (hydro, season) pairs to fall below the PACF significance threshold
at all lags — approaching NEWAVE's 93% zero-order rate.

### 6.3 Phase 3: Autocorrelation Refinements (Low Impact)

Align minor details with NEWAVE's autocorrelation computation:

- Switch from Bessel correction (1/(N-1)) to population divisor (1/N) in the
  autocorrelation computation
- Add cross-year lag adjustment: drop one observation when the lag crosses
  the year boundary

These are minor corrections that will improve numerical agreement with NEWAVE
but are unlikely to change order selection outcomes on their own.

### 6.4 Implementation Priority

Phase 1 and Phase 2 are both high-impact and somewhat independent. Phase 1 can
be implemented first because it only changes the PACF computation and order
selection — the coefficient estimation and LP integration remain unchanged.
Phase 2 is more invasive because it adds a new model parameter (`psi`) that
propagates through the LP construction.

**Recommended order:**

1. Phase 1 (periodic PACF) — moderate scope, high impact on order selection
2. Phase 2 (PAR-A annual component) — larger scope, highest impact on
   coefficient magnitudes
3. Phase 3 (autocorrelation details) — small scope, polish for numerical
   agreement

After Phase 1, re-run the convertido case to measure the improvement. If the
order distribution is substantially closer to NEWAVE's, Phase 2 can be scoped
more carefully. If Phase 1 alone is insufficient (likely, given the annual
component's large effect on coefficient magnitudes), Phase 2 becomes urgent.

---

## 7. Validation Strategy

After each phase, compare against the convertido case baseline:

1. **Per-(hydro, season) order distribution** — target: significantly more
   zero-order pairs (currently 8.6%, target >50% after Phase 1, >90% after
   Phase 2)
2. **Max |coefficient|** — target: < 5 after Phase 2 (currently 315)
3. **Simulation inflow nonnegativity slack** — target: near zero (currently
   3.5M m3/s reported in the original investigation)
4. **Training convergence** — lower bound and gap should remain reasonable
5. **Comparison with NEWAVE's `parpvaz.dat`** — per-hydro order comparison
   for the convertido case
