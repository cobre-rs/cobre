//! FPHA hyperplane fitting from reservoir geometry.
//!
//! This module implements the building blocks for fitting FPHA (Forebay-Height
//! Production Approximation) hyperplanes from a Volume-Height-Area (VHA) curve.
//! It evaluates the hydro production function `phi(v, q, s)` at grid points
//! and fits a piecewise-linear outer approximation.
//!
//! # Structure
//!
//! - [`FphaFittingError`] — validation errors for geometry table construction.
//! - [`ForebayTable`] — linear interpolation table for forebay height `h_fore(v)`
//!   and its derivative `dh_fore/dv` from VHA curve data.
//!
//! # Note on `dead_code`
//!
//! This module is scaffolded as part of Epic 01 of the FPHA fitting pipeline.
//! The types defined here are consumed by later tickets in the epic. The
//! `dead_code` allowance is intentional and will be removed when callers are
//! added in subsequent tickets.
#![allow(dead_code)]

use cobre_core::{EfficiencyModel, HydraulicLossesModel, Hydro, TailraceModel};
use cobre_io::extensions::{FphaConfig, HydroGeometryRow};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that arise during FPHA fitting geometry validation or evaluation.
///
/// Returned by [`ForebayTable::new`] when the supplied VHA curve data does not
/// satisfy the invariants required for linear interpolation.
#[derive(Debug)]
pub(crate) enum FphaFittingError {
    /// Fewer than 2 VHA curve points were provided for the named hydro plant.
    ///
    /// Linear interpolation requires at least 2 breakpoints. A single point
    /// defines only a trivial (constant) function and cannot represent the
    /// full volume-height relationship.
    InsufficientPoints {
        /// Name of the hydro plant whose curve was rejected.
        hydro_name: String,
        /// Number of points actually provided.
        count: usize,
    },

    /// The `volume_hm3` values are not strictly increasing between consecutive rows.
    ///
    /// Strict monotonicity is required so that each volume maps to a unique
    /// interpolation interval. Duplicate volumes produce a zero-length segment
    /// and undefined derivatives.
    NonMonotonicVolume {
        /// Name of the hydro plant whose curve was rejected.
        hydro_name: String,
        /// Zero-based index of the row whose volume is not strictly greater than
        /// the previous row's volume.
        index: usize,
        /// Volume at the previous row (hm³).
        v_prev: f64,
        /// Volume at the current row (hm³), which must satisfy `v_curr > v_prev`.
        v_curr: f64,
    },

    /// The `height_m` values decrease between consecutive rows.
    ///
    /// Heights must be monotonically non-decreasing because greater reservoir
    /// volume always corresponds to a higher or equal water surface elevation.
    NonMonotonicHeight {
        /// Name of the hydro plant whose curve was rejected.
        hydro_name: String,
        /// Zero-based index of the row whose height is strictly less than the
        /// previous row's height.
        index: usize,
        /// Height at the previous row (m).
        h_prev: f64,
        /// Height at the current row (m), which must satisfy `h_curr >= h_prev`.
        h_curr: f64,
    },

    /// Both absolute and percentile bounds were specified for the same dimension.
    ///
    /// `volume_min_hm3` and `volume_min_percentile` are mutually exclusive, as
    /// are `volume_max_hm3` and `volume_max_percentile`. Setting both for the
    /// same bound is ambiguous and is always rejected.
    ConflictingFittingWindow {
        /// Name of the hydro plant whose configuration was rejected.
        hydro_name: String,
        /// Human-readable description of the conflict.
        detail: String,
    },

    /// The resolved volume range is empty (`v_min >= v_max`).
    ///
    /// After applying the fitting window configuration, the lower bound was
    /// not strictly less than the upper bound. This can happen when absolute
    /// bounds are inverted, when percentile bounds yield a zero-width range,
    /// or when clamping collapses the window to a single point.
    EmptyFittingWindow {
        /// Name of the hydro plant whose configuration was rejected.
        hydro_name: String,
        /// Resolved lower bound (hm³).
        v_min: f64,
        /// Resolved upper bound (hm³).
        v_max: f64,
    },

    /// A discretization count was too small to define a valid grid interval.
    ///
    /// All three dimension counts (`n_volume_points`, `n_flow_points`,
    /// `n_spillage_points`) must be >= 2. `max_planes_per_hydro` must be >= 1.
    InsufficientDiscretization {
        /// Name of the hydro plant whose configuration was rejected.
        hydro_name: String,
        /// Which dimension was too small: `"volume"`, `"turbine"`, `"spillage"`,
        /// or `"max_planes_per_hydro"`.
        dimension: String,
        /// The value that was provided (< 2 for grid dimensions, < 1 for max planes).
        value: usize,
    },
}

impl std::fmt::Display for FphaFittingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientPoints { hydro_name, count } => write!(
                f,
                "hydro '{hydro_name}': VHA curve has {count} point(s); \
                 at least 2 are required for interpolation"
            ),
            Self::NonMonotonicVolume {
                hydro_name,
                index,
                v_prev,
                v_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': volume is not strictly increasing at index {index}: \
                 v[{index}]={v_curr} is not greater than v[{}]={v_prev}",
                index - 1
            ),
            Self::NonMonotonicHeight {
                hydro_name,
                index,
                h_prev,
                h_curr,
            } => write!(
                f,
                "hydro '{hydro_name}': height decreases at index {index}: \
                 h[{index}]={h_curr} < h[{}]={h_prev}",
                index - 1
            ),
            Self::ConflictingFittingWindow { hydro_name, detail } => write!(
                f,
                "hydro '{hydro_name}': conflicting fitting window configuration: {detail}"
            ),
            Self::EmptyFittingWindow {
                hydro_name,
                v_min,
                v_max,
            } => write!(
                f,
                "hydro '{hydro_name}': fitting window is empty after resolution: \
                 v_min={v_min} >= v_max={v_max}"
            ),
            Self::InsufficientDiscretization {
                hydro_name,
                dimension,
                value,
            } => write!(
                f,
                "hydro '{hydro_name}': discretization count for '{dimension}' is {value}, \
                 which is below the minimum required"
            ),
        }
    }
}

impl std::error::Error for FphaFittingError {}

// ── FittingBounds ─────────────────────────────────────────────────────────────

/// Resolved volume range and discretization counts for the FPHA fitting grid.
///
/// Produced by [`resolve_fitting_bounds`] from an [`FphaConfig`] and the hydro
/// plant entity. Consumed by the grid construction step in Epic 02.
#[derive(Debug)]
pub(crate) struct FittingBounds {
    /// Resolved lower bound of the fitting volume range (hm³).
    pub v_min: f64,
    /// Resolved upper bound of the fitting volume range (hm³).
    pub v_max: f64,
    /// Number of volume grid points (>= 2).
    pub n_volume_points: usize,
    /// Number of turbined-flow grid points (>= 2).
    pub n_flow_points: usize,
    /// Number of spillage grid points (>= 2).
    pub n_spillage_points: usize,
    /// Maximum number of hyperplanes retained after heuristic selection (>= 1).
    pub max_planes_per_hydro: usize,
}

/// Resolve the fitting volume range and discretization counts from configuration.
///
/// Combines the [`FphaConfig`] fitting window (if any), the hydro entity's
/// operating limits, and the forebay table's interpolation range to produce
/// a concrete [`FittingBounds`] for grid construction.
///
/// # Volume range resolution
///
/// The volume range is resolved in three mutually exclusive modes:
///
/// 1. **No fitting window** (`config.fitting_window` is `None`): use the full
///    forebay table range `[forebay.v_min(), forebay.v_max()]`.
/// 2. **Absolute bounds**: use `volume_min_hm3` / `volume_max_hm3` directly,
///    clamping to the forebay table range.
/// 3. **Percentile bounds**: compute
///    `v = entity_v_min + p * (entity_v_max - entity_v_min)` for each bound,
///    then clamp to the forebay table range.
///
/// Mixed modes (absolute min, percentile max or vice versa) are accepted as long
/// as neither the min bound nor the max bound has both absolute and percentile set.
///
/// # Errors
///
/// | Condition | Error variant |
/// |-----------|---------------|
/// | Both absolute and percentile set for the same bound (min or max) | [`FphaFittingError::ConflictingFittingWindow`] |
/// | Resolved `v_min >= v_max` | [`FphaFittingError::EmptyFittingWindow`] |
/// | Any discretization count < 2, or `max_planes_per_hydro` < 1 | [`FphaFittingError::InsufficientDiscretization`] |
pub(crate) fn resolve_fitting_bounds(
    config: &FphaConfig,
    hydro: &Hydro,
    forebay: &ForebayTable,
) -> Result<FittingBounds, FphaFittingError> {
    let hydro_name = &hydro.name;

    // ── Step 1: Resolve volume range ─────────────────────────────────────────

    let (v_min, v_max) = match &config.fitting_window {
        None => (forebay.v_min(), forebay.v_max()),
        Some(fw) => {
            // Check for conflicts on the min bound.
            if fw.volume_min_hm3.is_some() && fw.volume_min_percentile.is_some() {
                return Err(FphaFittingError::ConflictingFittingWindow {
                    hydro_name: hydro_name.clone(),
                    detail: "volume_min_hm3 and volume_min_percentile cannot both be set; \
                             use absolute bounds OR percentiles, not both for the same bound"
                        .to_owned(),
                });
            }
            // Check for conflicts on the max bound.
            if fw.volume_max_hm3.is_some() && fw.volume_max_percentile.is_some() {
                return Err(FphaFittingError::ConflictingFittingWindow {
                    hydro_name: hydro_name.clone(),
                    detail: "volume_max_hm3 and volume_max_percentile cannot both be set; \
                             use absolute bounds OR percentiles, not both for the same bound"
                        .to_owned(),
                });
            }

            let entity_v_min = hydro.min_storage_hm3;
            let entity_v_max = hydro.max_storage_hm3;
            let entity_range = entity_v_max - entity_v_min;

            // Resolve lower bound.
            let v_min_raw = if let Some(abs) = fw.volume_min_hm3 {
                abs
            } else if let Some(pct) = fw.volume_min_percentile {
                entity_v_min + pct * entity_range
            } else {
                forebay.v_min()
            };

            // Resolve upper bound.
            let v_max_raw = if let Some(abs) = fw.volume_max_hm3 {
                abs
            } else if let Some(pct) = fw.volume_max_percentile {
                entity_v_min + pct * entity_range
            } else {
                forebay.v_max()
            };

            // Clamp to forebay table range.
            let v_min = v_min_raw.clamp(forebay.v_min(), forebay.v_max());
            let v_max = v_max_raw.clamp(forebay.v_min(), forebay.v_max());

            (v_min, v_max)
        }
    };

    // ── Step 2: Validate volume range ────────────────────────────────────────

    if v_min >= v_max {
        return Err(FphaFittingError::EmptyFittingWindow {
            hydro_name: hydro_name.clone(),
            v_min,
            v_max,
        });
    }

    // ── Step 3: Resolve discretization counts ────────────────────────────────

    #[allow(clippy::cast_sign_loss)]
    let n_volume_points = config.volume_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let n_flow_points = config.turbine_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let n_spillage_points = config.spillage_discretization_points.unwrap_or(5) as usize;
    #[allow(clippy::cast_sign_loss)]
    let max_planes = config.max_planes_per_hydro.unwrap_or(10) as usize;

    if n_volume_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "volume".to_owned(),
            value: n_volume_points,
        });
    }
    if n_flow_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "turbine".to_owned(),
            value: n_flow_points,
        });
    }
    if n_spillage_points < 2 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "spillage".to_owned(),
            value: n_spillage_points,
        });
    }
    if max_planes < 1 {
        return Err(FphaFittingError::InsufficientDiscretization {
            hydro_name: hydro_name.clone(),
            dimension: "max_planes_per_hydro".to_owned(),
            value: max_planes,
        });
    }

    Ok(FittingBounds {
        v_min,
        v_max,
        n_volume_points,
        n_flow_points,
        n_spillage_points,
        max_planes_per_hydro: max_planes,
    })
}

// ── ForebayTable ──────────────────────────────────────────────────────────────

/// Linear interpolation table for forebay height `h_fore(v)`.
///
/// Stores the VHA curve for a single hydro plant as two parallel sorted vectors
/// of volume breakpoints (`volumes`, hm³) and corresponding surface elevations
/// (`heights`, m). All queries are clamped to `[v_min, v_max]`, so the table
/// never extrapolates and every method is infallible after construction.
///
/// # Construction
///
/// Build from a slice of [`HydroGeometryRow`] values (all rows for one hydro,
/// already sorted by ascending `volume_hm3` by the parser):
///
/// ```no_run
/// use cobre_io::extensions::HydroGeometryRow;
/// use cobre_core::EntityId;
///
/// // (ForebayTable is pub(crate); this example is for illustration only.)
/// let rows = vec![
///     HydroGeometryRow { hydro_id: EntityId::from(1), volume_hm3: 0.0,    height_m: 386.5, area_km2: 2.5 },
///     HydroGeometryRow { hydro_id: EntityId::from(1), volume_hm3: 2000.0, height_m: 390.0, area_km2: 3.1 },
/// ];
/// ```
#[derive(Debug, Clone)]
pub(crate) struct ForebayTable {
    /// Volume breakpoints (hm³), strictly increasing.
    volumes: Vec<f64>,
    /// Surface elevation breakpoints (m), monotonically non-decreasing.
    heights: Vec<f64>,
}

impl ForebayTable {
    /// Build a [`ForebayTable`] from a slice of VHA curve rows for one hydro plant.
    ///
    /// # Parameters
    ///
    /// - `rows` — all [`HydroGeometryRow`] entries for the hydro plant, sorted
    ///   by ascending `volume_hm3` (as returned by `cobre_io::extensions::parse_hydro_geometry`).
    /// - `hydro_name` — human-readable plant name used in error messages.
    ///
    /// # Errors
    ///
    /// | Condition | Error variant |
    /// |-----------|---------------|
    /// | Fewer than 2 rows | [`FphaFittingError::InsufficientPoints`] |
    /// | `volume_hm3` not strictly increasing | [`FphaFittingError::NonMonotonicVolume`] |
    /// | `height_m` decreasing | [`FphaFittingError::NonMonotonicHeight`] |
    pub(crate) fn new(
        rows: &[HydroGeometryRow],
        hydro_name: &str,
    ) -> Result<Self, FphaFittingError> {
        // Validate minimum point count.
        if rows.len() < 2 {
            return Err(FphaFittingError::InsufficientPoints {
                hydro_name: hydro_name.to_owned(),
                count: rows.len(),
            });
        }

        let mut volumes = Vec::with_capacity(rows.len());
        let mut heights = Vec::with_capacity(rows.len());

        volumes.push(rows[0].volume_hm3);
        heights.push(rows[0].height_m);

        for i in 1..rows.len() {
            let v_prev = rows[i - 1].volume_hm3;
            let v_curr = rows[i].volume_hm3;
            let h_prev = rows[i - 1].height_m;
            let h_curr = rows[i].height_m;

            // Volumes must be strictly increasing.
            if v_curr <= v_prev {
                return Err(FphaFittingError::NonMonotonicVolume {
                    hydro_name: hydro_name.to_owned(),
                    index: i,
                    v_prev,
                    v_curr,
                });
            }

            // Heights must be non-decreasing.
            if h_curr < h_prev {
                return Err(FphaFittingError::NonMonotonicHeight {
                    hydro_name: hydro_name.to_owned(),
                    index: i,
                    h_prev,
                    h_curr,
                });
            }

            volumes.push(v_curr);
            heights.push(h_curr);
        }

        Ok(Self { volumes, heights })
    }

    /// Minimum volume in the table (hm³).
    #[inline]
    pub(crate) fn v_min(&self) -> f64 {
        // INVARIANT: `volumes` has at least 2 elements (enforced by `new`).
        self.volumes[0]
    }

    /// Maximum volume in the table (hm³).
    #[inline]
    pub(crate) fn v_max(&self) -> f64 {
        // INVARIANT: `volumes` has at least 2 elements (enforced by `new`).
        self.volumes[self.volumes.len() - 1]
    }

    /// Interpolated forebay surface elevation at `volume_hm3` (m).
    ///
    /// The query volume is clamped to `[v_min, v_max]` before interpolation,
    /// so this method is infallible and never extrapolates. Values at exact
    /// breakpoints are returned without rounding error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cobre_io::extensions::HydroGeometryRow;
    /// use cobre_core::EntityId;
    ///
    /// // (ForebayTable is pub(crate); this example is for illustration only.)
    /// // let table = ForebayTable::new(&rows, "Sobradinho").unwrap();
    /// // assert!((table.height(1000.0) - 388.25).abs() < 1e-10);
    /// ```
    pub(crate) fn height(&self, volume_hm3: f64) -> f64 {
        let v = volume_hm3.clamp(self.v_min(), self.v_max());
        let (i, t) = self.locate(v);
        self.heights[i] + t * (self.heights[i + 1] - self.heights[i])
    }

    /// Derivative of forebay height with respect to volume at `volume_hm3` (m/hm³).
    ///
    /// Returns the slope `(h[i+1] - h[i]) / (v[i+1] - v[i])` of the piecewise-linear
    /// segment that contains the query volume. The query is clamped to `[v_min, v_max]`
    /// before lookup:
    ///
    /// - At interior breakpoints the right-segment slope is returned.
    /// - At `v_max` the last-segment slope is returned.
    /// - Queries below `v_min` return the first-segment slope.
    /// - Queries above `v_max` return the last-segment slope.
    pub(crate) fn height_derivative(&self, volume_hm3: f64) -> f64 {
        let v = volume_hm3.clamp(self.v_min(), self.v_max());
        let (i, _) = self.locate(v);
        (self.heights[i + 1] - self.heights[i]) / (self.volumes[i + 1] - self.volumes[i])
    }

    /// Find the segment index `i` and the fractional position `t` within it.
    ///
    /// Returns `(i, t)` such that:
    /// - `self.volumes[i] <= v <= self.volumes[i + 1]`
    /// - `t = (v - self.volumes[i]) / (self.volumes[i + 1] - self.volumes[i])`
    ///
    /// The caller must ensure `v` is already clamped to `[v_min, v_max]`.
    ///
    /// Uses `partition_point` (binary search) for O(log n) lookup.
    ///
    /// At `v_max` the last segment is returned (index `n - 2`) to avoid an
    /// out-of-bounds access on `i + 1`.
    fn locate(&self, v: f64) -> (usize, f64) {
        let n = self.volumes.len();
        // `partition_point` returns the first index where `volumes[idx] > v`.
        // Subtract 1 to get the left bracket. Saturate at `n - 2` so that
        // `i + 1` is always a valid index (handles the v == v_max case).
        let idx = self.volumes.partition_point(|&vk| vk <= v);
        let i = idx.saturating_sub(1).min(n - 2);
        let dv = self.volumes[i + 1] - self.volumes[i];
        let t = (v - self.volumes[i]) / dv;
        (i, t)
    }
}

// ── Tailrace and hydraulic loss evaluation ────────────────────────────────────

/// Tailrace elevation `h_tail(q_out)` for a total outflow of `outflow_m3s` (m).
///
/// - `Polynomial`: evaluates `c[0] + c[1]·q + c[2]·q² + …` via Horner's method.
/// - `Piecewise`: linearly interpolates between adjacent [`cobre_core::TailracePoint`]
///   breakpoints; the outflow is clamped to the table's range before lookup.
///
/// The function is infallible — the model invariants (≥ 1 coefficient; ≥ 2 points
/// sorted ascending) are enforced by the `cobre-io` parsing layer.
pub(crate) fn evaluate_tailrace(model: &TailraceModel, outflow_m3s: f64) -> f64 {
    match model {
        TailraceModel::Polynomial { coefficients } => {
            // Horner's method: evaluate from the highest-degree coefficient down.
            // For an empty slice (should not happen after IO validation) return 0.
            coefficients
                .iter()
                .rev()
                .fold(0.0_f64, |acc, c| acc * outflow_m3s + c)
        }
        TailraceModel::Piecewise { points } => {
            let n = points.len();
            if n == 0 {
                return 0.0;
            }
            if n == 1 {
                return points[0].height_m;
            }
            // Clamp outflow to [q_min, q_max].
            let q = outflow_m3s.clamp(points[0].outflow_m3s, points[n - 1].outflow_m3s);
            let (i, t) = locate_tailrace(points, q);
            points[i].height_m + t * (points[i + 1].height_m - points[i].height_m)
        }
    }
}

/// Derivative `dh_tail/dq_out` of the tailrace elevation at `outflow_m3s` (m/(m³/s)).
///
/// - `Polynomial`: evaluates the analytic derivative `c[1] + 2·c[2]·q + …` via
///   Horner's method. A single-coefficient (constant) polynomial returns `0.0`.
/// - `Piecewise`: returns the slope of the segment that contains the query outflow.
///   The outflow is clamped before lookup; out-of-range queries return the slope
///   of the nearest end segment.
pub(crate) fn evaluate_tailrace_derivative(model: &TailraceModel, outflow_m3s: f64) -> f64 {
    match model {
        TailraceModel::Polynomial { coefficients } => {
            // Build derivative coefficients: d[k] = (k+1) * c[k+1].
            // Evaluate via Horner's method from the highest-degree term down.
            let n = coefficients.len();
            if n <= 1 {
                return 0.0;
            }
            // Accumulate from the last coefficient down to index 1.
            // d[k] = (k+1)*c[k+1], so iterating rev over indices 1..n:
            //   term k: degree k contributes coefficient k * c[k]
            let mut acc = 0.0_f64;
            for k in (1..n).rev() {
                // At each step: acc = acc * q + k * c[k]
                // k is in 1..n; n <= coefficients.len() which is a usize bounded
                // in practice by the number of polynomial terms (always small).
                // We cast to u32 first to avoid clippy::cast_precision_loss.
                #[allow(clippy::cast_possible_truncation)]
                let k_f64 = f64::from(k as u32);
                acc = acc * outflow_m3s + k_f64 * coefficients[k];
            }
            acc
        }
        TailraceModel::Piecewise { points } => {
            let n = points.len();
            if n <= 1 {
                return 0.0;
            }
            let q = outflow_m3s.clamp(points[0].outflow_m3s, points[n - 1].outflow_m3s);
            let (i, _) = locate_tailrace(points, q);
            (points[i + 1].height_m - points[i].height_m)
                / (points[i + 1].outflow_m3s - points[i].outflow_m3s)
        }
    }
}

/// Head loss `h_loss` (m) for the given `gross_head` (m) and `turbined_m3s` (m³/s).
///
/// - `Factor { value }`: returns `value * gross_head` (fraction of gross head).
/// - `Constant { value_m }`: returns the fixed head loss; `gross_head` and
///   `turbined_m3s` are unused.
///
/// The `turbined_m3s` parameter is reserved for future flow-dependent loss variants
/// and is intentionally ignored for both current variants.
pub(crate) fn evaluate_losses(
    model: &HydraulicLossesModel,
    gross_head: f64,
    _turbined_m3s: f64,
) -> f64 {
    match model {
        HydraulicLossesModel::Factor { value } => value * gross_head,
        HydraulicLossesModel::Constant { value_m } => *value_m,
    }
}

/// Dimensionless loss factor for `Factor` variants; `0.0` for `Constant` variants.
///
/// Used by the net-head derivative computation to analytically propagate the loss
/// term through the production function gradient.
pub(crate) fn evaluate_losses_factor(model: &HydraulicLossesModel) -> f64 {
    match model {
        HydraulicLossesModel::Factor { value } => *value,
        HydraulicLossesModel::Constant { .. } => 0.0,
    }
}

/// Find segment index `i` and fractional position `t` in a piecewise tailrace table.
///
/// Returns `(i, t)` such that:
/// - `points[i].outflow_m3s <= q <= points[i+1].outflow_m3s`
/// - `t = (q - outflow[i]) / (outflow[i+1] - outflow[i])`
///
/// The caller must ensure `q` is already clamped to `[q_min, q_max]`. Uses
/// `partition_point` for O(log n) binary search; saturates at `n - 2` to keep
/// `i + 1` in bounds at `q == q_max`.
fn locate_tailrace(points: &[cobre_core::TailracePoint], q: f64) -> (usize, f64) {
    let n = points.len();
    let idx = points.partition_point(|p| p.outflow_m3s <= q);
    let i = idx.saturating_sub(1).min(n - 2);
    let dq = points[i + 1].outflow_m3s - points[i].outflow_m3s;
    let t = (q - points[i].outflow_m3s) / dq;
    (i, t)
}

// ── ProductionFunction ────────────────────────────────────────────────────────

/// Gravity times water density over unit conversion: g·ρ / 1000.
///
/// Used in the production function `phi = K * eta * q * h_net` to convert
/// from hydraulic power (W) to megawatts. The factor is `9.81 * 1000 / 1e6 = 9.81e-3`.
const K: f64 = 9.81 / 1000.0;

/// Complete hydro production function `phi(v, q, s)` with analytical derivatives.
///
/// Bundles the forebay interpolation table with the optional tailrace and hydraulic
/// loss models and a constant turbine efficiency into a single evaluable object.
/// Evaluation produces power output in MW; derivatives are used by the FPHA fitting
/// algorithm to compute tangent hyperplanes.
///
/// # Construction
///
/// Build from a validated [`ForebayTable`] and the optional model fields taken
/// directly from a hydro plant entity. All validation is done upstream; `new` is
/// infallible.
///
/// # Evaluation
///
/// All three public methods (`net_head`, `evaluate`, `partial_derivatives`) accept
/// `(v, q, s)` where:
/// - `v` — reservoir volume \[hm³\]
/// - `q` — turbined flow \[m³/s\]
/// - `s` — spillage flow \[m³/s\]
#[derive(Debug, Clone)]
pub(crate) struct ProductionFunction {
    /// Forebay height interpolation table.
    forebay: ForebayTable,
    /// Tailrace elevation model. `None` means zero tailrace height for all outflows.
    tailrace: Option<TailraceModel>,
    /// Hydraulic losses model. `None` means lossless penstock.
    hydraulic_losses: Option<HydraulicLossesModel>,
    /// Turbine efficiency (dimensionless, in `(0, 1]`). Defaults to `1.0` when the
    /// hydro entity has no `EfficiencyModel`.
    efficiency: f64,
    /// Maximum turbined flow \[m³/s\], carried for grid construction in the fitting
    /// algorithm.
    pub(crate) max_turbined_m3s: f64,
    /// Human-readable plant name for error messages.
    pub(crate) hydro_name: String,
}

impl ProductionFunction {
    /// Build a [`ProductionFunction`] from component models.
    ///
    /// # Parameters
    ///
    /// - `forebay` — pre-validated [`ForebayTable`] for this plant.
    /// - `tailrace` — optional reference to the plant's [`TailraceModel`]; cloned
    ///   into the struct. `None` = constant zero tailrace.
    /// - `hydraulic_losses` — optional reference to the plant's [`HydraulicLossesModel`];
    ///   copied into the struct. `None` = lossless.
    /// - `efficiency` — optional reference to the plant's [`EfficiencyModel`]; only
    ///   [`EfficiencyModel::Constant`] is supported. `None` = 1.0 (100% efficiency).
    /// - `max_turbined_m3s` — maximum turbined flow from the hydro entity \[m³/s\].
    /// - `hydro_name` — plant name used in diagnostic messages.
    pub(crate) fn new(
        forebay: ForebayTable,
        tailrace: Option<&TailraceModel>,
        hydraulic_losses: Option<&HydraulicLossesModel>,
        efficiency: Option<&EfficiencyModel>,
        max_turbined_m3s: f64,
        hydro_name: String,
    ) -> Self {
        let efficiency_value = match efficiency {
            Some(EfficiencyModel::Constant { value }) => *value,
            None => 1.0,
        };
        Self {
            forebay,
            tailrace: tailrace.cloned(),
            hydraulic_losses: hydraulic_losses.copied(),
            efficiency: efficiency_value,
            max_turbined_m3s,
            hydro_name,
        }
    }

    /// Net head available at the turbine \[m\].
    ///
    /// Computes `h_net = h_fore(v) - h_tail(q+s) - h_loss(gross_head, q)`, where:
    /// - `h_fore` is the interpolated forebay surface elevation,
    /// - `h_tail` is the tailrace elevation at total outflow `q + s` (0 if no model),
    /// - `h_loss` is the hydraulic head loss (0 if no model).
    ///
    /// For the [`HydraulicLossesModel::Factor`] variant, losses are proportional to
    /// gross head, which simplifies to `h_net = (1 - k) * (h_fore - h_tail)`.
    ///
    /// The result is clamped to `max(0.0, h_net)` — negative net head is physically
    /// impossible and arises only at out-of-range operating points.
    ///
    /// # Parameters
    ///
    /// - `v` — reservoir volume \[hm³\]
    /// - `q` — turbined flow \[m³/s\]
    /// - `s` — spillage flow \[m³/s\]
    pub(crate) fn net_head(&self, v: f64, q: f64, s: f64) -> f64 {
        let h_fore = self.forebay.height(v);
        let q_out = q + s;
        let h_tail = self
            .tailrace
            .as_ref()
            .map_or(0.0, |m| evaluate_tailrace(m, q_out));
        let gross_head = h_fore - h_tail;
        let h_loss = self
            .hydraulic_losses
            .as_ref()
            .map_or(0.0, |m| evaluate_losses(m, gross_head, q));
        let h_net = gross_head - h_loss;
        h_net.max(0.0)
    }

    /// Power output from the production function \[MW\].
    ///
    /// Evaluates `phi(v, q, s) = K * eta * q * h_net(v, q, s)` where
    /// `K = 9.81 / 1000` and `eta` is the turbine efficiency.
    ///
    /// The result is always non-negative because `q >= 0` and `h_net >= 0`.
    ///
    /// # Parameters
    ///
    /// - `v` — reservoir volume \[hm³\]
    /// - `q` — turbined flow \[m³/s\]
    /// - `s` — spillage flow \[m³/s\]
    pub(crate) fn evaluate(&self, v: f64, q: f64, s: f64) -> f64 {
        let h_net = self.net_head(v, q, s);
        K * self.efficiency * q * h_net
    }

    /// Analytical partial derivatives of the production function.
    ///
    /// Returns `(d_phi/dv, d_phi/dq, d_phi/ds)` evaluated at `(v, q, s)`.
    ///
    /// The derivative formulas depend on the loss model:
    ///
    /// **Constant losses or no losses** (`h_net = h_fore - h_tail - c`):
    /// ```text
    /// d_phi/dv = K·eta·q·dh_fore/dv
    /// d_phi/dq = K·eta·(h_net - q·dh_tail/dq_out)
    /// d_phi/ds = -K·eta·q·dh_tail/dq_out
    /// ```
    ///
    /// **Factor losses** (`h_net = (1-k)·(h_fore - h_tail)`):
    /// ```text
    /// d_phi/dv = K·eta·q·(1-k)·dh_fore/dv
    /// d_phi/dq = K·eta·(h_net - q·(1-k)·dh_tail/dq_out)
    /// d_phi/ds = -K·eta·q·(1-k)·dh_tail/dq_out
    /// ```
    ///
    /// # Sign conventions
    ///
    /// - `d_phi/dv > 0`: more storage raises forebay, increasing net head and power.
    /// - `d_phi/dq > 0` when net head is positive (turbining produces power).
    /// - `d_phi/ds <= 0`: spillage raises tailrace, reducing net head.
    ///   Equals zero when there is no tailrace model.
    ///
    /// # Parameters
    ///
    /// - `v` — reservoir volume \[hm³\]
    /// - `q` — turbined flow \[m³/s\]
    /// - `s` — spillage flow \[m³/s\]
    #[allow(clippy::similar_names)] // d_phi_dv / d_phi_dq / d_phi_ds are standard PDE notation
    pub(crate) fn partial_derivatives(&self, v: f64, q: f64, s: f64) -> (f64, f64, f64) {
        let h_fore = self.forebay.height(v);
        let dh_fore_dv = self.forebay.height_derivative(v);
        let q_out = q + s;

        let h_tail = self
            .tailrace
            .as_ref()
            .map_or(0.0, |m| evaluate_tailrace(m, q_out));
        let dh_tail_dq_out = self
            .tailrace
            .as_ref()
            .map_or(0.0, |m| evaluate_tailrace_derivative(m, q_out));

        let ke = K * self.efficiency;

        match self.hydraulic_losses {
            Some(HydraulicLossesModel::Factor { value: k_loss }) => {
                // h_net = (1 - k_loss) * (h_fore - h_tail)
                let one_minus_k = 1.0 - k_loss;
                let h_net = (one_minus_k * (h_fore - h_tail)).max(0.0);
                let d_phi_dv = ke * q * one_minus_k * dh_fore_dv;
                let d_phi_dq = ke * (h_net - q * one_minus_k * dh_tail_dq_out);
                let d_phi_ds = -ke * q * one_minus_k * dh_tail_dq_out;
                (d_phi_dv, d_phi_dq, d_phi_ds)
            }
            Some(HydraulicLossesModel::Constant { .. }) | None => {
                // h_net = h_fore - h_tail - h_loss_const   (h_loss_const may be 0)
                let h_net = self.net_head(v, q, s);
                let d_phi_dv = ke * q * dh_fore_dv;
                let d_phi_dq = ke * (h_net - q * dh_tail_dq_out);
                let d_phi_ds = -ke * q * dh_tail_dq_out;
                (d_phi_dv, d_phi_dq, d_phi_ds)
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::similar_names
)]
mod tests {
    use cobre_core::{
        EfficiencyModel, EntityId, HydraulicLossesModel, Hydro, HydroGenerationModel,
        HydroPenalties, TailraceModel, TailracePoint,
    };
    use cobre_io::extensions::{FittingWindow, FphaConfig, HydroGeometryRow};

    use super::{
        FittingBounds, ForebayTable, FphaFittingError, ProductionFunction, evaluate_losses,
        evaluate_losses_factor, evaluate_tailrace, evaluate_tailrace_derivative,
        resolve_fitting_bounds,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal [`HydroGeometryRow`] with `area_km2 = 0.0`.
    fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3,
            height_m,
            area_km2: 0.0,
        }
    }

    /// Sobradinho-style 5-point VHA curve used across several tests.
    fn sobradinho_rows() -> Vec<HydroGeometryRow> {
        vec![
            row(0.0, 386.5),
            row(2_000.0, 390.0),
            row(10_000.0, 396.0),
            row(24_500.0, 400.5),
            row(34_116.0, 401.3),
        ]
    }

    // ── AC: valid construction ────────────────────────────────────────────────

    #[test]
    fn valid_five_point_curve_construction_succeeds() {
        let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
        assert_eq!(table.v_min(), 0.0);
        assert_eq!(table.v_max(), 34_116.0);
    }

    // ── AC: interpolation at midpoint ─────────────────────────────────────────

    /// height(1000.0) on segment [0, 2000] with heights [386.5, 390.0] must equal
    /// 386.5 + (390.0 - 386.5) * 1000.0 / 2000.0 = 388.25 within 1e-10.
    #[test]
    fn interpolation_at_midpoint_segment_0_to_2000() {
        let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
        let expected = 386.5 + (390.0 - 386.5) * 1000.0 / 2000.0;
        assert!((table.height(1000.0) - expected).abs() < 1e-10);
    }

    // ── AC: interpolation at breakpoints ──────────────────────────────────────

    #[test]
    fn interpolation_at_breakpoints_returns_exact_values() {
        let rows = sobradinho_rows();
        let table = ForebayTable::new(&rows, "Sobradinho").unwrap();
        for r in &rows {
            assert!(
                (table.height(r.volume_hm3) - r.height_m).abs() < 1e-10,
                "breakpoint v={}: expected h={}, got h={}",
                r.volume_hm3,
                r.height_m,
                table.height(r.volume_hm3)
            );
        }
    }

    // ── AC: clamping below v_min ──────────────────────────────────────────────

    #[test]
    fn height_clamped_below_v_min() {
        let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
        let at_min = table.height(table.v_min());
        let below = table.height(table.v_min() - 100.0);
        assert!((at_min - below).abs() < 1e-10);
    }

    // ── AC: clamping above v_max ──────────────────────────────────────────────

    #[test]
    fn height_clamped_above_v_max() {
        let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
        let at_max = table.height(table.v_max());
        let above = table.height(table.v_max() + 999.0);
        assert!((at_max - above).abs() < 1e-10);
    }

    // ── AC: derivative on first segment ──────────────────────────────────────

    #[test]
    fn derivative_first_segment_correct() {
        let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
        let expected = (390.0 - 386.5) / 2_000.0;
        assert!((table.height_derivative(1000.0) - expected).abs() < 1e-10);
    }

    // ── AC: derivative on last segment ───────────────────────────────────────

    #[test]
    fn derivative_last_segment_and_at_v_max() {
        let rows = sobradinho_rows();
        let table = ForebayTable::new(&rows, "Sobradinho").unwrap();
        let n = rows.len();
        let expected = (rows[n - 1].height_m - rows[n - 2].height_m)
            / (rows[n - 1].volume_hm3 - rows[n - 2].volume_hm3);

        // Midpoint of last segment
        let v_mid = f64::midpoint(rows[n - 2].volume_hm3, rows[n - 1].volume_hm3);
        assert!(
            (table.height_derivative(v_mid) - expected).abs() < 1e-10,
            "derivative at last-segment midpoint"
        );

        // At v_max itself
        assert!(
            (table.height_derivative(table.v_max()) - expected).abs() < 1e-10,
            "derivative at v_max"
        );
    }

    // ── AC: derivative at interior breakpoints uses right segment ────────────

    #[test]
    fn derivative_at_interior_breakpoint_uses_right_segment() {
        let rows = sobradinho_rows();
        let table = ForebayTable::new(&rows, "Sobradinho").unwrap();

        // Breakpoint at index 1: v=2000, h=390.0; right segment is [2000, 10000].
        let expected_right =
            (rows[2].height_m - rows[1].height_m) / (rows[2].volume_hm3 - rows[1].volume_hm3);
        assert!(
            (table.height_derivative(rows[1].volume_hm3) - expected_right).abs() < 1e-10,
            "expected right-segment slope at breakpoint index 1"
        );
    }

    // ── AC: derivative clamping below v_min ──────────────────────────────────

    #[test]
    fn derivative_clamped_below_v_min_returns_first_segment_slope() {
        let rows = sobradinho_rows();
        let table = ForebayTable::new(&rows, "Sobradinho").unwrap();
        let first_slope =
            (rows[1].height_m - rows[0].height_m) / (rows[1].volume_hm3 - rows[0].volume_hm3);
        assert!((table.height_derivative(table.v_min() - 50.0) - first_slope).abs() < 1e-10);
    }

    // ── AC: insufficient points (0 points) ───────────────────────────────────

    #[test]
    fn insufficient_points_zero_rows() {
        let err = ForebayTable::new(&[], "Itaipu").unwrap_err();
        match err {
            FphaFittingError::InsufficientPoints { count, .. } => {
                assert_eq!(count, 0);
            }
            other => panic!("expected InsufficientPoints, got: {other:?}"),
        }
    }

    // ── AC: insufficient points (1 point) ────────────────────────────────────

    #[test]
    fn insufficient_points_one_row() {
        let err = ForebayTable::new(&[row(0.0, 386.5)], "Itaipu").unwrap_err();
        match err {
            FphaFittingError::InsufficientPoints { count, .. } => {
                assert_eq!(count, 1);
            }
            other => panic!("expected InsufficientPoints, got: {other:?}"),
        }
    }

    // ── AC: non-monotonic volume (duplicate) ──────────────────────────────────

    #[test]
    fn non_monotonic_volume_duplicate() {
        let rows = vec![row(0.0, 386.5), row(1000.0, 388.0), row(1000.0, 390.0)];
        let err = ForebayTable::new(&rows, "Tucurui").unwrap_err();
        match err {
            FphaFittingError::NonMonotonicVolume {
                index,
                v_prev,
                v_curr,
                ..
            } => {
                assert_eq!(index, 2);
                assert_eq!(v_prev, 1000.0);
                assert_eq!(v_curr, 1000.0);
            }
            other => panic!("expected NonMonotonicVolume, got: {other:?}"),
        }
    }

    // ── AC: non-monotonic volume (decreasing) ─────────────────────────────────

    #[test]
    fn non_monotonic_volume_decreasing() {
        let rows = vec![row(0.0, 386.5), row(2000.0, 390.0), row(1500.0, 392.0)];
        let err = ForebayTable::new(&rows, "Belo Monte").unwrap_err();
        match err {
            FphaFittingError::NonMonotonicVolume { index, .. } => {
                assert_eq!(index, 2);
            }
            other => panic!("expected NonMonotonicVolume, got: {other:?}"),
        }
    }

    // ── AC: non-monotonic height (decreasing) ─────────────────────────────────

    #[test]
    fn non_monotonic_height_decreasing() {
        let rows = vec![row(0.0, 390.0), row(1000.0, 392.0), row(2000.0, 388.0)];
        let err = ForebayTable::new(&rows, "Serra da Mesa").unwrap_err();
        match err {
            FphaFittingError::NonMonotonicHeight {
                index,
                h_prev,
                h_curr,
                ..
            } => {
                assert_eq!(index, 2);
                assert_eq!(h_prev, 392.0);
                assert_eq!(h_curr, 388.0);
            }
            other => panic!("expected NonMonotonicHeight, got: {other:?}"),
        }
    }

    // ── AC: plateau heights (equal consecutive) ───────────────────────────────

    #[test]
    fn equal_consecutive_heights_accepted() {
        let rows = vec![row(0.0, 386.5), row(1000.0, 386.5), row(2000.0, 390.0)];
        let result = ForebayTable::new(&rows, "Furnas");
        assert!(result.is_ok(), "plateau heights should be accepted");
    }

    // ── AC: Display messages are informative ──────────────────────────────────

    #[test]
    fn display_insufficient_points_contains_name_and_count() {
        let err = ForebayTable::new(&[], "Anta").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Anta"), "should contain hydro name: {msg}");
        assert!(msg.contains('0'), "should contain count: {msg}");
    }

    #[test]
    fn display_non_monotonic_volume_contains_name_and_index() {
        let rows = vec![row(0.0, 386.5), row(1000.0, 390.0), row(500.0, 392.0)];
        let err = ForebayTable::new(&rows, "Cachoeira").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Cachoeira"),
            "should contain hydro name: {msg}"
        );
        assert!(msg.contains('2'), "should contain index: {msg}");
    }

    #[test]
    fn display_non_monotonic_height_contains_name_and_index() {
        let rows = vec![row(0.0, 395.0), row(1000.0, 393.0)];
        let err = ForebayTable::new(&rows, "Marimbondo").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Marimbondo"),
            "should contain hydro name: {msg}"
        );
        assert!(msg.contains('1'), "should contain index: {msg}");
    }

    #[test]
    fn fpha_fitting_error_implements_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<FphaFittingError>();
    }

    // ── Tailrace evaluation: Polynomial ──────────────────────────────────────

    #[test]
    fn tailrace_polynomial_constant_one_coefficient() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![7.5],
        };
        assert!((evaluate_tailrace(&model, 0.0) - 7.5).abs() < 1e-10);
        assert!((evaluate_tailrace(&model, 5000.0) - 7.5).abs() < 1e-10);
    }

    #[test]
    fn tailrace_polynomial_linear_two_coefficients() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![3.0, 0.0005],
        };
        // At q = 2000: 3.0 + 0.0005 * 2000 = 4.0
        assert!((evaluate_tailrace(&model, 2000.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn tailrace_polynomial_quadratic_acceptance_criterion() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.001, -1e-7],
        };
        let expected = 5.0 + 0.001 * 3000.0 + (-1e-7) * 3000.0_f64.powi(2);
        assert!((evaluate_tailrace(&model, 3000.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn tailrace_polynomial_quartic_five_coefficients() {
        // h = 1 + 2q + 3q^2 + 4q^3 + 5q^4 at q = 1
        // = 1 + 2 + 3 + 4 + 5 = 15
        let model = TailraceModel::Polynomial {
            coefficients: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        assert!((evaluate_tailrace(&model, 1.0) - 15.0).abs() < 1e-10);
    }

    // ── Tailrace derivative: Polynomial ──────────────────────────────────────

    #[test]
    fn tailrace_polynomial_derivative_constant_is_zero() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![7.5],
        };
        assert!((evaluate_tailrace_derivative(&model, 1000.0)).abs() < 1e-10);
    }

    #[test]
    fn tailrace_polynomial_derivative_linear() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![3.0, 0.0005],
        };
        assert!((evaluate_tailrace_derivative(&model, 9999.0) - 0.0005).abs() < 1e-10);
    }

    #[test]
    fn tailrace_polynomial_derivative_quadratic_acceptance_criterion() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![5.0, 0.001, -1e-7],
        };
        let expected = 0.001 + 2.0 * (-1e-7) * 3000.0;
        assert!((evaluate_tailrace_derivative(&model, 3000.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn tailrace_polynomial_derivative_quartic() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        assert!((evaluate_tailrace_derivative(&model, 1.0) - 40.0).abs() < 1e-10);
    }

    // ── Tailrace evaluation: Piecewise ────────────────────────────────────────

    /// Build the 3-point piecewise model from the acceptance criteria.
    fn ac_piecewise() -> TailraceModel {
        use cobre_core::TailracePoint;
        TailraceModel::Piecewise {
            points: vec![
                TailracePoint {
                    outflow_m3s: 0.0,
                    height_m: 3.0,
                },
                TailracePoint {
                    outflow_m3s: 5000.0,
                    height_m: 4.5,
                },
                TailracePoint {
                    outflow_m3s: 15_000.0,
                    height_m: 6.2,
                },
            ],
        }
    }

    #[test]
    fn tailrace_piecewise_midpoint_first_segment_acceptance_criterion() {
        let model = ac_piecewise();
        let expected = 3.0 + (4.5 - 3.0) * 2500.0 / 5000.0;
        assert!((evaluate_tailrace(&model, 2500.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn tailrace_piecewise_at_breakpoints_exact() {
        let model = ac_piecewise();
        assert!((evaluate_tailrace(&model, 0.0) - 3.0).abs() < 1e-10);
        assert!((evaluate_tailrace(&model, 5000.0) - 4.5).abs() < 1e-10);
        assert!((evaluate_tailrace(&model, 15_000.0) - 6.2).abs() < 1e-10);
    }

    #[test]
    fn tailrace_piecewise_clamp_below_range() {
        let model = ac_piecewise();
        assert!((evaluate_tailrace(&model, -500.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn tailrace_piecewise_clamp_above_range() {
        let model = ac_piecewise();
        assert!((evaluate_tailrace(&model, 99_999.0) - 6.2).abs() < 1e-10);
    }

    // ── Tailrace derivative: Piecewise ────────────────────────────────────────

    #[test]
    fn tailrace_piecewise_derivative_first_segment_acceptance_criterion() {
        let model = ac_piecewise();
        let expected = (4.5 - 3.0) / 5000.0;
        assert!((evaluate_tailrace_derivative(&model, 2500.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn tailrace_piecewise_derivative_second_segment() {
        let model = ac_piecewise();
        let expected = (6.2 - 4.5) / (15_000.0 - 5000.0);
        assert!((evaluate_tailrace_derivative(&model, 10_000.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn tailrace_piecewise_derivative_at_q_max_returns_last_segment_slope() {
        let model = ac_piecewise();
        let expected = (6.2 - 4.5) / (15_000.0 - 5000.0);
        assert!((evaluate_tailrace_derivative(&model, 15_000.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn tailrace_piecewise_derivative_clamp_above_returns_last_segment_slope() {
        let model = ac_piecewise();
        let expected = (6.2 - 4.5) / (15_000.0 - 5000.0);
        assert!((evaluate_tailrace_derivative(&model, 99_999.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn tailrace_piecewise_derivative_clamp_below_returns_first_segment_slope() {
        let model = ac_piecewise();
        let expected = (4.5 - 3.0) / 5000.0;
        assert!((evaluate_tailrace_derivative(&model, -1.0) - expected).abs() < 1e-10);
    }

    // ── Hydraulic losses: Factor ──────────────────────────────────────────────

    #[test]
    fn losses_factor_acceptance_criterion() {
        let model = HydraulicLossesModel::Factor { value: 0.03 };
        assert!((evaluate_losses(&model, 100.0, 0.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn losses_factor_scales_with_gross_head() {
        let model = HydraulicLossesModel::Factor { value: 0.05 };
        assert!((evaluate_losses(&model, 200.0, 1000.0) - 10.0).abs() < 1e-10);
        assert!((evaluate_losses(&model, 0.0, 5000.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn losses_factor_turbined_has_no_effect() {
        let model = HydraulicLossesModel::Factor { value: 0.02 };
        let r1 = evaluate_losses(&model, 80.0, 0.0);
        let r2 = evaluate_losses(&model, 80.0, 99_999.0);
        assert!((r1 - r2).abs() < 1e-10);
    }

    // ── Hydraulic losses: Constant ────────────────────────────────────────────

    #[test]
    fn losses_constant_acceptance_criterion() {
        let model = HydraulicLossesModel::Constant { value_m: 2.5 };
        assert!((evaluate_losses(&model, 0.0, 0.0) - 2.5).abs() < 1e-10);
        assert!((evaluate_losses(&model, 999.0, 0.0) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn losses_constant_independent_of_all_inputs() {
        let model = HydraulicLossesModel::Constant { value_m: 1.25 };
        let r1 = evaluate_losses(&model, 50.0, 500.0);
        let r2 = evaluate_losses(&model, 200.0, 8000.0);
        assert!((r1 - 1.25).abs() < 1e-10);
        assert!((r2 - 1.25).abs() < 1e-10);
    }

    // ── Loss factor extraction ────────────────────────────────────────────────

    #[test]
    fn losses_factor_extraction_returns_factor() {
        let model = HydraulicLossesModel::Factor { value: 0.04 };
        assert!((evaluate_losses_factor(&model) - 0.04).abs() < 1e-10);
    }

    #[test]
    fn losses_factor_extraction_constant_returns_zero() {
        let model = HydraulicLossesModel::Constant { value_m: 5.0 };
        assert!((evaluate_losses_factor(&model)).abs() < 1e-10);
    }

    // ── AC: two-point minimum curve ───────────────────────────────────────────

    #[test]
    fn two_point_minimum_curve_works() {
        let rows = vec![row(0.0, 380.0), row(1000.0, 400.0)];
        let table = ForebayTable::new(&rows, "MinimalPlant").unwrap();
        assert_eq!(table.v_min(), 0.0);
        assert_eq!(table.v_max(), 1000.0);
        // Midpoint: 380 + 0.5 * (400 - 380) = 390
        assert!((table.height(500.0) - 390.0).abs() < 1e-10);
        // Derivative: (400 - 380) / 1000 = 0.02
        assert!((table.height_derivative(500.0) - 0.02).abs() < 1e-10);
    }

    // ── AC: second segment midpoint interpolation ─────────────────────────────

    #[test]
    fn interpolation_second_segment_correct() {
        let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
        // v = 6000: fraction = (6000 - 2000) / (10000 - 2000) = 0.5
        // h = 390.0 + 0.5 * (396.0 - 390.0) = 393.0
        let expected = 390.0 + 0.5 * (396.0 - 390.0);
        assert!((table.height(6_000.0) - expected).abs() < 1e-10);
    }

    // ── ProductionFunction helpers ────────────────────────────────────────────

    /// Build a simple 2-point ForebayTable that returns a constant 400 m for all v.
    /// Derivative is 0, but for these tests we usually fix v at 10000 hm3.
    fn flat_forebay_400m() -> ForebayTable {
        ForebayTable::new(
            &[
                HydroGeometryRow {
                    hydro_id: EntityId::from(1),
                    volume_hm3: 0.0,
                    height_m: 400.0,
                    area_km2: 0.0,
                },
                HydroGeometryRow {
                    hydro_id: EntityId::from(1),
                    volume_hm3: 20_000.0,
                    height_m: 400.0,
                    area_km2: 0.0,
                },
            ],
            "TestPlant",
        )
        .unwrap()
    }

    /// Build a sloped ForebayTable: h_fore = 380 + v * 2e-3 m (slope = 2e-3 m/hm3).
    /// At v = 10000 hm3: h_fore = 380 + 10000 * 2e-3 = 400 m.
    fn sloped_forebay() -> ForebayTable {
        ForebayTable::new(
            &[
                HydroGeometryRow {
                    hydro_id: EntityId::from(1),
                    volume_hm3: 0.0,
                    height_m: 380.0,
                    area_km2: 0.0,
                },
                HydroGeometryRow {
                    hydro_id: EntityId::from(1),
                    volume_hm3: 10_000.0,
                    height_m: 400.0,
                    area_km2: 0.0,
                },
            ],
            "TestPlant",
        )
        .unwrap()
    }

    /// Build a quadratic-style polynomial tailrace giving h_tail = 5.5 m at q_out = 3000.
    /// Coefficients: h = 5.0 + 0.001*q + (-1e-7)*q^2
    /// At q=3000: h = 5.0 + 3.0 - 0.9 = 7.1   (NOT 5.5, see below)
    ///
    /// For AC tests: use a linear model h = 5.5 / 3000 * q = 1.8333e-3 * q.
    /// At q=3000: h = 5.5. Derivative = 5.5 / 3000.
    fn linear_tailrace_5_5_at_3000() -> TailraceModel {
        let slope = 5.5 / 3000.0;
        TailraceModel::Polynomial {
            coefficients: vec![0.0, slope],
        }
    }

    /// Build the 3-point piecewise tailrace from the existing test helper (reused).
    fn piecewise_tailrace() -> TailraceModel {
        TailraceModel::Piecewise {
            points: vec![
                TailracePoint {
                    outflow_m3s: 0.0,
                    height_m: 3.0,
                },
                TailracePoint {
                    outflow_m3s: 5000.0,
                    height_m: 4.5,
                },
                TailracePoint {
                    outflow_m3s: 15_000.0,
                    height_m: 6.2,
                },
            ],
        }
    }

    // ── net_head tests ────────────────────────────────────────────────────────

    /// Net head with no tailrace, no losses: h_net = h_fore(v).
    #[test]
    fn net_head_no_tailrace_no_losses_equals_h_fore() {
        let forebay = sloped_forebay();
        let pf =
            ProductionFunction::new(forebay, None, None, None, 12_600.0, "TestPlant".to_owned());
        // At v=10000: h_fore = 400. h_tail = 0. h_loss = 0. h_net = 400.
        assert!((pf.net_head(10_000.0, 3000.0, 0.0) - 400.0).abs() < 1e-10);
    }

    #[test]
    fn net_head_polynomial_tailrace_constant_losses_acceptance_criterion() {
        let forebay = flat_forebay_400m();
        let tailrace = linear_tailrace_5_5_at_3000();
        let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            12_600.0,
            "TestPlant".to_owned(),
        );
        let h_net = pf.net_head(10_000.0, 3000.0, 0.0);
        let expected = 400.0 - 5.5 - 2.0; // 392.5
        assert!(
            (h_net - expected).abs() < 1e-6,
            "h_net={h_net}, expected={expected}"
        );
    }

    #[test]
    fn net_head_piecewise_tailrace_factor_losses() {
        let forebay = flat_forebay_400m();
        let tailrace = piecewise_tailrace();
        let losses = HydraulicLossesModel::Factor { value: 0.03 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            None,
            12_600.0,
            "TestPlant".to_owned(),
        );
        // q_out = q + s = 2500 + 0 = 2500
        let h_tail = 3.0 + (4.5 - 3.0) * 2500.0 / 5000.0; // 3.75
        let gross_head = 400.0 - h_tail; // 396.25
        let expected = (1.0 - 0.03) * gross_head; // 384.3625
        let h_net = pf.net_head(0.0, 2500.0, 0.0);
        assert!(
            (h_net - expected).abs() < 1e-6,
            "h_net={h_net}, expected={expected}"
        );
    }

    #[test]
    fn net_head_clamped_to_zero_when_losses_exceed_forebay() {
        // h_fore = 400, but h_loss = 500 (absurd but tests clamping).
        let forebay = flat_forebay_400m();
        let losses = HydraulicLossesModel::Constant { value_m: 500.0 };
        let pf = ProductionFunction::new(
            forebay,
            None,
            Some(&losses),
            None,
            12_600.0,
            "TestPlant".to_owned(),
        );
        let h_net = pf.net_head(10_000.0, 3000.0, 0.0);
        assert!(h_net >= 0.0, "net head must be non-negative, got {h_net}");
        assert!(
            (h_net - 0.0).abs() < 1e-10,
            "h_net should be exactly 0.0, got {h_net}"
        );
    }

    // ── evaluate tests ────────────────────────────────────────────────────────

    #[test]
    fn evaluate_acceptance_criterion() {
        let forebay = flat_forebay_400m();
        let tailrace = linear_tailrace_5_5_at_3000();
        let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            12_600.0,
            "TestPlant".to_owned(),
        );
        let phi = pf.evaluate(10_000.0, 3000.0, 0.0);
        let expected = (9.81 * 0.92 * 3000.0 * 392.5) / 1000.0;
        assert!(
            (phi - expected).abs() < 1e-2,
            "phi={phi}, expected={expected}"
        );
    }

    // ── partial_derivatives: no tailrace tests ────────────────────────────────

    #[test]
    fn partial_derivatives_no_tailrace_ds_is_zero() {
        let forebay = sloped_forebay();
        let pf =
            ProductionFunction::new(forebay, None, None, None, 12_600.0, "TestPlant".to_owned());
        let (_, _, d_phi_ds) = pf.partial_derivatives(10_000.0, 3000.0, 0.0);
        assert!(
            d_phi_ds.abs() < 1e-10,
            "d_phi_ds should be 0.0 with no tailrace, got {d_phi_ds}"
        );
    }

    #[test]
    fn partial_derivatives_no_tailrace_dv_is_positive() {
        let forebay = sloped_forebay(); // slope = 2e-3 m/hm3
        let pf =
            ProductionFunction::new(forebay, None, None, None, 12_600.0, "TestPlant".to_owned());
        let (d_phi_dv, _, _) = pf.partial_derivatives(10_000.0, 3000.0, 0.0);
        assert!(d_phi_dv > 0.0, "d_phi_dv must be positive, got {d_phi_dv}");
    }

    // ── partial_derivatives: polynomial tailrace + constant losses ────────────

    /// Analytical derivatives with polynomial tailrace and constant losses.
    ///
    /// Setup: sloped_forebay (slope = 2e-3 m/hm3), linear tailrace (slope = 5.5/3000),
    /// constant loss = 2.0 m, efficiency = 0.92.
    ///
    /// At (v=10000, q=3000, s=0):
    ///   h_fore = 400, dh_fore_dv = 2e-3
    ///   h_tail = 5.5, dh_tail_dq_out = 5.5/3000
    ///   h_net = 400 - 5.5 - 2.0 = 392.5
    ///   K = 9.81e-3, eta = 0.92, ke = K*eta
    ///   d_phi/dv = ke * q * dh_fore_dv = ke * 3000 * 2e-3
    ///   d_phi/dq = ke * (h_net - q * dh_tail_dq_out) = ke * (392.5 - 3000*5.5/3000)
    ///            = ke * (392.5 - 5.5) = ke * 387.0
    ///   d_phi/ds = -ke * q * dh_tail_dq_out = -ke * 3000 * 5.5/3000 = -ke * 5.5
    #[test]
    fn partial_derivatives_polynomial_tailrace_constant_losses() {
        let forebay = sloped_forebay();
        let tailrace = linear_tailrace_5_5_at_3000();
        let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
        let efficiency = EfficiencyModel::Constant { value: 0.92 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            Some(&efficiency),
            12_600.0,
            "TestPlant".to_owned(),
        );

        let (d_phi_dv, d_phi_dq, d_phi_ds) = pf.partial_derivatives(10_000.0, 3000.0, 0.0);

        let ke = 9.81e-3 * 0.92;
        let dh_fore_dv = 2e-3_f64; // slope of sloped_forebay
        let dh_tail_dq_out = 5.5 / 3000.0;
        let h_net = 392.5_f64;

        let expected_dv = ke * 3000.0 * dh_fore_dv;
        let expected_dq = ke * (h_net - 3000.0 * dh_tail_dq_out);
        let expected_ds = -ke * 3000.0 * dh_tail_dq_out;

        assert!(
            (d_phi_dv - expected_dv).abs() < 1e-10,
            "d_phi_dv={d_phi_dv}, expected={expected_dv}"
        );
        assert!(
            (d_phi_dq - expected_dq).abs() < 1e-10,
            "d_phi_dq={d_phi_dq}, expected={expected_dq}"
        );
        assert!(
            (d_phi_ds - expected_ds).abs() < 1e-10,
            "d_phi_ds={d_phi_ds}, expected={expected_ds}"
        );
    }

    #[test]
    fn partial_derivatives_polynomial_tailrace_constant_losses_dv_positive() {
        let forebay = sloped_forebay();
        let tailrace = linear_tailrace_5_5_at_3000();
        let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            None,
            12_600.0,
            "TestPlant".to_owned(),
        );
        let (d_phi_dv, _, _) = pf.partial_derivatives(10_000.0, 3000.0, 0.0);
        assert!(d_phi_dv > 0.0, "d_phi_dv must be positive, got {d_phi_dv}");
    }

    #[test]
    fn partial_derivatives_polynomial_tailrace_ds_nonpositive() {
        let forebay = sloped_forebay();
        let tailrace = linear_tailrace_5_5_at_3000();
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            None,
            None,
            12_600.0,
            "TestPlant".to_owned(),
        );
        let (_, _, d_phi_ds) = pf.partial_derivatives(10_000.0, 3000.0, 0.0);
        assert!(
            d_phi_ds <= 0.0,
            "d_phi_ds must be <= 0 with tailrace, got {d_phi_ds}"
        );
    }

    // ── partial_derivatives: piecewise tailrace + factor losses ───────────────

    #[test]
    fn partial_derivatives_factor_losses_dv_accounts_for_k_factor() {
        let forebay = sloped_forebay();
        let losses = HydraulicLossesModel::Factor { value: 0.03 };
        let pf_factor = ProductionFunction::new(
            forebay.clone(),
            None,
            Some(&losses),
            None,
            12_600.0,
            "TestPlant".to_owned(),
        );
        let pf_noloss =
            ProductionFunction::new(forebay, None, None, None, 12_600.0, "TestPlant".to_owned());
        let (dv_factor, _, _) = pf_factor.partial_derivatives(10_000.0, 3000.0, 0.0);
        let (dv_noloss, _, _) = pf_noloss.partial_derivatives(10_000.0, 3000.0, 0.0);
        assert!(
            dv_factor > 0.0,
            "d_phi_dv must be positive, got {dv_factor}"
        );
        let expected_ratio = 0.97; // (1 - 0.03)
        let ratio = dv_factor / dv_noloss;
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "ratio of d_phi_dv factor/noloss should be 0.97, got {ratio}"
        );
    }

    /// Analytical derivatives with piecewise tailrace and factor losses.
    ///
    /// At (v=0, q=2000, s=500), q_out=2500:
    ///   h_fore = 380 (flat at v=0 end of sloped table, actually = 380.0 at v=0)
    ///   dh_fore_dv = 2e-3 (slope of sloped_forebay)
    ///   h_tail at q_out=2500 = 3.0 + (4.5-3.0)*2500/5000 = 3.75
    ///   dh_tail_dq_out = (4.5-3.0)/5000 = 3e-4
    ///   gross_head = 380 - 3.75 = 376.25
    ///   h_net = (1 - 0.03) * 376.25 = 0.97 * 376.25 = 364.9625
    ///   ke = 9.81e-3 * 1.0
    ///   d_phi/dv = ke * q * (1-k) * dh_fore_dv = ke * 2000 * 0.97 * 2e-3
    ///   d_phi/dq = ke * (h_net - q * (1-k) * dh_tail) = ke * (364.9625 - 2000*0.97*3e-4)
    ///   d_phi/ds = -ke * q * (1-k) * dh_tail = -ke * 2000 * 0.97 * 3e-4
    #[test]
    fn partial_derivatives_piecewise_tailrace_factor_losses() {
        let forebay = sloped_forebay();
        let tailrace = piecewise_tailrace();
        let losses = HydraulicLossesModel::Factor { value: 0.03 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            None,
            12_600.0,
            "TestPlant".to_owned(),
        );

        let (d_phi_dv, d_phi_dq, d_phi_ds) = pf.partial_derivatives(0.0, 2000.0, 500.0);

        let ke = 9.81e-3_f64;
        let k = 0.03_f64;
        let one_minus_k = 1.0 - k;
        let dh_fore_dv = 2e-3_f64;
        let dh_tail_dq_out = (4.5 - 3.0) / 5000.0; // 3e-4
        let h_fore = 380.0_f64; // sloped_forebay at v=0
        let h_tail = 3.0 + (4.5 - 3.0) * 2500.0 / 5000.0; // 3.75
        let h_net = one_minus_k * (h_fore - h_tail); // 0.97 * 376.25
        let q = 2000.0_f64;

        let expected_dv = ke * q * one_minus_k * dh_fore_dv;
        let expected_dq = ke * (h_net - q * one_minus_k * dh_tail_dq_out);
        let expected_ds = -ke * q * one_minus_k * dh_tail_dq_out;

        assert!(
            (d_phi_dv - expected_dv).abs() < 1e-10,
            "d_phi_dv={d_phi_dv}, expected={expected_dv}"
        );
        assert!(
            (d_phi_dq - expected_dq).abs() < 1e-10,
            "d_phi_dq={d_phi_dq}, expected={expected_dq}"
        );
        assert!(
            (d_phi_ds - expected_ds).abs() < 1e-10,
            "d_phi_ds={d_phi_ds}, expected={expected_ds}"
        );
    }

    #[test]
    fn partial_derivatives_piecewise_tailrace_ds_negative() {
        let forebay = sloped_forebay();
        let tailrace = piecewise_tailrace();
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            None,
            None,
            12_600.0,
            "TestPlant".to_owned(),
        );
        let (_, _, d_phi_ds) = pf.partial_derivatives(10_000.0, 2000.0, 500.0);
        assert!(
            d_phi_ds < 0.0,
            "d_phi_ds must be < 0 with piecewise tailrace, got {d_phi_ds}"
        );
    }

    // ── Finite-difference cross-checks ────────────────────────────────────────

    /// Finite-difference helper: central difference derivative along one dimension.
    fn fd_derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    /// Cross-check: analytical d_phi/dv matches finite difference within 1e-4 relative.
    #[test]
    fn finite_difference_cross_check_dv() {
        let forebay = sloped_forebay();
        let tailrace = piecewise_tailrace();
        let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            12_600.0,
            "TestPlant".to_owned(),
        );

        let v = 5_000.0_f64;
        let q = 2000.0_f64;
        let s = 300.0_f64;
        let h = 1e-3_f64; // 1e-3 hm3 step for volume

        let (d_phi_dv_analytical, _, _) = pf.partial_derivatives(v, q, s);
        let d_phi_dv_fd = fd_derivative(|vi| pf.evaluate(vi, q, s), v, h);

        let rel_err = (d_phi_dv_analytical - d_phi_dv_fd).abs() / d_phi_dv_fd.abs().max(1e-12);
        assert!(
            rel_err < 1e-4,
            "FD check d_phi_dv: analytical={d_phi_dv_analytical}, fd={d_phi_dv_fd}, rel_err={rel_err}"
        );
    }

    /// Cross-check: analytical d_phi/dq matches finite difference within 1e-4 relative.
    #[test]
    fn finite_difference_cross_check_dq() {
        let forebay = sloped_forebay();
        let tailrace = piecewise_tailrace();
        let losses = HydraulicLossesModel::Factor { value: 0.03 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            Some(&EfficiencyModel::Constant { value: 0.92 }),
            12_600.0,
            "TestPlant".to_owned(),
        );

        let v = 7_000.0_f64;
        let q = 3000.0_f64;
        let s = 500.0_f64;
        let h = 1e-4_f64; // 1e-4 m3/s step

        let (_, d_phi_dq_analytical, _) = pf.partial_derivatives(v, q, s);
        let d_phi_dq_fd = fd_derivative(|qi| pf.evaluate(v, qi, s), q, h);

        let rel_err = (d_phi_dq_analytical - d_phi_dq_fd).abs() / d_phi_dq_fd.abs().max(1e-12);
        assert!(
            rel_err < 1e-4,
            "FD check d_phi_dq: analytical={d_phi_dq_analytical}, fd={d_phi_dq_fd}, rel_err={rel_err}"
        );
    }

    /// Cross-check: analytical d_phi/ds matches finite difference within 1e-4 relative.
    #[test]
    fn finite_difference_cross_check_ds() {
        let forebay = sloped_forebay();
        let tailrace = piecewise_tailrace();
        let losses = HydraulicLossesModel::Constant { value_m: 1.5 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            Some(&EfficiencyModel::Constant { value: 0.90 }),
            12_600.0,
            "TestPlant".to_owned(),
        );

        let v = 8_000.0_f64;
        let q = 2500.0_f64;
        let s = 200.0_f64;
        let h = 1e-4_f64;

        let (_, _, d_phi_ds_analytical) = pf.partial_derivatives(v, q, s);
        let d_phi_ds_fd = fd_derivative(|si| pf.evaluate(v, q, si), s, h);

        let rel_err = (d_phi_ds_analytical - d_phi_ds_fd).abs() / d_phi_ds_fd.abs().max(1e-12);
        assert!(
            rel_err < 1e-4,
            "FD check d_phi_ds: analytical={d_phi_ds_analytical}, fd={d_phi_ds_fd}, rel_err={rel_err}"
        );
    }

    /// Cross-check all three derivatives simultaneously with Factor losses.
    #[test]
    fn finite_difference_cross_check_all_derivatives_factor_losses() {
        let forebay = sloped_forebay();
        let tailrace = linear_tailrace_5_5_at_3000();
        let losses = HydraulicLossesModel::Factor { value: 0.05 };
        let pf = ProductionFunction::new(
            forebay,
            Some(&tailrace),
            Some(&losses),
            Some(&EfficiencyModel::Constant { value: 0.95 }),
            12_600.0,
            "TestPlant".to_owned(),
        );

        let v = 6_000.0_f64;
        let q = 1500.0_f64;
        let s = 100.0_f64;

        let (dv, dq, ds) = pf.partial_derivatives(v, q, s);
        let h_v = 1e-3_f64;
        let h_qs = 1e-4_f64;

        let dv_fd = fd_derivative(|vi| pf.evaluate(vi, q, s), v, h_v);
        let dq_fd = fd_derivative(|qi| pf.evaluate(v, qi, s), q, h_qs);
        let ds_fd = fd_derivative(|si| pf.evaluate(v, q, si), s, h_qs);

        let rel_err_v = (dv - dv_fd).abs() / dv_fd.abs().max(1e-12);
        let rel_err_q = (dq - dq_fd).abs() / dq_fd.abs().max(1e-12);
        let rel_err_s = (ds - ds_fd).abs() / ds_fd.abs().max(1e-12);

        assert!(
            rel_err_v < 1e-4,
            "dv: analytical={dv}, fd={dv_fd}, rel_err={rel_err_v}"
        );
        assert!(
            rel_err_q < 1e-4,
            "dq: analytical={dq}, fd={dq_fd}, rel_err={rel_err_q}"
        );
        assert!(
            rel_err_s < 1e-4,
            "ds: analytical={ds}, fd={ds_fd}, rel_err={rel_err_s}"
        );
    }

    // ── Helpers for resolve_fitting_bounds tests ──────────────────────────────

    /// Build a minimal `Hydro` with the given storage bounds.
    fn make_hydro(min_storage_hm3: f64, max_storage_hm3: f64) -> Hydro {
        let zero_penalties = HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        };
        Hydro {
            id: EntityId::from(1),
            name: "TestHydro".to_owned(),
            bus_id: EntityId::from(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3,
            max_storage_hm3,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::Fpha,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 12_600.0,
            min_generation_mw: 0.0,
            max_generation_mw: 14_000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        }
    }

    /// Build a `ForebayTable` spanning `[0.0, 34_116.0]` hm³.
    fn sobradinho_forebay() -> ForebayTable {
        ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap()
    }

    /// Build a simple 2-point `ForebayTable` spanning `[100.0, 2000.0]` hm³.
    fn small_forebay() -> ForebayTable {
        let small_rows = vec![row(100.0, 386.5), row(2_000.0, 390.0)];
        ForebayTable::new(&small_rows, "SmallHydro").unwrap()
    }

    /// Build a default `FphaConfig` (all fields `None` / defaults).
    fn default_config() -> FphaConfig {
        FphaConfig {
            source: "computed".to_owned(),
            volume_discretization_points: None,
            turbine_discretization_points: None,
            spillage_discretization_points: None,
            max_planes_per_hydro: None,
            fitting_window: None,
        }
    }

    // ── AC: no fitting window — forebay defaults, all discretization = 5 ─────

    /// Given config with no fitting window and all discretization fields None,
    /// resolve_fitting_bounds uses forebay range and defaults all counts to 5,
    /// max_planes_per_hydro to 10.
    #[test]
    fn no_fitting_window_uses_forebay_defaults() {
        let config = default_config();
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds: FittingBounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.v_min, 0.0);
        assert_eq!(bounds.v_max, 34_116.0);
        assert_eq!(bounds.n_volume_points, 5);
        assert_eq!(bounds.n_flow_points, 5);
        assert_eq!(bounds.n_spillage_points, 5);
        assert_eq!(bounds.max_planes_per_hydro, 10);
    }

    // ── AC: absolute bounds — both set ────────────────────────────────────────

    /// Given config with absolute volume_min_hm3 = 1000 and volume_max_hm3 = 30000,
    /// resolve_fitting_bounds returns those bounds unchanged (within forebay range).
    #[test]
    fn absolute_bounds_both_set() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(1_000.0),
                volume_max_hm3: Some(30_000.0),
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.v_min, 1_000.0);
        assert_eq!(bounds.v_max, 30_000.0);
    }

    // ── AC: absolute bounds — only min set ───────────────────────────────────

    /// Given only volume_min_hm3, v_max falls back to forebay.v_max().
    #[test]
    fn absolute_bounds_only_min() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(5_000.0),
                volume_max_hm3: None,
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.v_min, 5_000.0);
        assert_eq!(bounds.v_max, 34_116.0);
    }

    // ── AC: absolute bounds — only max set ───────────────────────────────────

    /// Given only volume_max_hm3, v_min falls back to forebay.v_min().
    #[test]
    fn absolute_bounds_only_max() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: None,
                volume_max_hm3: Some(20_000.0),
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.v_min, 0.0);
        assert_eq!(bounds.v_max, 20_000.0);
    }

    // ── AC: percentile bounds ─────────────────────────────────────────────────

    /// Given percentile bounds 0.1 and 0.9 on a hydro with range [100, 2000],
    /// v_min = 100 + 0.1 * 1900 = 290, v_max = 100 + 0.9 * 1900 = 1810.
    #[test]
    fn percentile_bounds_both_set() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: None,
                volume_max_hm3: None,
                volume_min_percentile: Some(0.1),
                volume_max_percentile: Some(0.9),
            }),
            ..default_config()
        };
        let hydro = make_hydro(100.0, 2_000.0);
        let forebay = small_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        let expected_min = 100.0 + 0.1 * 1_900.0; // 290.0
        let expected_max = 100.0 + 0.9 * 1_900.0; // 1810.0
        assert!((bounds.v_min - expected_min).abs() < 1e-10);
        assert!((bounds.v_max - expected_max).abs() < 1e-10);
    }

    // ── AC: mixed — absolute min, percentile max (non-conflicting) ────────────

    /// Absolute min bound + percentile max bound is accepted (different dimensions).
    #[test]
    fn mixed_absolute_min_percentile_max() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(200.0),
                volume_max_hm3: None,
                volume_min_percentile: None,
                volume_max_percentile: Some(0.9),
            }),
            ..default_config()
        };
        let hydro = make_hydro(100.0, 2_000.0);
        let forebay = small_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        let expected_max = 100.0 + 0.9 * 1_900.0; // 1810.0
        assert_eq!(bounds.v_min, 200.0);
        assert!((bounds.v_max - expected_max).abs() < 1e-10);
    }

    // ── AC: conflicting — both absolute and percentile for min ───────────────

    /// volume_min_hm3 and volume_min_percentile both set -> ConflictingFittingWindow.
    #[test]
    fn conflicting_min_bound_returns_error() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(500.0),
                volume_max_hm3: None,
                volume_min_percentile: Some(0.1),
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(100.0, 2_000.0);
        let forebay = small_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(err, FphaFittingError::ConflictingFittingWindow { .. }),
            "expected ConflictingFittingWindow, got: {err:?}"
        );
    }

    // ── AC: conflicting — both absolute and percentile for max ───────────────

    /// volume_max_hm3 and volume_max_percentile both set -> ConflictingFittingWindow.
    #[test]
    fn conflicting_max_bound_returns_error() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: None,
                volume_max_hm3: Some(1_800.0),
                volume_min_percentile: None,
                volume_max_percentile: Some(0.9),
            }),
            ..default_config()
        };
        let hydro = make_hydro(100.0, 2_000.0);
        let forebay = small_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(err, FphaFittingError::ConflictingFittingWindow { .. }),
            "expected ConflictingFittingWindow, got: {err:?}"
        );
    }

    // ── AC: empty range — v_min >= v_max ─────────────────────────────────────

    /// Absolute bounds with v_min > v_max -> EmptyFittingWindow.
    #[test]
    fn inverted_absolute_bounds_returns_empty_window_error() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(1_500.0),
                volume_max_hm3: Some(1_000.0),
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(err, FphaFittingError::EmptyFittingWindow { .. }),
            "expected EmptyFittingWindow, got: {err:?}"
        );
    }

    /// Equal absolute bounds (v_min == v_max) -> EmptyFittingWindow.
    #[test]
    fn equal_absolute_bounds_returns_empty_window_error() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(1_000.0),
                volume_max_hm3: Some(1_000.0),
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(err, FphaFittingError::EmptyFittingWindow { .. }),
            "expected EmptyFittingWindow, got: {err:?}"
        );
    }

    // ── AC: clamping — absolute bounds outside forebay range ─────────────────

    /// Absolute v_min below forebay.v_min() gets clamped to forebay.v_min().
    #[test]
    fn absolute_min_below_forebay_gets_clamped() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(-500.0),
                volume_max_hm3: Some(20_000.0),
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        // Clamped to forebay.v_min() = 0.0
        assert_eq!(bounds.v_min, 0.0);
        assert_eq!(bounds.v_max, 20_000.0);
    }

    /// Absolute v_max above forebay.v_max() gets clamped to forebay.v_max().
    #[test]
    fn absolute_max_above_forebay_gets_clamped() {
        let config = FphaConfig {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(1_000.0),
                volume_max_hm3: Some(50_000.0),
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.v_min, 1_000.0);
        // Clamped to forebay.v_max() = 34_116.0
        assert_eq!(bounds.v_max, 34_116.0);
    }

    // ── AC: discretization defaults ──────────────────────────────────────────

    /// All discretization fields None -> defaults to 5 for each dimension.
    #[test]
    fn discretization_all_none_defaults_to_five() {
        let config = default_config();
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.n_volume_points, 5);
        assert_eq!(bounds.n_flow_points, 5);
        assert_eq!(bounds.n_spillage_points, 5);
    }

    /// Explicit discretization values are passed through unchanged.
    #[test]
    fn discretization_explicit_values_passed_through() {
        let config = FphaConfig {
            volume_discretization_points: Some(8),
            turbine_discretization_points: Some(6),
            spillage_discretization_points: Some(10),
            max_planes_per_hydro: Some(20),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.n_volume_points, 8);
        assert_eq!(bounds.n_flow_points, 6);
        assert_eq!(bounds.n_spillage_points, 10);
        assert_eq!(bounds.max_planes_per_hydro, 20);
    }

    // ── AC: insufficient discretization ──────────────────────────────────────

    /// volume_discretization_points = 1 -> InsufficientDiscretization { dimension: "volume", value: 1 }.
    #[test]
    fn volume_discretization_one_returns_error() {
        let config = FphaConfig {
            volume_discretization_points: Some(1),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        match &err {
            FphaFittingError::InsufficientDiscretization {
                dimension, value, ..
            } => {
                assert_eq!(dimension, "volume");
                assert_eq!(*value, 1);
            }
            other => panic!("expected InsufficientDiscretization, got: {other:?}"),
        }
    }

    /// volume_discretization_points = 0 -> InsufficientDiscretization.
    #[test]
    fn volume_discretization_zero_returns_error() {
        let config = FphaConfig {
            volume_discretization_points: Some(0),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(
                err,
                FphaFittingError::InsufficientDiscretization { ref dimension, value: 0, .. }
                if dimension == "volume"
            ),
            "expected InsufficientDiscretization for volume=0, got: {err:?}"
        );
    }

    /// turbine_discretization_points = 1 -> InsufficientDiscretization { dimension: "turbine", value: 1 }.
    #[test]
    fn turbine_discretization_one_returns_error() {
        let config = FphaConfig {
            turbine_discretization_points: Some(1),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(
                err,
                FphaFittingError::InsufficientDiscretization { ref dimension, value: 1, .. }
                if dimension == "turbine"
            ),
            "expected InsufficientDiscretization for turbine=1, got: {err:?}"
        );
    }

    /// spillage_discretization_points = 1 -> InsufficientDiscretization { dimension: "spillage", value: 1 }.
    #[test]
    fn spillage_discretization_one_returns_error() {
        let config = FphaConfig {
            spillage_discretization_points: Some(1),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(
                err,
                FphaFittingError::InsufficientDiscretization { ref dimension, value: 1, .. }
                if dimension == "spillage"
            ),
            "expected InsufficientDiscretization for spillage=1, got: {err:?}"
        );
    }

    // ── AC: max_planes_per_hydro ──────────────────────────────────────────────

    /// max_planes_per_hydro = None -> defaults to 10.
    #[test]
    fn max_planes_per_hydro_none_defaults_to_ten() {
        let config = default_config();
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.max_planes_per_hydro, 10);
    }

    /// max_planes_per_hydro = Some(5) -> 5.
    #[test]
    fn max_planes_per_hydro_explicit_value() {
        let config = FphaConfig {
            max_planes_per_hydro: Some(5),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

        assert_eq!(bounds.max_planes_per_hydro, 5);
    }

    /// max_planes_per_hydro = Some(0) -> InsufficientDiscretization.
    #[test]
    fn max_planes_per_hydro_zero_returns_error() {
        let config = FphaConfig {
            max_planes_per_hydro: Some(0),
            ..default_config()
        };
        let hydro = make_hydro(0.0, 34_116.0);
        let forebay = sobradinho_forebay();

        let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
        assert!(
            matches!(
                err,
                FphaFittingError::InsufficientDiscretization { ref dimension, value: 0, .. }
                if dimension == "max_planes_per_hydro"
            ),
            "expected InsufficientDiscretization for max_planes_per_hydro=0, got: {err:?}"
        );
    }
}
