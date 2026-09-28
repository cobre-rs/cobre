//! Entity system/local index vocabulary.
//!
//! A bare `usize` does not distinguish an entity's canonical *system* position
//! (its `0..n_h`-style slot in `System::hydros`/`thermals`/`lines`) from its
//! *local* position within a per-stage sparse identity list (e.g. its slot in
//! `fpha_hydro_indices`) — the `fpha_local` vs `h_sys` confusion this
//! vocabulary forbids still compiles and silently addresses the wrong LP
//! column. [`HydroSys`], [`ThermalSys`], [`LineSys`], [`BusSys`], [`NcsSys`],
//! and [`PumpingSys`] carry a system position; [`FphaLocal`], [`EvapLocal`],
//! [`FillingTargetLocal`],
//! [`FloorLocal`], and [`AnticipatedLocal`] each carry a position within their
//! own named local list, and [`FphaCellLocal`] a position within the stage's
//! plant-major FPHA-*cell* sequence (finer than [`FphaLocal`]: a split FPHA plant
//! spans several). [`HydroCell`] carries neither: it is a position within
//! the study-scope [`HydroCellIndex`](super::HydroCellIndex) partition, addressed
//! by its own CSR ranges rather than a per-stage list. None of these types
//! carries arithmetic: offset formulas stay with the owning value type — these
//! types only gate which `usize` crosses which boundary.
//!
//! `ContractSys` is deliberately NOT introduced: contract columns are addressed
//! by the per-direction slot from `contract_family_slot`, never by a system
//! index, so a system-index type would have no call site.
//!
//! ## Cross-family assignment pins
//!
//! No type here coerces into another — a copy-paste swap between
//! two entity families (e.g. handing an `FphaLocal` where a `HydroSys` is
//! required) is a compile error, not a silently wrong LP column:
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{FphaLocal, HydroSys};
//!
//! let _wrong: HydroSys = FphaLocal::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{EvapLocal, HydroSys};
//!
//! let _wrong: HydroSys = EvapLocal::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{HydroSys, ThermalSys};
//!
//! let _wrong: HydroSys = ThermalSys::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{AnticipatedLocal, ThermalSys};
//!
//! let _wrong: ThermalSys = AnticipatedLocal::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{EvapLocal, FphaLocal};
//!
//! let _wrong: FphaLocal = EvapLocal::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{HydroSys, LineSys};
//!
//! let _wrong: HydroSys = LineSys::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{HydroCell, HydroSys};
//!
//! let _wrong: HydroCell = HydroSys::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{FphaCellLocal, FphaLocal};
//!
//! let _wrong: FphaLocal = FphaCellLocal::new(0);
//! ```
//!
//! ```compile_fail
//! use cobre_sddp::indexer::{NcsSys, PumpingSys};
//!
//! let _wrong: NcsSys = PumpingSys::new(0);
//! ```

/// A hydro's canonical system position (its slot in `System::hydros`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct HydroSys(usize);

impl HydroSys {
    /// Wrap a raw hydro system-index position.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw hydro system-index position.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A `bus_id` equivalence class over one plant's unit groups — a cell is a
/// class of groups, never a group itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct HydroCell(usize);

impl HydroCell {
    /// Wrap a raw hydro-cell index position.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw hydro-cell index position.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A thermal's canonical system position (its slot in `System::thermals`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ThermalSys(usize);

impl ThermalSys {
    /// Wrap a raw thermal system-index position.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw thermal system-index position.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A line's canonical system position (its slot in `System::lines`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct LineSys(usize);

impl LineSys {
    /// Wrap a raw line system-index position.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw line system-index position.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A bus's canonical system position (its slot in `System::buses`, declaration
/// order) — distinct from a stochastic load slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BusSys(usize);

impl BusSys {
    /// Wrap a raw bus system-index position.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw bus system-index position.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A non-controllable source's canonical system position (its slot in the
/// full, id-sorted NCS list — the dense layout's column order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct NcsSys(usize);

impl NcsSys {
    /// Wrap a raw NCS system-index position.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw NCS system-index position.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A pumping station's canonical system position (its slot in the full
/// system station list — the dense layout's column order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct PumpingSys(usize);

impl PumpingSys {
    /// Wrap a raw pumping-station system-index position.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw pumping-station system-index position.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A hydro's position within a stage's `fpha_hydro_indices` local list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FphaLocal(usize);

impl FphaLocal {
    /// Wrap a raw FPHA-local index.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw FPHA-local index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A cell's position among the cells of FPHA plants, in the same plant-major
/// order `fpha_hydro_indices`-driven callers already use — distinct from
/// [`FphaLocal`] (a PLANT's position among FPHA plants): a split FPHA plant
/// contributes more than one [`FphaCellLocal`] but exactly one [`FphaLocal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FphaCellLocal(usize);

impl FphaCellLocal {
    /// Wrap a raw FPHA-cell-local index.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw FPHA-cell-local index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A hydro's position within a stage's evaporation-model local list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct EvapLocal(usize);

impl EvapLocal {
    /// Wrap a raw evaporation-local index.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw evaporation-local index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A hydro's position within a stage's filling-target local list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FillingTargetLocal(usize);

impl FillingTargetLocal {
    /// Wrap a raw filling-target-local index.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw filling-target-local index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A hydro's position within a stage's storage-floor local list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FloorLocal(usize);

impl FloorLocal {
    /// Wrap a raw storage-floor-local index.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw storage-floor-local index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A thermal's position within a stage's anticipated-commitment local list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct AnticipatedLocal(usize);

impl AnticipatedLocal {
    /// Wrap a raw anticipated-local index.
    #[inline]
    #[must_use]
    pub fn new(v: usize) -> Self {
        Self(v)
    }

    /// Extract the raw anticipated-local index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AnticipatedLocal, BusSys, EvapLocal, FillingTargetLocal, FloorLocal, FphaCellLocal,
        FphaLocal, HydroCell, HydroSys, LineSys, NcsSys, PumpingSys, ThermalSys,
    };

    #[test]
    fn hydro_sys_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<HydroSys>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(HydroSys::new(3).get(), 3);
    }

    #[test]
    fn hydro_cell_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<HydroCell>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(HydroCell::new(7).get(), 7);
    }

    #[test]
    fn thermal_sys_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<ThermalSys>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(ThermalSys::new(11).get(), 11);
    }

    #[test]
    fn line_sys_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<LineSys>(), std::mem::size_of::<usize>());
        assert_eq!(LineSys::new(4).get(), 4);
    }

    #[test]
    fn bus_sys_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<BusSys>(), std::mem::size_of::<usize>());
        assert_eq!(BusSys::new(5).get(), 5);
    }

    #[test]
    fn ncs_sys_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<NcsSys>(), std::mem::size_of::<usize>());
        assert_eq!(NcsSys::new(6).get(), 6);
    }

    #[test]
    fn pumping_sys_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<PumpingSys>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(PumpingSys::new(7).get(), 7);
    }

    #[test]
    fn fpha_local_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<FphaLocal>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(FphaLocal::new(2).get(), 2);
    }

    #[test]
    fn fpha_cell_local_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<FphaCellLocal>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(FphaCellLocal::new(9).get(), 9);
    }

    #[test]
    fn evap_local_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<EvapLocal>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(EvapLocal::new(6).get(), 6);
    }

    #[test]
    fn filling_target_local_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<FillingTargetLocal>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(FillingTargetLocal::new(8).get(), 8);
    }

    #[test]
    fn floor_local_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<FloorLocal>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(FloorLocal::new(1).get(), 1);
    }

    #[test]
    fn anticipated_local_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<AnticipatedLocal>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(AnticipatedLocal::new(5).get(), 5);
    }
}
