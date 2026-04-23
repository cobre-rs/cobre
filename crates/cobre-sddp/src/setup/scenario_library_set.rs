//! Nested scenario-library containers for training and simulation phases.
//!
//! [`ScenarioLibraries`] groups the per-phase sampling configuration and
//! optional pre-built libraries into two [`PhaseLibraries`] values — one for
//! training and one for simulation — eliminating the flat `sim_`-prefix
//! duplication that previously existed on [`super::StudySetup`].

use cobre_core::scenario::SamplingScheme;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary};

/// Sampling schemes and optional pre-built libraries for a single execution
/// phase (training or simulation).
///
/// The seven fields cover the three entity classes (inflow, load, NCS): one
/// [`SamplingScheme`] and one optional library per class, plus the optional
/// historical inflow library.
///
/// Field names drop the redundant `_library` / `_scheme` suffix because the
/// enclosing struct name already conveys the "library" context.
#[derive(Debug)]
pub struct PhaseLibraries {
    /// Forward-pass noise source scheme for the inflow entity class.
    pub inflow_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the load entity class.
    pub load_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the NCS entity class.
    pub ncs_scheme: SamplingScheme,
    /// Pre-standardized historical inflow windows library.
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::Historical`, else `None`.
    pub historical: Option<HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library.
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::External`, else `None`.
    pub external_inflow: Option<ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library.
    ///
    /// `Some` when `load_scheme == SamplingScheme::External`, else `None`.
    pub external_load: Option<ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library.
    ///
    /// `Some` when `ncs_scheme == SamplingScheme::External`, else `None`.
    pub external_ncs: Option<ExternalScenarioLibrary>,
}

/// Training and simulation [`PhaseLibraries`] grouped as a pair.
///
/// Replaces the 14 flat `inflow_scheme` / `sim_inflow_scheme` / … fields
/// previously held directly on [`super::StudySetup`]. The training/simulation
/// distinction is now structural rather than name-prefix-based.
///
/// ## Simulation `None` fields
///
/// For each entity class, `simulation.<library>` is `None` when the
/// simulation scheme is identical to the training scheme — the
/// simulation context falls back to the training library in that case.
/// This asymmetry is explicit: `PhaseLibraries` always stores `None` for the
/// simulation phase when it would share the training library.
#[derive(Debug)]
pub struct ScenarioLibraries {
    /// Libraries and schemes used during the training (backward-pass) phase.
    pub training: PhaseLibraries,
    /// Libraries and schemes used during the simulation phase.
    ///
    /// Optional libraries that equal their training counterparts are stored as
    /// `None`; the simulation context resolves them via fallback.
    pub simulation: PhaseLibraries,
}
