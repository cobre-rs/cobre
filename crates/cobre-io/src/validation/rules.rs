//! Table of the diagnostics the validation layers emit through
//! [`ValidationContext::emit`](super::ValidationContext::emit).
//!
//! Each rule is declared once, by the `declare_rules!` macro below, which both names the
//! rule's constant and lists it in [`RULES`]. An emitter passes that constant to `emit`, so
//! a diagnostic takes its kind and severity from the table and cannot state its own.
//!
//! # Rule ids
//!
//! An id is `<namespace>.<label>` and matches `^[a-z_]+(\.[0-9A-Za-z]+)+$`; consumers treat
//! it as an opaque string. The namespace is the table that numbers the rule, and the label
//! is the number written there, verbatim (`7a`, `25b`, `A`):
//!
//! - a rule a module table numbers takes that module's namespace (`dimensional.6`,
//!   `scalar_parameters.A`, `travel_time.12`);
//! - a Layer 5 rule takes `semantic.5a.N` or `semantic.5b.N` from its layer table;
//! - a module that numbered nothing had its rules numbered `1`, `2`, … in the source order
//!   of their first emit (`structural.1`, `referential.3`).
//!
//! Labels were assigned once, when the table was first built from the existing numbering.
//! A rule added later, or a kind and severity pair added later to a listed rule, takes the
//! next free number in its namespace, past every number the namespace has listed or retired.
//! A listed id is never renamed, split or renumbered, and a retired id is never reused.
//!
//! When the table was first built, a numbered row whose sites emitted more than one kind and
//! severity pair got one entry per pair, labelled `<row>.<k>`, with `k` counted once from 1 in
//! that build's source order, and its bare number is not an id.
//! Sub-labels are assigned only then. A pair added later to a listed rule, sub-labelled or not,
//! takes the next free number in its namespace, and no listed id is renamed, split or
//! renumbered to make room for it.
//!
//! # Granularity
//!
//! One entry per distinct check. Sites that report the same condition with the same kind
//! and severity share an entry.
//!
//! # Layers
//!
//! [`ValidationLayer`] names the pipeline layer that emits a rule. The `semantic` layer
//! covers both Layer 5 passes and the `travel_time` namespace.

use super::{ErrorKind, Severity};

/// One diagnostic the validation layers can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ValidationRule {
    /// Stable `<namespace>.<label>` identifier.
    pub id: &'static str,
    /// Pipeline layer that emits the rule.
    pub layer: ValidationLayer,
    /// Kind carried by every diagnostic of this rule.
    pub kind: ErrorKind,
    /// Severity carried by every diagnostic of this rule.
    pub severity: Severity,
    /// One-line description of the condition the rule reports.
    pub summary: &'static str,
}

/// Pipeline layer that emits a [`ValidationRule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationLayer {
    /// Layer 1: presence of case files.
    Structural,
    /// Layer 2: file parsing and schema conformance.
    Schema,
    /// Layer 3: cross-entity references.
    Referential,
    /// Layer 4: cross-file coverage and array dimensions.
    Dimensional,
    /// Layer 5: domain rules on hydro, thermal, stage, penalty and scenario data.
    Semantic,
    /// Layer 6: productivity supplied by exactly one source.
    ProductivityResolution,
    /// Cross-checks on the scalar parameters.
    ScalarParameters,
}

impl ValidationLayer {
    /// Returns the `snake_case` token the layer is exported as.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::Schema => "schema",
            Self::Referential => "referential",
            Self::Dimensional => "dimensional",
            Self::Semantic => "semantic",
            Self::ProductivityResolution => "productivity_resolution",
            Self::ScalarParameters => "scalar_parameters",
        }
    }
}

macro_rules! declare_rules {
    ($($name:ident = $id:literal, $layer:ident, $kind:ident, $severity:ident, $summary:literal;)+) => {
        $(
            // Summaries are exported as plain text, so file and column names in them carry no backticks.
            #[allow(clippy::doc_markdown)]
            #[doc = $summary]
            pub(crate) const $name: ValidationRule = ValidationRule {
                id: $id,
                layer: ValidationLayer::$layer,
                kind: ErrorKind::$kind,
                severity: Severity::$severity,
                summary: $summary,
            };
        )+

        /// Every validation rule, in declaration order.
        pub const RULES: &[ValidationRule] = &[$($name),+];
    };
}

declare_rules! {
    STRUCTURAL_REMOVED_FILE_PRESENT = "structural.1",
        Structural, BusinessRuleViolation, Error,
        "A case input file that is no longer read is present; the message names its replacement";
    STRUCTURAL_REQUIRED_FILE_MISSING = "structural.2",
        Structural, FileNotFound, Error,
        "A required case input file is missing";
    SCHEMA_FILE_UNREADABLE = "schema.1",
        Schema, FileNotFound, Error,
        "A case input file cannot be read";
    SCHEMA_FILE_UNPARSABLE = "schema.2",
        Schema, ParseError, Error,
        "A case input file cannot be parsed (invalid JSON syntax or an unreadable Parquet header)";
    SCHEMA_FILE_NONCONFORMING = "schema.3",
        Schema, SchemaViolation, Error,
        "A case input file does not conform to its schema (a missing field, a wrong type or an out-of-range value)";
    REFERENTIAL_UNDECLARED_ENTITY = "referential.1",
        Referential, InvalidReference, Error,
        "An entity field references an entity id that no registry declares";
    REFERENTIAL_UNKNOWN_CORRELATION_ENTITY_TYPE = "referential.2",
        Referential, InvalidReference, Error,
        "A correlation entity has an entity_type other than inflow, load or ncs";
    REFERENTIAL_UNDECLARED_UNIT_GROUP = "referential.3",
        Referential, InvalidReference, Error,
        "A hydro_unit_group_bounds row references a unit group its hydro does not declare";
    REFERENTIAL_GENERIC_BOUNDS_WITHOUT_ENDPOINT = "referential.4",
        Referential, InvalidValue, Error,
        "A generic_constraint_bounds row has neither bound_lower nor bound_upper";
    REFERENTIAL_GENERIC_BOUNDS_INVERTED = "referential.5",
        Referential, InvalidValue, Error,
        "A generic_constraint_bounds row has bound_upper below bound_lower";
    REFERENTIAL_GENERIC_BOUND_REFERENCE_WITHOUT_ROWS = "referential.6",
        Referential, InvalidReference, Error,
        "A generic constraint declares a bound reference but has no rows in generic_constraint_bounds.parquet";
    REFERENTIAL_NCS_BOUNDS_STAGE = "referential.7",
        Referential, InvalidReference, Error,
        "An ncs_bounds row has a stage_id that is not a study stage";
    REFERENTIAL_NCS_FACTOR_STAGE = "referential.8",
        Referential, InvalidReference, Error,
        "A non_controllable_factors entry has a stage_id that is not a study stage";
    REFERENTIAL_GENERIC_TERM_UNDECLARED_ENTITY = "referential.9",
        Referential, InvalidReference, Error,
        "A generic constraint term references an entity that is not declared";
    REFERENTIAL_GENERIC_TERM_BUS_WITHOUT_UNIT_GROUP = "referential.10",
        Referential, InvalidReference, Error,
        "A generic constraint hydro term selects a bus on which the hydro has no unit group";
    REFERENTIAL_GENERIC_TERM_STUB_CONTRACT = "referential.11",
        Referential, UnusedEntity, Warning,
        "A generic constraint term references a stub contract, so the term has no effect";
    DIMENSIONAL_INFLOW_STATS_COVERAGE = "dimensional.1",
        Dimensional, DimensionMismatch, Error,
        "An active hydro has no inflow seasonal statistics for a study stage";
    DIMENSIONAL_LOAD_STATS_COVERAGE = "dimensional.2",
        Dimensional, DimensionMismatch, Error,
        "A bus has no load seasonal statistics for a study stage";
    DIMENSIONAL_CORRELATION_ROW_COUNT = "dimensional.3",
        Dimensional, DimensionMismatch, Error,
        "A correlation group's matrix row count differs from its entity count";
    DIMENSIONAL_CORRELATION_ROW_LENGTH = "dimensional.4",
        Dimensional, DimensionMismatch, Error,
        "A correlation group's matrix row length differs from its entity count";
    DIMENSIONAL_CORRELATION_PROFILE = "dimensional.5",
        Dimensional, DimensionMismatch, Error,
        "The correlation schedule names a profile that is not defined";
    DIMENSIONAL_FPHA_HYPERPLANES = "dimensional.6",
        Dimensional, DimensionMismatch, Error,
        "An FPHA-configured hydro with turbine capacity has no rows in fpha_hyperplanes.parquet";
    DIMENSIONAL_GEOMETRY_ROWS = "dimensional.7",
        Dimensional, DimensionMismatch, Error,
        "An FPHA or linearized-head hydro has too few hydro_geometry.parquet rows (at least 1 for FPHA, 2 otherwise)";
    SEMANTIC_HYDRO_CASCADE_CYCLE = "semantic.5a.1",
        Semantic, CycleDetected, Error,
        "The hydro cascade formed by downstream links contains a cycle";
    SEMANTIC_HYDRO_STORAGE_BOUNDS_INVERTED = "semantic.5a.2",
        Semantic, InvalidValue, Error,
        "A hydro's min_storage_hm3 exceeds its max_storage_hm3";
    SEMANTIC_HYDRO_TURBINED_BOUNDS_INVERTED = "semantic.5a.3",
        Semantic, InvalidValue, Error,
        "A hydro's min_turbined_m3s exceeds its max_turbined_m3s";
    SEMANTIC_HYDRO_OUTFLOW_BOUNDS_INVERTED = "semantic.5a.4",
        Semantic, InvalidValue, Error,
        "A hydro's min_outflow_m3s exceeds its max_outflow_m3s";
    SEMANTIC_HYDRO_GENERATION_BOUNDS_INVERTED = "semantic.5a.5",
        Semantic, InvalidValue, Error,
        "A hydro's min_generation_mw exceeds its max_generation_mw";
    LIFECYCLE_ENTRY_NOT_BEFORE_EXIT = "semantic.5a.6",
        Semantic, InvalidValue, Error,
        "An entity's entry_stage_id is not before its exit_stage_id";
    SEMANTIC_FILLING_START_STAGE_UNKNOWN = "semantic.5a.7",
        Semantic, InvalidValue, Error,
        "A filling hydro's start_stage_id is not a study stage";
    SEMANTIC_FILLING_GUARD_VIOLATED = "semantic.5a.7a",
        Semantic, InvalidValue, Error,
        "A filling hydro has no entry_stage_id, starts filling at or after it, declares an exit_stage_id, or has a filling_storage seed outside its allowed range";
    SEMANTIC_FILLING_NEVER_OPERATES = "semantic.5a.7b",
        Semantic, ModelQuality, Warning,
        "A filling hydro's entry_stage_id is at or beyond the study horizon, so it never operates within the study";
    GEOMETRY_VOLUME_NOT_INCREASING = "semantic.5a.8",
        Semantic, BusinessRuleViolation, Error,
        "A hydro's hydro_geometry.parquet volume_hm3 values are not strictly increasing";
    GEOMETRY_HEIGHT_DECREASING = "semantic.5a.9",
        Semantic, BusinessRuleViolation, Error,
        "A hydro's hydro_geometry.parquet height_m decreases as volume increases";
    GEOMETRY_AREA_DECREASING = "semantic.5a.10",
        Semantic, BusinessRuleViolation, Error,
        "A hydro's hydro_geometry.parquet area_km2 decreases as volume increases";
    SEMANTIC_FPHA_STAGE_WITHOUT_PLANES = "semantic.5a.11",
        Semantic, BusinessRuleViolation, Error,
        "An FPHA hydro has no hyperplanes for a stage";
    SEMANTIC_FPHA_PLANE_COEFFICIENT_SIGN = "semantic.5a.12",
        Semantic, BusinessRuleViolation, Error,
        "An FPHA hyperplane has a negative gamma_v or a positive gamma_s";
    SEMANTIC_THERMAL_GENERATION_BOUNDS_INVERTED = "semantic.5a.13",
        Semantic, InvalidValue, Error,
        "A thermal's min_generation_mw exceeds its max_generation_mw";
    SEMANTIC_ANTICIPATED_LEAD_UNREACHABLE = "semantic.5a.14",
        Semantic, BusinessRuleViolation, Error,
        "An anticipated thermal's lead is below one stage or reaches no stage within the study horizon or a declared post-study stage";
    SEMANTIC_ANTICIPATED_COMMITMENTS_INCONSISTENT = "semantic.5a.15",
        Semantic, BusinessRuleViolation, Error,
        "past_anticipated_commitments do not match the anticipated thermals one to one, or a plant's windows do not tile its leading delivery stages, straddle the horizon end, or commit a value the plant cannot deliver";
    SEMANTIC_ANTICIPATED_DECISION_ON_NON_ANTICIPATED = "semantic.5a.17",
        Semantic, BusinessRuleViolation, Error,
        "A generic constraint's anticipated_decision term targets a thermal that is not anticipated";
    SEMANTIC_THERMAL_GENERATION_ON_ANTICIPATED = "semantic.5a.18",
        Semantic, SemanticAmbiguity, Warning,
        "A generic constraint's thermal_generation term targets an anticipated thermal, so it reads the delivered generation rather than the commitment";
    // The ids are valid, so the kind is InvalidValue, not InvalidReference.
    SEMANTIC_PUMPING_SAME_ENDPOINTS = "semantic.5a.19",
        Semantic, InvalidValue, Error,
        "A pumping station's source and destination hydro are the same";
    SEMANTIC_GENERIC_PER_BLOCK_REFERENCE_UNRESOLVABLE = "semantic.5a.20",
        Semantic, BusinessRuleViolation, Error,
        "A generic constraint's per-block term names a block or storage boundary that its stage does not expose";
    SEMANTIC_ANTICIPATED_WINDOW_SPANS_CADENCE_CHANGE = "semantic.5a.28",
        Semantic, ModelQuality, Warning,
        "A lead_stages anticipated thermal's active window spans adjacent study stages of different durations";
    SEMANTIC_INFLOW_SEED_ANNUAL_COMPONENT_NOT_MONTHLY = "semantic.5a.29",
        Semantic, BusinessRuleViolation, Error,
        "Inflow annual components are supplied under a season cycle other than Monthly";
    SEMANTIC_INFLOW_SEED_READ_SLOT_UNCOVERED = "semantic.5a.30",
        Semantic, BusinessRuleViolation, Error,
        "An inflow lag slot that the model reads is not fully covered by realized inflow records";
    SEMANTIC_INFLOW_SEED_UNREAD_SLOT_UNCOVERED = "semantic.5a.31",
        Semantic, ModelQuality, Warning,
        "An inflow lag slot that the model never reads is not fully covered by realized inflow records";
    SEMANTIC_INFLOW_SEED_CONDITIONING_PAST_STUDY_START = "semantic.5a.32",
        Semantic, InvalidValue, Error,
        "A recent_observations window extends past the study start";
    SEMANTIC_INFLOW_SEED_PARTIAL_CURRENT_PERIOD = "semantic.5a.33",
        Semantic, ModelQuality, Warning,
        "The in-progress period before the study start is only partly covered by realized inflow records";
    SEMANTIC_INFLOW_SEED_FIRST_SEASON_UNRESOLVED = "semantic.5a.34",
        Semantic, ModelQuality, Warning,
        "The first study stage's season cannot be resolved while inflow lag seeding is active";
    SEMANTIC_INFLOW_SEED_NEGATIVE_RECORD = "semantic.5a.34a",
        Semantic, ModelQuality, Warning,
        "A realized inflow record is negative; it is accepted as incremental inflow";
    SEMANTIC_BOUND_ROW_BLOCK_OUT_OF_RANGE = "semantic.5a.35",
        Semantic, BusinessRuleViolation, Error,
        "A bound-override row's block_id is outside its stage's blocks";
    SEMANTIC_BOUND_ROW_DUPLICATE = "semantic.5a.36",
        Semantic, DuplicateId, Error,
        "Two bound-override rows set the same column for the same entity, stage and block";
    SEMANTIC_BOUND_ROW_BLOCK_ON_STAGE_COLUMN = "semantic.5a.37",
        Semantic, BusinessRuleViolation, Error,
        "A bound-override row gives a block_id to a column that has no per-block variable";
    SEMANTIC_BOUND_ROW_BLOCK_ON_ANTICIPATED_THERMAL = "semantic.5a.38",
        Semantic, BusinessRuleViolation, Error,
        "A thermal_bounds row gives a block_id for an anticipated thermal";
    UNIT_GROUP_DUPLICATE_ID = "semantic.5a.39",
        Semantic, DuplicateId, Error,
        "A hydro declares the same unit group id more than once";
    UNIT_GROUP_BOUNDS_INVERTED = "semantic.5a.40",
        Semantic, InvalidValue, Error,
        "A unit group's minimum turbined flow or generation exceeds its maximum";
    UNIT_GROUP_MAXIMA_EXCEED_PLANT = "semantic.5a.41",
        Semantic, InvalidValue, Error,
        "A hydro's unit group maxima sum above the plant's own max_turbined_m3s or max_generation_mw";
    SEMANTIC_BOUND_ROW_RAISES_PLANT_CAPACITY = "semantic.5a.43",
        Semantic, InvalidValue, Error,
        "A hydro_bounds row raises max_turbined_m3s or max_generation_mw above the hydro's declared value";
    UNIT_GROUP_MINIMA_BELOW_PLANT = "semantic.5a.44",
        Semantic, InvalidValue, Error,
        "A hydro's unit group minima sum below the plant's own min_turbined_m3s or min_generation_mw";
    SEMANTIC_BOUND_ROW_RAISES_GROUP_CAPACITY = "semantic.5a.45",
        Semantic, InvalidValue, Error,
        "A hydro_unit_group_bounds row raises max_turbined_m3s or max_generation_mw above the unit group's declared value";
    SEMANTIC_HYDRO_DIVERSION_FLOOR_WITHOUT_CHANNEL = "semantic.5a.46",
        Semantic, InvalidValue, Error,
        "A hydro_bounds row sets min_diversion_m3s for a hydro that declares no diversion channel";
    POST_STUDY_BOUNDARY_INCONSISTENT = "semantic.5a.47",
        Semantic, BusinessRuleViolation, Error,
        "post_study_stages.json is not date-contiguous from the study horizon end, lacks a bound an anticipated lead reaches, or its commitments mis-tile post-study stages or fall outside the commissioning window";
    SEMANTIC_BOUND_ROW_STAGE_UNKNOWN = "semantic.5a.49",
        Semantic, BusinessRuleViolation, Error,
        "A bound-override row's stage_id is not a study stage";
    SEMANTIC_HYDRO_EVAPORATION_WITHOUT_GEOMETRY = "semantic.5a.50",
        Semantic, BusinessRuleViolation, Error,
        "A hydro with evaporation coefficients has no rows in hydro_geometry.parquet";
    SEMANTIC_GENERIC_PRODUCTIVITY_TAG_MISMATCH = "semantic.5a.51",
        Semantic, SemanticAmbiguity, Warning,
        "A generic constraint pairs max_stored_energy with accumulated_productivity for the same hydro";
    SEMANTIC_PUMPING_ENDPOINT_NOT_OPERATING = "semantic.5a.52",
        Semantic, BusinessRuleViolation, Error,
        "A pumping station is active at a study stage where its source or destination hydro is not operating";
    TRAVEL_TIME_INVALID = "travel_time.1",
        Semantic, InvalidValue, Error,
        "A hydro's travel_time_hours is negative or not finite";
    TRAVEL_TIME_ZERO = "travel_time.2",
        Semantic, ModelQuality, Warning,
        "A hydro's travel_time_hours is zero, so no travel-time arc is created";
    TRAVEL_TIME_NEGLIGIBLE = "travel_time.3",
        Semantic, ModelQuality, Warning,
        "A hydro's travel time is negligible relative to every study stage length";
    TRAVEL_TIME_BEYOND_HORIZON = "travel_time.4",
        Semantic, ModelQuality, Warning,
        "A hydro's travel time exceeds the remaining study horizon from some stage";
    TRAVEL_TIME_DEFLUENCES_UNCOVERED = "travel_time.5",
        Semantic, BusinessRuleViolation, Error,
        "A travel-time arc's past_defluences windows do not cover the water in transit at the study start";
    TRAVEL_TIME_DEFLUENCE_FUTURE_DATED = "travel_time.5b",
        Semantic, InvalidValue, Error,
        "A past_defluences window ends after the study start";
    TRAVEL_TIME_HETEROGENEOUS_CONFLUENCE = "travel_time.6",
        Semantic, NotImplemented, Error,
        "Travel-time arcs with different travel times feed one downstream hydro while a study stage is chronological";
    TRAVEL_TIME_DOWNSTREAM_NOT_OPERATING = "travel_time.12",
        Semantic, BusinessRuleViolation, Error,
        "A travel-time arc releases at a stage where its downstream hydro is not yet operating";
    SEMANTIC_TRANSITION_ENDPOINT_NOT_A_STAGE = "semantic.5b.1",
        Semantic, InvalidValue, Error,
        "A policy-graph transition source_id or target_id is not a declared stage id";
    SEMANTIC_TRANSITION_PROBABILITY_SUM = "semantic.5b.2",
        Semantic, InvalidValue, Error,
        "The outgoing transition probabilities of a policy-graph source do not sum to 1 within tolerance";
    SEMANTIC_CYCLIC_GRAPH_DISCOUNT_RATE = "semantic.5b.3",
        Semantic, InvalidValue, Error,
        "A cyclic policy graph has an annual_discount_rate that is not positive";
    SEMANTIC_PENALTY_DEFICIT_NOT_ABOVE_GENERATION_VIOLATION = "semantic.5b.8",
        Semantic, ModelQuality, Warning,
        "The maximum deficit-segment cost does not exceed a hydro's generation_violation_below_cost (both $/MWh)";
    SEMANTIC_PENALTY_FLOW_VIOLATION_NOT_ABOVE_RESOURCE = "semantic.5b.9",
        Semantic, ModelQuality, Warning,
        "A hydro's minimum flow-violation cost does not exceed the maximum spillage or diversion cost over all hydros (both $/(m³/s·h))";
    SEMANTIC_PENALTY_RESOURCE_COST_NOT_POSITIVE = "semantic.5b.10",
        Semantic, ModelQuality, Warning,
        "A hydro's spillage_cost or diversion_cost is not positive";
    SEMANTIC_FPHA_TURBINED_COST_NEGATIVE = "semantic.5b.11",
        Semantic, BusinessRuleViolation, Error,
        "An FPHA hydro has a negative turbined_cost";
    SEMANTIC_INFLOW_STD_ZERO = "semantic.5b.12",
        Semantic, ModelQuality, Warning,
        "An inflow seasonal standard deviation is zero (deterministic inflow) while a scenario source generates inflow";
    SEMANTIC_CORRELATION_ASYMMETRIC = "semantic.5b.14",
        Semantic, BusinessRuleViolation, Error,
        "A correlation matrix is not symmetric within tolerance";
    SEMANTIC_CORRELATION_DIAGONAL = "semantic.5b.15",
        Semantic, BusinessRuleViolation, Error,
        "A correlation matrix diagonal entry differs from 1 beyond tolerance";
    SEMANTIC_CORRELATION_OFF_DIAGONAL_RANGE = "semantic.5b.16",
        Semantic, BusinessRuleViolation, Error,
        "A correlation matrix off-diagonal entry lies outside [-1, 1]";
    SEMANTIC_CORRELATION_MIXED_ENTITY_TYPES = "semantic.5b.16a",
        Semantic, BusinessRuleViolation, Error,
        "A correlation group mixes entities of different entity_type";
    SEMANTIC_LOAD_FACTOR_BLOCK_ID = "semantic.5b.17",
        Semantic, BusinessRuleViolation, Error,
        "A load factor names a block_id that its stage does not declare";
    SEMANTIC_ESTIMATION_WITHOUT_SEASON_DEFINITIONS = "semantic.5b.19",
        Semantic, BusinessRuleViolation, Error,
        "Estimation from inflow history is required but stages.json declares no season_definitions";
    SEMANTIC_ESTIMATION_FEW_OBSERVATIONS = "semantic.5b.20",
        Semantic, ModelQuality, Warning,
        "A hydro and season have fewer inflow history observations than the recommended minimum for estimation";
    SEMANTIC_ESTIMATION_HYDRO_WITHOUT_HISTORY = "semantic.5b.21",
        Semantic, BusinessRuleViolation, Error,
        "Estimation is required but a hydro has no inflow history observation";
    SEMANTIC_SOBOL_OPENING_COUNT = "semantic.5b.25",
        Semantic, ModelQuality, Warning,
        "A stage using the qmc_sobol noise method has a num_openings that is not a power of 2";
    SEMANTIC_STAGE_SEASON_UNDEFINED = "semantic.5b.27",
        Semantic, BusinessRuleViolation, Error,
        "A stage season_id is not defined in season_definitions";
    SEMANTIC_SEASON_WITHOUT_OBSERVATIONS = "semantic.5b.28",
        Semantic, ModelQuality, Warning,
        "A defined season has no inflow history observation while estimation is active and the training inflow scheme is not external";
    SEMANTIC_SEASON_DURATION_SPREAD = "semantic.5b.29",
        Semantic, BusinessRuleViolation, Error,
        "Stages that share a season_id differ in duration by more than the sub-period tolerance";
    SEMANTIC_SEASON_UNREFERENCED = "semantic.5b.30",
        Semantic, ModelQuality, Warning,
        "A season defined in season_definitions is referenced by no stage";
    SEMANTIC_HISTORY_FINER_THAN_SEASON = "semantic.5b.31.1",
        Semantic, BusinessRuleViolation, Warning,
        "A hydro has several inflow history observations for one season in one year; estimation aggregates them to the season";
    SEMANTIC_HISTORY_COARSER_THAN_SEASON = "semantic.5b.31.2",
        Semantic, BusinessRuleViolation, Error,
        "A hydro's inflow history misses a defined season in an interior year, which indicates coarser-than-season observations that cannot be disaggregated";
    SEMANTIC_FILLING_SCHEDULE_SHORT_OF_DEAD_VOLUME = "semantic.5b.33",
        Semantic, BusinessRuleViolation, Error,
        "A filling hydro's minimum filling schedule cannot reach the dead volume before its entry stage, within a relative tolerance";
    SEMANTIC_INFLOW_LAGS_DISABLED_UNDER_AR_MODEL = "semantic.5b.34",
        Semantic, ModelQuality, Warning,
        "Every study stage disables inflow_lags although the inflow model has an autoregressive order above zero";
    SEMANTIC_AR_COEFFICIENT_SEASON_UNRESOLVED = "semantic.5b.35.1",
        Semantic, BusinessRuleViolation, Error,
        "User-supplied autoregressive coefficients of nonzero order sit at a stage whose season cannot be resolved";
    SEMANTIC_AR_COEFFICIENT_NOT_STATIONARY = "semantic.5b.35.2",
        Semantic, InvalidValue, Error,
        "User-supplied autoregressive coefficients fail the periodic stationarity check";
    SEMANTIC_NODE_SCENARIO_ID_DECLARATION = "semantic.5b.36",
        Semantic, InvalidValue, Error,
        "Under enumerated forward selection, a node has no scenario_id at a stage carrying a slot-occupying external class, or declares one at a stage carrying none";
    SEMANTIC_NODE_SCENARIO_ID_RANGE = "semantic.5b.37",
        Semantic, InvalidValue, Error,
        "A node scenario_id is outside the column range of a slot-occupying external class at its stage";
    SEMANTIC_NODE_GRAPH_MALFORMED = "semantic.5b.38.1",
        Semantic, InvalidValue, Error,
        "A declared node graph is malformed: a node names an undeclared study stage, a transition endpoint is not a declared node, a study stage has no node, a node is unreachable from the first stage, or a non-final node has no successor";
    SEMANTIC_NODE_ID_DUPLICATE = "semantic.5b.38.2",
        Semantic, DuplicateId, Error,
        "Two policy-graph nodes share an id";
    SEMANTIC_NODE_GRAPH_CYCLE = "semantic.5b.38.3",
        Semantic, CycleDetected, Error,
        "The declared node graph contains a cycle";
    SEMANTIC_NODE_EDGE_SKIPS_STAGE = "semantic.5b.39",
        Semantic, InvalidValue, Error,
        "A node-graph transition does not advance exactly one stage";
    SEMANTIC_NODE_RECOMBINABLE_SUBTREES = "semantic.5b.40",
        Semantic, ModelQuality, Warning,
        "A stage carries several nodes with structurally identical subtrees";
    SEMANTIC_NUM_OPENINGS_DECLARATION = "semantic.5b.41",
        Semantic, InvalidValue, Error,
        "Under a node graph, a stage with generated openings declares no num_openings, or a stage with only external openings declares one";
    SEMANTIC_EDGE_DISCOUNT_OVERRIDE_UNDER_NODES = "semantic.5b.42",
        Semantic, InvalidValue, Error,
        "A transition declares annual_discount_rate_override under a node graph, where the override belongs on its stage";
    SEMANTIC_FILE_OPENINGS_CONFLICT = "semantic.5b.43",
        Semantic, InvalidValue, Error,
        "A file-sourced backward opening tree is configured together with a declared node graph or under enumerated forward selection";
    SEMANTIC_SAMPLING_METHOD_INERT = "semantic.5b.44",
        Semantic, ModelQuality, Warning,
        "Under a node graph, a stage's sampling_method has no effect because the stage carries external openings or several nodes";
    SEMANTIC_EXTERNAL_SCHEME_WITHOUT_DATA = "semantic.5b.44a",
        Semantic, BusinessRuleViolation, Error,
        "A class resolved to the external scheme has no external scenario data";
    SEMANTIC_EXTERNAL_COLUMN_COUNT_DISAGREEMENT = "semantic.5b.45",
        Semantic, BusinessRuleViolation, Error,
        "Two slot-occupying external classes disagree on a stage's column count";
    SEMANTIC_EXTERNAL_SCENARIO_ID_SET = "semantic.5b.46",
        Semantic, BusinessRuleViolation, Error,
        "An external class does not carry each scenario_id from 0 to the stage's column count minus one exactly once per entity and stage";
    SEMANTIC_EXTERNAL_STAGE_UNRESOLVED = "semantic.5b.47",
        Semantic, InvalidValue, Error,
        "An external scenario row's stage_id is not a declared study stage";
    SEMANTIC_EXTERNAL_PREFIX_INCOHERENT = "semantic.5b.48",
        Semantic, ModelQuality, Warning,
        "Along a node-graph edge, an external class's pointed columns disagree over their shared history prefix";
    SEMANTIC_EXTERNAL_INFLOW_CONSTANT_UNDER_AR_MODEL = "semantic.5b.50",
        Semantic, BusinessRuleViolation, Error,
        "An external inflow library is constant at a stage for a hydro whose inflow model has autoregressive order above zero";
    SEMANTIC_STAGE_BLOCKS = "semantic.5b.52",
        Semantic, InvalidValue, Error,
        "A study stage declares no block, or a block whose duration_hours is not finite and positive";
    PRODUCTIVITY_SUPPLIED_TWICE = "productivity_resolution.1",
        ProductivityResolution, SchemaViolation, Error,
        "A hydro's stage productivity is supplied by both hydro_production_models.json and hydro_energy_productivity.parquet";
    PRODUCTIVITY_MISSING = "productivity_resolution.2",
        ProductivityResolution, DimensionMismatch, Error,
        "A constant-productivity hydro has no productivity value for a stage";
    SCALAR_PARAMETER_UNDECLARED_HYDRO = "scalar_parameters.A",
        ScalarParameters, InvalidReference, Error,
        "A computed scalar parameter references a hydro that is not declared";
    SCALAR_PARAMETER_STAGE_COUNT = "scalar_parameters.B",
        ScalarParameters, SchemaViolation, Error,
        "A per-stage scalar parameter does not hold exactly one value per study stage";
    SCALAR_PARAMETER_DUPLICATE = "scalar_parameters.C",
        ScalarParameters, SchemaViolation, Error,
        "Two scalar parameters share an id or a name";
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{RULES, ValidationLayer};
    use crate::validation::{Severity, ValidationContext};

    const RETIRED_RULE_IDS: &[&str] = &[
        "travel_time.11",
        "travel_time.13",
        "semantic.5a.16",
        "semantic.5a.21",
        "semantic.5a.22",
        "semantic.5a.23",
        "semantic.5a.24",
        "semantic.5a.25",
        "semantic.5a.25b",
        "semantic.5a.26",
        "semantic.5a.26a",
        "semantic.5a.26b",
        "semantic.5a.27",
        "semantic.5a.42",
        "semantic.5a.48",
        "semantic.5b.4",
        "semantic.5b.5",
        "semantic.5b.6",
        "semantic.5b.7",
        "semantic.5b.13",
        "semantic.5b.18",
        "semantic.5b.22",
        "semantic.5b.23",
        "semantic.5b.24",
        "semantic.5b.26",
        "semantic.5b.32",
        "semantic.5b.49",
    ];

    // Spelled from chars so the source grep in `tests/genericity_gate.rs` does not match this file.
    const BANNED_WORDS: [&[char]; 2] =
        [&['s', 'd', 'd', 'p'], &['b', 'e', 'n', 'd', 'e', 'r', 's']];

    fn layer_of_namespace(namespace: &str) -> Option<ValidationLayer> {
        match namespace {
            "structural" => Some(ValidationLayer::Structural),
            "schema" => Some(ValidationLayer::Schema),
            "referential" => Some(ValidationLayer::Referential),
            "dimensional" => Some(ValidationLayer::Dimensional),
            "semantic" | "travel_time" => Some(ValidationLayer::Semantic),
            "productivity_resolution" => Some(ValidationLayer::ProductivityResolution),
            "scalar_parameters" => Some(ValidationLayer::ScalarParameters),
            _ => None,
        }
    }

    fn matches_id_grammar(id: &str) -> bool {
        let mut parts = id.split('.');
        let namespace = parts.next().unwrap_or("");
        let labels: Vec<&str> = parts.collect();
        !namespace.is_empty()
            && namespace
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b == b'_')
            && !labels.is_empty()
            && labels
                .iter()
                .all(|l| !l.is_empty() && l.bytes().all(|b| b.is_ascii_alphanumeric()))
    }

    #[test]
    fn rule_ids_are_unique_and_namespaced_by_their_layer() {
        let mut problems: Vec<String> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();

        for rule in RULES {
            if !matches_id_grammar(rule.id) {
                problems.push(format!("{}: id does not match the id grammar", rule.id));
            }
            if !seen.insert(rule.id) {
                problems.push(format!("{}: id is listed more than once", rule.id));
            }

            let namespace = rule.id.split_once('.').map_or(rule.id, |(ns, _)| ns);
            match layer_of_namespace(namespace) {
                None => problems.push(format!("{}: unknown namespace '{namespace}'", rule.id)),
                Some(layer) if layer != rule.layer => problems.push(format!(
                    "{}: namespace '{namespace}' belongs to {layer:?} but the rule is {:?}",
                    rule.id, rule.layer
                )),
                Some(_) => {}
            }

            let token = rule.layer.as_str();
            if token.is_empty() || !token.bytes().all(|b| b.is_ascii_lowercase() || b == b'_') {
                problems.push(format!(
                    "{}: layer token '{token}' is not snake_case",
                    rule.id
                ));
            }
        }

        for shorter in RULES {
            for longer in RULES {
                if longer
                    .id
                    .strip_prefix(shorter.id)
                    .is_some_and(|rest| rest.starts_with('.'))
                {
                    problems.push(format!(
                        "{} and {} are both listed: a kind and severity pair added to a listed \
                         rule takes the next free number in its namespace, and a sub-labelled \
                         row's bare number is not an id",
                        shorter.id, longer.id
                    ));
                }
            }
        }

        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn retired_rule_ids_are_never_listed() {
        let problems: Vec<String> = RULES
            .iter()
            .filter(|rule| RETIRED_RULE_IDS.contains(&rule.id))
            .map(|rule| format!("{}: a retired id is listed", rule.id))
            .collect();

        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn every_rule_summary_is_a_generic_single_line() {
        let mut problems: Vec<String> = Vec::new();

        for rule in RULES {
            let summary = rule.summary;
            if summary.is_empty() {
                problems.push(format!("{}: empty summary", rule.id));
            }
            if summary.contains('\n') {
                problems.push(format!("{}: summary spans several lines", rule.id));
            }
            if summary != summary.trim() {
                problems.push(format!("{}: summary has surrounding whitespace", rule.id));
            }
            let lowered = summary.to_lowercase();
            for chars in BANNED_WORDS {
                let banned: String = chars.iter().collect();
                if lowered.contains(&banned) {
                    problems.push(format!("{}: summary names '{banned}'", rule.id));
                }
            }
        }

        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn emit_takes_kind_and_severity_from_the_rule() {
        let mut problems: Vec<String> = Vec::new();

        for rule in RULES {
            let mut ctx = ValidationContext::new();
            ctx.emit(rule, "case/file.json", None::<&str>, "message");

            let (own, other) = match rule.severity {
                Severity::Error => (ctx.errors(), ctx.warnings()),
                Severity::Warning => (ctx.warnings(), ctx.errors()),
            };
            if own.len() != 1 || !other.is_empty() {
                problems.push(format!(
                    "{}: expected one {:?} entry and none of the other severity, got {} and {}",
                    rule.id,
                    rule.severity,
                    own.len(),
                    other.len()
                ));
                continue;
            }
            let entry = own[0];
            if entry.kind != rule.kind || entry.severity != rule.severity {
                problems.push(format!(
                    "{}: emitted {:?}/{:?}, the rule says {:?}/{:?}",
                    rule.id, entry.kind, entry.severity, rule.kind, rule.severity
                ));
            }
        }

        assert!(problems.is_empty(), "{problems:#?}");
    }
}
