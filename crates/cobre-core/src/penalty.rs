//! Penalty resolution and pre-resolved penalty structures.
//!
//! The penalty system uses a three-tier resolution cascade: global defaults,
//! entity-level overrides, and stage-level overrides (DEC-006). After resolution,
//! penalties are stored as pre-computed per-(entity, stage) values so the training
//! loop does not need to re-evaluate the cascade on each iteration.
