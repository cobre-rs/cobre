# Epic 03: Validation and Testing

## Goal

Implement cross-reference validation, cascade DAG validation, bus connectivity checks, and comprehensive integration tests including declaration-order invariance. After this epic, `SystemBuilder::build()` performs full validation and the crate has thorough test coverage demonstrating correctness.

## Scope

- Cross-reference validation: verify all bus_id, downstream_id, source_hydro_id, destination_hydro_id references point to existing entities
- Cascade DAG validation: detect cycles in the hydro cascade graph
- Filling config validation: verify `start_stage_id < entry_stage_id` and other filling constraints
- Penalty value validation: non-negative costs, FPHA ordering rule
- Bus connectivity check (warning-level): buses with no connections
- Declaration-order invariance tests: same entities in different input orders produce identical System
- Integration tests: realistic multi-entity System construction and querying
- Penalty ordering validation (warning-level): checks from Penalty System spec section 2

## Non-Goals

- Stage-varying validation (requires stages from Phase 2)
- JSON schema validation (Phase 2: cobre-io)
- Performance benchmarks (not yet meaningful)

## Tickets

1. **ticket-009**: Implement cross-reference validation in SystemBuilder
2. **ticket-010**: Implement cascade cycle detection and filling config validation
3. **ticket-011**: Implement integration tests and declaration-order invariance tests

## Dependencies

- Blocked by: Epic 02 (System struct and topology)
- Blocks: Nothing within Phase 1 -- this completes Phase 1

## Success Criteria

- `SystemBuilder::build()` rejects invalid cross-references with descriptive errors
- `SystemBuilder::build()` rejects cascade cycles
- `SystemBuilder::build()` rejects invalid filling configs
- Declaration-order invariance test passes
- Full `cargo test`, `cargo clippy`, and `cargo doc` clean
