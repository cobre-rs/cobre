//! Parsing for `constraints/generic_constraints.json` — user-defined linear constraints.
//!
//! [`parse_generic_constraints`] reads `constraints/generic_constraints.json` and
//! returns a sorted `Vec<GenericConstraint>`.
//!
//! ## JSON structure (spec SS3)
//!
//! ```json
//! {
//!   "constraints": [
//!     {
//!       "id": 0,
//!       "name": "min_southeast_hydro",
//!       "description": "...",
//!       "expression": "hydro_generation(0) + hydro_generation(1) <= @demanda_sin",
//!       "slack": { "enabled": true, "penalty": 5000.0 }
//!     }
//!   ]
//! }
//! ```
//!
//! An RHS endpoint is authored inline, in the `expression` string itself — see
//! the relational grammar below — rather than as a separate JSON field.
//!
//! ## Expression grammar (spec SS3)
//!
//! ```text
//! relation   ::= side (('<=' | '>=' | '==') side)?
//! side       ::= side_term (('+' | '-') side_term)*
//! side_term  ::= term
//!              | coefficient                             (only in a relation with an operator)
//! term       ::= coefficient '*' '@' name '*' variable   (parameter coefficient, scaled)
//!              | '@' name '*' variable                   (parameter coefficient)
//!              | coefficient '*' '@' name                (named-expression reference, scaled)
//!              | '@' name                                (named-expression reference)
//!              | coefficient '*' variable
//!              | coefficient '*' group
//!              | group
//!              | variable
//! group      ::= '(' term (('+' | '-') term)* ')'
//! variable   ::= var_name '(' entity_id (',' block_id)? (',' 'bus' '=' bus_id)? ')'
//! ```
//!
//! An `@name` immediately followed by `* variable` is a scalar-parameter
//! coefficient, resolved against `name_to_id` supplied by the caller (pass
//! `&HashMap::new()` when no parameters are loaded; such a coefficient token then
//! errors as an unknown parameter). A standalone `@name`, or a `coefficient *
//! @name` with no trailing variable, is a named-expression reference, resolved
//! against the file's own `expressions` table, not `name_to_id`. `@param * @name`
//! (two references in one term) is rejected — it has no linear core representation.
//!
//! A `group` — bare, or scaled by a leading literal `coefficient '*'` — distributes
//! that coefficient into every inner term's scale at parse time (nesting to
//! arbitrary depth), yielding the same flat term list as the hand-expanded form; a
//! `group` may contain any `term` form, including nested groups, but not a bare
//! `coefficient`; a constant inside a group is rejected on either side of an
//! operator. Only a literal coefficient may scale a `group` — `@param * (...)` is
//! rejected for the same reason as `@param * @name`.
//!
//! A bare `coefficient` — a standalone numeric literal with no trailing `*` — is
//! valid only as a `side_term` of a relation with an operator, outside every group;
//! it has no linear-core representation, so an operator-free `expression` rejects it.
//! At most one top-level relational operator (`<=`, `>=`, `==`, outside any
//! parenthesis or variable argument list) is accepted; a second one is a
//! descriptive error — an inline double-relational range (`LI <= expr <= LS`) is
//! not supported. With no operator, `expression` is the flat one-sided LHS form.
//! With one, [`normalize`] folds every
//! variable term onto the merged left-hand side (the right side's sign-flipped,
//! same-variable terms merged, a net-zero column dropped) and collects every
//! non-variable term — a literal or a `@name` that resolves as a parameter, never
//! a named expression — into the affine remainder the operator assigns to
//! `bound_lower_affine` (`>=`), `bound_upper_affine` (`<=`), or both (`==`).
//!
//! An optional top-level `expressions` array declares named linear expressions
//! (see [`parse_named_expressions`]); a constraint or another expression
//! references one with `@name`, and references are flattened into the referring
//! term list at load — see the `named_expression_inline` module for the
//! substitution and cycle-detection rules.
//!
//! Every name in the LP variable catalog is recognised. Block-capable
//! variables accept an optional second argument (`hydro_turbined`, `hydro_spillage`,
//! `hydro_diversion`, `hydro_outflow`, `hydro_generation`, `hydro_inflow`,
//! `hydro_evaporation`, `hydro_storage_initial`, `hydro_storage_final`,
//! `hydro_useful_volume_initial`, `hydro_useful_volume_final`, …);
//! stage-only variables (`hydro_storage`, `hydro_withdrawal`,
//! `anticipated_decision`) must not have a block argument. Use
//! `anticipated_decision(thermal_id)` to reference the per-stage commitment column of an
//! anticipated thermal unit (no block index accepted).
//!
//! `hydro_turbined` and `hydro_generation` additionally accept a named `bus=`
//! argument selecting one cell of a plant split across several buses; no other
//! variable accepts it. All four forms parse: `f(e)`, `f(e, b)`, `f(e, bus=n)`,
//! `f(e, b, bus=n)` — the named argument may follow a positional block but never
//! precede one.
//!
//! `line_exchange` alone also accepts a `(source_bus=X, target_bus=Y)` addressing
//! form in place of the positional line id: the parser resolves the endpoint pair
//! to the connecting line via the caller-supplied [`LineBusPairIndex`], folding the
//! orientation sign into the term (a pair reversed against the line's declared
//! direction contributes `-1.0`). It desugars to the same `LineExchange { line_id }`
//! a direct `line_exchange(line_id)` produces.
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! - No two constraints share the same `id`.
//! - `slack.enabled = true` requires `slack.penalty` to be present and > 0.0.
//! - Each `expression` string must parse without error.
//!
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in entity registries — Layer 3.
//! - Block ID validity for the referenced stage — Layer 3/5.

use cobre_core::{
    ConstraintExpression, EntityId, GenericConstraint, Line, LinearTerm, SlackConfig, VariableRef,
};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::named_expression_inline::{
    ParsedExpression, ParsedTerm, detect_cycles, inline, validate_references_resolve,
};
use super::rhs_normalize::{RelOp, SideTerm, normalize};
use crate::LoadError;

#[cfg(test)]
use cobre_core::AffineBound;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Root object of `generic_constraints.json`.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RawGenericConstraintsFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of constraint entries.
    constraints: Vec<RawConstraint>,

    /// Named linear-expression declarations sharing the `@name` namespace with
    /// scalar parameters. Absent ⇒ empty ⇒ no named expressions.
    #[serde(default)]
    expressions: Vec<RawNamedExpression>,
}

/// A single constraint entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
struct RawConstraint {
    /// Constraint identifier. Must be unique within the file.
    id: i32,

    /// Short name used in reports and log output.
    name: String,

    /// Optional human-readable description.
    description: Option<String>,

    /// Expression string to be parsed. E.g. `"2.5 * thermal_generation(5) - hydro_generation(3)"`.
    /// To constrain an anticipated thermal's commitment, use `"anticipated_decision(5)"` (stage-level scalar, no block index).
    /// To constrain one bus of a plant split across several buses, use `"hydro_turbined(5, bus=2)"`
    /// or `"hydro_generation(5, bus=2)"` — the only two variables accepting a `bus=` selector.
    /// An optional inline relational operator (`<=`, `>=`, `==`) followed by a
    /// right-hand side normalizes onto the F3 interval — see the module doc.
    expression: String,

    /// Slack variable configuration.
    slack: RawSlackConfig,
}

/// The slack configuration.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
struct RawSlackConfig {
    /// Whether a slack variable is allowed.
    enabled: bool,

    /// Penalty per unit of violation. Must be > 0.0 when `enabled` is `true`.
    penalty: Option<f64>,
}

/// A named linear-expression declaration.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
struct RawNamedExpression {
    /// Name bound to the expression; unique in the file and distinct from every scalar-parameter name.
    name: String,

    /// Expression string, parsed under the variable + `@param`-coefficient grammar.
    expression: String,

    /// Optional human-readable description.
    #[serde(rename = "description")]
    _description: Option<String>,
}

// ── Line bus-pair addressing ────────────────────────────────────────────────

/// Direction of a `(source_bus, target_bus)` pair relative to a line's declared
/// source→target orientation. `Reversed` folds `-1.0` into the resolved term's scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinePairOrientation {
    Forward,
    Reversed,
}

/// Resolves a `(source_bus, target_bus)` pair to the `line_exchange` line it addresses
/// and the sign that pair takes relative to the line's declared direction.
///
/// Built from `&[Line]` via [`build_line_bus_pair_index`]. The default is empty (no pair
/// resolves) — the form a direct [`parse_generic_constraints`] caller with no line
/// topology passes.
#[derive(Debug, Clone, Default)]
pub struct LineBusPairIndex {
    by_pair: HashMap<(EntityId, EntityId), (EntityId, LinePairOrientation)>,
}

impl LineBusPairIndex {
    fn resolve(
        &self,
        source_bus: EntityId,
        target_bus: EntityId,
    ) -> Option<(EntityId, LinePairOrientation)> {
        self.by_pair.get(&(source_bus, target_bus)).copied()
    }
}

/// Build the `(source_bus, target_bus) → (line_id, orientation)` index from the loaded
/// lines. Each line contributes its declared pair (`Forward`) and the reversed pair
/// (`Reversed`), so an author may address a line from either endpoint.
///
/// # Errors
///
/// [`LoadError::SchemaError`] when two distinct lines share one unordered bus pair: the
/// pair no longer identifies a single line, so it is rejected loudly (pointing the author
/// at a named-expression sum) rather than resolved to an arbitrary line.
pub fn build_line_bus_pair_index(lines: &[Line]) -> Result<LineBusPairIndex, LoadError> {
    let mut by_pair: HashMap<(EntityId, EntityId), (EntityId, LinePairOrientation)> =
        HashMap::with_capacity(lines.len().saturating_mul(2));
    for line in lines {
        insert_line_pair(
            &mut by_pair,
            (line.source_bus_id, line.target_bus_id),
            line.id,
            LinePairOrientation::Forward,
        )?;
        insert_line_pair(
            &mut by_pair,
            (line.target_bus_id, line.source_bus_id),
            line.id,
            LinePairOrientation::Reversed,
        )?;
    }
    Ok(LineBusPairIndex { by_pair })
}

/// Insert one directed pair, rejecting a key already claimed by a different line. A key
/// re-inserted by the same line (a degenerate `source == target` self-loop) keeps its
/// first entry rather than erroring.
fn insert_line_pair(
    by_pair: &mut HashMap<(EntityId, EntityId), (EntityId, LinePairOrientation)>,
    key: (EntityId, EntityId),
    line_id: EntityId,
    orientation: LinePairOrientation,
) -> Result<(), LoadError> {
    match by_pair.get(&key) {
        Some((existing, _)) if *existing != line_id => Err(duplicate_line_pair_error(key)),
        Some(_) => Ok(()),
        None => {
            by_pair.insert(key, (line_id, orientation));
            Ok(())
        }
    }
}

/// Order-invariant duplicate-pair error: names the buses sorted so the message is
/// identical regardless of which line or direction triggered the collision.
fn duplicate_line_pair_error(key: (EntityId, EntityId)) -> LoadError {
    let (a, b) = if key.0.0 <= key.1.0 {
        (key.0, key.1)
    } else {
        (key.1, key.0)
    };
    LoadError::SchemaError {
        path: PathBuf::from("system/lines.json"),
        field: format!("lines[source_bus={a},target_bus={b}]"),
        message: format!(
            "buses {a} and {b} are connected by more than one line, so \
             (source_bus={a}, target_bus={b}) does not identify a single line; \
             reference the intended line by its id, or sum the lines with a named expression"
        ),
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `constraints/generic_constraints.json` from `path`, returning
/// constraints sorted by `id` ascending (declaration-order invariance).
///
/// # Errors
///
/// | Condition                                      | Error variant              |
/// | ---------------------------------------------- | -------------------------- |
/// | File not found / read failure                  | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field  | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the constraints array    | [`LoadError::SchemaError`] |
/// | `slack.enabled = true` with absent or <= 0 penalty | [`LoadError::SchemaError`] |
/// | Expression syntax error                        | [`LoadError::SchemaError`] |
/// | Unknown variable name in expression            | [`LoadError::SchemaError`] |
/// | `@param` coefficient not found in `name_to_id` | [`LoadError::SchemaError`] |
/// | Named expression: duplicate or parameter-colliding name | [`LoadError::SchemaError`] |
/// | Named expression: definition fails to parse    | [`LoadError::SchemaError`] |
/// | Named expression: reference to an undeclared name | [`LoadError::SchemaError`] |
/// | Named expression: reference cycle              | [`LoadError::SchemaError`] |
///
/// # Parameters
///
/// `name_to_id` maps parameter definition names to their [`EntityId`]. Pass
/// `&HashMap::new()` when no parameters have been loaded; expressions that
/// contain `@name` tokens will then fail with a schema error. The real mapping
/// is wired in by the caller once the parameter loader output is available.
///
/// `line_index` resolves the `line_exchange(source_bus=X, target_bus=Y)` addressing
/// form to a line id; pass [`LineBusPairIndex::default`] when no line topology is
/// available and any pair form then errors as an unmatched pair.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::{LineBusPairIndex, parse_generic_constraints};
/// use std::collections::HashMap;
/// use std::path::Path;
///
/// let constraints = parse_generic_constraints(
///     Path::new("case/constraints/generic_constraints.json"),
///     &HashMap::new(),
///     &LineBusPairIndex::default(),
/// ).expect("valid generic constraints file");
/// println!("loaded {} generic constraints", constraints.len());
/// ```
#[allow(clippy::implicit_hasher)]
pub fn parse_generic_constraints(
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<Vec<GenericConstraint>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawGenericConstraintsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path, name_to_id, line_index)?;

    let table = parse_named_expressions(&raw.expressions, path, name_to_id, line_index)?;
    detect_cycles(&table).map_err(|message| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: "expressions".to_string(),
        message,
    })?;
    validate_expression_references_resolve(&table, path)?;

    convert(raw, path, name_to_id, &table, line_index)
}

/// Report a reference to an undeclared name even when no constraint uses the
/// declaration, via a cheap name-existence walk that does not materialize terms —
/// so an exponentially-expanding declaration is not force-expanded here (expansion,
/// and the [`inline`] term-budget cap, apply only when a constraint references it).
/// Cycles are already excluded by [`detect_cycles`].
fn validate_expression_references_resolve(
    table: &[(String, ParsedExpression)],
    path: &Path,
) -> Result<(), LoadError> {
    validate_references_resolve(table).map_err(|(i, message)| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: format!("expressions[{i}].expression"),
        message,
    })
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized constraint data.
fn validate_raw(
    raw: &RawGenericConstraintsFile,
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<(), LoadError> {
    validate_no_duplicate_ids(&raw.constraints, path)?;
    for (i, constraint) in raw.constraints.iter().enumerate() {
        validate_slack(&constraint.slack, i, path)?;
        // Parsed here for accurate field paths; the result is discarded and the
        // expression is re-parsed and inlined during convert(). This is a syntax
        // check only — reference resolution needs the full table, built later.
        parse_relation(&constraint.expression, name_to_id, line_index).map_err(|msg| {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("constraints[{i}].expression"),
                message: msg,
            }
        })?;
    }
    Ok(())
}

/// Check that no two constraints share the same `id`.
fn validate_no_duplicate_ids(constraints: &[RawConstraint], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, constraint) in constraints.iter().enumerate() {
        if !seen.insert(constraint.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("constraints[{i}].id"),
                message: format!("duplicate id {} in constraints array", constraint.id),
            });
        }
    }
    Ok(())
}

/// Check slack config consistency: `enabled = true` requires `penalty > 0.0`.
fn validate_slack(
    slack: &RawSlackConfig,
    constraint_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if slack.enabled {
        match slack.penalty {
            None => {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("constraints[{constraint_index}].slack.penalty"),
                    message: "slack.enabled is true but slack.penalty is absent".to_string(),
                });
            }
            Some(p) if p <= 0.0 => {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("constraints[{constraint_index}].slack.penalty"),
                    message: format!("slack.penalty must be > 0.0 when enabled, got {p}"),
                });
            }
            Some(_) => {}
        }
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert the validated raw data to `Vec<GenericConstraint>`, sorted by `id`.
///
/// An operator-free expression is parsed and its `@name` references inlined
/// against `table` to produce the flat `ConstraintExpression` the core
/// consumes. An expression carrying a relational operator instead resolves
/// each side to [`SideTerm`]s and hands them to [`normalize`], which supplies
/// the merged LHS and the operator's targeted affine-bound endpoint(s).
fn convert(
    raw: RawGenericConstraintsFile,
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    table: &[(String, ParsedExpression)],
    line_index: &LineBusPairIndex,
) -> Result<Vec<GenericConstraint>, LoadError> {
    let mut result = Vec::with_capacity(raw.constraints.len());

    for (i, c) in raw.constraints.into_iter().enumerate() {
        let field = || format!("constraints[{i}].expression");
        let expression_err = |message: String| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field(),
            message,
        };

        let relation =
            parse_relation(&c.expression, name_to_id, line_index).map_err(expression_err)?;

        let (terms, bound_lower_affine, bound_upper_affine) = match relation {
            ParsedRelation::Plain(parsed) => {
                let terms = inline(&parsed, table).map_err(expression_err)?;
                (terms, None, None)
            }
            ParsedRelation::Relational { lhs, op, rhs } => {
                let lhs = resolve_split_side(lhs, name_to_id, table).map_err(expression_err)?;
                let rhs = resolve_split_side(rhs, name_to_id, table).map_err(expression_err)?;
                normalize(lhs, rhs, op)
            }
        };

        let slack = SlackConfig {
            enabled: c.slack.enabled,
            penalty: c.slack.penalty,
        };

        let mut expression = ConstraintExpression { terms };
        expression.canonicalize();

        result.push(GenericConstraint {
            id: EntityId::from(c.id),
            name: c.name,
            description: c.description,
            expression,
            slack,
            bound_lower_affine,
            bound_upper_affine,
        });
    }

    result.sort_by_key(|gc| gc.id.0);

    Ok(result)
}

// ── Expression parser ─────────────────────────────────────────────────────────

/// Parse an expression string (grammar: module doc) into a [`ParsedExpression`],
/// which may carry unresolved `@name` references — [`inline`] flattens them.
///
/// An `@param * variable` coefficient token is resolved against `name_to_id`; pass
/// `&HashMap::new()` when no parameters are loaded and such a token then errors.
/// Returns `Err(String)` on parse failure; the caller wraps it in
/// `LoadError::SchemaError` with the field path.
fn parse_expression_terms(
    input: &str,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<ParsedExpression, String> {
    let tokens = tokenize(input)?;
    parse_terms(&tokens, name_to_id, line_index)
}

// ── Named-expression declarations ───────────────────────────────────────────

/// Parse and validate the `expressions` declarations into a `(name, parsed)` table.
/// Rejects a duplicate name or a name colliding with a scalar parameter in
/// `name_to_id`, and parses each definition to a [`ParsedExpression`] (which may
/// carry unresolved references). Reference resolution and cycle detection run over
/// the full table after it is built.
///
/// # Errors
///
/// [`LoadError::SchemaError`] — `field = "expressions[i].name"` for a duplicate or
/// parameter-colliding name; `field = "expressions[i].expression"` for a definition
/// that fails to parse.
fn parse_named_expressions(
    entries: &[RawNamedExpression],
    path: &Path,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<Vec<(String, ParsedExpression)>, LoadError> {
    let mut table: Vec<(String, ParsedExpression)> = Vec::with_capacity(entries.len());
    let mut seen: HashSet<&str> = HashSet::new();

    for (i, entry) in entries.iter().enumerate() {
        if !seen.insert(entry.name.as_str()) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("expressions[{i}].name"),
                message: format!(
                    "duplicate named expression \"{}\" in expressions array",
                    entry.name
                ),
            });
        }

        if name_to_id.contains_key(&entry.name) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("expressions[{i}].name"),
                message: format!(
                    "name \"{}\" is declared as both a scalar parameter and a named expression; the \"@name\" namespace is shared",
                    entry.name
                ),
            });
        }

        let parsed = parse_expression_terms(&entry.expression, name_to_id, line_index).map_err(
            |message| LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("expressions[{i}].expression"),
                message,
            },
        )?;

        table.push((entry.name.clone(), parsed));
    }

    Ok(table)
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Tokens produced by the expression tokenizer.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Plus,
    /// Also handles unary negation of a term coefficient.
    Minus,
    Star,
    LParen,
    RParen,
    /// Separator between `entity_id` and `block_id`.
    Comma,
    /// The `=` in a named `bus=` argument.
    Equals,
    /// `<=` — a relational upper-bound split point.
    Le,
    /// `>=` — a relational lower-bound split point.
    Ge,
    /// `==` — a relational both-endpoints split point.
    EqEq,
    /// A non-negative literal — the tokenizer never emits a sign.
    Number(f64),
    Ident(String),
    /// A `@name` parameter reference; holds the identifier after `@`.
    ParamRef(String),
}

/// Tokenize an expression string into a `Vec<Token>`.
fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        match c {
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '=' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::EqEq);
                    i += 2;
                } else {
                    tokens.push(Token::Equals);
                    i += 1;
                }
            }
            '<' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Le);
                    i += 2;
                } else {
                    return Err(format!(
                        "bare '<' at position {i} is not a valid operator; use '<=' for a relational bound"
                    ));
                }
            }
            '>' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    return Err(format!(
                        "bare '>' at position {i} is not a valid operator; use '>=' for a relational bound"
                    ));
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && i > start
                            && (chars[i - 1] == 'e' || chars[i - 1] == 'E')))
                {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let val: f64 = s
                    .parse()
                    .map_err(|_| format!("invalid number literal \"{s}\" at position {start}"))?;
                tokens.push(Token::Number(val));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            '@' => {
                let at_pos = i;
                i += 1;
                if i >= chars.len() || !(chars[i].is_alphabetic() || chars[i] == '_') {
                    return Err(format!(
                        "@ must be followed by an identifier at position {at_pos}"
                    ));
                }
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                tokens.push(Token::ParamRef(name));
            }
            other => {
                return Err(format!("unexpected character '{other}' at position {i}"));
            }
        }
    }

    Ok(tokens)
}

// ── Term parser ───────────────────────────────────────────────────────────────

/// Parse a token stream into a [`ParsedExpression`] (grammar: module doc).
fn parse_terms(
    tokens: &[Token],
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<ParsedExpression, String> {
    if tokens.is_empty() {
        return Err("expression must not be empty".to_string());
    }

    let mut terms = Vec::new();
    let mut pos = 0;

    let mut sign: f64 = 1.0;
    match tokens.first() {
        Some(Token::Plus) => pos += 1,
        Some(Token::Minus) => {
            sign = -1.0;
            pos += 1;
        }
        _ => {}
    }

    let (mut group, next_pos) = parse_single_term(tokens, pos, sign, name_to_id, line_index)?;
    terms.append(&mut group);
    pos = next_pos;

    while pos < tokens.len() {
        let op_sign = match &tokens[pos] {
            Token::Plus => 1.0,
            Token::Minus => -1.0,
            other => {
                return Err(format!(
                    "expected '+' or '-' between terms, got {other:?} at position {pos}"
                ));
            }
        };
        pos += 1;

        let (mut group, next_pos) =
            parse_single_term(tokens, pos, op_sign, name_to_id, line_index)?;
        terms.append(&mut group);
        pos = next_pos;
    }

    Ok(terms)
}

/// Parse one term starting at `tokens[pos]` with the given sign prefix, returning
/// the term(s) it expands to — more than one only for a parenthesized group — and
/// the token position after the term.
///
/// An `@name` followed by `*` is a scalar-parameter coefficient; an `@name` with no
/// trailing `*` (standalone or after `coefficient *`) is a named-expression
/// reference. A parenthesized group — bare, or preceded by `coefficient '*'` —
/// distributes its leading coefficient into every inner term via
/// [`distribute_into_group`] (module doc grammar).
fn parse_single_term(
    tokens: &[Token],
    pos: usize,
    sign: f64,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<(Vec<ParsedTerm>, usize), String> {
    if pos >= tokens.len() {
        return Err(format!(
            "unexpected end of expression: expected a term at position {pos}"
        ));
    }

    match &tokens[pos] {
        Token::Number(coeff_val) => {
            let literal = *coeff_val * sign;
            let next = pos + 1;

            if next >= tokens.len() {
                return Err(format!(
                    "expected '*' after coefficient {coeff_val}, got end of expression"
                ));
            }
            if tokens[next] != Token::Star {
                return Err(format!(
                    "expected '*' after coefficient {coeff_val}, got {:?}",
                    tokens[next]
                ));
            }

            let after_star = next + 1;
            if after_star >= tokens.len() {
                return Err(
                    "expected variable name, @parameter, or '(' after '*', got end of expression"
                        .to_string(),
                );
            }

            if tokens[after_star] == Token::LParen {
                let (group, end_pos) =
                    parse_group_terms(tokens, after_star, name_to_id, line_index)?;
                return Ok((distribute_into_group(group, literal), end_pos));
            }

            if let Token::ParamRef(name) = &tokens[after_star] {
                let star2 = after_star + 1;
                if star2 < tokens.len() && tokens[star2] == Token::Star {
                    let var_pos = star2 + 1;
                    if tokens.get(var_pos) == Some(&Token::LParen) {
                        return Err(param_scaled_group_error(name));
                    }
                    let id = resolve_param_ref(name, name_to_id)?;
                    if let Some(Token::ParamRef(second)) = tokens.get(var_pos) {
                        return Err(format!(
                            "only one @parameter reference is allowed per term; found \"@{name}\" and \"@{second}\""
                        ));
                    }
                    let (variable, orientation, end_pos) =
                        parse_variable_ref(tokens, var_pos, line_index)?;
                    Ok((
                        vec![ParsedTerm::Flat(LinearTerm::parameter(
                            id,
                            literal * orientation,
                            variable,
                        ))],
                        end_pos,
                    ))
                } else {
                    Ok((
                        vec![ParsedTerm::Ref {
                            name: name.clone(),
                            scale: literal,
                        }],
                        after_star + 1,
                    ))
                }
            } else {
                let (variable, orientation, end_pos) =
                    parse_variable_ref(tokens, after_star, line_index)?;
                Ok((
                    vec![ParsedTerm::Flat(LinearTerm::literal(
                        literal * orientation,
                        variable,
                    ))],
                    end_pos,
                ))
            }
        }
        Token::ParamRef(name) => {
            let star = pos + 1;
            if star < tokens.len() && tokens[star] == Token::Star {
                let var_pos = star + 1;
                if tokens.get(var_pos) == Some(&Token::LParen) {
                    return Err(param_scaled_group_error(name));
                }
                let id = resolve_param_ref(name, name_to_id)?;
                if var_pos >= tokens.len() {
                    return Err(format!(
                        "parameter \"@{name}\" must multiply a variable, e.g. \"@{name} * hydro_generation(0)\""
                    ));
                }
                if let Token::ParamRef(second) = &tokens[var_pos] {
                    return Err(format!(
                        "only one @parameter reference is allowed per term; found \"@{name}\" and \"@{second}\""
                    ));
                }
                let (variable, orientation, end_pos) =
                    parse_variable_ref(tokens, var_pos, line_index)?;
                Ok((
                    vec![ParsedTerm::Flat(LinearTerm::parameter(
                        id,
                        sign * orientation,
                        variable,
                    ))],
                    end_pos,
                ))
            } else {
                Ok((
                    vec![ParsedTerm::Ref {
                        name: name.clone(),
                        scale: sign,
                    }],
                    pos + 1,
                ))
            }
        }
        Token::Ident(_) => {
            let (variable, orientation, end_pos) = parse_variable_ref(tokens, pos, line_index)?;
            Ok((
                vec![ParsedTerm::Flat(LinearTerm::literal(
                    sign * orientation,
                    variable,
                ))],
                end_pos,
            ))
        }
        Token::LParen => {
            let (group, end_pos) = parse_group_terms(tokens, pos, name_to_id, line_index)?;
            Ok((distribute_into_group(group, sign), end_pos))
        }
        other => Err(format!(
            "expected a coefficient, variable name, or '(' at position {pos}, got {other:?}"
        )),
    }
}

/// Parse a parenthesized group `'(' term (('+'|'-') term)* ')'` starting at
/// `tokens[pos]` (the `'('`), returning the un-distributed inner terms and the
/// position after the matching `')'`. The caller folds the group's leading
/// coefficient in via [`distribute_into_group`].
fn parse_group_terms(
    tokens: &[Token],
    pos: usize,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<(Vec<ParsedTerm>, usize), String> {
    let mut cursor = pos + 1;

    if tokens.get(cursor) == Some(&Token::RParen) {
        return Err(format!("empty parenthesized group at position {pos}"));
    }

    let mut sign: f64 = 1.0;
    match tokens.get(cursor) {
        Some(Token::Plus) => cursor += 1,
        Some(Token::Minus) => {
            sign = -1.0;
            cursor += 1;
        }
        _ => {}
    }

    let (mut terms, next_pos) = parse_single_term(tokens, cursor, sign, name_to_id, line_index)?;
    cursor = next_pos;

    loop {
        match tokens.get(cursor) {
            Some(Token::RParen) => {
                cursor += 1;
                break;
            }
            Some(Token::Plus | Token::Minus) => {
                let op_sign = if tokens[cursor] == Token::Plus {
                    1.0
                } else {
                    -1.0
                };
                cursor += 1;
                let (mut group, next_pos) =
                    parse_single_term(tokens, cursor, op_sign, name_to_id, line_index)?;
                terms.append(&mut group);
                cursor = next_pos;
            }
            Some(other) => {
                return Err(format!(
                    "expected '+', '-', or ')' inside parenthesized group opened at position {pos}, got {other:?} at position {cursor}"
                ));
            }
            None => {
                return Err(format!(
                    "unexpected end of expression: expected ')' to close parenthesized group opened at position {pos}"
                ));
            }
        }
    }

    Ok((terms, cursor))
}

/// Multiply `coeff` into every inner term of a parsed group — [`LinearTerm::scale`]
/// for a [`ParsedTerm::Flat`], the reference `scale` for a [`ParsedTerm::Ref`] — the
/// same rule the named-expression reference substitution behind [`inline`] uses:
/// coefficient kind preserved, duplicates left un-merged.
fn distribute_into_group(mut terms: Vec<ParsedTerm>, coeff: f64) -> Vec<ParsedTerm> {
    for term in &mut terms {
        match term {
            ParsedTerm::Flat(lt) => lt.scale *= coeff,
            ParsedTerm::Ref { scale, .. } => *scale *= coeff,
        }
    }
    terms
}

/// The `@param * (` rejection: a parenthesized group has no flat linear-core
/// representation when scaled by a runtime parameter, mirroring the `@param *
/// @name` precedent — only a literal coefficient may scale a group.
fn param_scaled_group_error(name: &str) -> String {
    format!(
        "parameter \"@{name}\" cannot scale a parenthesized group: a group takes only a literal coefficient, not \"@{name} * (...)\""
    )
}

/// Look up `name` in `name_to_id`, returning its [`EntityId`] or a descriptive error.
fn resolve_param_ref(
    name: &str,
    name_to_id: &HashMap<String, EntityId>,
) -> Result<EntityId, String> {
    name_to_id.get(name).copied().ok_or_else(|| {
        format!("unknown parameter \"@{name}\": no definition with this name was loaded")
    })
}

// ── Relational split ──────────────────────────────────────────────────────────

/// One term of a relational side before named-expression/parameter
/// resolution: either the shape [`parse_single_term`] already produces, or a
/// standalone numeric literal with no variable — the leaf a plain, one-sided
/// expression rejects but a relational side's affine remainder may carry.
#[derive(Debug)]
enum SplitTerm {
    /// The existing parsed-term shape: a flat variable term, or an
    /// unresolved named-expression/parameter reference.
    Named(ParsedTerm),
    /// A standalone numeric literal with no variable.
    Constant(f64),
}

/// A constraint expression, optionally split by exactly one top-level
/// relational operator (grammar: module doc).
#[derive(Debug)]
enum ParsedRelation {
    /// No relational operator: the flat one-sided form.
    Plain(ParsedExpression),
    /// Exactly one top-level relational operator; each side is parsed but
    /// not yet resolved against the named-expression table.
    Relational {
        lhs: Vec<SplitTerm>,
        op: RelOp,
        rhs: Vec<SplitTerm>,
    },
}

/// Tokenize `input` and, when it carries exactly one top-level relational
/// operator, split it into [`ParsedRelation::Relational`]; otherwise parse it
/// as the one-sided [`ParsedRelation::Plain`] form.
///
/// # Errors
///
/// Returns `Err(String)` on a tokenizer or parse failure, or when the
/// expression carries two or more top-level relational operators.
fn parse_relation(
    input: &str,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<ParsedRelation, String> {
    let tokens = tokenize(input)?;
    match split_relational(&tokens)? {
        None => Ok(ParsedRelation::Plain(parse_terms(
            &tokens, name_to_id, line_index,
        )?)),
        Some((lhs_tokens, op, rhs_tokens)) => {
            let lhs = parse_split_side(lhs_tokens, name_to_id, line_index)?;
            let rhs = parse_split_side(rhs_tokens, name_to_id, line_index)?;
            Ok(ParsedRelation::Relational { lhs, op, rhs })
        }
    }
}

/// A tokenized expression split at its single top-level relational operator:
/// the left slice, the operator, and the right slice.
type RelationalSplit<'a> = (&'a [Token], RelOp, &'a [Token]);

/// Scan `tokens` for a top-level (paren-depth-0) relational operator — the
/// depth check also excludes any operator-shaped token inside a variable's
/// argument list, since every argument list opens its own real `(`/`)` pair.
///
/// Returns `Ok(None)` for zero operators, `Ok(Some((lhs, op, rhs)))` split
/// around the single operator found; `Err` for two or more.
fn split_relational(tokens: &[Token]) -> Result<Option<RelationalSplit<'_>>, String> {
    let mut depth: i32 = 0;
    let mut found: Option<(usize, RelOp)> = None;

    for (i, tok) in tokens.iter().enumerate() {
        match tok {
            Token::LParen => depth += 1,
            Token::RParen => depth -= 1,
            _ => {}
        }

        if depth != 0 {
            continue;
        }
        let op = match tok {
            Token::Le => RelOp::Le,
            Token::Ge => RelOp::Ge,
            Token::EqEq => RelOp::Eq,
            _ => continue,
        };
        if found.is_some() {
            return Err(
                "a generic constraint expression carries at most one relational operator \
                 (<=, >=, ==); an inline double-relational range is not supported — split \
                 into two constraints"
                    .to_string(),
            );
        }
        found = Some((i, op));
    }

    Ok(found.map(|(i, op)| (&tokens[..i], op, &tokens[i + 1..])))
}

/// Parse one side of a relational split (grammar: module doc) — the same
/// leading-sign-then-`(+|-)`-separated-list structure as [`parse_terms`],
/// extended with the standalone-constant leaf a relational side's affine
/// remainder may carry (`... <= 73`).
fn parse_split_side(
    tokens: &[Token],
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<Vec<SplitTerm>, String> {
    if tokens.is_empty() {
        return Err("expression must not be empty".to_string());
    }

    let mut terms = Vec::new();
    let mut pos = 0;

    let mut sign: f64 = 1.0;
    match tokens.first() {
        Some(Token::Plus) => pos += 1,
        Some(Token::Minus) => {
            sign = -1.0;
            pos += 1;
        }
        _ => {}
    }

    let (mut group, next_pos) = parse_split_single_term(tokens, pos, sign, name_to_id, line_index)?;
    terms.append(&mut group);
    pos = next_pos;

    while pos < tokens.len() {
        let op_sign = match &tokens[pos] {
            Token::Plus => 1.0,
            Token::Minus => -1.0,
            other => {
                return Err(format!(
                    "expected '+' or '-' between terms, got {other:?} at position {pos}"
                ));
            }
        };
        pos += 1;

        let (mut group, next_pos) =
            parse_split_single_term(tokens, pos, op_sign, name_to_id, line_index)?;
        terms.append(&mut group);
        pos = next_pos;
    }

    Ok(terms)
}

/// Parse one term of a relational side: a standalone numeric literal with no
/// following `*` is the new affine-remainder constant leaf; every other shape
/// delegates to [`parse_single_term`] unchanged (paren groups, `@param`
/// coefficients, and named-expression/parameter references all work exactly
/// as they do in the plain grammar).
fn parse_split_single_term(
    tokens: &[Token],
    pos: usize,
    sign: f64,
    name_to_id: &HashMap<String, EntityId>,
    line_index: &LineBusPairIndex,
) -> Result<(Vec<SplitTerm>, usize), String> {
    if let Some(Token::Number(v)) = tokens.get(pos)
        && tokens.get(pos + 1) != Some(&Token::Star)
    {
        return Ok((vec![SplitTerm::Constant(v * sign)], pos + 1));
    }

    let (parsed, next_pos) = parse_single_term(tokens, pos, sign, name_to_id, line_index)?;
    Ok((parsed.into_iter().map(SplitTerm::Named).collect(), next_pos))
}

/// Resolve one already-parsed relational side into [`SideTerm`]s: a
/// [`ParsedTerm::Flat`] term is already a variable term; a [`ParsedTerm::Ref`]
/// inlines through `table` when declared there (so a genuine named-expression
/// reference still expands into variable terms that fold, exactly as the
/// plain grammar's `@expr` does), or — since a bound-position `@name` names a
/// parameter, never a named expression — resolves directly against
/// `name_to_id` when it is not, so an undeclared name here reports as
/// "unknown parameter", never "undeclared reference".
fn resolve_split_side(
    raw: Vec<SplitTerm>,
    name_to_id: &HashMap<String, EntityId>,
    table: &[(String, ParsedExpression)],
) -> Result<Vec<SideTerm>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for term in raw {
        match term {
            SplitTerm::Constant(v) => out.push(SideTerm::Constant(v)),
            SplitTerm::Named(ParsedTerm::Flat(lt)) => out.push(SideTerm::Variable(lt)),
            SplitTerm::Named(ParsedTerm::Ref { name, scale }) => {
                if table.iter().any(|(declared, _)| declared == &name) {
                    let inlined = inline(&vec![ParsedTerm::Ref { name, scale }], table)?;
                    out.extend(inlined.into_iter().map(SideTerm::Variable));
                } else {
                    let id = resolve_param_ref(&name, name_to_id)?;
                    out.push(SideTerm::Param(scale, id));
                }
            }
        }
    }
    Ok(out)
}

// ── Integer conversion helpers ────────────────────────────────────────────────

/// Convert an `f64` token value to `i32` if it represents an exact integer in `[0, i32::MAX]`.
///
/// Entity IDs are non-negative integers. The tokenizer stores all numeric literals
/// as `f64`, so we need to verify the value round-trips exactly through `i32`.
fn token_f64_to_i32(v: f64) -> Option<i32> {
    if v < 0.0 || v > f64::from(i32::MAX) || v.fract() != 0.0 {
        return None;
    }
    // SAFETY: v is in [0, i32::MAX] with zero fractional part; the cast is exact.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(v as i32)
}

/// Convert an `f64` token value to `usize` if it represents an exact non-negative integer.
///
/// Block IDs are non-negative integers. Same rationale as [`token_f64_to_i32`].
fn token_f64_to_usize(v: f64) -> Option<usize> {
    if v < 0.0 || v.fract() != 0.0 {
        return None;
    }
    // SAFETY: v >= 0 and has zero fractional part; usize can represent all
    // non-negative f64 values that fit within platform pointer width.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(v as usize)
}

/// Parse a variable reference
/// `var_name '(' entity_id (',' block_id)? (',' 'bus' '=' bus_id)? ')'`, or the
/// `line_exchange` pair form `line_exchange '(' 'source_bus' '=' id ',' 'target_bus' '='
/// id ')'`, from `tokens[pos]`.
///
/// Returns the [`VariableRef`], the orientation sign (`+1.0`, or `-1.0` when the pair
/// resolves to a line reversed relative to its declared direction), and the position of
/// the next unconsumed token. Non-pair forms always carry `+1.0`.
fn parse_variable_ref(
    tokens: &[Token],
    pos: usize,
    line_index: &LineBusPairIndex,
) -> Result<(VariableRef, f64, usize), String> {
    let var_name = match tokens.get(pos) {
        Some(Token::Ident(name)) => name.clone(),
        Some(other) => {
            return Err(format!(
                "expected variable name, got {other:?} at position {pos}"
            ));
        }
        None => return Err("expected variable name, got end of expression".to_string()),
    };

    if tokens.get(pos + 1) != Some(&Token::LParen) {
        return Err(format!(
            "expected '(' after variable name \"{var_name}\" at position {}",
            pos + 1
        ));
    }

    if let Some(Token::Ident(arg)) = tokens.get(pos + 2)
        && (arg == "source_bus" || arg == "target_bus")
    {
        return parse_line_bus_pair_ref(tokens, pos, &var_name, line_index);
    }

    let entity_id = match tokens.get(pos + 2) {
        Some(Token::Number(n)) => {
            let n_i32 = token_f64_to_i32(*n).ok_or_else(|| {
                format!(
                    "entity_id must be a non-negative integer, got {n} in variable \"{var_name}\""
                )
            })?;
            EntityId::from(n_i32)
        }
        Some(other) => {
            return Err(format!(
                "expected integer entity_id in variable \"{var_name}\", got {other:?} at position {}",
                pos + 2
            ));
        }
        None => {
            return Err(format!(
                "unexpected end of expression: expected entity_id in variable \"{var_name}\""
            ));
        }
    };

    let mut cursor = pos + 3;
    let mut block_id: Option<usize> = None;
    let mut bus_id: Option<EntityId> = None;
    let mut seen_bus = false;

    loop {
        match tokens.get(cursor) {
            Some(Token::RParen) => break,
            Some(Token::Comma) => {
                cursor += 1;
                match tokens.get(cursor) {
                    Some(Token::Number(b)) => {
                        if seen_bus {
                            return Err(format!(
                                "positional block argument must precede the named \"bus=\" argument in variable \"{var_name}\""
                            ));
                        }
                        if block_id.is_some() {
                            return Err(format!(
                                "repeated block argument in variable \"{var_name}\""
                            ));
                        }
                        let b_usize = token_f64_to_usize(*b).ok_or_else(|| {
                            format!(
                                "block_id must be a non-negative integer, got {b} in variable \"{var_name}\""
                            )
                        })?;
                        block_id = Some(b_usize);
                        cursor += 1;
                    }
                    Some(Token::Ident(name)) if name == "bus" => {
                        if seen_bus {
                            return Err(format!(
                                "repeated \"bus=\" argument in variable \"{var_name}\""
                            ));
                        }
                        seen_bus = true;
                        cursor += 1;
                        if tokens.get(cursor) != Some(&Token::Equals) {
                            return Err(format!(
                                "expected '=' after \"bus\" in variable \"{var_name}\""
                            ));
                        }
                        cursor += 1;
                        match tokens.get(cursor) {
                            Some(Token::Number(b)) => {
                                let b_i32 = token_f64_to_i32(*b).ok_or_else(|| {
                                    format!(
                                        "bus_id must be a non-negative integer, got {b} in variable \"{var_name}\""
                                    )
                                })?;
                                bus_id = Some(EntityId::from(b_i32));
                                cursor += 1;
                            }
                            Some(other) => {
                                return Err(format!(
                                    "expected integer bus_id after \"bus=\" in variable \"{var_name}\", got {other:?} at position {cursor}"
                                ));
                            }
                            None => {
                                return Err(format!(
                                    "unexpected end of expression: expected bus_id after \"bus=\" in variable \"{var_name}\""
                                ));
                            }
                        }
                    }
                    Some(Token::Ident(name)) if name == "source_bus" || name == "target_bus" => {
                        return Err(format!(
                            "the (source_bus=, target_bus=) pair form is only accepted on \"line_exchange\" as its sole argument, e.g. \"line_exchange(source_bus=3, target_bus=7)\"; it cannot follow a positional argument in \"{var_name}\""
                        ));
                    }
                    Some(Token::Ident(name)) => {
                        return Err(format!(
                            "unknown named argument \"{name}\" in variable \"{var_name}\": only \"bus\" is supported"
                        ));
                    }
                    Some(other) => {
                        return Err(format!(
                            "expected block_id, \"bus=\", or ')' in variable \"{var_name}\" argument list, got {other:?} at position {cursor}"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "unexpected end of expression: expected an argument after ',' in variable \"{var_name}\""
                        ));
                    }
                }
            }
            Some(other) => {
                return Err(format!(
                    "expected ',' or ')' in variable \"{var_name}\" argument list, got {other:?} at position {cursor}"
                ));
            }
            None => {
                return Err(format!(
                    "unexpected end of expression: expected ')' to close variable \"{var_name}\""
                ));
            }
        }
    }
    cursor += 1;

    let variable = build_variable_ref(&var_name, entity_id, block_id, bus_id)?;

    Ok((variable, 1.0, cursor))
}

/// Parse and resolve the `line_exchange(source_bus=X, target_bus=Y)` addressing form.
///
/// Rejects the form on any variable other than `line_exchange`, and requires both
/// `source_bus=` and `target_bus=`. Resolves the pair via `line_index` to the existing
/// `LineExchange { line_id }` term, returning the orientation sign (`+1.0` forward,
/// `-1.0` reversed) for the caller to fold into the term scale.
fn parse_line_bus_pair_ref(
    tokens: &[Token],
    pos: usize,
    var_name: &str,
    line_index: &LineBusPairIndex,
) -> Result<(VariableRef, f64, usize), String> {
    if var_name != "line_exchange" {
        return Err(format!(
            "the (source_bus=, target_bus=) pair form addresses a line by its endpoint buses and is only accepted on \"line_exchange\", not \"{var_name}\""
        ));
    }

    let mut source_bus: Option<EntityId> = None;
    let mut target_bus: Option<EntityId> = None;
    let mut cursor = pos + 2;

    loop {
        let arg = match tokens.get(cursor) {
            Some(Token::Ident(name)) if name == "source_bus" || name == "target_bus" => {
                name.clone()
            }
            Some(other) => {
                return Err(format!(
                    "expected \"source_bus\" or \"target_bus\" in \"{var_name}\" pair form, got {other:?} at position {cursor}"
                ));
            }
            None => {
                return Err(format!(
                    "unexpected end of expression in \"{var_name}\" pair form"
                ));
            }
        };
        cursor += 1;

        if tokens.get(cursor) != Some(&Token::Equals) {
            return Err(format!(
                "expected '=' after \"{arg}\" in \"{var_name}\" pair form"
            ));
        }
        cursor += 1;

        let bus = match tokens.get(cursor) {
            Some(Token::Number(n)) => {
                let n_i32 = token_f64_to_i32(*n).ok_or_else(|| {
                    format!("{arg} must be a non-negative integer, got {n} in \"{var_name}\"")
                })?;
                EntityId::from(n_i32)
            }
            Some(other) => {
                return Err(format!(
                    "expected an integer bus id after \"{arg}=\" in \"{var_name}\", got {other:?} at position {cursor}"
                ));
            }
            None => {
                return Err(format!(
                    "unexpected end of expression: expected a bus id after \"{arg}=\" in \"{var_name}\""
                ));
            }
        };
        cursor += 1;

        if arg == "source_bus" {
            if source_bus.is_some() {
                return Err(format!(
                    "repeated \"source_bus=\" argument in \"{var_name}\""
                ));
            }
            source_bus = Some(bus);
        } else {
            if target_bus.is_some() {
                return Err(format!(
                    "repeated \"target_bus=\" argument in \"{var_name}\""
                ));
            }
            target_bus = Some(bus);
        }

        match tokens.get(cursor) {
            Some(Token::Comma) => cursor += 1,
            Some(Token::RParen) => {
                cursor += 1;
                break;
            }
            Some(other) => {
                return Err(format!(
                    "expected ',' or ')' in \"{var_name}\" pair form, got {other:?} at position {cursor}"
                ));
            }
            None => {
                return Err(format!(
                    "unexpected end of expression: expected ')' to close \"{var_name}\""
                ));
            }
        }
    }

    let (Some(source_bus), Some(target_bus)) = (source_bus, target_bus) else {
        return Err(format!(
            "the \"{var_name}\" pair form requires both \"source_bus=\" and \"target_bus=\"; write \"line_exchange(source_bus=X, target_bus=Y)\""
        ));
    };

    let (line_id, orientation) = line_index.resolve(source_bus, target_bus).ok_or_else(|| {
        format!("no line connects buses source_bus={source_bus} and target_bus={target_bus}")
    })?;

    let sign = match orientation {
        LinePairOrientation::Forward => 1.0,
        LinePairOrientation::Reversed => -1.0,
    };

    Ok((
        VariableRef::LineExchange {
            line_id,
            block_id: None,
        },
        sign,
        cursor,
    ))
}

/// Build a [`VariableRef`] from the parsed variable name, entity ID, optional block
/// ID, and optional bus selector.
///
/// Returns `Err(String)` if the variable name is not a recognised LP variable name, if a
/// block argument is provided for a stage-only variable, or if a `bus=` selector is
/// provided for any variable other than `hydro_turbined`/`hydro_generation`.
// Rationale: one exhaustive match over the LP variable catalog; splitting would scatter the
// canonical catalog and drop the compile-time exhaustiveness check.
#[allow(clippy::too_many_lines)]
fn build_variable_ref(
    name: &str,
    entity_id: EntityId,
    block_id: Option<usize>,
    bus_id: Option<EntityId>,
) -> Result<VariableRef, String> {
    let variable = match name {
        // Stage-only variables (block_id must be None).
        "hydro_storage" => {
            if block_id.is_some() {
                return Err(format!(
                    "variable \"{name}\" does not accept a block argument"
                ));
            }
            Ok(VariableRef::HydroStorage {
                hydro_id: entity_id,
            })
        }
        "hydro_withdrawal" => {
            if block_id.is_some() {
                return Err(format!(
                    "variable \"{name}\" does not accept a block argument"
                ));
            }
            Ok(VariableRef::HydroWithdrawal {
                hydro_id: entity_id,
            })
        }
        // Block-capable variables.
        "hydro_evaporation" => Ok(VariableRef::HydroEvaporation {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_inflow" => Ok(VariableRef::HydroInflow {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_storage_initial" => Ok(VariableRef::HydroStorageInitial {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_storage_final" => Ok(VariableRef::HydroStorageFinal {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_useful_volume_initial" => Ok(VariableRef::HydroUsefulVolumeInitial {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_useful_volume_final" => Ok(VariableRef::HydroUsefulVolumeFinal {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_turbined" => Ok(VariableRef::HydroTurbined {
            hydro_id: entity_id,
            block_id,
            bus_id,
        }),
        "hydro_spillage" => Ok(VariableRef::HydroSpillage {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_diversion" => Ok(VariableRef::HydroDiversion {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_outflow" => Ok(VariableRef::HydroOutflow {
            hydro_id: entity_id,
            block_id,
        }),
        "hydro_generation" => Ok(VariableRef::HydroGeneration {
            hydro_id: entity_id,
            block_id,
            bus_id,
        }),
        "thermal_generation" => Ok(VariableRef::ThermalGeneration {
            thermal_id: entity_id,
            block_id,
        }),
        "line_direct" => Ok(VariableRef::LineDirect {
            line_id: entity_id,
            block_id,
        }),
        "line_reverse" => Ok(VariableRef::LineReverse {
            line_id: entity_id,
            block_id,
        }),
        "line_exchange" => Ok(VariableRef::LineExchange {
            line_id: entity_id,
            block_id,
        }),
        "bus_deficit" => Ok(VariableRef::BusDeficit {
            bus_id: entity_id,
            block_id,
        }),
        "bus_excess" => Ok(VariableRef::BusExcess {
            bus_id: entity_id,
            block_id,
        }),
        "pumping_flow" => Ok(VariableRef::PumpingFlow {
            station_id: entity_id,
            block_id,
        }),
        "pumping_power" => Ok(VariableRef::PumpingPower {
            station_id: entity_id,
            block_id,
        }),
        "contract_import" => Ok(VariableRef::ContractImport {
            contract_id: entity_id,
            block_id,
        }),
        "contract_export" => Ok(VariableRef::ContractExport {
            contract_id: entity_id,
            block_id,
        }),
        "non_controllable_generation" => Ok(VariableRef::NonControllableGeneration {
            source_id: entity_id,
            block_id,
        }),
        "non_controllable_curtailment" => Ok(VariableRef::NonControllableCurtailment {
            source_id: entity_id,
            block_id,
        }),
        "anticipated_decision" => {
            if block_id.is_some() {
                return Err(format!(
                    "variable \"anticipated_decision\" is a stage-level scalar and \
                     does not accept a block_id — write \"anticipated_decision({})\", \
                     not \"anticipated_decision({}, ...)\"",
                    entity_id.0, entity_id.0,
                ));
            }
            Ok(VariableRef::AnticipatedDecision {
                thermal_id: entity_id,
            })
        }
        other => Err(format!(
            "unknown variable name \"{other}\": not a recognised LP variable name"
        )),
    }?;

    if bus_id.is_some()
        && !matches!(
            variable,
            VariableRef::HydroTurbined { .. } | VariableRef::HydroGeneration { .. }
        )
    {
        return Err(format!(
            "variable \"{name}\" does not accept a bus selector; only \"hydro_turbined\" and \"hydro_generation\" accept a \"bus=\" argument"
        ));
    }

    Ok(variable)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support::write_json;
    use cobre_core::CoefficientRef;
    use std::fmt::Write as _;

    fn lit(term: &LinearTerm) -> f64 {
        match term.coefficient {
            CoefficientRef::Literal(v) => v,
            CoefficientRef::Parameter(_) => panic!("expected literal"),
        }
    }

    /// Parse a reference-free expression to a [`ConstraintExpression`], the shape
    /// the flat-grammar tests assert on. Any surviving `@name` reference resolves
    /// against an empty table and therefore errors as undeclared.
    fn parse_expression(
        input: &str,
        name_to_id: &HashMap<String, EntityId>,
    ) -> Result<ConstraintExpression, String> {
        parse_expression_with_index(input, name_to_id, &LineBusPairIndex::default())
    }

    fn parse_expression_with_index(
        input: &str,
        name_to_id: &HashMap<String, EntityId>,
        line_index: &LineBusPairIndex,
    ) -> Result<ConstraintExpression, String> {
        let parsed = parse_expression_terms(input, name_to_id, line_index)?;
        let terms = inline(&parsed, &[])?;
        Ok(ConstraintExpression { terms })
    }

    /// Build a one-line index: a line `id` connecting `source_bus_id`→`target_bus_id`.
    fn line(id: i32, source: i32, target: i32) -> Line {
        Line {
            id: EntityId(id),
            name: format!("L{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .expect("valid date"),
            source_bus_id: EntityId(source),
            target_bus_id: EntityId(target),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn flat(term: &ParsedTerm) -> &LinearTerm {
        match term {
            ParsedTerm::Flat(lt) => lt,
            ParsedTerm::Ref { name, .. } => panic!("expected a Flat term, got reference @{name}"),
        }
    }

    fn param_id(term: &LinearTerm) -> EntityId {
        match term.coefficient {
            CoefficientRef::Parameter(id) => id,
            CoefficientRef::Literal(_) => panic!("expected Parameter coefficient"),
        }
    }

    fn one_param_table() -> std::collections::HashMap<String, EntityId> {
        let mut m = std::collections::HashMap::new();
        m.insert("rho_eq".to_string(), EntityId(7));
        m.insert("rho".to_string(), EntityId(7));
        m
    }

    const VALID_JSON: &str = r#"{
  "constraints": [
    {
      "id": 1,
      "name": "min_hydro",
      "expression": "hydro_generation(10) + hydro_generation(11)",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "max_thermal",
      "expression": "2.5 * thermal_generation(5) - hydro_generation(3)",
      "slack": { "enabled": true, "penalty": 5000.0 }
    }
  ]
}"#;

    // ── Expression parser unit tests ──────────────────────────────────────────

    /// AC-1: Simple single-term expression with implicit coefficient.
    #[test]
    fn test_expr_simple_single_term() {
        let expr = parse_expression("hydro_generation(10)", &HashMap::new()).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert!((lit(&expr.terms[0]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Addition: two terms, both coefficient 1.0.
    #[test]
    fn test_expr_addition_two_terms() {
        let expr = parse_expression(
            "hydro_generation(10) + hydro_generation(11)",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((lit(&expr.terms[0]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
                bus_id: None,
            }
        );
        assert!((lit(&expr.terms[1]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(11),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// AC-2: Coefficient with `*` and subtraction (negation of second term).
    #[test]
    fn test_expr_coefficient_and_subtraction() {
        let expr = parse_expression(
            "2.5 * thermal_generation(5) - hydro_generation(3)",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((lit(&expr.terms[0]) - 2.5).abs() < 1e-10);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
        assert!((lit(&expr.terms[1]) - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Subtraction: second term has coefficient -1.0.
    #[test]
    fn test_expr_subtraction_negates_coefficient() {
        let expr = parse_expression(
            "thermal_generation(5) - hydro_generation(3)",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((lit(&expr.terms[0]) - 1.0).abs() < f64::EPSILON);
        assert!((lit(&expr.terms[1]) - (-1.0)).abs() < f64::EPSILON);
    }

    /// Block-specific variable: `hydro_turbined(5, 0)` → `block_id: Some(0)`.
    #[test]
    fn test_expr_block_specific_variable() {
        let expr = parse_expression("hydro_turbined(5, 0)", &HashMap::new()).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroTurbined {
                hydro_id: EntityId(5),
                block_id: Some(0),
                bus_id: None,
            }
        );
    }

    /// Block-specific line_exchange: `line_exchange(0, 1)` → `block_id: Some(1)`.
    #[test]
    fn test_expr_line_exchange_with_block() {
        let expr = parse_expression("line_exchange(0, 1)", &HashMap::new()).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::LineExchange {
                line_id: EntityId(0),
                block_id: Some(1),
            }
        );
    }

    /// Stage-only variable: `hydro_storage(7)` → no block.
    #[test]
    fn test_expr_stage_only_hydro_storage() {
        let expr = parse_expression("hydro_storage(7)", &HashMap::new()).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroStorage {
                hydro_id: EntityId(7),
            }
        );
    }

    /// Stage-only variable with block argument → error.
    #[test]
    fn test_expr_stage_only_with_block_is_error() {
        let err = parse_expression("hydro_storage(7, 0)", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("does not accept a block argument"),
            "expected block argument error, got: {err}"
        );
    }

    /// `hydro_inflow` is block-capable: a bare entity id parses with `block_id: None`.
    #[test]
    fn test_build_hydro_inflow_no_block() {
        let var = build_variable_ref("hydro_inflow", EntityId(3), None, None).unwrap();
        assert_eq!(
            var,
            VariableRef::HydroInflow {
                hydro_id: EntityId(3),
                block_id: None,
            }
        );
    }

    /// `hydro_inflow` accepts a block argument: the upstream-release terms are per-block,
    /// so `hydro_inflow(h, k)` threads `block_id: Some(k)` through.
    #[test]
    fn test_build_hydro_inflow_with_block() {
        let var = build_variable_ref("hydro_inflow", EntityId(3), Some(0), None).unwrap();
        assert_eq!(
            var,
            VariableRef::HydroInflow {
                hydro_id: EntityId(3),
                block_id: Some(0),
            }
        );
    }

    /// `hydro_storage_initial` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_storage_initial_block_none_and_some() {
        let none = build_variable_ref("hydro_storage_initial", EntityId(4), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(4),
                block_id: None,
            }
        );
        let some = build_variable_ref("hydro_storage_initial", EntityId(4), Some(2), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(4),
                block_id: Some(2),
            }
        );
    }

    /// `hydro_storage_final` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_storage_final_block_none_and_some() {
        let none = build_variable_ref("hydro_storage_final", EntityId(4), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(4),
                block_id: None,
            }
        );
        let some = build_variable_ref("hydro_storage_final", EntityId(4), Some(1), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(4),
                block_id: Some(1),
            }
        );
    }

    /// `hydro_useful_volume_initial` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_useful_volume_initial_block_none_and_some() {
        let none =
            build_variable_ref("hydro_useful_volume_initial", EntityId(4), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroUsefulVolumeInitial {
                hydro_id: EntityId(4),
                block_id: None,
            }
        );
        let some =
            build_variable_ref("hydro_useful_volume_initial", EntityId(4), Some(2), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroUsefulVolumeInitial {
                hydro_id: EntityId(4),
                block_id: Some(2),
            }
        );
    }

    /// `hydro_useful_volume_final` is block-capable: a bare entity id threads
    /// `block_id: None`, and a block argument threads `block_id: Some(k)`.
    #[test]
    fn test_build_hydro_useful_volume_final_block_none_and_some() {
        let none =
            build_variable_ref("hydro_useful_volume_final", EntityId(3), None, None).unwrap();
        assert_eq!(
            none,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(3),
                block_id: None,
            }
        );
        let some =
            build_variable_ref("hydro_useful_volume_final", EntityId(3), Some(0), None).unwrap();
        assert_eq!(
            some,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(3),
                block_id: Some(0),
            }
        );
    }

    /// Neither useful-volume boundary variable accepts a `bus=` selector — the
    /// post-match guard rejects it the same as for the storage-pair variables.
    #[test]
    fn test_build_hydro_useful_volume_bus_selector_rejected() {
        for name in ["hydro_useful_volume_final", "hydro_useful_volume_initial"] {
            assert!(build_variable_ref(name, EntityId(3), None, Some(EntityId(1))).is_err());
        }
    }

    /// `hydro_useful_volume_final(3, 0) >= 0`, parsed through the full expression
    /// parser, yields a single term referencing `HydroUsefulVolumeFinal`.
    #[test]
    fn test_expr_hydro_useful_volume_final_full_expression() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_useful_volume_final(3, 0) >= 0", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result[0].expression.terms.len(), 1);
        assert_eq!(
            result[0].expression.terms[0].variable,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(3),
                block_id: Some(0),
            }
        );
    }

    /// A two-term ramp expression `hydro_storage_final(5, 1) - hydro_storage_initial(5, 1)`
    /// parses to the two block-qualified storage-boundary terms.
    #[test]
    fn test_expr_storage_ramp_two_terms() {
        let expr = parse_expression(
            "hydro_storage_final(5, 1) - hydro_storage_initial(5, 1)",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((lit(&expr.terms[0]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(5),
                block_id: Some(1),
            }
        );
        assert!((lit(&expr.terms[1]) - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(5),
                block_id: Some(1),
            }
        );
    }

    /// AC-3: Unknown variable name → error.
    #[test]
    fn test_expr_unknown_variable_name() {
        let err = parse_expression("invalid_var(0)", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("unknown variable name"),
            "expected unknown variable error, got: {err}"
        );
    }

    /// Missing closing parenthesis → error.
    #[test]
    fn test_expr_missing_closing_paren() {
        let err = parse_expression("hydro_generation(10", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("expected ')'") || err.contains("unexpected end"),
            "expected paren error, got: {err}"
        );
    }

    /// Empty expression → error.
    #[test]
    fn test_expr_empty_is_error() {
        let err = parse_expression("", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("empty"),
            "expected empty expression error, got: {err}"
        );
    }

    // ── Relational split unit tests ────────────────────────────────────────────

    /// The three two-character relational lexemes tokenize to their own tokens,
    /// distinct from the single-character `Equals` a `bus=` argument emits.
    #[test]
    fn tokenize_recognises_relational_operators() {
        assert_eq!(tokenize("<=").unwrap(), vec![Token::Le]);
        assert_eq!(tokenize(">=").unwrap(), vec![Token::Ge]);
        assert_eq!(tokenize("==").unwrap(), vec![Token::EqEq]);
        assert_eq!(tokenize("=").unwrap(), vec![Token::Equals]);
    }

    /// A bare `<` or `>` (not part of a two-character operator) is a tokenizer error.
    #[test]
    fn tokenize_bare_relational_char_is_error() {
        for input in ["hydro_generation(0) < 5", "hydro_generation(0) > 5"] {
            let err = tokenize(input).unwrap_err();
            assert!(
                err.contains("not a valid operator"),
                "expected a bare-operator error for \"{input}\", got: {err}"
            );
        }
    }

    /// An operator-free expression parses to [`ParsedRelation::Plain`] with the
    /// same terms `parse_terms` alone produces — the split adds no observable
    /// difference for the zero-operator case.
    #[test]
    fn parse_relation_zero_operators_is_plain() {
        let input = "hydro_generation(10) + hydro_generation(11)";
        let ParsedRelation::Plain(parsed) =
            parse_relation(input, &HashMap::new(), &LineBusPairIndex::default()).unwrap()
        else {
            panic!("expected ParsedRelation::Plain for an operator-free expression");
        };
        let tokens = tokenize(input).unwrap();
        let direct = parse_terms(&tokens, &HashMap::new(), &LineBusPairIndex::default()).unwrap();
        assert_eq!(parsed, direct);
    }

    /// A single top-level operator splits into [`ParsedRelation::Relational`]
    /// with the right operator and non-empty sides.
    #[test]
    fn parse_relation_single_operator_splits() {
        for (input, expected_op) in [
            ("hydro_generation(0) <= 5", RelOp::Le),
            ("hydro_generation(0) >= 5", RelOp::Ge),
            ("hydro_generation(0) == 5", RelOp::Eq),
        ] {
            let ParsedRelation::Relational { lhs, op, rhs } =
                parse_relation(input, &HashMap::new(), &LineBusPairIndex::default()).unwrap()
            else {
                panic!("expected ParsedRelation::Relational for \"{input}\"");
            };
            assert_eq!(op, expected_op, "wrong operator for \"{input}\"");
            assert_eq!(lhs.len(), 1, "lhs for \"{input}\"");
            assert_eq!(rhs.len(), 1, "rhs for \"{input}\"");
        }
    }

    /// A relational operator nested inside a variable's argument list (the
    /// `bus=` single `=`) is not mistaken for a top-level split point — the
    /// depth check excludes it.
    #[test]
    fn parse_relation_ignores_operator_shaped_tokens_inside_parens() {
        let input = "hydro_turbined(5, bus=2) <= 10";
        let ParsedRelation::Relational { op, .. } =
            parse_relation(input, &HashMap::new(), &LineBusPairIndex::default()).unwrap()
        else {
            panic!("expected ParsedRelation::Relational for \"{input}\"");
        };
        assert_eq!(op, RelOp::Le);
    }

    /// Two top-level relational operators (an inline double-relational range)
    /// is a descriptive error naming the one-operator limit.
    #[test]
    fn parse_relation_two_operators_is_error() {
        let err = parse_relation(
            "line_exchange(3) <= 10 >= 2",
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap_err();
        assert!(
            err.contains("at most one relational operator"),
            "expected the too-many-operators error, got: {err}"
        );
    }

    /// A standalone numeric literal with no trailing `*` parses as a
    /// [`SplitTerm::Constant`] on a relational side.
    #[test]
    fn parse_split_side_standalone_constant() {
        let tokens = tokenize("73").unwrap();
        let terms =
            parse_split_side(&tokens, &HashMap::new(), &LineBusPairIndex::default()).unwrap();
        assert_eq!(terms.len(), 1);
        match &terms[0] {
            SplitTerm::Constant(v) => assert!((v - 73.0).abs() < f64::EPSILON),
            SplitTerm::Named(t) => panic!("expected a Constant term, got: {t:?}"),
        }
    }

    /// A leading minus on a standalone constant negates it.
    #[test]
    fn parse_split_side_negated_standalone_constant() {
        let tokens = tokenize("10 - 73").unwrap();
        let terms =
            parse_split_side(&tokens, &HashMap::new(), &LineBusPairIndex::default()).unwrap();
        assert_eq!(terms.len(), 2);
        for (term, expected) in terms.iter().zip([10.0, -73.0]) {
            match term {
                SplitTerm::Constant(v) => assert!((v - expected).abs() < f64::EPSILON),
                SplitTerm::Named(t) => panic!("expected a Constant term, got: {t:?}"),
            }
        }
    }

    /// `thermal_generation(5) <= 0.87 * hydro_generation(140) + 73` normalizes
    /// to the same terms as the hand-flattened LHS
    /// `thermal_generation(5) - 0.87 * hydro_generation(140)` after
    /// canonicalize; `bound_upper_affine` carries the constant,
    /// `bound_lower_affine` stays `None`.
    #[test]
    fn ac1_inline_constant_and_variable_rhs_normalizes() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "thermal_generation(5) <= 0.87 * hydro_generation(140) + 73", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();

        let hand_flattened = parse_expression(
            "thermal_generation(5) - 0.87 * hydro_generation(140)",
            &HashMap::new(),
        )
        .unwrap();
        let mut hand_flattened = hand_flattened;
        hand_flattened.canonicalize();

        assert_eq!(result[0].expression.terms, hand_flattened.terms);
        assert_eq!(
            result[0].bound_upper_affine,
            Some(AffineBound {
                constant: 73.0,
                terms: vec![]
            })
        );
        assert_eq!(result[0].bound_lower_affine, None);
    }

    /// `line_exchange(3) >= @target` where `target` is a loaded parameter
    /// resolves the lower endpoint to that parameter, leaving the upper `None`.
    #[test]
    fn ac2_inline_lower_bound_bare_param_reference() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "line_exchange(3) >= @target", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let mut tbl = HashMap::new();
        tbl.insert("target".to_string(), EntityId(99));
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        assert_eq!(
            result[0].bound_lower_affine,
            Some(AffineBound::single(EntityId(99)))
        );
        assert_eq!(result[0].bound_upper_affine, None);
    }

    /// `hydro_generation(0) - hydro_generation(0) <= 5` merges to an empty
    /// term list (the net-zero column dropped) and `bound_upper_affine` carries
    /// the constant `5`.
    #[test]
    fn ac3_net_zero_column_drops_and_constant_bound_survives() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_generation(0) - hydro_generation(0) <= 5", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert!(result[0].expression.terms.is_empty());
        assert_eq!(
            result[0].bound_upper_affine,
            Some(AffineBound {
                constant: 5.0,
                terms: vec![]
            })
        );
        assert_eq!(result[0].bound_lower_affine, None);
    }

    /// DECOMP-shaped form 1: a decision-variable RHS mixing two variable terms
    /// and a constant (`RSE <= -0.316*FBIPS + 0.5*GPC + 12352`).
    #[test]
    fn decomp_shaped_decision_variable_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "rse", "expression": "bus_deficit(0) <= -0.316 * bus_deficit(1) + 0.5 * bus_deficit(2) + 12352", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3);
        let find = |bus_id: i32| {
            terms
                .iter()
                .find(|t| {
                    t.variable
                        == VariableRef::BusDeficit {
                            bus_id: EntityId(bus_id),
                            block_id: None,
                        }
                })
                .unwrap_or_else(|| panic!("expected bus_deficit({bus_id}) in {terms:?}"))
        };
        assert!((effective(find(0)) - 1.0).abs() < f64::EPSILON);
        assert!((effective(find(1)) - 0.316).abs() < 1e-10);
        assert!((effective(find(2)) - (-0.5)).abs() < 1e-10);
        assert_eq!(
            result[0].bound_upper_affine,
            Some(AffineBound {
                constant: 12352.0,
                terms: vec![]
            })
        );
        assert_eq!(result[0].bound_lower_affine, None);
    }

    /// DECOMP-shaped form 2: affine parameters plus a constant, no decision
    /// variable on the RHS at all (`RNE <= 11000 - 0.04*val_demanda(3) +
    /// 0.06*ger_pee(11)`, authored with cobre's own `@name` scalar-parameter
    /// grammar).
    #[test]
    fn decomp_shaped_affine_param_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "rne", "expression": "bus_deficit(0) <= 11000 - 0.04 * @val_demanda_3 + 0.06 * @ger_pee_11", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let mut tbl = HashMap::new();
        tbl.insert("val_demanda_3".to_string(), EntityId(50));
        tbl.insert("ger_pee_11".to_string(), EntityId(51));
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();

        assert_eq!(result[0].expression.terms.len(), 1);
        assert!((lit(&result[0].expression.terms[0]) - 1.0).abs() < f64::EPSILON);

        let bound = result[0]
            .bound_upper_affine
            .as_ref()
            .expect("upper bound present");
        assert!((bound.constant - 11000.0).abs() < f64::EPSILON);
        assert_eq!(bound.terms.len(), 2);
        assert!(bound.terms.contains(&(-0.04, EntityId(50))));
        assert!(bound.terms.contains(&(0.06, EntityId(51))));
        assert_eq!(result[0].bound_lower_affine, None);
    }

    /// DECOMP-shaped form 3: a pure-constant RHS (`FSIOR <= 73`).
    #[test]
    fn decomp_shaped_pure_constant_rhs() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "fsior", "expression": "bus_deficit(0) <= 73", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result[0].expression.terms.len(), 1);
        assert_eq!(
            result[0].bound_upper_affine,
            Some(AffineBound {
                constant: 73.0,
                terms: vec![]
            })
        );
        assert_eq!(result[0].bound_lower_affine, None);
    }

    // ── Parenthesized group unit tests ────────────────────────────────────────

    /// A distributed group's terms equal the hand-flattened form's terms, each on
    /// effective coefficient (`coefficient * scale`) and variable.
    #[test]
    fn test_expr_group_distribution_equals_hand_flattened() {
        let grouped = parse_expression(
            "2 * (hydro_generation(0) - hydro_generation(1))",
            &HashMap::new(),
        )
        .unwrap();
        let flattened = parse_expression(
            "2 * hydro_generation(0) - 2 * hydro_generation(1)",
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(grouped.terms.len(), flattened.terms.len());
        for (g, f) in grouped.terms.iter().zip(flattened.terms.iter()) {
            assert!(
                (effective(g) - effective(f)).abs() < f64::EPSILON,
                "effective coefficients differ: {} vs {}",
                effective(g),
                effective(f)
            );
            assert_eq!(g.variable, f.variable);
        }
    }

    /// Positive inner signs inside a group distribute the same way as negative
    /// ones.
    #[test]
    fn test_expr_group_distribution_positive_inner_signs() {
        let expr = parse_expression(
            "3 * (hydro_generation(0) + hydro_generation(1))",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((effective(&expr.terms[0]) - 3.0).abs() < f64::EPSILON);
        assert!((effective(&expr.terms[1]) - 3.0).abs() < f64::EPSILON);
    }

    /// Nested groups multiply coefficients inward — `2 * (a - 3 * (b + c))`
    /// distributes `2` then `2*3` — term 0 keeps effective coefficient `2.0`,
    /// terms 1 and 2 land on `-6.0`.
    #[test]
    fn test_expr_group_nested_distribution_multiplies_inward() {
        let expr = parse_expression(
            "2 * (hydro_generation(0) - 3 * (hydro_generation(1) + hydro_generation(2)))",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 3);
        assert!((effective(&expr.terms[0]) - 2.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert!((effective(&expr.terms[1]) - (-6.0)).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
        assert!((effective(&expr.terms[2]) - (-6.0)).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[2].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(2),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// A bare group with the implicit `1.0` coefficient reproduces the
    /// un-grouped term list unchanged.
    #[test]
    fn test_expr_bare_group_implicit_unit_coefficient() {
        let grouped = parse_expression(
            "(hydro_generation(0) + hydro_generation(1))",
            &HashMap::new(),
        )
        .unwrap();
        let flat =
            parse_expression("hydro_generation(0) + hydro_generation(1)", &HashMap::new()).unwrap();
        assert_eq!(grouped.terms, flat.terms);
    }

    /// A leading `-` before a bare group negates every inner term.
    #[test]
    fn test_expr_negated_bare_group() {
        let expr = parse_expression(
            "hydro_generation(5) - (hydro_generation(0) + hydro_generation(1))",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 3);
        assert!((effective(&expr.terms[1]) - (-1.0)).abs() < f64::EPSILON);
        assert!((effective(&expr.terms[2]) - (-1.0)).abs() < f64::EPSILON);
    }

    /// `@param * (...)` is rejected, naming the parameter and stating a group
    /// takes only a literal coefficient.
    #[test]
    fn test_expr_param_scaled_group_is_rejected() {
        let tbl = one_param_table();
        let err = parse_expression("@rho * (hydro_generation(0) + hydro_generation(1))", &tbl)
            .unwrap_err();
        assert!(
            err.contains("rho"),
            "error should name the parameter, got: {err}"
        );
        assert!(
            err.contains("literal coefficient"),
            "error should state a group takes only a literal coefficient, got: {err}"
        );
    }

    /// `coefficient * @param * (...)` is rejected the same way as the bare
    /// `@param * (...)` form.
    #[test]
    fn test_expr_literal_times_param_scaled_group_is_rejected() {
        let tbl = one_param_table();
        let err = parse_expression(
            "2.0 * @rho * (hydro_generation(0) + hydro_generation(1))",
            &tbl,
        )
        .unwrap_err();
        assert!(
            err.contains("rho"),
            "error should name the parameter, got: {err}"
        );
        assert!(
            err.contains("literal coefficient"),
            "error should state a group takes only a literal coefficient, got: {err}"
        );
    }

    /// An empty group `()` is rejected.
    #[test]
    fn test_expr_empty_group_is_rejected() {
        let err = parse_expression("3 * ()", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("empty parenthesized group"),
            "expected empty group error, got: {err}"
        );
    }

    /// A bare numeric constant inside a group is rejected, scaled or not.
    #[test]
    fn test_expr_constant_inside_group_is_rejected() {
        for expr in [
            "hydro_generation(0) + (hydro_generation(1) + 5)",
            "2 * (hydro_generation(1) - 3)",
        ] {
            let err = parse_expression(expr, &HashMap::new()).unwrap_err();
            assert!(
                err.contains("expected '*' after coefficient"),
                "expected constant-inside-group error for {expr:?}, got: {err}"
            );
        }
    }

    /// An unterminated group (no closing `)`) is rejected.
    #[test]
    fn test_expr_unterminated_group_is_rejected() {
        let err = parse_expression(
            "hydro_generation(0) + (hydro_generation(1) + hydro_generation(2)",
            &HashMap::new(),
        )
        .unwrap_err();
        assert!(
            err.contains("expected ')'") || err.contains("unexpected end"),
            "expected unterminated-group error, got: {err}"
        );
    }

    /// A stray token where an operator or `)` is expected inside a group is
    /// rejected.
    #[test]
    fn test_expr_group_stray_token_is_rejected() {
        let err = parse_expression(
            "2 * (hydro_generation(0) hydro_generation(1))",
            &HashMap::new(),
        )
        .unwrap_err();
        assert!(
            err.contains("expected '+', '-', or ')'"),
            "expected stray-token-in-group error, got: {err}"
        );
    }

    /// A group containing a `@param * variable` term distributes the literal
    /// coefficient into the term's `scale` while preserving the `Parameter`
    /// coefficient kind.
    #[test]
    fn test_expr_group_with_param_term_preserves_parameter_kind() {
        let tbl = one_param_table();
        let expr = parse_expression("2.0 * (@rho * hydro_generation(0))", &tbl).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(param_id(&expr.terms[0]), EntityId(7));
        assert!((expr.terms[0].scale - 2.0).abs() < f64::EPSILON);
    }

    /// All stage-only and block-capable variable names parse to the right
    /// [`VariableRef`]. Covers the 21 entity-keyed keywords (the stage-level
    /// `anticipated_decision` is exercised by its own dedicated tests).
    #[test]
    fn test_expr_all_21_entity_keyed_variable_types_recognised() {
        let cases: &[(&str, VariableRef)] = &[
            (
                "hydro_storage(0)",
                VariableRef::HydroStorage {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "hydro_turbined(0)",
                VariableRef::HydroTurbined {
                    hydro_id: EntityId(0),
                    block_id: None,
                    bus_id: None,
                },
            ),
            (
                "hydro_spillage(0)",
                VariableRef::HydroSpillage {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "hydro_diversion(0)",
                VariableRef::HydroDiversion {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "hydro_outflow(0)",
                VariableRef::HydroOutflow {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "hydro_generation(0)",
                VariableRef::HydroGeneration {
                    hydro_id: EntityId(0),
                    block_id: None,
                    bus_id: None,
                },
            ),
            (
                "hydro_evaporation(0)",
                VariableRef::HydroEvaporation {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "hydro_withdrawal(0)",
                VariableRef::HydroWithdrawal {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "hydro_inflow(0)",
                VariableRef::HydroInflow {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "thermal_generation(0)",
                VariableRef::ThermalGeneration {
                    thermal_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "line_direct(0)",
                VariableRef::LineDirect {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "line_reverse(0)",
                VariableRef::LineReverse {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "line_exchange(0)",
                VariableRef::LineExchange {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "bus_deficit(0)",
                VariableRef::BusDeficit {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "bus_excess(0)",
                VariableRef::BusExcess {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "pumping_flow(0)",
                VariableRef::PumpingFlow {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "pumping_power(0)",
                VariableRef::PumpingPower {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "contract_import(0)",
                VariableRef::ContractImport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "contract_export(0)",
                VariableRef::ContractExport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "non_controllable_generation(0)",
                VariableRef::NonControllableGeneration {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "non_controllable_curtailment(0)",
                VariableRef::NonControllableCurtailment {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
        ];

        assert_eq!(cases.len(), 21, "must have exactly 21 variable types");

        for (input, expected) in cases {
            let expr = parse_expression(input, &HashMap::new())
                .unwrap_or_else(|e| panic!("parse failed for \"{input}\": {e}"));
            assert_eq!(expr.terms.len(), 1, "single term for \"{input}\"");
            assert_eq!(
                &expr.terms[0].variable, expected,
                "wrong VariableRef for \"{input}\""
            );
        }
    }

    // ── Bus selector unit tests ────────────────────────────────────────────────

    /// All four argument forms parse: bare, positional block only, named `bus=`
    /// only, and both together — `bus=` is the grammar's first named argument, so
    /// a bare positional third slot (rejected by design) must fail to parse.
    #[test]
    fn parse_bus_selector_all_four_forms() {
        let cases: &[(&str, Option<usize>, Option<i32>)] = &[
            ("hydro_turbined(5)", None, None),
            ("hydro_turbined(5, 0)", Some(0), None),
            ("hydro_turbined(5, bus=2)", None, Some(2)),
            ("hydro_turbined(5, 0, bus=2)", Some(0), Some(2)),
        ];

        for (input, expected_block, expected_bus) in cases {
            let expr = parse_expression(input, &HashMap::new())
                .unwrap_or_else(|e| panic!("parse failed for \"{input}\": {e}"));
            assert_eq!(expr.terms.len(), 1, "single term for \"{input}\"");
            match &expr.terms[0].variable {
                VariableRef::HydroTurbined {
                    hydro_id,
                    block_id,
                    bus_id,
                } => {
                    assert_eq!(*hydro_id, EntityId(5), "hydro_id for \"{input}\"");
                    assert_eq!(block_id, expected_block, "block_id for \"{input}\"");
                    assert_eq!(bus_id.map(|b| b.0), *expected_bus, "bus_id for \"{input}\"");
                }
                other => panic!("expected HydroTurbined for \"{input}\", got {other:?}"),
            }
        }
    }

    /// `bus=` is accepted only by `hydro_turbined`/`hydro_generation`; every
    /// malformed named-argument shape is rejected; and an unknown variable name
    /// reports itself, not the bus selector — the post-match guard runs AFTER
    /// the name match, never before it.
    #[test]
    fn parse_bus_selector_rejections() {
        for expr in [
            "hydro_storage(5, bus=2)",
            "hydro_spillage(5, bus=2)",
            "hydro_outflow(5, bus=2)",
        ] {
            let err = parse_expression(expr, &HashMap::new()).unwrap_err();
            assert!(
                err.contains("hydro_turbined") && err.contains("hydro_generation"),
                "expected message naming hydro_turbined/hydro_generation for \"{expr}\", got: {err}"
            );
        }

        for expr in [
            "hydro_turbined(5, foo=2)",
            "hydro_turbined(5, bus=2, bus=3)",
            "hydro_turbined(5, bus=2, 0)",
            "hydro_turbined(5, bus=)",
            "hydro_turbined(5, bus 2)",
            "hydro_turbined(5, bus=2.5)",
            "hydro_turbined(5, bus=-1)",
        ] {
            assert!(
                parse_expression(expr, &HashMap::new()).is_err(),
                "expected Err for \"{expr}\""
            );
        }

        let err = parse_expression("not_a_variable(5, bus=2)", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("not_a_variable"),
            "expected the unknown variable name in the message, got: {err}"
        );
        assert!(
            !err.contains("bus selector"),
            "an unknown variable must not be reported as a bus-selector error, got: {err}"
        );
    }

    /// A second positional block argument is refused, equal or not, with or without `bus=`.
    #[test]
    fn test_expr_repeated_block_argument_is_rejected() {
        for expr in [
            "hydro_turbined(5, 0, 0)",
            "hydro_turbined(5, 0, 1)",
            "hydro_generation(5, 1, 2, bus=3)",
            "thermal_generation(0, 0, 0)",
            "2 * (hydro_turbined(5, 0, 1))",
        ] {
            let err = parse_expression(expr, &HashMap::new()).expect_err(expr);
            assert!(
                err.contains("repeated block argument in variable"),
                "expected the repeated-block message for \"{expr}\", got: {err}"
            );
        }

        let err = parse_expression("hydro_turbined(5, 0, 1)", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("\"hydro_turbined\""),
            "expected the variable name in the message, got: {err}"
        );
    }

    /// One block argument per variable stays accepted, including across separate terms.
    #[test]
    fn test_expr_single_block_argument_per_variable_is_accepted() {
        for expr in ["hydro_turbined(5, 1)", "hydro_turbined(5, 1, bus=2)"] {
            let parsed = parse_expression(expr, &HashMap::new());
            assert!(
                parsed.is_ok(),
                "expected Ok for \"{expr}\", got: {parsed:?}"
            );
        }

        for (expr, expected) in [
            (
                "hydro_turbined(5, 0) + hydro_turbined(5, 1)",
                [Some(0), Some(1)],
            ),
            (
                "hydro_turbined(5, 0) + hydro_turbined(5, 0)",
                [Some(0), Some(0)],
            ),
        ] {
            let parsed = parse_expression(expr, &HashMap::new()).unwrap();
            let blocks: Vec<Option<usize>> = parsed
                .terms
                .iter()
                .map(|term| match term.variable {
                    VariableRef::HydroTurbined { block_id, .. } => block_id,
                    ref other => panic!("expected HydroTurbined for \"{expr}\", got {other:?}"),
                })
                .collect();
            assert_eq!(blocks, expected, "for \"{expr}\"");
        }
    }

    // ── Line bus-pair addressing unit tests ────────────────────────────────────

    /// A pair matching a line's declared source→target resolves to that line with a
    /// positive orientation.
    #[test]
    fn line_bus_pair_forward_resolves_to_line_positive() {
        let index = build_line_bus_pair_index(&[line(5, 3, 7)]).expect("index builds");
        let expr = parse_expression_with_index(
            "line_exchange(source_bus=3, target_bus=7)",
            &HashMap::new(),
            &index,
        )
        .expect("pair form parses");
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::LineExchange {
                line_id: EntityId(5),
                block_id: None,
            }
        );
        assert!((lit(&expr.terms[0]) * expr.terms[0].scale - 1.0).abs() < f64::EPSILON);
    }

    /// The reversed pair resolves to the same line with a negative orientation.
    #[test]
    fn line_bus_pair_reversed_resolves_to_same_line_negative() {
        let index = build_line_bus_pair_index(&[line(5, 3, 7)]).expect("index builds");
        let expr = parse_expression_with_index(
            "line_exchange(source_bus=7, target_bus=3)",
            &HashMap::new(),
            &index,
        )
        .expect("reversed pair form parses");
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::LineExchange {
                line_id: EntityId(5),
                block_id: None,
            }
        );
        assert!((lit(&expr.terms[0]) * expr.terms[0].scale - (-1.0)).abs() < f64::EPSILON);
    }

    /// The pair form is rejected on any variable other than `line_exchange`, both after a
    /// positional argument and as the leading form.
    #[test]
    fn line_bus_pair_on_wrong_variable_is_rejected() {
        let index = build_line_bus_pair_index(&[line(5, 3, 7)]).expect("index builds");
        for input in [
            "hydro_generation(0, source_bus=3, target_bus=7)",
            "hydro_generation(source_bus=3, target_bus=7)",
        ] {
            let err = parse_expression_with_index(input, &HashMap::new(), &index).unwrap_err();
            assert!(
                err.contains("line_exchange"),
                "message must state the pair form is only accepted on line_exchange for \"{input}\", got: {err}"
            );
        }
    }

    /// A pair form carrying only one of the two named arguments is rejected.
    #[test]
    fn line_bus_pair_partial_form_is_rejected() {
        let index = build_line_bus_pair_index(&[line(5, 3, 7)]).expect("index builds");
        for input in ["line_exchange(source_bus=3)", "line_exchange(target_bus=7)"] {
            let err = parse_expression_with_index(input, &HashMap::new(), &index).unwrap_err();
            assert!(
                err.contains("both") && err.contains("source_bus") && err.contains("target_bus"),
                "expected a both-required error for \"{input}\", got: {err}"
            );
        }
    }

    /// A pair with no matching line is a descriptive error naming the pair.
    #[test]
    fn line_bus_pair_no_matching_line_is_error() {
        let index = build_line_bus_pair_index(&[line(5, 3, 7)]).expect("index builds");
        let err = parse_expression_with_index(
            "line_exchange(source_bus=1, target_bus=2)",
            &HashMap::new(),
            &index,
        )
        .unwrap_err();
        assert!(
            err.contains("no line connects") && err.contains('1') && err.contains('2'),
            "expected an unmatched-pair error naming the buses, got: {err}"
        );
    }

    /// Two distinct lines sharing one unordered bus pair make the index build fail loudly,
    /// naming the buses and pointing at named expressions — regardless of the lines'
    /// declared orientations or their order in the input.
    #[test]
    fn line_bus_pair_duplicate_pair_is_loud_error() {
        for lines in [
            [line(1, 3, 7), line(2, 3, 7)],
            [line(2, 7, 3), line(1, 3, 7)],
        ] {
            let err = build_line_bus_pair_index(&lines).unwrap_err();
            match err {
                LoadError::SchemaError { message, .. } => {
                    assert!(
                        message.contains("buses 3 and 7"),
                        "message must name buses 3 and 7: {message}"
                    );
                    assert!(
                        message.contains("named expression"),
                        "message must point at named expressions: {message}"
                    );
                }
                other => panic!("expected SchemaError, got: {other:?}"),
            }
        }
    }

    /// The forward pair form desugars to the exact terms a direct `line_exchange(line_id)`
    /// produces — byte-identity of the addressing sugar.
    #[test]
    fn line_bus_pair_byte_identical_to_direct_form() {
        let index = build_line_bus_pair_index(&[line(5, 3, 7)]).expect("index builds");
        let pair = parse_expression_with_index(
            "line_exchange(source_bus=3, target_bus=7)",
            &HashMap::new(),
            &index,
        )
        .expect("pair form parses");
        let direct = parse_expression_with_index("line_exchange(5)", &HashMap::new(), &index)
            .expect("direct form parses");
        assert_eq!(pair.terms, direct.terms);
    }

    // ── @name parameter reference unit tests ─────────────────────────────────

    /// AC-1: `@rho_eq * hydro_generation(0)` with implicit scale 1.0.
    #[test]
    fn parses_param_ref_implicit_scale() {
        let tbl = one_param_table();
        let expr = parse_expression("@rho_eq * hydro_generation(0)", &tbl).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(param_id(&expr.terms[0]), EntityId(7));
        assert!((expr.terms[0].scale - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// AC-2: `2.5 * @rho_eq * hydro_generation(0)` with explicit literal scale.
    #[test]
    fn parses_param_ref_with_literal_scale() {
        let tbl = one_param_table();
        let expr = parse_expression("2.5 * @rho_eq * hydro_generation(0)", &tbl).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(param_id(&expr.terms[0]), EntityId(7));
        assert!((expr.terms[0].scale - 2.5).abs() < 1e-10);
    }

    /// AC-3: Unknown `@name` produces an error naming the identifier.
    #[test]
    fn unknown_param_ref_returns_named_error() {
        let err = parse_expression("@unknown * hydro_generation(0)", &HashMap::new()).unwrap_err();
        assert!(
            err.contains("unknown parameter"),
            "error should contain 'unknown parameter', got: {err}"
        );
        assert!(
            err.contains("@unknown"),
            "error should name the offending ref, got: {err}"
        );
    }

    /// A standalone `@name` (no trailing `* variable`) parses as a named-expression
    /// reference; against no matching declaration it errors as undeclared.
    #[test]
    fn bare_reference_without_declaration_is_undeclared() {
        let tbl = one_param_table();
        let err = parse_expression("@rho_eq + hydro_generation(0)", &tbl).unwrap_err();
        assert!(
            err.contains("undeclared") && err.contains("rho_eq"),
            "error should name the undeclared reference, got: {err}"
        );
    }

    /// A minus sign before `@name` negates the referenced scale.
    #[test]
    fn param_ref_after_minus_negates_scale() {
        let tbl = one_param_table();
        let expr =
            parse_expression("thermal_generation(5) - @rho * hydro_generation(3)", &tbl).unwrap();
        assert_eq!(expr.terms.len(), 2);
        // First term: literal coefficient 1.0
        assert!((lit(&expr.terms[0]) - 1.0).abs() < f64::EPSILON);
        // Second term: parameter coefficient with scale -1.0
        assert_eq!(param_id(&expr.terms[1]), EntityId(7));
        assert!((expr.terms[1].scale - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Two `@name` references in a single term is rejected.
    #[test]
    fn two_param_refs_in_one_term_is_error() {
        let mut tbl = one_param_table();
        tbl.insert("b".to_string(), EntityId(8));
        let err = parse_expression("@rho_eq * @b * hydro_generation(0)", &tbl).unwrap_err();
        assert!(
            err.contains("only one @parameter reference"),
            "error should say 'only one @parameter reference', got: {err}"
        );
    }

    /// `2.0 * @rho * hydro_generation(0)` yields scale 2.0 and Parameter coefficient.
    #[test]
    fn nested_literal_and_param_ref() {
        let tbl = one_param_table();
        let expr = parse_expression("2.0 * @rho * hydro_generation(0)", &tbl).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(param_id(&expr.terms[0]), EntityId(7));
        assert!((expr.terms[0].scale - 2.0).abs() < f64::EPSILON);
    }

    // ── parse_generic_constraints integration tests ───────────────────────────

    /// Valid 2-constraint file. First has 2 hydro_generation terms.
    #[test]
    fn test_parse_valid_two_constraints() {
        let f = write_json(VALID_JSON);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();

        // Should be 2 constraints, sorted by id ascending: 0, 1.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, EntityId(0)); // id=0 was second in JSON
        assert_eq!(result[1].id, EntityId(1)); // id=1 was first in JSON

        // The first constraint (id=1, "min_hydro") has 2 hydro_generation terms.
        // After sorting, result[1] is the "min_hydro" constraint.
        let min_hydro = &result[1];
        assert_eq!(min_hydro.expression.terms.len(), 2);
        assert!((lit(&min_hydro.expression.terms[0]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            min_hydro.expression.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            min_hydro.expression.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(11),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Authoring-order invariance: two constraints differing only in the order their
    /// same terms are written parse to element-for-element-equal `terms`, because
    /// `convert` canonicalizes every expression.
    #[test]
    fn canonical_parse_order_invariance_hydro_generation() {
        let forward = r#"{
  "constraints": [
    { "id": 0, "name": "c", "expression": "hydro_generation(0) + hydro_generation(1)", "slack": { "enabled": false } }
  ]
}"#;
        let reversed = r#"{
  "constraints": [
    { "id": 0, "name": "c", "expression": "hydro_generation(1) + hydro_generation(0)", "slack": { "enabled": false } }
  ]
}"#;
        let ff = write_json(forward);
        let rf = write_json(reversed);
        let a = parse_generic_constraints(ff.path(), &HashMap::new(), &LineBusPairIndex::default())
            .unwrap();
        let b = parse_generic_constraints(rf.path(), &HashMap::new(), &LineBusPairIndex::default())
            .unwrap();
        assert_eq!(a[0].expression.terms, b[0].expression.terms);
        // Canonical order is by entity id: hydro_generation(0) then hydro_generation(1).
        assert_eq!(
            a[0].expression.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            a[0].expression.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Expression `"2.5 * thermal_generation(5) - hydro_generation(3)"` — after
    /// canonicalization the terms are content-ordered: hydro_generation (variant tag 5)
    /// precedes thermal_generation (tag 8), regardless of authoring order.
    #[test]
    fn test_parse_coefficient_and_subtraction_expression() {
        let f = write_json(VALID_JSON);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();

        // result[0] is id=0 "max_thermal"
        let max_thermal = &result[0];
        assert_eq!(max_thermal.expression.terms.len(), 2);
        assert!((lit(&max_thermal.expression.terms[0]) - (-1.0)).abs() < f64::EPSILON);
        assert!((max_thermal.expression.terms[0].scale - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            max_thermal.expression.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
                bus_id: None,
            }
        );
        assert!((lit(&max_thermal.expression.terms[1]) - 2.5).abs() < 1e-10);
        assert!((max_thermal.expression.terms[1].scale - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            max_thermal.expression.terms[1].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
    }

    /// A JSON constraint mixing a literal term and an `@rho_eq`-scaled parameter term
    /// parses; after canonicalization the terms are content-ordered, so
    /// hydro_generation (tag 5) precedes thermal_generation (tag 8).
    #[test]
    fn test_parse_param_ref_in_json_constraint() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "mixed",
      "expression": "thermal_generation(5) - @rho_eq * hydro_generation(3)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let tbl = one_param_table();
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        assert_eq!(result.len(), 1);
        let expr = &result[0].expression;
        assert_eq!(expr.terms.len(), 2);
        // First term (canonical): hydro_generation(3) with a parameter coef, scale -1.0.
        assert_eq!(param_id(&expr.terms[0]), EntityId(7));
        assert!((expr.terms[0].scale - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
                bus_id: None,
            }
        );
        // Second term (canonical): thermal_generation(5) literal 1.0.
        assert!((lit(&expr.terms[1]) - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
    }

    /// An inline `<=` against a bare `@name` resolves to
    /// `Some(AffineBound::single(id))` on the upper endpoint; the lower endpoint
    /// stays `None`.
    #[test]
    fn test_parse_inline_upper_bound_resolves_to_entity_id() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "demand_cap",
      "expression": "hydro_generation(3) <= @rho_eq",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let tbl = one_param_table();
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].bound_upper_affine,
            Some(AffineBound::single(EntityId(7)))
        );
        assert_eq!(result[0].bound_lower_affine, None);
    }

    /// An inline `>=` against a bare `@name` resolves the same way, on the lower
    /// endpoint.
    #[test]
    fn test_parse_inline_lower_bound_resolves_to_entity_id() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "demand_floor",
      "expression": "hydro_generation(3) >= @rho_eq",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let tbl = one_param_table();
        let result =
            parse_generic_constraints(f.path(), &tbl, &LineBusPairIndex::default()).unwrap();
        assert_eq!(
            result[0].bound_lower_affine,
            Some(AffineBound::single(EntityId(7)))
        );
    }

    /// A bound-position `@name` naming a parameter absent from `name_to_id` is a
    /// `SchemaError` on the constraint's `expression` field, naming the missing
    /// parameter with the "unknown parameter" wording — never the named-expression
    /// "undeclared reference" wording, since a bound-position `@name` never
    /// consults the named-expression table.
    #[test]
    fn test_parse_unknown_inline_bound_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "bad_bound",
      "expression": "hydro_generation(0) >= @missing_param",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("expression"),
                    "field should name the expression, got: {field}"
                );
                assert!(
                    message.contains("unknown parameter") && message.contains("missing_param"),
                    "message should be the unknown-parameter wording naming the missing \
                     parameter, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A relational side referencing a genuinely-declared named expression still
    /// inlines it — named-expression inlining runs on each side before
    /// normalization — so its variable terms fold into the merged LHS rather than
    /// the affine remainder; the bound-position-names-a-parameter carve-out
    /// applies only to a name that does NOT resolve in the table (the previous
    /// test).
    #[test]
    fn test_parse_bound_side_named_expression_inlines_into_merged_lhs() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(10) + hydro_generation(11)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_generation(0) >= @fnese", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].bound_lower_affine,
            Some(AffineBound {
                constant: 0.0,
                terms: vec![]
            }),
            "an all-variable RHS supplies an explicit zero bound, not \"no bound\""
        );
        assert_eq!(result[0].bound_upper_affine, None);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3, "hydro_generation(0) plus fnese's two terms");
        for hydro_id in [0, 10, 11] {
            assert!(
                terms.iter().any(|t| t.variable
                    == VariableRef::HydroGeneration {
                        hydro_id: EntityId(hydro_id),
                        block_id: None,
                        bus_id: None,
                    }),
                "expected hydro_generation({hydro_id}) in the merged terms, got: {terms:?}"
            );
        }
    }

    /// The retired `bound_upper_ref` JSON field is now an unknown field under
    /// `deny_unknown_fields` — a symbolic endpoint is authored inline instead.
    #[test]
    fn test_parse_retired_bound_upper_ref_field_is_unknown_field_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false },
      "bound_upper_ref": "demanda"
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for the retired \"bound_upper_ref\" field, got: {err:?}"
        );
    }

    /// A constraint declaring neither bound reference parses with both fields `None`
    /// (existing files are unaffected).
    #[test]
    fn test_parse_absent_bound_refs_are_none() {
        let f = write_json(VALID_JSON);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        for gc in &result {
            assert_eq!(gc.bound_lower_affine, None);
            assert_eq!(gc.bound_upper_affine, None);
        }
    }

    /// Unknown `@param` in JSON expression → SchemaError with "expression" field and
    /// "unknown parameter" in the message.
    #[test]
    fn test_parse_unknown_param_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "bad_ref",
      "expression": "@missing * hydro_generation(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("expression"),
                    "field should contain 'expression', got: {field}"
                );
                assert!(
                    message.contains("unknown parameter"),
                    "message should contain 'unknown parameter', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Invalid expression → SchemaError with "expression" in field.
    #[test]
    fn test_parse_invalid_expression_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "bad",
      "expression": "invalid_var(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("expression"),
                    "field should contain 'expression', got: {field}"
                );
                assert!(
                    message.contains("unknown variable"),
                    "message should contain 'unknown variable', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A repeated block argument → SchemaError on the constraint and on the named-expression field.
    #[test]
    fn parse_generic_constraints_repeated_block_argument_is_schema_error() {
        let in_constraint = r#"{
  "constraints": [
    { "id": 0, "name": "bad", "expression": "thermal_generation(0, 0, 1) <= 10", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(in_constraint);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "constraints[0].expression");
                assert!(
                    message.contains("repeated block argument in variable \"thermal_generation\""),
                    "message should name the repeated block argument, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }

        let in_named_expression = r#"{
  "expressions": [
    { "name": "e", "expression": "hydro_turbined(5, 0, 1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "@e <= 10", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(in_named_expression);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "expressions[0].expression");
                assert!(
                    message.contains("repeated block argument in variable \"hydro_turbined\""),
                    "message should name the repeated block argument, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A constant inside a group on a relational side → SchemaError on the expression field.
    #[test]
    fn parse_generic_constraints_relational_constant_inside_group_is_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "bad",
      "expression": "thermal_generation(5) <= 2 * (hydro_generation(140) + 73)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "constraints[0].expression");
                assert!(
                    message.contains("expected '*' after coefficient 73"),
                    "message should name the rejected constant, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Duplicate IDs → SchemaError.
    #[test]
    fn test_parse_duplicate_ids_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "b",
      "expression": "thermal_generation(1)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("id"),
                    "field should contain 'id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Slack enabled without penalty → SchemaError.
    #[test]
    fn test_parse_slack_enabled_without_penalty_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": true }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("penalty"),
                    "field should contain 'penalty', got: {field}"
                );
                assert!(
                    message.contains("absent") || message.contains("enabled"),
                    "message should explain the issue, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Clean break: a constraint object still carrying a `"sense"` key is rejected
    /// (`RawConstraint` is `deny_unknown_fields`, no alias) — shape is derived from
    /// the bounds parquet's endpoint pair, never authored on the constraint.
    #[test]
    fn test_parse_unknown_sense_field_returns_parse_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "sense": ">=",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for an unknown 'sense' field, got: {err:?}"
        );
    }

    /// `None` path → `Ok(Vec::new())` (tested via `load_generic_constraints`).
    /// The `load_*` wrapper is in `mod.rs`; here we test the `parse_*` function returns Ok
    /// for a valid empty constraints array.
    #[test]
    fn test_parse_empty_constraints_array() {
        let json = r#"{ "constraints": [] }"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert!(result.is_empty());
    }

    /// Sorted output: constraints come back in id-ascending order regardless of JSON order.
    #[test]
    fn test_parse_sorted_by_id() {
        let json = r#"{
  "constraints": [
    {
      "id": 5,
      "name": "c",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    },
    {
      "id": 2,
      "name": "b",
      "expression": "thermal_generation(0)",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "a",
      "expression": "line_direct(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, EntityId(0));
        assert_eq!(result[1].id, EntityId(2));
        assert_eq!(result[2].id, EntityId(5));
    }

    /// Full JSON constraint with `line_exchange` parses correctly.
    #[test]
    fn test_parse_line_exchange_json_constraint() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "net_exchange",
      "expression": "line_exchange(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "net_exchange");
        assert_eq!(result[0].expression.terms.len(), 1);
        assert_eq!(
            result[0].expression.terms[0].variable,
            VariableRef::LineExchange {
                line_id: EntityId(0),
                block_id: None,
            }
        );
    }

    /// Slack with zero penalty → SchemaError.
    #[test]
    fn test_parse_slack_zero_penalty_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": true, "penalty": 0.0 }
    }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("penalty"),
                    "field should contain 'penalty', got: {field}"
                );
                assert!(
                    message.contains("> 0.0"),
                    "message should mention > 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Description is optional — absent means `None`.
    #[test]
    fn test_parse_description_optional() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "nodesc",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].description.is_none());
    }

    // ── AC-3..AC-6: anticipated_decision parser tests ─────────────────────────

    /// AC-3: `anticipated_decision(5)` produces a single `AnticipatedDecision` term.
    #[test]
    fn anticipated_decision_simple_parse() {
        let expr = parse_expression("anticipated_decision(5)", &HashMap::new()).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert!(
            (lit(&expr.terms[0]) - 1.0).abs() < f64::EPSILON,
            "coefficient must be 1.0"
        );
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(5),
            }
        );
    }

    /// AC-4: `2.5 * anticipated_decision(5)` applies the literal scale.
    #[test]
    fn anticipated_decision_with_coefficient() {
        let expr = parse_expression("2.5 * anticipated_decision(5)", &HashMap::new()).unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert!(
            (lit(&expr.terms[0]) - 2.5).abs() < f64::EPSILON,
            "coefficient must be 2.5"
        );
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(5),
            }
        );
    }

    /// AC-5: `anticipated_decision(5, 0)` returns an error naming the variable
    /// and explaining it does not accept a block_id.
    #[test]
    fn anticipated_decision_rejects_block_id() {
        let result = parse_expression("anticipated_decision(5, 0)", &HashMap::new());
        assert!(
            result.is_err(),
            "should reject block_id for anticipated_decision"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("anticipated_decision"),
            "error message must name the variable: {msg}"
        );
        // Should mention that it doesn't accept a block_id
        assert!(
            msg.contains("block_id") || msg.contains("stage-level"),
            "error message should explain no block_id accepted: {msg}"
        );
    }

    /// AC-6: Whitespace insensitivity — `anticipated_decision( 5 )` produces the
    /// same term as `anticipated_decision(5)`.
    #[test]
    fn anticipated_decision_whitespace_insensitive() {
        let expr1 = parse_expression("anticipated_decision(5)", &HashMap::new()).unwrap();
        let expr2 = parse_expression("anticipated_decision( 5 )", &HashMap::new()).unwrap();
        assert_eq!(expr1.terms.len(), 1);
        assert_eq!(expr2.terms.len(), 1);
        assert_eq!(
            expr1.terms[0].variable, expr2.terms[0].variable,
            "whitespace must not affect the parsed variable"
        );
    }

    /// AC-6 (multi-term): `anticipated_decision(5) + anticipated_decision(6)` works.
    #[test]
    fn anticipated_decision_multi_term_expression() {
        let expr = parse_expression(
            "anticipated_decision(5) + anticipated_decision(6)",
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(5),
            }
        );
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(6),
            }
        );
    }

    // ── Named-expression declaration tests ────────────────────────────────────

    fn raw_named(name: &str, expression: &str) -> RawNamedExpression {
        RawNamedExpression {
            name: name.to_string(),
            expression: expression.to_string(),
            _description: None,
        }
    }

    /// Byte-neutral: no declarations ⇒ empty table, no error.
    #[test]
    fn parse_named_expressions_empty_is_ok() {
        let table = parse_named_expressions(
            &[],
            Path::new("generic_constraints.json"),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert!(table.is_empty());
    }

    /// Two distinct valid definitions ⇒ a two-entry table, each carrying its parsed expression.
    #[test]
    fn parse_named_expressions_two_valid_definitions() {
        let entries = [
            raw_named("fnese", "hydro_generation(0) + hydro_generation(1)"),
            raw_named("fns", "2.5 * thermal_generation(5) - hydro_generation(3)"),
        ];
        let table = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].0, "fnese");
        assert_eq!(table[0].1.len(), 2);
        assert_eq!(
            flat(&table[0].1[0]).variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(table[1].0, "fns");
        assert_eq!(table[1].1.len(), 2);
        assert!((lit(flat(&table[1].1[0])) - 2.5).abs() < 1e-10);
        assert_eq!(
            flat(&table[1].1[1]).variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// Two declarations sharing a name ⇒ SchemaError naming the duplicate, field under `expressions`.
    #[test]
    fn parse_named_expressions_duplicate_name_is_error() {
        let entries = [
            raw_named("fnese", "hydro_generation(0)"),
            raw_named("fnese", "hydro_generation(1)"),
        ];
        let err = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("expressions"), "field: {field}");
                assert!(message.contains("fnese"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A declaration whose name is already a scalar parameter ⇒ SchemaError naming it.
    #[test]
    fn parse_named_expressions_param_collision_is_error() {
        let entries = [raw_named("rho", "hydro_generation(0)")];
        let err = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &one_param_table(),
            &LineBusPairIndex::default(),
        )
        .unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("expressions"), "field: {field}");
                assert!(message.contains("rho"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A standalone `@name` in a declaration parses to a [`ParsedTerm::Ref`]; the
    /// referenced expression stays unresolved until the full table inlines it.
    #[test]
    fn parse_named_expressions_reference_parses_to_ref() {
        let entries = [raw_named("combo", "@fnese + hydro_generation(0)")];
        let table = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].1.len(), 2);
        match &table[0].1[0] {
            ParsedTerm::Ref { name, scale } => {
                assert_eq!(name, "fnese");
                assert!((scale - 1.0).abs() < f64::EPSILON);
            }
            ParsedTerm::Flat(lt) => panic!("expected a reference term, got Flat: {lt:?}"),
        }
        assert_eq!(
            flat(&table[0].1[1]).variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// The parser does not reclassify a standalone `@name` by consulting the scalar
    /// parameter table: even a live parameter name standalone is a reference term
    /// (unresolvable as an expression, hence an undeclared error at load).
    #[test]
    fn parse_named_expressions_standalone_param_name_parses_to_ref() {
        let entries = [raw_named("combo", "@rho")];
        let table = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &one_param_table(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert_eq!(table.len(), 1);
        match &table[0].1[0] {
            ParsedTerm::Ref { name, .. } => assert_eq!(name, "rho"),
            ParsedTerm::Flat(lt) => panic!("expected a reference term, got Flat: {lt:?}"),
        }
    }

    /// A `@param` coefficient in a definition parses to a `Flat` parameter term.
    #[test]
    fn parse_named_expressions_param_coefficient_is_allowed() {
        let entries = [raw_named("scaled", "@rho * hydro_generation(0)")];
        let table = parse_named_expressions(
            &entries,
            Path::new("generic_constraints.json"),
            &one_param_table(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(param_id(flat(&table[0].1[0])), EntityId(7));
    }

    /// A file carrying both constraints and a valid `expressions` array parses; the
    /// returned constraints are unaffected by the declarations.
    #[test]
    fn parse_generic_constraints_with_expressions_key() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(10) + hydro_generation(11)", "description": "SE net flow" }
  ],
  "constraints": [
    {
      "id": 0,
      "name": "c0",
      "expression": "hydro_generation(0)",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, EntityId(0));
        assert_eq!(result[0].expression.terms.len(), 1);
    }

    /// A declaration referencing an undeclared name fails the load even when every
    /// constraint is valid and no constraint uses the declaration.
    #[test]
    fn parse_generic_constraints_bad_expression_declaration_fails_load() {
        let json = r#"{
  "expressions": [
    { "name": "bad", "expression": "@other + hydro_generation(0)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { message, .. } if message.contains("undeclared") && message.contains("other")),
            "expected an undeclared-reference error naming \"other\", got: {err:?}"
        );
    }

    /// An empty `expressions` array is byte-neutral — same constraints as with the key absent.
    #[test]
    fn parse_generic_constraints_empty_expressions_is_byte_neutral() {
        let with_key = r#"{ "expressions": [], "constraints": [ { "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": { "enabled": false } } ] }"#;
        let without_key = r#"{ "constraints": [ { "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": { "enabled": false } } ] }"#;
        let with_file = write_json(with_key);
        let without_file = write_json(without_key);
        let a = parse_generic_constraints(
            with_file.path(),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        let b = parse_generic_constraints(
            without_file.path(),
            &HashMap::new(),
            &LineBusPairIndex::default(),
        )
        .unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].id, b[0].id);
        assert_eq!(a[0].expression.terms.len(), b[0].expression.terms.len());
    }

    /// An unknown key inside a declaration is rejected (`deny_unknown_fields`).
    #[test]
    fn parse_generic_constraints_unknown_expression_field_returns_parse_error() {
        let json = r#"{
  "expressions": [
    { "name": "x", "expression": "hydro_generation(0)", "unexpected": 1 }
  ],
  "constraints": []
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for an unknown declaration field, got: {err:?}"
        );
    }

    // ── Named-expression inlining tests ───────────────────────────────────────

    fn effective(term: &LinearTerm) -> f64 {
        lit(term) * term.scale
    }

    /// Single-reference distribution: `2.0 * @fnese - thermal_generation(3)` with
    /// `fnese = hydro_generation(0) + hydro_generation(1)` flattens to the
    /// hand-written expansion — 3 terms, effective coefficients 2.0, 2.0, -1.0.
    #[test]
    fn parse_generic_constraints_inlines_single_reference() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(0) + hydro_generation(1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "2.0 * @fnese - thermal_generation(3)", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3);
        assert!((effective(&terms[0]) - 2.0).abs() < 1e-10);
        assert!((effective(&terms[1]) - 2.0).abs() < 1e-10);
        assert!((effective(&terms[2]) - (-1.0)).abs() < 1e-10);
        assert_eq!(
            terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            terms[2].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(3),
                block_id: None,
            }
        );
    }

    /// A group's inner term list may contain a standalone `@name` reference,
    /// exactly as a top-level term list may: the group's leading coefficient
    /// distributes into the reference's own `scale`, composing with the
    /// reference's own expansion at inline time.
    #[test]
    fn parse_generic_constraints_group_distributes_into_named_reference() {
        let json = r#"{
  "expressions": [
    { "name": "fnese", "expression": "hydro_generation(0) + hydro_generation(1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "2.0 * (@fnese - hydro_generation(2))", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 3);
        assert!((effective(&terms[0]) - 2.0).abs() < 1e-10);
        assert!((effective(&terms[1]) - 2.0).abs() < 1e-10);
        assert!((effective(&terms[2]) - (-2.0)).abs() < 1e-10);
    }

    /// Composition resolves transitively: a constraint referencing `outer`, which
    /// references `inner`, which references a flat `base`.
    #[test]
    fn parse_generic_constraints_inlines_composition() {
        let json = r#"{
  "expressions": [
    { "name": "base", "expression": "hydro_generation(0)" },
    { "name": "inner", "expression": "3.0 * @base" },
    { "name": "outer", "expression": "@inner + hydro_generation(1)" }
  ],
  "constraints": [
    { "id": 0, "name": "c0", "expression": "2.0 * @outer", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        let terms = &result[0].expression.terms;
        assert_eq!(terms.len(), 2);
        assert!((effective(&terms[0]) - 6.0).abs() < 1e-10);
        assert!((effective(&terms[1]) - 2.0).abs() < 1e-10);
        assert_eq!(
            terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
    }

    /// A 2-node reference cycle ⇒ SchemaError whose message names both nodes and
    /// the cycle path.
    #[test]
    fn parse_generic_constraints_two_node_cycle_errors() {
        let json = r#"{
  "expressions": [
    { "name": "a", "expression": "@b" },
    { "name": "b", "expression": "@a" }
  ],
  "constraints": []
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains('a') && message.contains('b'),
                    "message: {message}"
                );
                assert!(message.contains("a -> b -> a"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A self-reference is a cycle of length 1 ⇒ SchemaError naming the node.
    #[test]
    fn parse_generic_constraints_self_reference_errors() {
        let json = r#"{
  "expressions": [
    { "name": "e", "expression": "@e" }
  ],
  "constraints": []
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { message, .. } if message.contains("e -> e")),
            "expected a self-reference cycle naming \"e\", got: {err:?}"
        );
    }

    /// A reference to an undeclared name inside a constraint ⇒ SchemaError naming
    /// the missing name, on the constraint's expression field.
    #[test]
    fn parse_generic_constraints_undeclared_reference_in_constraint_errors() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "@nope", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("expression"), "field: {field}");
                assert!(
                    message.contains("undeclared") && message.contains("nope"),
                    "message: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `@param * @name` (a parameter coefficient in front of a reference) is
    /// rejected, reusing the "only one @parameter reference" error class.
    #[test]
    fn parse_generic_constraints_param_times_reference_rejected() {
        let json = r#"{
  "constraints": [
    { "id": 0, "name": "c0", "expression": "@rho * @fnese", "slack": { "enabled": false } }
  ]
}"#;
        let f = write_json(json);
        let err =
            parse_generic_constraints(f.path(), &one_param_table(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { message, .. } if message.contains("only one @parameter reference")),
            "expected the two-reference error class, got: {err:?}"
        );
    }

    /// A `levels`-deep doubling chain `e0 = hydro_generation(0)`,
    /// `e_k = @e_{k-1} + @e_{k-1}` as a JSON `expressions` array body.
    fn doubling_chain_expressions(levels: u32) -> String {
        let mut exprs = String::from(r#"{ "name": "e0", "expression": "hydro_generation(0)" }"#);
        for k in 1..=levels {
            let _ = write!(
                exprs,
                r#", {{ "name": "e{k}", "expression": "@e{prev} + @e{prev}" }}"#,
                prev = k - 1
            );
        }
        exprs
    }

    /// A declared-but-unreferenced doubling chain must parse without expanding: if
    /// declaration-time validation force-expanded it, this 2^60-term chain would
    /// exhaust memory. Completing quickly proves the cheap resolvability walk.
    #[test]
    fn parse_generic_constraints_unreferenced_doubling_chain_parses_fast() {
        let json = format!(
            r#"{{ "expressions": [ {} ], "constraints": [ {{ "id": 0, "name": "c0", "expression": "hydro_generation(0)", "slack": {{ "enabled": false }} }} ] }}"#,
            doubling_chain_expressions(60)
        );
        let f = write_json(&json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].expression.terms.len(), 1);
    }

    /// A constraint that references the top of the doubling chain triggers the
    /// term-budget cap — loudly and fast, not by exhausting memory.
    #[test]
    fn parse_generic_constraints_referenced_doubling_chain_hits_budget_fast() {
        let json = format!(
            r#"{{ "expressions": [ {} ], "constraints": [ {{ "id": 0, "name": "c0", "expression": "@e60", "slack": {{ "enabled": false }} }} ] }}"#,
            doubling_chain_expressions(60)
        );
        let f = write_json(&json);
        let err =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, message, .. }
                if field.contains("constraints[0].expression")
                    && message.contains("more than")
                    && message.contains("100000")),
            "expected a budget-exceeded error on the constraint, got: {err:?}"
        );
    }

    /// A legitimate large-but-under-cap flat constraint expression parses: the cap
    /// rejects only pathological exponential expansion, never a big flat sum.
    #[test]
    fn parse_generic_constraints_large_flat_expression_under_cap_parses() {
        let mut expr = String::from("hydro_generation(0)");
        for id in 1..1000 {
            let _ = write!(expr, " + hydro_generation({id})");
        }
        let json = format!(
            r#"{{ "constraints": [ {{ "id": 0, "name": "big", "expression": "{expr}", "slack": {{ "enabled": false }} }} ] }}"#
        );
        let f = write_json(&json);
        let result =
            parse_generic_constraints(f.path(), &HashMap::new(), &LineBusPairIndex::default())
                .unwrap();
        assert_eq!(result[0].expression.terms.len(), 1000);
    }
}
