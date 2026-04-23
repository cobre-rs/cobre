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
//!       "expression": "hydro_generation(0) + hydro_generation(1)",
//!       "sense": ">=",
//!       "slack": { "enabled": true, "penalty": 5000.0 }
//!     }
//!   ]
//! }
//! ```
//!
//! ## Expression grammar (spec SS3)
//!
//! ```text
//! expression ::= term (('+' | '-') term)*
//! term       ::= coefficient '*' variable | variable | number
//! variable   ::= var_name '(' entity_id (',' block_id)? ')'
//! ```
//!
//! All 20 variable names from the variable catalog are recognised. Block-specific
//! variables accept an optional second argument; stage-only variables (`hydro_storage`,
//! `hydro_evaporation`, `hydro_withdrawal`) must not have a block argument.
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! - No two constraints share the same `id`.
//! - `sense` must be `">="`, `"<="`, or `"=="`.
//! - `slack.enabled = true` requires `slack.penalty` to be present and > 0.0.
//! - Each `expression` string must parse without error.
//!
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in entity registries — Layer 3.
//! - Block ID validity for the referenced stage — Layer 3/5.

use cobre_core::{
    ConstraintExpression, ConstraintSense, EntityId, GenericConstraint, LinearTerm, SlackConfig,
    VariableRef,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `generic_constraints.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawGenericConstraintsFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of constraint entries.
    constraints: Vec<RawConstraint>,
}

/// Intermediate type for a single constraint entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct RawConstraint {
    /// Constraint identifier. Must be unique within the file.
    id: i32,

    /// Short name used in reports and log output.
    name: String,

    /// Optional human-readable description.
    description: Option<String>,

    /// Expression string to be parsed. E.g. `"2.5 * thermal_generation(5) - hydro_generation(3)"`.
    expression: String,

    /// Comparison sense: `">="`, `"<="`, or `"=="`.
    sense: String,

    /// Slack variable configuration.
    slack: RawSlackConfig,
}

/// Intermediate type for the slack configuration.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct RawSlackConfig {
    /// Whether a slack variable is allowed.
    enabled: bool,

    /// Penalty per unit of violation. Must be > 0.0 when `enabled` is `true`.
    penalty: Option<f64>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `constraints/generic_constraints.json` from `path`.
///
/// Reads the JSON file, deserialises it through intermediate serde types,
/// performs post-deserialization validation (including expression parsing),
/// then converts to `Vec<GenericConstraint>`. The result is sorted by `id`
/// ascending to satisfy declaration-order invariance.
///
/// # Errors
///
/// | Condition                                      | Error variant              |
/// | ---------------------------------------------- | -------------------------- |
/// | File not found / read failure                  | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field  | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the constraints array    | [`LoadError::SchemaError`] |
/// | Invalid `sense` value                          | [`LoadError::SchemaError`] |
/// | `slack.enabled = true` with absent or <= 0 penalty | [`LoadError::SchemaError`] |
/// | Expression syntax error                        | [`LoadError::SchemaError`] |
/// | Unknown variable name in expression            | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_generic_constraints;
/// use std::path::Path;
///
/// let constraints = parse_generic_constraints(
///     Path::new("case/constraints/generic_constraints.json")
/// ).expect("valid generic constraints file");
/// println!("loaded {} generic constraints", constraints.len());
/// ```
pub fn parse_generic_constraints(path: &Path) -> Result<Vec<GenericConstraint>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawGenericConstraintsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    convert(raw, path)
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized constraint data.
fn validate_raw(raw: &RawGenericConstraintsFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_ids(&raw.constraints, path)?;
    for (i, constraint) in raw.constraints.iter().enumerate() {
        validate_sense(&constraint.sense, i, path)?;
        validate_slack(&constraint.slack, i, path)?;
        // Expression is validated here to get accurate field paths; actual
        // parsed result is discarded — re-parsed during convert().
        parse_expression(&constraint.expression).map_err(|msg| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("constraints[{i}].expression"),
            message: msg,
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

/// Check that the sense string is one of the three allowed values.
fn validate_sense(sense: &str, constraint_index: usize, path: &Path) -> Result<(), LoadError> {
    match sense {
        ">=" | "<=" | "==" => Ok(()),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("constraints[{constraint_index}].sense"),
            message: format!("unknown variant \"{other}\": expected one of \">=\", \"<=\", \"==\""),
        }),
    }
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
/// Expression parsing is repeated here (after validation) to produce the final
/// `ConstraintExpression`. The validation pass guarantees no errors will occur here.
fn convert(
    raw: RawGenericConstraintsFile,
    path: &Path,
) -> Result<Vec<GenericConstraint>, LoadError> {
    let mut result = Vec::with_capacity(raw.constraints.len());

    for (i, c) in raw.constraints.into_iter().enumerate() {
        // Expression already validated; re-parse is infallible at this stage.
        let expression = parse_expression(&c.expression).map_err(|msg| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("constraints[{i}].expression"),
            message: msg,
        })?;

        let sense = match c.sense.as_str() {
            ">=" => ConstraintSense::GreaterEqual,
            "<=" => ConstraintSense::LessEqual,
            "==" => ConstraintSense::Equal,
            // Unreachable: already validated.
            other => {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("constraints[{i}].sense"),
                    message: format!("unknown sense value \"{other}\""),
                });
            }
        };

        let slack = SlackConfig {
            enabled: c.slack.enabled,
            penalty: c.slack.penalty,
        };

        result.push(GenericConstraint {
            id: EntityId::from(c.id),
            name: c.name,
            description: c.description,
            expression,
            sense,
            slack,
        });
    }

    // Sort by id for declaration-order invariance.
    result.sort_by_key(|gc| gc.id.0);

    Ok(result)
}

// ── Expression parser ─────────────────────────────────────────────────────────

/// Parse an expression string into a [`ConstraintExpression`].
///
/// Grammar (spec SS3):
/// ```text
/// expression ::= term (('+' | '-') term)*
/// term       ::= coefficient '*' variable | variable | number
/// variable   ::= var_name '(' entity_id (',' block_id)? ')'
/// ```
///
/// Returns `Err(String)` with a human-readable error message on parse failure.
/// The caller wraps this in `LoadError::SchemaError` with the appropriate
/// field path.
pub(crate) fn parse_expression(input: &str) -> Result<ConstraintExpression, String> {
    let tokens = tokenize(input)?;
    let terms = parse_terms(&tokens)?;
    Ok(ConstraintExpression { terms })
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Tokens produced by the expression tokenizer.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// `+` operator.
    Plus,
    /// `-` operator (also handles unary negation of a term coefficient).
    Minus,
    /// `*` operator (coefficient × variable).
    Star,
    /// `(` opening parenthesis.
    LParen,
    /// `)` closing parenthesis.
    RParen,
    /// `,` separator between `entity_id` and `block_id`.
    Comma,
    /// A non-negative floating-point or integer literal.
    Number(f64),
    /// An identifier: variable name.
    Ident(String),
}

/// Tokenize an expression string into a `Vec<Token>`.
///
/// Splits on whitespace and special characters (`+`, `-`, `*`, `(`, `)`, `,`).
/// Numbers and identifiers are recognized as multi-character tokens.
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
            c if c.is_ascii_digit() || c == '.' => {
                // Parse a number literal (integer or floating-point).
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
                // Parse an identifier.
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            other => {
                return Err(format!("unexpected character '{other}' at position {i}"));
            }
        }
    }

    Ok(tokens)
}

// ── Term parser ───────────────────────────────────────────────────────────────

/// Parse a token stream into a list of `LinearTerm` entries.
///
/// Grammar:
/// ```text
/// expression ::= term (('+' | '-') term)*
/// term       ::= coefficient '*' variable | variable
/// ```
fn parse_terms(tokens: &[Token]) -> Result<Vec<LinearTerm>, String> {
    if tokens.is_empty() {
        return Err("expression must not be empty".to_string());
    }

    let mut terms = Vec::new();
    let mut pos = 0;

    // Parse the sign and optional leading coefficient before the first term.
    // The first term may have a unary `-` or `+` prefix.
    let mut sign: f64 = 1.0;
    if pos < tokens.len() {
        match &tokens[pos] {
            Token::Plus => {
                pos += 1;
            }
            Token::Minus => {
                sign = -1.0;
                pos += 1;
            }
            _ => {}
        }
    }

    // Parse the first term.
    let (term, next_pos) = parse_single_term(tokens, pos, sign)?;
    terms.push(term);
    pos = next_pos;

    // Parse remaining `('+' | '-') term` pairs.
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

        let (term, next_pos) = parse_single_term(tokens, pos, op_sign)?;
        terms.push(term);
        pos = next_pos;
    }

    Ok(terms)
}

/// Parse one term starting at `tokens[pos]` with the given sign prefix.
///
/// A term is either:
/// - `coefficient '*' variable(...)` — explicit coefficient
/// - `variable(...)` — implicit coefficient 1.0 (multiplied by sign)
///
/// Returns the parsed `LinearTerm` and the new token position after the term.
fn parse_single_term(
    tokens: &[Token],
    pos: usize,
    sign: f64,
) -> Result<(LinearTerm, usize), String> {
    if pos >= tokens.len() {
        return Err(format!(
            "unexpected end of expression: expected a term at position {pos}"
        ));
    }

    match &tokens[pos] {
        Token::Number(coeff_val) => {
            // Coefficient followed by '*' and then a variable reference.
            let coefficient = coeff_val * sign;
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

            let var_pos = next + 1;
            if var_pos >= tokens.len() {
                return Err("expected variable name after '*', got end of expression".to_string());
            }

            let (variable, end_pos) = parse_variable_ref(tokens, var_pos)?;
            Ok((
                LinearTerm {
                    coefficient,
                    variable,
                },
                end_pos,
            ))
        }
        Token::Ident(_) => {
            // Variable reference with implicit coefficient 1.0 × sign.
            let coefficient = sign;
            let (variable, end_pos) = parse_variable_ref(tokens, pos)?;
            Ok((
                LinearTerm {
                    coefficient,
                    variable,
                },
                end_pos,
            ))
        }
        other => Err(format!(
            "expected a coefficient or variable name at position {pos}, got {other:?}"
        )),
    }
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

/// Parse a variable reference `var_name '(' entity_id (',' block_id)? ')'` from `tokens[pos]`.
///
/// Returns the [`VariableRef`] and the position of the next unconsumed token.
fn parse_variable_ref(tokens: &[Token], pos: usize) -> Result<(VariableRef, usize), String> {
    // Expect identifier.
    let var_name = match tokens.get(pos) {
        Some(Token::Ident(name)) => name.clone(),
        Some(other) => {
            return Err(format!(
                "expected variable name, got {other:?} at position {pos}"
            ));
        }
        None => return Err("expected variable name, got end of expression".to_string()),
    };

    // Expect '('.
    if tokens.get(pos + 1) != Some(&Token::LParen) {
        return Err(format!(
            "expected '(' after variable name \"{var_name}\" at position {}",
            pos + 1
        ));
    }

    // Expect entity_id (integer).
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

    // Peek at next token: either ',' (block_id follows) or ')' (no block).
    let mut cursor = pos + 3;
    let block_id: Option<usize> = match tokens.get(cursor) {
        Some(Token::Comma) => {
            cursor += 1;
            // Expect block_id.
            match tokens.get(cursor) {
                Some(Token::Number(b)) => {
                    let b_usize = token_f64_to_usize(*b).ok_or_else(|| {
                        format!(
                            "block_id must be a non-negative integer, got {b} in variable \"{var_name}\""
                        )
                    })?;
                    cursor += 1;
                    Some(b_usize)
                }
                Some(other) => {
                    return Err(format!(
                        "expected integer block_id after comma in variable \"{var_name}\", got {other:?} at position {cursor}"
                    ));
                }
                None => {
                    return Err(format!(
                        "unexpected end of expression: expected block_id in variable \"{var_name}\""
                    ));
                }
            }
        }
        Some(Token::RParen) => None,
        Some(other) => {
            return Err(format!(
                "expected ',' or ')' in variable \"{var_name}\" argument list, got {other:?} at position {cursor}"
            ));
        }
        None => {
            return Err(format!(
                "unexpected end of expression: expected ')' after entity_id in variable \"{var_name}\""
            ));
        }
    };

    // Expect ')'.
    if tokens.get(cursor) != Some(&Token::RParen) {
        return Err(format!(
            "expected ')' to close variable \"{var_name}\", got {:?} at position {cursor}",
            tokens.get(cursor)
        ));
    }
    cursor += 1;

    // Map variable name to VariableRef variant.
    let variable = build_variable_ref(&var_name, entity_id, block_id)?;

    Ok((variable, cursor))
}

/// Build a [`VariableRef`] from the parsed variable name, entity ID, and optional block ID.
///
/// Returns `Err(String)` if the variable name is not one of the 20 known names, or
/// if a block argument is provided for a stage-only variable (no block argument expected).
#[allow(clippy::too_many_lines)]
fn build_variable_ref(
    name: &str,
    entity_id: EntityId,
    block_id: Option<usize>,
) -> Result<VariableRef, String> {
    match name {
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
        "hydro_evaporation" => {
            if block_id.is_some() {
                return Err(format!(
                    "variable \"{name}\" does not accept a block argument"
                ));
            }
            Ok(VariableRef::HydroEvaporation {
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
        "hydro_turbined" => Ok(VariableRef::HydroTurbined {
            hydro_id: entity_id,
            block_id,
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
        other => Err(format!(
            "unknown variable name \"{other}\": not one of the 20 supported LP variable types"
        )),
    }
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
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(content.as_bytes()).expect("write");
        f
    }

    const VALID_JSON: &str = r#"{
  "constraints": [
    {
      "id": 1,
      "name": "min_hydro",
      "expression": "hydro_generation(10) + hydro_generation(11)",
      "sense": ">=",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "max_thermal",
      "expression": "2.5 * thermal_generation(5) - hydro_generation(3)",
      "sense": "<=",
      "slack": { "enabled": true, "penalty": 5000.0 }
    }
  ]
}"#;

    // ── Expression parser unit tests ──────────────────────────────────────────

    /// AC-1: Simple single-term expression with implicit coefficient.
    #[test]
    fn test_expr_simple_single_term() {
        let expr = parse_expression("hydro_generation(10)").unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert!((expr.terms[0].coefficient - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
            }
        );
    }

    /// Addition: two terms, both coefficient 1.0.
    #[test]
    fn test_expr_addition_two_terms() {
        let expr = parse_expression("hydro_generation(10) + hydro_generation(11)").unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((expr.terms[0].coefficient - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
            }
        );
        assert!((expr.terms[1].coefficient - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(11),
                block_id: None,
            }
        );
    }

    /// AC-2: Coefficient with `*` and subtraction (negation of second term).
    #[test]
    fn test_expr_coefficient_and_subtraction() {
        let expr = parse_expression("2.5 * thermal_generation(5) - hydro_generation(3)").unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((expr.terms[0].coefficient - 2.5).abs() < 1e-10);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
        assert!((expr.terms[1].coefficient - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(
            expr.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
            }
        );
    }

    /// Subtraction: second term has coefficient -1.0.
    #[test]
    fn test_expr_subtraction_negates_coefficient() {
        let expr = parse_expression("thermal_generation(5) - hydro_generation(3)").unwrap();
        assert_eq!(expr.terms.len(), 2);
        assert!((expr.terms[0].coefficient - 1.0).abs() < f64::EPSILON);
        assert!((expr.terms[1].coefficient - (-1.0)).abs() < f64::EPSILON);
    }

    /// Block-specific variable: `hydro_turbined(5, 0)` → `block_id: Some(0)`.
    #[test]
    fn test_expr_block_specific_variable() {
        let expr = parse_expression("hydro_turbined(5, 0)").unwrap();
        assert_eq!(expr.terms.len(), 1);
        assert_eq!(
            expr.terms[0].variable,
            VariableRef::HydroTurbined {
                hydro_id: EntityId(5),
                block_id: Some(0),
            }
        );
    }

    /// Block-specific line_exchange: `line_exchange(0, 1)` → `block_id: Some(1)`.
    #[test]
    fn test_expr_line_exchange_with_block() {
        let expr = parse_expression("line_exchange(0, 1)").unwrap();
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
        let expr = parse_expression("hydro_storage(7)").unwrap();
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
        let err = parse_expression("hydro_storage(7, 0)").unwrap_err();
        assert!(
            err.contains("does not accept a block argument"),
            "expected block argument error, got: {err}"
        );
    }

    /// AC-3: Unknown variable name → error.
    #[test]
    fn test_expr_unknown_variable_name() {
        let err = parse_expression("invalid_var(0)").unwrap_err();
        assert!(
            err.contains("unknown variable name"),
            "expected unknown variable error, got: {err}"
        );
    }

    /// Missing closing parenthesis → error.
    #[test]
    fn test_expr_missing_closing_paren() {
        let err = parse_expression("hydro_generation(10").unwrap_err();
        assert!(
            err.contains("expected ')'") || err.contains("unexpected end"),
            "expected paren error, got: {err}"
        );
    }

    /// Empty expression → error.
    #[test]
    fn test_expr_empty_is_error() {
        let err = parse_expression("").unwrap_err();
        assert!(
            err.contains("empty"),
            "expected empty expression error, got: {err}"
        );
    }

    /// All 19 variable types are recognised.
    #[test]
    fn test_expr_all_19_variable_types_recognised() {
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
                },
            ),
            (
                "hydro_evaporation(0)",
                VariableRef::HydroEvaporation {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "hydro_withdrawal(0)",
                VariableRef::HydroWithdrawal {
                    hydro_id: EntityId(0),
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

        assert_eq!(cases.len(), 20, "must have exactly 20 variable types");

        for (input, expected) in cases {
            let expr = parse_expression(input)
                .unwrap_or_else(|e| panic!("parse failed for \"{input}\": {e}"));
            assert_eq!(expr.terms.len(), 1, "single term for \"{input}\"");
            assert_eq!(
                &expr.terms[0].variable, expected,
                "wrong VariableRef for \"{input}\""
            );
        }
    }

    // ── parse_generic_constraints integration tests ───────────────────────────

    /// Valid 2-constraint file. First has 2 hydro_generation terms.
    #[test]
    fn test_parse_valid_two_constraints() {
        let f = write_json(VALID_JSON);
        let result = parse_generic_constraints(f.path()).unwrap();

        // Should be 2 constraints, sorted by id ascending: 0, 1.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, EntityId(0)); // id=0 was second in JSON
        assert_eq!(result[1].id, EntityId(1)); // id=1 was first in JSON

        // The first constraint (id=1, "min_hydro") has 2 hydro_generation terms.
        // After sorting, result[1] is the "min_hydro" constraint.
        let min_hydro = &result[1];
        assert_eq!(min_hydro.expression.terms.len(), 2);
        assert!((min_hydro.expression.terms[0].coefficient - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            min_hydro.expression.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
            }
        );
        assert_eq!(
            min_hydro.expression.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(11),
                block_id: None,
            }
        );
        assert_eq!(min_hydro.sense, ConstraintSense::GreaterEqual);
    }

    /// Expression `"2.5 * thermal_generation(5) - hydro_generation(3)"`.
    #[test]
    fn test_parse_coefficient_and_subtraction_expression() {
        let f = write_json(VALID_JSON);
        let result = parse_generic_constraints(f.path()).unwrap();

        // result[0] is id=0 "max_thermal"
        let max_thermal = &result[0];
        assert_eq!(max_thermal.expression.terms.len(), 2);
        assert!((max_thermal.expression.terms[0].coefficient - 2.5).abs() < 1e-10);
        assert_eq!(
            max_thermal.expression.terms[0].variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            }
        );
        assert!((max_thermal.expression.terms[1].coefficient - (-1.0)).abs() < f64::EPSILON);
        assert_eq!(
            max_thermal.expression.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(3),
                block_id: None,
            }
        );
        assert_eq!(max_thermal.sense, ConstraintSense::LessEqual);
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
      "sense": ">=",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err = parse_generic_constraints(f.path()).unwrap_err();
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

    /// Duplicate IDs → SchemaError.
    #[test]
    fn test_parse_duplicate_ids_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "sense": ">=",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "b",
      "expression": "thermal_generation(1)",
      "sense": "<=",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err = parse_generic_constraints(f.path()).unwrap_err();
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
      "sense": ">=",
      "slack": { "enabled": true }
    }
  ]
}"#;
        let f = write_json(json);
        let err = parse_generic_constraints(f.path()).unwrap_err();
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

    /// Invalid sense string → SchemaError.
    #[test]
    fn test_parse_invalid_sense_returns_schema_error() {
        let json = r#"{
  "constraints": [
    {
      "id": 0,
      "name": "a",
      "expression": "hydro_generation(0)",
      "sense": "!=",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let err = parse_generic_constraints(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("sense"),
                    "field should contain 'sense', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `None` path → `Ok(Vec::new())` (tested via `load_generic_constraints`).
    /// The `load_*` wrapper is in `mod.rs`; here we test the `parse_*` function returns Ok
    /// for a valid empty constraints array.
    #[test]
    fn test_parse_empty_constraints_array() {
        let json = r#"{ "constraints": [] }"#;
        let f = write_json(json);
        let result = parse_generic_constraints(f.path()).unwrap();
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
      "sense": ">=",
      "slack": { "enabled": false }
    },
    {
      "id": 2,
      "name": "b",
      "expression": "thermal_generation(0)",
      "sense": "<=",
      "slack": { "enabled": false }
    },
    {
      "id": 0,
      "name": "a",
      "expression": "line_direct(0)",
      "sense": "==",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result = parse_generic_constraints(f.path()).unwrap();
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
      "sense": "==",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result = parse_generic_constraints(f.path()).unwrap();
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
      "sense": ">=",
      "slack": { "enabled": true, "penalty": 0.0 }
    }
  ]
}"#;
        let f = write_json(json);
        let err = parse_generic_constraints(f.path()).unwrap_err();
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
      "sense": "==",
      "slack": { "enabled": false }
    }
  ]
}"#;
        let f = write_json(json);
        let result = parse_generic_constraints(f.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].description.is_none());
    }
}
