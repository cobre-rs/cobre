//! Terminal banner for the `cobre` CLI.
//!
//! Renders the three-line copper busbar banner with the Cobre brand identity.
//! Color output is gated on `NO_COLOR` and terminal color support, following
//! the [no-color.org](https://no-color.org) convention.
//!
//! # Example
//!
//! ```rust,no_run
//! use console::Term;
//! use cobre::banner;
//!
//! banner::print_banner(&Term::buffered_stderr());
//! ```

use console::Term;

/// Render the three-line banner as a `String`.
///
/// When `use_color` is `true`, the returned string contains ANSI 256-color
/// escape sequences. When `false`, the same Unicode characters are returned
/// without any escape codes.
///
/// This function is intentionally side-effect-free so that the banner content
/// can be verified in unit tests without requiring a real terminal.
pub(crate) fn render_banner_string(use_color: bool) -> String {
    let version = env!("CARGO_PKG_VERSION");

    if use_color {
        let bar = "\x1b[38;5;172m\u{257a}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{257b}\x1b[0m";
        let dot = "\x1b[38;5;179m\u{25cf}\x1b[0m";
        let spark = "\x1b[38;5;214m\u{26a1}\x1b[0m";
        let cobre = format!("\x1b[1;38;5;253mCOBRE v{version}\x1b[0m");
        let tagline = "\x1b[38;5;245mPower systems in Rust\x1b[0m";
        format!(" {bar}{dot}\n {bar}{dot}{spark}  {cobre}\n {bar}{dot}   {tagline}\n")
    } else {
        let bar = "\u{257a}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{257b}";
        let dot = "\u{25cf}";
        let spark = "\u{26a1}";
        format!(
            " {bar}{dot}\n {bar}{dot}{spark}  COBRE v{version}\n {bar}{dot}   Power systems in Rust\n"
        )
    }
}

/// Write the three-line Cobre banner followed by an empty line to `stderr`.
///
/// Color is enabled when [`console::colors_enabled_stderr`] returns `true`
/// and the `NO_COLOR` environment variable is absent. This follows the
/// [no-color.org](https://no-color.org) convention.
///
/// Write errors are silently ignored — banner rendering is fire-and-forget.
/// The caller is responsible for the display conditions (`--quiet`,
/// `--no-banner`, terminal detection).
pub fn print_banner(stderr: &Term) {
    let use_color = console::colors_enabled_stderr() && std::env::var_os("NO_COLOR").is_none();
    let banner = render_banner_string(use_color);
    for line in banner.lines() {
        let _ = stderr.write_line(line);
    }
    let _ = stderr.write_line("");
}

#[cfg(test)]
mod tests {
    use console::Term;

    use super::{print_banner, render_banner_string};

    #[test]
    fn test_render_banner_colored_contains_ansi_escapes() {
        let banner = render_banner_string(true);
        assert!(banner.contains("\x1b[38;5;172m"));
        assert!(banner.contains("\x1b[0m"));
    }

    #[test]
    fn test_render_banner_plain_no_ansi_escapes() {
        let banner = render_banner_string(false);
        assert!(!banner.contains("\x1b["));
    }

    #[test]
    fn test_render_banner_contains_version() {
        let banner = render_banner_string(false);
        let idx = banner
            .find("COBRE v")
            .expect("banner must contain 'COBRE v'");
        let after = &banner[idx + "COBRE v".len()..];
        assert!(after.chars().next().is_some_and(|c| c.is_ascii_digit()));
    }

    #[test]
    fn test_render_banner_contains_unicode_busbars() {
        let banner = render_banner_string(false);
        assert!(banner.contains('\u{2501}'));
        assert!(banner.contains('\u{25cf}'));
    }

    #[test]
    fn test_print_banner_does_not_panic() {
        // Smoke test: calling print_banner with a buffered stderr must not panic.
        print_banner(&Term::buffered_stderr());
    }
}
