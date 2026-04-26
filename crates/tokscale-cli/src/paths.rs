//! Cross-platform resolution for tokscale's user config directory.
//!
//! macOS users following the docs expect `~/.config/tokscale/` because that
//! is what `auth.rs`, `cursor.rs`, and `antigravity.rs` already write to.
//! `dirs::config_dir()` would instead return `~/Library/Application Support/`
//! on macOS, splitting state across two roots and silently ignoring
//! settings.json edits the user made via the documented path. This module
//! enforces the unified `~/.config/tokscale/` location on macOS + Linux,
//! while keeping the platform default on Windows.

use std::path::PathBuf;

/// Resolve the tokscale config dir, honoring `TOKSCALE_CONFIG_DIR` first.
///
/// Resolution order:
/// 1. `TOKSCALE_CONFIG_DIR` (absolute path, taken verbatim).
/// 2. macOS / Linux: `$HOME/.config/tokscale`.
/// 3. Windows (and any other platform): `dirs::config_dir().join("tokscale")`.
/// 4. Last-ditch fallback: `./.tokscale` so a missing HOME never panics.
pub fn get_config_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("TOKSCALE_CONFIG_DIR") {
        return PathBuf::from(custom);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        if let Some(home) = dirs::home_dir() {
            return home.join(".config").join("tokscale");
        }
    }

    dirs::config_dir()
        .map(|d| d.join("tokscale"))
        .unwrap_or_else(|| PathBuf::from(".tokscale"))
}

/// Whether `TOKSCALE_CONFIG_DIR` is explicitly set in the environment.
///
/// Callers that want to read a legacy on-disk location during the macOS
/// transition MUST gate that fallback on this returning `false`. When the
/// override is set (CI sandbox, tests, isolated profile), the user has
/// asked for an explicit, hermetic config root — silently ingesting
/// `~/Library/Application Support/tokscale/` defeats that contract.
pub fn is_config_dir_overridden() -> bool {
    std::env::var_os("TOKSCALE_CONFIG_DIR").is_some_and(|v| !v.is_empty())
}

/// Legacy macOS config dir (`~/Library/Application Support/tokscale`).
///
/// Returns `None` off macOS, when HOME cannot be resolved, or when
/// `TOKSCALE_CONFIG_DIR` is set (so the env override stays hermetic).
/// Used by `Settings::load()` and `load_star_cache()` so users upgrading
/// from a release that wrote files under `~/Library/Application Support/`
/// keep their preferences on first launch after upgrade.
#[cfg(target_os = "macos")]
pub fn legacy_macos_config_dir() -> Option<PathBuf> {
    if is_config_dir_overridden() {
        return None;
    }
    dirs::config_dir().map(|d| d.join("tokscale"))
}

#[cfg(not(target_os = "macos"))]
pub fn legacy_macos_config_dir() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::env;

    #[test]
    #[serial]
    fn env_override_is_returned_verbatim() {
        let prev = env::var_os("TOKSCALE_CONFIG_DIR");
        unsafe {
            env::set_var("TOKSCALE_CONFIG_DIR", "/tmp/tokscale-custom");
        }
        assert_eq!(get_config_dir(), PathBuf::from("/tmp/tokscale-custom"));
        unsafe {
            match prev {
                Some(v) => env::set_var("TOKSCALE_CONFIG_DIR", v),
                None => env::remove_var("TOKSCALE_CONFIG_DIR"),
            }
        }
    }

    #[test]
    #[serial]
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn unix_default_is_dot_config_tokscale_under_home() {
        let prev_override = env::var_os("TOKSCALE_CONFIG_DIR");
        let prev_home = env::var_os("HOME");
        unsafe {
            env::remove_var("TOKSCALE_CONFIG_DIR");
            env::set_var("HOME", "/tmp/tokscale-home-test");
        }
        assert_eq!(
            get_config_dir(),
            PathBuf::from("/tmp/tokscale-home-test/.config/tokscale"),
        );
        unsafe {
            match prev_override {
                Some(v) => env::set_var("TOKSCALE_CONFIG_DIR", v),
                None => env::remove_var("TOKSCALE_CONFIG_DIR"),
            }
            match prev_home {
                Some(v) => env::set_var("HOME", v),
                None => env::remove_var("HOME"),
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn legacy_helper_returns_none_off_macos() {
        assert!(legacy_macos_config_dir().is_none());
    }
}
