//! Never blind-exec repo code. `trust_remote_code=True` (or any custom
//! `.py`) triggers the modal: `[View Files] [Run in Sandbox] [Cancel]`
//! (docs/09 §9.4). Override is allowed but never silent.

use crate::core::{NexoraError, Result};
use serde::{Deserialize, Serialize};

/// Trusted (official/compatible) | Community | Unverified (odd code) |
/// Blocked (dangerous/incompatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustLevel {
    Trusted,
    Community,
    Unverified,
    Blocked,
}

impl TrustLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trusted => "trusted",
            Self::Community => "community",
            Self::Unverified => "unverified",
            Self::Blocked => "blocked",
        }
    }
}

/// The user's explicit modal choice. Only `RunInSandbox`/`Cancel` resolve
/// the gate; `ViewFiles` keeps it pending (caller re-prompts afterwards).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserDecision {
    ViewFiles,
    RunInSandbox,
    Cancel,
}

/// Heuristic over publisher + file signals. Conservative by design:
/// unknown publishers with executable code are at best Unverified.
pub fn classify_trust(
    publisher_official: bool,
    has_custom_py: bool,
    has_suspicious_patterns: bool,
) -> TrustLevel {
    if has_suspicious_patterns {
        return TrustLevel::Blocked;
    }
    if has_custom_py {
        return if publisher_official {
            TrustLevel::Community
        } else {
            TrustLevel::Unverified
        };
    }
    if publisher_official {
        TrustLevel::Trusted
    } else {
        TrustLevel::Community
    }
}

/// Enforce the custom-code consent gate. Returns `Ok` only on explicit
/// `RunInSandbox` consent; `Blocked` repos always fail, even with consent.
pub fn gate_custom_code(
    repo: &str,
    trust: TrustLevel,
    has_custom_py: bool,
    trust_remote_code: bool,
    decision: Option<UserDecision>,
) -> Result<()> {
    if trust == TrustLevel::Blocked {
        return Err(NexoraError::CustomCode {
            repo: repo.into(),
            detail: "blocked: dangerous or incompatible content".into(),
        });
    }
    if !(has_custom_py || trust_remote_code) {
        return Ok(());
    }
    match decision {
        Some(UserDecision::RunInSandbox) => Ok(()),
        Some(UserDecision::Cancel) | None => Err(NexoraError::CustomCode {
            repo: repo.into(),
            detail: "custom repo code requires explicit [Run in Sandbox] consent".into(),
        }),
        // View Files = inspect first, gate stays closed until re-confirmed.
        Some(UserDecision::ViewFiles) => Err(NexoraError::CustomCode {
            repo: repo.into(),
            detail: "review requested — confirm [Run in Sandbox] or [Cancel] afterwards".into(),
        }),
    }
}
