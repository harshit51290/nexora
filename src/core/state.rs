//! Persisted state machines (docs/04-DATA-MODEL.md §4.2–4.3).
//!
//! Every transition must be persisted to SQLite by the owning manager;
//! `ERROR` states must always carry a [`crate::core::NexoraError`] code +
//! human recommendation (AGENTS.md rule 7).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Model lifecycle:
/// `DISCOVERED -> ANALYZING -> SUPPORTED -> DOWNLOADING -> INSTALLED
///  -> VALIDATING -> READY -> LOADED -> RUNNING`, failure -> `ERROR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelState {
    Discovered,
    Analyzing,
    Supported,
    Downloading,
    Installed,
    Validating,
    Ready,
    Loaded,
    Running,
    Error,
}

impl ModelState {
    /// Machine-readable status string persisted in `models.status`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "DISCOVERED",
            Self::Analyzing => "ANALYZING",
            Self::Supported => "SUPPORTED",
            Self::Downloading => "DOWNLOADING",
            Self::Installed => "INSTALLED",
            Self::Validating => "VALIDATING",
            Self::Ready => "READY",
            Self::Loaded => "LOADED",
            Self::Running => "RUNNING",
            Self::Error => "ERROR",
        }
    }

    /// Legal outgoing transitions (ERROR reachable from any non-terminal).
    pub fn can_transition(self, next: Self) -> bool {
        use ModelState as M;
        matches!(
            (self, next),
            (M::Discovered, M::Analyzing | M::Error)
                | (M::Analyzing, M::Supported | M::Error)
                | (M::Supported, M::Downloading | M::Error)
                | (M::Downloading, M::Installed | M::Error)
                | (M::Installed, M::Validating | M::Error)
                | (M::Validating, M::Ready | M::Error)
                | (M::Ready, M::Loaded | M::Error)
                | (M::Loaded, M::Running | M::Ready | M::Error)
                | (M::Running, M::Loaded | M::Ready | M::Error)
                | (M::Error, M::Discovered | M::Analyzing | M::Supported)
        )
    }

    pub fn is_terminal_active(self) -> bool {
        matches!(self, Self::Running | Self::Loaded)
    }
}

impl fmt::Display for ModelState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ModelState {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "DISCOVERED" => Ok(Self::Discovered),
            "ANALYZING" => Ok(Self::Analyzing),
            "SUPPORTED" => Ok(Self::Supported),
            "DOWNLOADING" => Ok(Self::Downloading),
            "INSTALLED" => Ok(Self::Installed),
            "VALIDATING" => Ok(Self::Validating),
            "READY" => Ok(Self::Ready),
            "LOADED" => Ok(Self::Loaded),
            "RUNNING" => Ok(Self::Running),
            "ERROR" => Ok(Self::Error),
            other => Err(format!("unknown ModelState: {other}")),
        }
    }
}

/// Runtime lifecycle (supervised child process; crash -> ERROR, app survives):
/// `NOT_INSTALLED -> INSTALLING -> READY -> STARTING -> RUNNING
///  -> STOPPING -> STOPPED`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RuntimeState {
    NotInstalled,
    Installing,
    Ready,
    Starting,
    Running,
    Stopping,
    Stopped,
    Error,
}

impl RuntimeState {
    /// Machine-readable status string persisted in `runtimes.status`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotInstalled => "NOT_INSTALLED",
            Self::Installing => "INSTALLING",
            Self::Ready => "READY",
            Self::Starting => "STARTING",
            Self::Running => "RUNNING",
            Self::Stopping => "STOPPING",
            Self::Stopped => "STOPPED",
            Self::Error => "ERROR",
        }
    }

    pub fn can_transition(self, next: Self) -> bool {
        use RuntimeState as R;
        matches!(
            (self, next),
            (R::NotInstalled, R::Installing | R::Error)
                | (R::Installing, R::Ready | R::Error)
                | (R::Ready, R::Starting | R::Error)
                | (R::Starting, R::Running | R::Error)
                | (R::Running, R::Stopping | R::Error)
                | (R::Stopping, R::Stopped | R::Error)
                | (R::Stopped, R::Starting | R::Error)
                | (R::Error, R::Installing | R::NotInstalled)
        )
    }
}

impl fmt::Display for RuntimeState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for RuntimeState {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "NOT_INSTALLED" => Ok(Self::NotInstalled),
            "INSTALLING" => Ok(Self::Installing),
            "READY" => Ok(Self::Ready),
            "STARTING" => Ok(Self::Starting),
            "RUNNING" => Ok(Self::Running),
            "STOPPING" => Ok(Self::Stopping),
            "STOPPED" => Ok(Self::Stopped),
            "ERROR" => Ok(Self::Error),
            other => Err(format!("unknown RuntimeState: {other}")),
        }
    }
}
