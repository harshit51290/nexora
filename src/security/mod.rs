//! Trust levels + `trust_remote_code` gate (docs/09 §9.4, AGENTS.md rule 5).

pub mod pickle;
pub mod trust;

pub use trust::{classify_trust, gate_custom_code, TrustLevel, UserDecision};
