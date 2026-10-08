//! Third-party runtime plugin system (docs/06 §6.9).
//!
//! Layout: `plugins/<name>/manifest.json` + entrypoint. Manifest shape:
//! `{name, version, supported_models[], entrypoint}`. Third-party adapters
//! (whisper/kokoro/ollama/custom) extend the registry without core changes.
//!
//! Orchestration-only (AGENTS.md rules 1, 5): plugins execute as sandboxed
//! child processes ([`spawn_plugin`]) — never linked in, never run in-process.
//! A plugin runs only after its manifest validates AND the user consents for
//! its trust level (see `src/security/trust.rs`).
//!
//! SCOPE NOTE: this file is intentionally **not wired into `src/lib.rs` yet**
//! (editing `lib.rs` / the runtime registry is out of scope for this change).
//! Wiring needs: `pub mod plugins;` in `src/lib.rs`, plus a lookup hook in
//! `src/runtime` (`adapter_for`) that resolves unknown ids through
//! [`PluginRegistry`] before returning `None`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// API version this host speaks. A plugin whose manifest declares a newer
/// `api_version` is rejected rather than run against a contract it was not
/// built for (docs/06 §6.9: versioned API).
pub const PLUGIN_API_VERSION: u32 = 1;

/// Validated `plugins/<name>/manifest.json`
/// (`{name, version, supported_models[], entrypoint}`, docs/06 §6.9).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub supported_models: Vec<String>,
    pub entrypoint: String,
    /// Plugin API contract the entrypoint was built against.
    /// Defaults to 1 when omitted by older manifests.
    #[serde(default = "default_api_version")]
    pub api_version: u32,
}

fn default_api_version() -> u32 {
    1
}

/// Coded plugin failure (AGENTS.md rule 7: code + human fix, no bare traces).
#[derive(Debug, Clone)]
pub struct PluginError {
    pub code: &'static str,
    pub message: String,
    pub hint: String,
}

impl PluginError {
    pub fn new(code: &'static str, message: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            hint: hint.into(),
        }
    }
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} Fix: {}", self.code, self.message, self.hint)
    }
}

impl std::error::Error for PluginError {}

/// Read + validate `manifest.json` inside `plugin_dir`
/// (`plugins/<name>/manifest.json`).
pub fn load_manifest(plugin_dir: &Path) -> Result<PluginManifest, PluginError> {
    let path = plugin_dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        PluginError::new(
            "E-PLUGIN-NO-MANIFEST",
            format!("cannot read {}: {e}", path.display()),
            "Reinstall the plugin — its folder must contain manifest.json \
             (docs/06 §6.9). If you removed it by hand, delete the whole plugin folder.",
        )
    })?;
    let manifest: PluginManifest = serde_json::from_str(&text).map_err(|e| {
        PluginError::new(
            "E-PLUGIN-BAD-MANIFEST",
            format!("{} is not valid JSON: {e}", path.display()),
            "The manifest must be {name, version, supported_models[], entrypoint}. \
             Compare it against plugins/example-tts/manifest.json.",
        )
    })?;
    validate_manifest(&manifest, plugin_dir)?;
    Ok(manifest)
}

/// Validate manifest fields against the plugin dir on disk:
/// non-empty slug name, semver version, supported API contract, and an
/// entrypoint that exists *inside* the plugin dir (absolute paths and `..`
/// escapes are rejected — the child process is jailed to this dir).
pub fn validate_manifest(manifest: &PluginManifest, plugin_dir: &Path) -> Result<(), PluginError> {
    if manifest.name.trim().is_empty()
        || !manifest
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(PluginError::new(
            "E-PLUGIN-BAD-MANIFEST",
            format!("plugin name '{}' must be a non-empty slug [a-z0-9-_]", manifest.name),
            "Rename the plugin key to lowercase letters, digits, dashes or underscores.",
        ));
    }
    if semver::Version::parse(manifest.version.trim()).is_err() {
        return Err(PluginError::new(
            "E-PLUGIN-BAD-VERSION",
            format!("plugin '{}' version '{}' is not semver", manifest.name, manifest.version),
            "Use MAJOR.MINOR.PATCH (e.g. \"0.3.1\"). Prereleases like \"1.0.0-beta.1\" are accepted.",
        ));
    }
    if manifest.api_version > PLUGIN_API_VERSION {
        return Err(PluginError::new(
            "E-PLUGIN-API-MISMATCH",
            format!(
                "plugin '{}' needs plugin API v{}, this host speaks v{}",
                manifest.name, manifest.api_version, PLUGIN_API_VERSION
            ),
            "Update Nexora, or install an older release of the plugin built for this API version.",
        ));
    }
    let entry = Path::new(&manifest.entrypoint);
    if entry.is_absolute() || entry.components().any(|c| c == std::path::Component::ParentDir) {
        return Err(PluginError::new(
            "E-PLUGIN-BAD-ENTRYPOINT",
            format!("plugin '{}' entrypoint escapes its folder: '{}'", manifest.name, manifest.entrypoint),
            "The entrypoint must be a relative path inside the plugin folder (e.g. \"serve.py\").",
        ));
    }
    if !plugin_dir.join(entry).is_file() {
        return Err(PluginError::new(
            "E-PLUGIN-NO-ENTRYPOINT",
            format!(
                "plugin '{}' entrypoint not found: {}",
                manifest.name,
                plugin_dir.join(entry).display()
            ),
            "Reinstall the plugin — its entrypoint script is missing from the plugin folder.",
        ));
    }
    Ok(())
}

/// In-memory registry: the extension point through which third-party adapters
/// join without core changes (docs/06 §6.9). Persisted rows live in the
/// `plugins` table — see [`register_row`] / [`load_registry`].
#[derive(Debug, Default)]
pub struct PluginRegistry {
    entries: HashMap<String, (PathBuf, PluginManifest)>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a validated manifest + its folder. Re-registering the same
    /// name replaces the entry (upgrade path).
    pub fn register(&mut self, plugin_dir: PathBuf, manifest: PluginManifest) {
        self.entries.insert(manifest.name.clone(), (plugin_dir, manifest));
    }

    pub fn unregister(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }

    pub fn get(&self, name: &str) -> Option<&PluginManifest> {
        self.entries.get(name).map(|(_, m)| m)
    }

    pub fn plugin_dir(&self, name: &str) -> Option<&Path> {
        self.entries.get(name).map(|(d, _)| d.as_path())
    }

    /// All registered plugin names (registry keys for the runtime lookup).
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.entries.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Find plugins claiming a model (`supported_models` holds repo ids and/or
    /// `"*"` wildcard). Empty claim list = claims nothing.
    pub fn find_for_model(&self, repository: &str) -> Vec<&PluginManifest> {
        let mut hits: Vec<&PluginManifest> = self
            .entries
            .values()
            .map(|(_, m)| m)
            .filter(|m| {
                m.supported_models
                    .iter()
                    .any(|s| s == "*" || s.eq_ignore_ascii_case(repository))
            })
            .collect();
        hits.sort_by(|a, b| a.name.cmp(&b.name));
        hits
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Output captured from one sandboxed plugin invocation.
#[derive(Debug, Clone)]
pub struct PluginOut {
    /// Raw stdout bytes (protocol payloads are JSON on top of this).
    pub stdout: Vec<u8>,
    /// Last bytes of stderr (for crash translation, not for display raw).
    pub stderr_tail: String,
}

/// Run a plugin entrypoint as a sandboxed child process.
///
/// Sandbox contract (AGENTS.md rule 5):
/// * no shell — `entrypoint` is spawned directly with `args` (no `cmd /c`,
///   no `sh -c`), so plugin input can never inject shell syntax;
/// * jailed cwd — the child runs with `plugin_dir` as its working directory
///   and receives entrypoint/material paths only inside it;
/// * piped stdio — `payload_json` goes over stdin; stdout/stderr are captured,
///   never inherited, so a plugin cannot scribble on the host terminal;
/// * tagged env — the child gets `NEXORA_PLUGIN_API=1`; secrets from the host
///   process env are NOT forwarded (pass only what the payload carries);
/// * crash isolation — non-zero exits map to `E-PLUGIN-CRASHED`; the host
///   stays alive (same supervision rule as runtime adapters, docs/04 §4.3).
pub fn spawn_plugin(
    plugin_dir: &Path,
    manifest: &PluginManifest,
    args: &[String],
    payload_json: &str,
) -> Result<PluginOut, PluginError> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    validate_manifest(manifest, plugin_dir)?;
    let entry = plugin_dir.join(&manifest.entrypoint);
    let mut child = Command::new(&entry)
        .args(args)
        .current_dir(plugin_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("NEXORA_PLUGIN_API", PLUGIN_API_VERSION.to_string())
        .spawn()
        .map_err(|e| {
            PluginError::new(
                "E-PLUGIN-SPAWN-FAILED",
                format!("could not launch plugin '{}': {e}", manifest.name),
                "Reinstall the plugin and check antivirus quarantine — fresh interpreter \
                 shims are sometimes flagged. The host itself is unaffected.",
            )
        })?;
    if let Some(mut stdin) = child.stdin.take() {
        // A broken pipe here just means the plugin exited without reading.
        let _ = stdin.write_all(payload_json.as_bytes());
    }
    let output = child.wait_with_output().map_err(|e| {
        PluginError::new(
            "E-PLUGIN-CRASHED",
            format!("plugin '{}' I/O error while waiting: {e}", manifest.name),
            "Retry once; if it repeats, reinstall the plugin or report it to the plugin author.",
        )
    })?;
    if !output.status.success() {
        let tail: String = String::from_utf8_lossy(&output.stderr)
            .chars()
            .rev()
            .take(800)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(PluginError::new(
            "E-PLUGIN-CRASHED",
            format!("plugin '{}' exited with {}{}", manifest.name, output.status, {
                let t = tail.trim();
                if t.is_empty() { String::new() } else { format!(" — stderr: {t}") }
            }),
            "The plugin crashed but Nexora is still running. Check Logs, then reinstall \
             the plugin or try a built-in runtime instead.",
        ));
    }
    let stderr_tail: String = String::from_utf8_lossy(&output.stderr)
        .chars()
        .rev()
        .take(800)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    Ok(PluginOut {
        stdout: output.stdout,
        stderr_tail,
    })
}

/// Persist one registry entry to the `plugins` table (upsert by name).
/// Same pool-passing pattern as [`crate::model::ModelManager`].
pub async fn register_row(
    pool: &sqlx::SqlitePool,
    manifest: &PluginManifest,
) -> Result<(), PluginError> {
    sqlx::query(
        "INSERT INTO plugins(name, version, entrypoint, supported_models) VALUES(?,?,?,?) \
         ON CONFLICT(name) DO UPDATE SET version=excluded.version, entrypoint=excluded.entrypoint, supported_models=excluded.supported_models",
    )
    .bind(&manifest.name)
    .bind(&manifest.version)
    .bind(&manifest.entrypoint)
    .bind(serde_json::to_string(&manifest.supported_models).unwrap_or_else(|_| "[]".into()))
    .execute(pool)
    .await
    .map_err(|e| {
        PluginError::new(
            "E-PLUGIN-DB",
            format!("cannot persist plugin '{}': {e}", manifest.name),
            "Restart Nexora; if it persists, restore settings.db from backup.",
        )
    })?;
    Ok(())
}

/// Remove one registry row from the `plugins` table.
pub async fn unregister_row(pool: &sqlx::SqlitePool, name: &str) -> Result<(), PluginError> {
    sqlx::query("DELETE FROM plugins WHERE name = ?")
        .bind(name)
        .execute(pool)
        .await
        .map_err(|e| {
            PluginError::new(
                "E-PLUGIN-DB",
                format!("cannot remove plugin '{name}': {e}"),
                "Restart Nexora; if it persists, restore settings.db from backup.",
            )
        })?;
    Ok(())
}

/// Load persisted plugin rows back into a registry. `plugins_base_dir` is the
/// storage `plugins/` dir, so each row resolves to `<base>/<name>/`.
pub async fn load_registry(
    pool: &sqlx::SqlitePool,
    plugins_base_dir: &Path,
) -> Result<PluginRegistry, PluginError> {
    use sqlx::Row;
    let rows = sqlx::query("SELECT name, version, entrypoint, supported_models FROM plugins")
        .fetch_all(pool)
        .await
        .map_err(|e| {
            PluginError::new(
                "E-PLUGIN-DB",
                format!("cannot load plugin registry: {e}"),
                "Restart Nexora; if it persists, restore settings.db from backup.",
            )
        })?;
    let mut registry = PluginRegistry::new();
    for row in rows {
        let name: String = row.get("name");
        let supported: String = row.get("supported_models");
        registry.register(
            plugins_base_dir.join(&name),
            PluginManifest {
                name,
                version: row.get("version"),
                supported_models: serde_json::from_str(&supported).unwrap_or_default(),
                entrypoint: row.get("entrypoint"),
                api_version: PLUGIN_API_VERSION,
            },
        );
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("plugins/example-tts")
    }

    #[test]
    fn example_manifest_loads() {
        let m = load_manifest(&example_dir()).expect("example manifest must validate");
        assert_eq!(m.name, "example-tts");
        assert!(semver::Version::parse(&m.version).is_ok());
        assert!(!m.supported_models.is_empty());
    }

    #[test]
    fn bad_version_and_escape_rejected() {
        let dir = example_dir();
        let mut m = load_manifest(&dir).unwrap();
        m.version = "not-a-version".into();
        assert!(validate_manifest(&m, &dir).is_err());
        m.version = "0.1.0".into();
        m.entrypoint = "../evil.py".into();
        let err = validate_manifest(&m, &dir).unwrap_err();
        assert_eq!(err.code, "E-PLUGIN-BAD-ENTRYPOINT");
        m.entrypoint = "missing.py".into();
        let err = validate_manifest(&m, &dir).unwrap_err();
        assert_eq!(err.code, "E-PLUGIN-NO-ENTRYPOINT");
    }

    #[test]
    fn registry_find_for_model() {
        let mut reg = PluginRegistry::new();
        reg.register(
            PathBuf::from("/x/a"),
            PluginManifest {
                name: "a".into(),
                version: "1.0.0".into(),
                supported_models: vec!["owner/model".into()],
                entrypoint: "serve.py".into(),
                api_version: 1,
            },
        );
        reg.register(
            PathBuf::from("/x/b"),
            PluginManifest {
                name: "b".into(),
                version: "1.0.0".into(),
                supported_models: vec!["*".into()],
                entrypoint: "serve.py".into(),
                api_version: 1,
            },
        );
        assert_eq!(reg.names(), vec!["a", "b"]);
        assert_eq!(reg.find_for_model("owner/model").len(), 2);
        assert_eq!(reg.find_for_model("other/x").len(), 1);
        assert!(reg.unregister("a"));
        assert_eq!(reg.len(), 1);
    }

    #[tokio::test]
    async fn registry_rows_round_trip() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        let schema = include_str!("../migrations/001_init.sql");
        for stmt in schema.split(';') {
            let stmt = stmt.trim();
            if !stmt.is_empty() {
                sqlx::query(stmt).execute(&pool).await.unwrap();
            }
        }
        let m = load_manifest(&example_dir()).unwrap();
        register_row(&pool, &m).await.unwrap();
        let reg = load_registry(&pool, Path::new("/plugins")).await.unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get("example-tts").unwrap().version, m.version);
        unregister_row(&pool, "example-tts").await.unwrap();
        let reg = load_registry(&pool, Path::new("/plugins")).await.unwrap();
        assert!(reg.is_empty());
    }
}
