//! HF import: URL parsing (`owner/repo@rev`), Hub API file +
//! metadata + license fetch. Private/gated repos authenticate with a token
//! read from the OS environment (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`).

use crate::core::{NexoraError, Result};
use serde::{Deserialize, Serialize};

/// Parsed repo reference. `rev` is a branch, tag, or commit SHA.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfRepo {
    pub owner: String,
    pub repo: String,
    pub rev: String,
}

impl HfRepo {
    pub fn id(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }
    pub fn api_url(&self) -> String {
        format!("https://huggingface.co/api/models/{}", self.id())
    }
}

/// Accept `https://huggingface.co/<owner>/<repo>` plus `@rev`, trailing
/// `.git`, and `/tree|/blob|/resolve/<rev>` deep links.
pub fn parse_hf_url(url: &str) -> Result<HfRepo> {
    let url = url
        .trim()
        .trim_end_matches(".git")
        .trim_end_matches('/')
        .to_string();
    let (url, at_rev) = match url.rsplit_once('@') {
        Some((base, rev)) if !rev.contains('/') => (base.to_string(), Some(rev.to_string())),
        _ => (url, None),
    };
    let path = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))
        .or_else(|| url.strip_prefix("huggingface.co/"))
        .ok_or_else(|| anyhow::anyhow!("not a huggingface.co URL: {url}"))?;
    let segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if segs.len() < 2 {
        return Err(anyhow::anyhow!("HF URL must be https://huggingface.co/<owner>/<repo>").into());
    }
    let (owner, repo) = (segs[0].to_string(), segs[1].to_string());
    // /owner/repo/{tree,blob,resolve}/<rev...>
    let mut rev = at_rev.clone().unwrap_or_else(|| "main".to_string());
    if segs.len() > 3 && matches!(segs[2], "tree" | "blob" | "resolve") {
        rev = segs[3..].join("/");
    } else if segs.len() > 2 && !matches!(segs[2], "tree" | "blob" | "resolve") && at_rev.is_none()
    {
        return Err(anyhow::anyhow!("unexpected HF URL suffix: /{}", segs[2..].join("/")).into());
    }
    if owner.is_empty() || repo.is_empty() || rev.is_empty() {
        return Err(anyhow::anyhow!("invalid HF repo reference").into());
    }
    Ok(HfRepo { owner, repo, rev })
}

/// Token from OS environment only. Never read from config files or the DB.
pub fn hf_token_from_env() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"] {
        if let Ok(v) = std::env::var(key) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfFileEntry {
    #[serde(default)]
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default, rename = "lfs")]
    pub lfs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoMetadata {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    #[serde(default, rename = "library_name")]
    pub library_name: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub siblings: Vec<HfFileEntry>,
    #[serde(default)]
    pub card: Option<String>,
}

impl RepoMetadata {
    /// License from tags (`license:apache-2.0`) or the `license` field.
    pub fn license(&self) -> Option<String> {
        if let Some(l) = &self.license {
            return Some(l.clone());
        }
        self.tags
            .iter()
            .find_map(|t| t.strip_prefix("license:").map(str::to_string))
    }

    pub fn file_names(&self) -> Vec<String> {
        self.siblings.iter().map(|s| s.rfilename.clone()).collect()
    }
}

fn authed_client() -> reqwest::Client {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(tok) = hf_token_from_env() {
        if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {tok}")) {
            headers.insert(reqwest::header::AUTHORIZATION, v);
        }
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .user_agent("nexora/0.1")
        .build()
        .unwrap_or_default()
}

/// Resolve a bare model name through the Hub alias endpoint
/// (`gpt2` → `openai-community/gpt2`, rev `main`). Used when the input is
/// neither a URL nor `owner/model`.
pub async fn resolve_bare_name(name: &str) -> Result<HfRepo> {
    let name = name.trim();
    let doc: serde_json::Value = reqwest::Client::builder()
        .user_agent("nexora/0.1")
        .build()
        .unwrap_or_default()
        .get(format!("https://huggingface.co/api/models/{name}"))
        .send()
        .await
        .map_err(NexoraError::Http)?
        .error_for_status()
        .map_err(|_| {
            NexoraError::Other(anyhow::anyhow!(
                "unknown model {name} (fix: pass owner/model or a full HF URL)"
            ))
        })?
        .json()
        .await
        .map_err(NexoraError::Http)?;
    let id = doc.get("id").and_then(|v| v.as_str()).unwrap_or(name);
    let (owner, repo) = id.split_once('/').unwrap_or(("", id));
    if owner.is_empty() {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "Hub did not resolve {name} to owner/model"
        )));
    }
    Ok(HfRepo {
        owner: owner.to_string(),
        repo: repo.to_string(),
        rev: "main".to_string(),
    })
}
/// URL of a single repo file at `rev` (Hub `resolve` endpoint). Used to
/// fetch analyzer inputs (`config.json`, `model_index.json`) without
/// cloning the repo.
pub fn repo_file_url(repo: &HfRepo, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/{}/resolve/{}/{}",
        repo.owner,
        repo.repo,
        repo.rev,
        filename.trim_start_matches('/')
    )
}

fn gated_error(repo: &HfRepo) -> NexoraError {
    NexoraError::Other(anyhow::anyhow!(
        "HF repo {} is private/gated: set HF_TOKEN in the OS environment and retry",
        repo.id()
    ))
}

/// GET one text file (`config.json` / `model_index.json`) with the env
/// token when present. Returns the raw text; parsing into
/// `TransformersConfig` / diffusion index lives in `crate::analyze` so this
/// module stays free of analyzer types.
pub async fn fetch_repo_text(repo: &HfRepo, filename: &str) -> Result<String> {
    let res = authed_client()
        .get(repo_file_url(repo, filename))
        .send()
        .await
        .map_err(NexoraError::Http)?;
    match res.status() {
        s if s == reqwest::StatusCode::UNAUTHORIZED || s == reqwest::StatusCode::FORBIDDEN => {
            return Err(gated_error(repo))
        }
        _ => {}
    }
    res.error_for_status()
        .map_err(NexoraError::Http)?
        .text()
        .await
        .map_err(NexoraError::Http)
}

/// GET one JSON file (`config.json` / `model_index.json`) as a value.
pub async fn fetch_repo_json(repo: &HfRepo, filename: &str) -> Result<serde_json::Value> {
    let text = fetch_repo_text(repo, filename).await?;
    serde_json::from_str(&text).map_err(|e| {
        NexoraError::Other(anyhow::anyhow!("{filename} is not valid JSON: {e}"))
    })
}

/// GET /api/models/{owner}/{repo} — metadata, tags, siblings, license.
pub async fn fetch_metadata(repo: &HfRepo) -> Result<RepoMetadata> {
    let res = authed_client()
        .get(repo.api_url())
        .send()
        .await
        .map_err(NexoraError::Http)?;
    if res.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "HF repo {} is private/gated: set HF_TOKEN in the OS environment and retry",
            repo.id()
        )));
    }
    res.error_for_status()
        .map_err(NexoraError::Http)?
        .json()
        .await
        .map_err(NexoraError::Http)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> HfRepo {
        HfRepo {
            owner: "Qwen".to_string(),
            repo: "Qwen2-7B".to_string(),
            rev: "main".to_string(),
        }
    }

    #[test]
    fn parse_bare_url() {
        let r = parse_hf_url("https://huggingface.co/Qwen/Qwen2-7B").unwrap();
        assert_eq!(r.owner, "Qwen");
        assert_eq!(r.repo, "Qwen2-7B");
        assert_eq!(r.rev, "main");
    }

    #[test]
    fn parse_at_rev_and_tree_link() {
        let r = parse_hf_url("https://huggingface.co/Qwen/Qwen2-7B@v1.2").unwrap();
        assert_eq!(r.rev, "v1.2");
        let r = parse_hf_url("https://huggingface.co/Qwen/Qwen2-7B/tree/q4_k_m").unwrap();
        assert_eq!(r.rev, "q4_k_m");
    }

    #[test]
    fn file_url_points_at_resolve_rev() {
        assert_eq!(
            repo_file_url(&repo(), "config.json"),
            "https://huggingface.co/Qwen/Qwen2-7B/resolve/main/config.json"
        );
        assert_eq!(
            repo_file_url(&repo(), "/model_index.json"),
            "https://huggingface.co/Qwen/Qwen2-7B/resolve/main/model_index.json"
        );
    }

    #[test]
    fn license_prefers_field_then_tag() {
        let mut m = RepoMetadata {
            id: "x".to_string(),
            tags: vec!["license:apache-2.0".to_string()],
            pipeline_tag: None,
            library_name: None,
            license: None,
            siblings: vec![],
            card: None,
        };
        assert_eq!(m.license(), Some("apache-2.0".to_string()));
        m.license = Some("mit".to_string());
        assert_eq!(m.license(), Some("mit".to_string()));
    }
}
