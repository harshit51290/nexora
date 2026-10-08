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
    let mut rev = at_rev.unwrap_or_else(|| "main".to_string());
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
    Ok(res
        .error_for_status()
        .map_err(NexoraError::Http)?
        .json()
        .await
        .map_err(NexoraError::Http)?)
}
