//! Pre-download memory estimation, ported from `alvarobartt/hf-mem`
//! (MIT, © Álvaro Bartt { https://github.com/alvarobartt/hf-mem }).
//!
//! Reads Safetensors headers / GGUF metadata through HTTP Range requests —
//! never full downloads — and returns weight bytes, KV-cache bytes, and
//! totals. Powers Nexora's pre-download estimate card (docs/07 §7.4),
//! the VRAM gate input, and `uar estimate`.

pub mod gguf;
pub mod safetensors;

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use crate::core::{NexoraError, Result};
use crate::hf::{fetch_metadata, fetch_repo_json, repo_file_url, HfRepo};

/// Fetch `bytes=start-end` (inclusive) with the env token. Requires 206;
/// a 200 fallback is sliced defensively (Hub always honors ranges).
pub async fn range_get(url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        let tok = tok.trim().to_string();
        if !tok.is_empty() {
            if let Ok(v) =
                reqwest::header::HeaderValue::from_str(&format!("Bearer {tok}"))
            {
                headers.insert(reqwest::header::AUTHORIZATION, v);
            }
        }
    }
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .user_agent("nexora/0.1 (mem-estimate; hf-mem port)")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();
    let res = client
        .get(url)
        .header("Range", format!("bytes={start}-{end}"))
        .send()
        .await
        .map_err(NexoraError::Http)?;
    let status = res.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_GATED: Hub refused range read (fix: set HF_TOKEN for private/gated repos)"
        )));
    }
    let body = res.error_for_status().map_err(NexoraError::Http)?;
    let bytes = body.bytes().await.map_err(NexoraError::Http)?.to_vec();
    if status != reqwest::StatusCode::PARTIAL_CONTENT {
        // Server ignored Range: only usable when the whole body fits.
        let want = (end - start + 1) as usize;
        if bytes.len() <= want {
            return Ok(bytes);
        }
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_RANGE: server ignored Range and sent {} bytes (fix: retry — Hub range support is required)",
            bytes.len()
        )));
    }
    Ok(bytes)
}

#[derive(Debug, Clone)]
pub struct EstimateOpts {
    pub experimental: bool,
    pub max_model_len: Option<u64>,
    pub batch_size: u64,
    pub kv_cache_dtype: String,
    pub gguf_file: Option<String>,
}

impl Default for EstimateOpts {
    fn default() -> Self {
        Self {
            experimental: false,
            max_model_len: None,
            batch_size: 1,
            kv_cache_dtype: "auto".into(),
            gguf_file: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEstimate {
    pub name: String,
    pub bytes: u64,
    pub params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemEstimate {
    pub model_id: String,
    pub revision: String,
    pub filename: Option<String>,
    pub weights_bytes: u64,
    pub param_count: u64,
    pub kv_bytes: Option<u64>,
    pub kv_dtype: Option<String>,
    pub total_bytes: Option<u64>,
    pub per_file: Vec<FileEstimate>,
    pub moe: Option<safetensors::MoeStat>,
    pub experimental: bool,
}

fn is_gguf_shard(name: &str) -> Option<(String, u64)> {
    // `base-00001-of-00046.gguf` → (base.gguf, 1). Mirrors `_SHARD_PATTERN`.
    let stripped = name.strip_suffix(".gguf")?;
    let (base, tail) = stripped.rsplit_once("-of-")?;
    let (base, num) = base.rsplit_once('-')?;
    let n: u64 = num.parse().ok()?;
    let _total: u64 = tail.parse().ok()?;
    Some((format!("{base}.gguf"), n))
}

async fn estimate_gguf(
    repo: &HfRepo,
    paths: Vec<String>,
    opts: &EstimateOpts,
) -> Result<MemEstimate> {
    if !safetensors::KV_CACHE_DTYPE_CHOICES.contains(&opts.kv_cache_dtype.as_str())
        && gguf::GGUF_KV_DTYPES.iter().all(|d| *d != opts.kv_cache_dtype)
        && opts.kv_cache_dtype != "auto"
    {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_KV_DTYPE: --kv-cache-dtype={} invalid for GGUF (fix: one of {} or auto)",
            opts.kv_cache_dtype,
            gguf::GGUF_KV_DTYPES.join(", ")
        )));
    }
    let mut set: JoinSet<Result<(String, gguf::GgufStat, Option<(String, u64)>)>> = JoinSet::new();
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(16));
    for path in paths {
        let permit = sem.clone().acquire_owned().await.map_err(|_| {
            NexoraError::Other(anyhow::anyhow!("E_MEM_INTERNAL: semaphore closed"))
        })?;
        let url = repo_file_url(repo, &path);
        let shard = is_gguf_shard(&path);
        // Shards > 1 often drop KV fields — only parse cache on shard 1.
        let want_kv = opts.experimental && shard.as_ref().map(|(_, n)| *n == 1).unwrap_or(true);
        let kv_dtype = if opts.kv_cache_dtype == "auto" {
            "F16".to_string()
        } else {
            opts.kv_cache_dtype.clone()
        };
        let batch = opts.batch_size;
        let max_len = opts.max_model_len;
        set.spawn(async move {
            let _permit = permit;
            let mut stat = gguf::fetch(&url, want_kv).await?;
            if want_kv {
                let ctx = max_len.or_else(|| stat.kv_fields.get("context_length").copied());
                let size = gguf::kv_cache_size(&stat.kv_fields, &kv_dtype, ctx, batch).ok();
                stat.kv_fields.insert("__kv_bytes__".into(), size.unwrap_or(0));
                if ctx.is_none() {
                    tracing::warn!("GGUF {path}: no context_length and no --max-model-len; KV skipped");
                }
            }
            Ok((path, stat, shard))
        });
    }
    // Merge shards by base name (upstream `_collect_gguf_results`).
    let mut merged: HashMap<String, gguf::GgufStat> = HashMap::new();
    let mut kv_total: HashMap<String, u64> = HashMap::new();
    while let Some(res) = set.join_next().await {
        let (path, stat, shard) = res.map_err(|e| {
            NexoraError::Other(anyhow::anyhow!("E_MEM_INTERNAL: estimate task failed: {e}"))
        })??;
        let kv = stat.kv_fields.get("__kv_bytes__").copied().unwrap_or(0);
        let key = shard.map(|(b, _)| b).unwrap_or(path);
        if kv > 0 {
            kv_total.insert(key.clone(), kv);
        }
        merged
            .entry(key)
            .and_modify(|m| *m = gguf::merge(m.clone(), &stat))
            .or_insert(stat);
    }
    if opts.gguf_file.is_some() {
        let (name, stat) = merged.into_iter().next().ok_or_else(|| {
            NexoraError::Other(anyhow::anyhow!("E_MEM_EMPTY: no GGUF files matched"))
        })?;
        let kv = kv_total.get(&name).copied();
        let total = kv.map(|k| stat.bytes_count + k);
        return Ok(MemEstimate {
            model_id: repo.id(),
            revision: repo.rev.clone(),
            filename: Some(name.clone()),
            weights_bytes: stat.bytes_count,
            param_count: stat.param_count,
            kv_bytes: kv,
            kv_dtype: kv.map(|_| {
                if opts.kv_cache_dtype == "auto" {
                    "F16".into()
                } else {
                    opts.kv_cache_dtype.clone()
                }
            }),
            total_bytes: total,
            per_file: vec![FileEstimate {
                name,
                bytes: stat.bytes_count,
                params: stat.param_count,
            }],
            moe: None,
            experimental: opts.experimental,
        });
    }
    let per_file: Vec<FileEstimate> = merged
        .iter()
        .map(|(name, s)| FileEstimate {
            name: name.clone(),
            bytes: s.bytes_count,
            params: s.param_count,
        })
        .collect();
    let weights: u64 = per_file.iter().map(|f| f.bytes).sum();
    let params: u64 = per_file.iter().map(|f| f.params).sum();
    Ok(MemEstimate {
        model_id: repo.id(),
        revision: repo.rev.clone(),
        filename: None,
        weights_bytes: weights,
        param_count: params,
        kv_bytes: None,
        kv_dtype: None,
        total_bytes: None,
        per_file,
        moe: None,
        experimental: opts.experimental,
    })
}

async fn fetch_component(
    repo: &HfRepo,
    urls: Vec<String>,
) -> Result<safetensors::ComponentStat> {
    let mut set: JoinSet<Result<safetensors::ComponentStat>> = JoinSet::new();
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(16));
    for url in urls {
        let permit = sem.clone().acquire_owned().await.map_err(|_| {
            NexoraError::Other(anyhow::anyhow!("E_MEM_INTERNAL: semaphore closed"))
        })?;
        set.spawn(async move {
            let _permit = permit;
            let header = safetensors::fetch_header(&url).await?;
            let mut comp = safetensors::ComponentStat::default();
            safetensors::accumulate_header(&mut comp, &header)?;
            Ok(comp)
        });
    }
    let mut merged = safetensors::ComponentStat::default();
    while let Some(res) = set.join_next().await {
        let comp = res
            .map_err(|e| {
                NexoraError::Other(anyhow::anyhow!("E_MEM_INTERNAL: estimate task failed: {e}"))
            })??;
        for (dtype, m) in comp.dtypes {
            let entry = merged.dtypes.entry(dtype).or_default();
            entry.param_count = entry.param_count.saturating_add(m.param_count);
            entry.bytes_count = entry.bytes_count.saturating_add(m.bytes_count);
        }
        merged.param_count = merged.param_count.saturating_add(comp.param_count);
        merged.bytes_count = merged.bytes_count.saturating_add(comp.bytes_count);
    }
    Ok(merged)
}

/// Estimate one repo: GGUF when the repo is GGUF-only (or `--gguf-file`),
/// else Safetensors single / sharded / diffusers-components (upstream
/// `arun` routing, incl. the `mmproj` exclusion and the both-formats warn).
pub async fn estimate_repo(repo: &HfRepo, opts: &EstimateOpts) -> Result<MemEstimate> {
    let meta = fetch_metadata(repo).await?;
    let paths: Vec<String> = meta.file_names();
    let has = |n: &str| paths.iter().any(|p| p == n);
    let gguf_paths: Vec<String> = paths
        .iter()
        .filter(|p| p.ends_with(".gguf") && !p.contains("mmproj-"))
        .cloned()
        .collect();
    let has_st = has("model.safetensors")
        || has("model.safetensors.index.json")
        || has("model_index.json");
    let want_gguf = opts.gguf_file.is_some() || (!gguf_paths.is_empty() && !has_st);
    if has_st && !gguf_paths.is_empty() && opts.gguf_file.is_none() {
        tracing::warn!(
            "repo {} has both Safetensors and GGUF weights; estimating Safetensors (pass --gguf-file for a GGUF file)",
            repo.id()
        );
    }

    if want_gguf {
        if gguf_paths.is_empty() {
            return Err(NexoraError::Other(anyhow::anyhow!(
                "E_MEM_EMPTY: no GGUF files in {} (fix: drop --gguf-file or pick a GGUF repo)",
                repo.id()
            )));
        }
        let mut selected = gguf_paths;
        if let Some(want) = &opts.gguf_file {
            // Sharded request (`base-00001-of-00005.gguf`) keeps the whole
            // shard family; otherwise suffix-match a single file.
            let family: Option<String> = want
                .strip_suffix(".gguf")
                .and_then(|b| b.rsplit_once("-of-"))
                .and_then(|(left, total)| {
                    total.parse::<u64>().ok()?;
                    let (base, num) = left.rsplit_once('-')?;
                    num.parse::<u64>().ok()?;
                    Some(format!("{base}-"))
                });
            selected = match family {
                Some(prefix) => selected
                    .into_iter()
                    .filter(|p| {
                        p.starts_with(&prefix)
                            && p.ends_with(".gguf")
                            && is_gguf_shard(p).is_some()
                    })
                    .collect(),
                None => {
                    let hits: Vec<String> =
                        selected.into_iter().filter(|p| p.ends_with(want)).collect();
                    if hits.len() > 1 {
                        return Err(NexoraError::Other(anyhow::anyhow!(
                            "E_MEM_AMBIGUOUS: multiple GGUF files match `{want}` (fix: pass the full sharded name)"
                        )));
                    }
                    hits
                }
            };
            if selected.is_empty() {
                return Err(NexoraError::Other(anyhow::anyhow!(
                    "E_MEM_EMPTY: no GGUF file matching `{want}` in {}",
                    repo.id()
                )));
            }
        }
        return estimate_gguf(repo, selected, opts).await;
    }

    // ---- Safetensors routing ----
    // component name -> weight-file URLs
    let mut components: HashMap<String, Vec<String>> = HashMap::new();
    if has("model.safetensors") {
        components.insert(
            "Transformer".into(),
            vec![repo_file_url(repo, "model.safetensors")],
        );
    } else if has("model.safetensors.index.json") {
        let index: serde_json::Value = fetch_repo_json(repo, "model.safetensors.index.json").await?;
        let mut urls = HashSet::new();
        if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for f in map.values().filter_map(|v| v.as_str()) {
                urls.insert(repo_file_url(repo, f));
            }
        }
        components.insert("Transformer".into(), urls.into_iter().collect());
    } else if has("model_index.json") {
        let index: serde_json::Value = fetch_repo_json(repo, "model_index.json").await?;
        for (key, _) in index.as_object().map(|o| o.clone()).unwrap_or_default() {
            if key.starts_with('_') {
                continue;
            }
            let single = [&format!("{key}/diffusion_pytorch_model.safetensors"), &format!("{key}/model.safetensors")];
            let mut urls = vec![];
            for cand in single {
                if has(cand) {
                    urls.push(repo_file_url(repo, cand));
                }
            }
            for suffix in [
                "diffusion_pytorch_model.safetensors.index.json",
                "model.safetensors.index.json",
            ] {
                let idx_name = format!("{key}/{suffix}");
                if has(&idx_name) {
                    let idx: serde_json::Value = fetch_repo_json(repo, &idx_name).await?;
                    if let Some(map) = idx.get("weight_map").and_then(|v| v.as_object()) {
                        for f in map.values().filter_map(|v| v.as_str()) {
                            urls.push(repo_file_url(repo, &format!("{key}/{f}")));
                        }
                    }
                }
            }
            if !urls.is_empty() {
                components.insert(key, urls);
            }
        }
    } else {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_EMPTY: no model.safetensors / sharded index / model_index.json in {} (fix: GGUF repos need --gguf-file semantics — already handled; this repo has neither)",
            repo.id()
        )));
    }

    // Sentence-transformers dense heads (upstream extra component).
    if has("config_sentence_transformers.json") && has("modules.json") {
        if let Ok(modules) = fetch_repo_json(repo, "modules.json").await {
            if let Some(arr) = modules.as_array() {
                for m in arr {
                    let is_dense = m.get("type").and_then(|v| v.as_str())
                        == Some("sentence_transformers.models.Dense");
                    if let (true, Some(path)) =
                        (is_dense, m.get("path").and_then(|v| v.as_str()))
                    {
                        components.insert(
                            path.to_string(),
                            vec![repo_file_url(repo, &format!("{path}/model.safetensors"))],
                        );
                    }
                }
            }
        }
    }

    let mut stats: HashMap<String, safetensors::ComponentStat> = HashMap::new();
    for (name, urls) in &components {
        stats.insert(name.clone(), fetch_component(repo, urls.clone()).await?);
    }
    let weights: u64 = stats.values().map(|c| c.bytes_count).sum();
    let params: u64 = stats.values().map(|c| c.param_count).sum();

    // Experimental: MoE + KV cache for CausalLM / ConditionalGeneration.
    let mut moe = None;
    let mut kv_bytes = None;
    let mut kv_dtype = None;
    if opts.experimental && has("config.json") {
        let config: serde_json::Value = fetch_repo_json(repo, "config.json").await.unwrap_or_default();
        let mut config = config;
        let archs: Vec<String> = config
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        let is_lm = archs.iter().any(|a| a.contains("ForCausalLM") || a.contains("ForConditionalGeneration"));
        if is_lm {
            if archs.iter().any(|a| a.contains("ForConditionalGeneration")) {
                if let Some(text) = config.get("text_config").cloned() {
                    // Follow `_name_or_path` like upstream, inheriting
                    // dtype/quantization keys from the outer config.
                    let mut text = text;
                    if let Some(r) = text.get("_name_or_path").and_then(|v| v.as_str()) {
                        let referenced = HfRepo {
                            owner: r.split('/').next().unwrap_or("").into(),
                            repo: r.split('/').nth(1).unwrap_or("").into(),
                            rev: repo.rev.clone(),
                        };
                        if !referenced.owner.is_empty() && !referenced.repo.is_empty() {
                            if let Ok(rc) = fetch_repo_json(&referenced, "config.json").await {
                                if let (Some(t), Some(r)) =
                                    (text.as_object_mut(), rc.as_object())
                                {
                                    for (k, v) in r {
                                        t.entry(k.clone()).or_insert(v.clone());
                                    }
                                }
                            }
                        }
                    }
                    if let Some(obj) = text.as_object() {
                        for key in ["dtype", "torch_dtype", "quantization_config"] {
                            if let (Some(cfg), Some(v)) = (
                                config.as_object_mut(),
                                obj.get(key),
                            ) {
                                cfg.entry(key.to_string()).or_insert(v.clone());
                            }
                        }
                    }
                    config = text;
                }
            }
            // NOTE: MoE needs per-tensor names; the aggregated stats above
            // dropped them, so re-fetch headers (small) for the split.
            let mut raw: HashMap<String, serde_json::Map<String, serde_json::Value>> =
                HashMap::new();
            for (name, urls) in &components {
                for url in urls {
                    if let Ok(h) = safetensors::fetch_header(url).await {
                        raw.entry(name.clone()).or_default().extend(h);
                    }
                }
            }
            moe = safetensors::parse_moe(&raw, &config).ok().flatten();
            let max_len = opts.max_model_len.or_else(|| {
                config
                    .get("max_position_embeddings")
                    .or_else(|| config.get("n_positions"))
                    .or_else(|| config.get("max_seq_len"))
                    .and_then(|v| v.as_u64())
            });
            let needed = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
                .iter()
                .all(|k| config.get(*k).and_then(|v| v.as_u64()).is_some());
            if let (Some(ml), true) = (max_len, needed) {
                let observed: Vec<String> = stats
                    .values()
                    .flat_map(|c| c.dtypes.keys().cloned())
                    .collect();
                match safetensors::resolve_kv_dtype(&config, &opts.kv_cache_dtype, &observed) {
                    Ok(dtype) => {
                        match safetensors::kv_cache_size(&config, &dtype, ml, opts.batch_size) {
                            Ok(bytes) => {
                                kv_bytes = Some(bytes);
                                kv_dtype = Some(dtype);
                            }
                            Err(e) => tracing::warn!("KV estimate failed: {e}"),
                        }
                    }
                    Err(e) => tracing::warn!("KV dtype unresolved: {e}"),
                }
            } else {
                tracing::warn!("KV skipped: config lacks max-len or attention dims");
            }
        } else {
            tracing::warn!("--experimental set but architecture is not CausalLM/ConditionalGeneration; KV skipped");
        }
    }

    let per_file = vec![FileEstimate {
        name: "weights".into(),
        bytes: weights,
        params,
    }];
    let total = kv_bytes.map(|k| weights + k);
    Ok(MemEstimate {
        model_id: repo.id(),
        revision: repo.rev.clone(),
        filename: None,
        weights_bytes: weights,
        param_count: params,
        kv_bytes,
        kv_dtype,
        total_bytes: total,
        per_file,
        moe,
        experimental: opts.experimental,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_pattern_splits_base_and_index() {
        let (base, n) = is_gguf_shard("Kimi-K2.5-BF16-00001-of-00046.gguf").unwrap();
        assert_eq!(base, "Kimi-K2.5-BF16.gguf");
        assert_eq!(n, 1);
        assert!(is_gguf_shard("model.gguf").is_none());
    }

    #[test]
    fn default_opts_are_conservative() {
        let o = EstimateOpts::default();
        assert!(!o.experimental && o.batch_size == 1 && o.kv_cache_dtype == "auto");
    }
}
