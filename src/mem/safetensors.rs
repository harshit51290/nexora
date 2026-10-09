//! Safetensors side of memory estimation.
//!
//! Ported from `alvarobartt/hf-mem` (MIT, © Álvaro Bartt {
//! https://github.com/alvarobartt/hf-mem }): header fetch via HTTP Range
//! requests, tensor accumulation, MoE breakdown, KV-cache math. Errors are
//! Nexora-coded; every number matches the upstream formulas.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::{NexoraError, Result};

/// Bytes fetched per weight file on the first pass; larger headers trigger
/// one follow-up ranged request for the remainder (upstream
/// `MAX_METADATA_SIZE`).
pub const MAX_METADATA_SIZE: u64 = 100_000;

/// Valid `--kv-cache-dtype` choices for Safetensors (upstream
/// `KV_CACHE_DTYPE_CHOICES`).
pub const KV_CACHE_DTYPE_CHOICES: &[&str] = &[
    "auto",
    "bfloat16",
    "fp8",
    "fp8_ds_mla",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
];

/// Bytes per element (upstream `get_safetensors_dtype_bytes`).
pub fn dtype_bytes(dtype: &str) -> Result<u64> {
    match dtype {
        "F64" | "I64" | "U64" => Ok(8),
        "F32" | "I32" | "U32" => Ok(4),
        "F16" | "BF16" | "I16" | "U16" => Ok(2),
        "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "I8" | "U8" => Ok(1),
        other => Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_DTYPE: safetensors dtype {other} not handled (fix: file an issue — upstream hf-mem would raise here too)"
        ))),
    }
}

/// Upstream `torch_dtype_to_safetensors_dtype` (falls back to `F16`).
pub fn torch_dtype_to_safetensors(dtype: &str) -> String {
    let d = dtype.strip_prefix("torch.").unwrap_or(dtype);
    match d {
        "float32" => "F32",
        "float16" => "F16",
        "bfloat16" => "BF16",
        "float8_e4m3" | "float8_e4m3fn" => "F8_E4M3",
        "float8_e5m2" => "F8_E5M2",
        "int8" => "I8",
        _ => "F16",
    }
    .to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DtypeStat {
    pub param_count: u64,
    pub bytes_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComponentStat {
    pub dtypes: HashMap<String, DtypeStat>,
    pub param_count: u64,
    pub bytes_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeightsStat {
    pub components: HashMap<String, ComponentStat>,
    pub param_count: u64,
    pub bytes_count: u64,
}

/// Accumulate one tensor (upstream `_accumulate_tensor`).
pub fn accumulate(
    comp: &mut ComponentStat,
    dtype: &str,
    shape: &[u64],
) -> Result<(u64, u64)> {
    let bytes_each = dtype_bytes(dtype)?;
    let params: u64 = shape.iter().product();
    let bytes = params.saturating_mul(bytes_each);
    let entry = comp.dtypes.entry(dtype.to_string()).or_default();
    entry.param_count = entry.param_count.saturating_add(params);
    entry.bytes_count = entry.bytes_count.saturating_add(bytes);
    comp.param_count = comp.param_count.saturating_add(params);
    comp.bytes_count = comp.bytes_count.saturating_add(bytes);
    Ok((params, bytes))
}

/// Sum one weight file's header (`{tensor: {dtype, shape}}`, skipping
/// `__metadata__`) into a component (upstream `parse_safetensors_metadata`
/// inner loop).
pub fn accumulate_header(
    comp: &mut ComponentStat,
    header: &serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                NexoraError::Other(anyhow::anyhow!(
                    "E_MEM_HEADER: tensor {name} has no dtype string"
                ))
            })?;
        let shape: Vec<u64> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|n| n.as_u64()).collect())
            .unwrap_or_default();
        accumulate(comp, dtype, &shape)?;
    }
    Ok(())
}

/// Fetch one weight file's JSON header with at most two ranged requests:
/// `bytes=0-100000`, then the remainder when the LE-u64 length prefix says
/// the header is larger (upstream `fetch_safetensors_metadata`).
pub async fn fetch_header(url: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
    let first = super::range_get(url, 0, MAX_METADATA_SIZE).await?;
    if first.len() < 8 {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_HEADER: short read on {url} (fix: retry — the Hub may have throttled the range request)"
        )));
    }
    let len = u64::from_le_bytes(first[0..8].try_into().unwrap()) as usize;
    if len <= MAX_METADATA_SIZE as usize {
        let body = first
            .get(8..8 + len)
            .ok_or_else(|| {
                NexoraError::Other(anyhow::anyhow!("E_MEM_HEADER: truncated header on {url}"))
            })?;
        return serde_json::from_slice(body).map_err(|e| {
            NexoraError::Other(anyhow::anyhow!("E_MEM_HEADER: invalid safetensors header on {url}: {e}"))
        });
    }
    let mut buf = first[8..].to_vec();
    let rest = super::range_get(url, MAX_METADATA_SIZE + 1, len as u64 + 7).await?;
    buf.extend_from_slice(&rest);
    serde_json::from_slice(&buf).map_err(|e| {
        NexoraError::Other(anyhow::anyhow!("E_MEM_HEADER: invalid safetensors header on {url}: {e}"))
    })
}

// ---------------------------------------------------------------------------
// KV cache (upstream `kv_cache.py`)
// ---------------------------------------------------------------------------

/// (full-attention layers, sliding-window layers). Gemma3-style
/// `sliding_window_pattern`, explicit `layer_types`, else all-full
/// (upstream `_resolve_attention_layer_counts`).
pub fn attention_layer_counts(config: &serde_json::Value) -> (u64, u64) {
    let total = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    // `checked_div` doubles as the zero guard (`None` on `/0`).
    let full = config
        .get("sliding_window_pattern")
        .and_then(|v| v.as_u64())
        .and_then(|n| total.checked_div(n));
    if let Some(full) = full {
        return (full, total.saturating_sub(full));
    }
    if let Some(types) = config.get("layer_types").and_then(|v| v.as_array()) {
        let full = types
            .iter()
            .filter(|t| {
                matches!(
                    t.as_str(),
                    Some("attention" | "full_attention" | "global_attention")
                )
            })
            .count() as u64;
        return (full, total.saturating_sub(full));
    }
    (total, 0)
}

/// Resolve the KV dtype: explicit override (validated), quantization-config
/// branches, else `torch_dtype`/`dtype` mapping (upstream
/// `resolve_kv_cache_dtype`, compacted).
pub fn resolve_kv_dtype(
    config: &serde_json::Value,
    kv_cache_dtype: &str,
    observed_dtypes: &[String],
) -> Result<String> {
    if !KV_CACHE_DTYPE_CHOICES.contains(&kv_cache_dtype) {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_KV_DTYPE: --kv-cache-dtype={kv_cache_dtype} invalid (fix: one of {})",
            KV_CACHE_DTYPE_CHOICES.join(", ")
        )));
    }
    match kv_cache_dtype {
        "fp8_e5m2" => return Ok("F8_E5M2".into()),
        "fp8_e4m3" => return Ok("F8_E4M3".into()),
        "fp8" | "fp8_ds_mla" | "fp8_inc" => return Ok("F8_E4M3".into()),
        "bfloat16" => return Ok("BF16".into()),
        _ => {}
    }
    if let Some(qc) = config.get("quantization_config").and_then(|v| v.as_object()) {
        let method = qc.get("quant_method").and_then(|v| v.as_str()).unwrap_or("");
        if method == "fp8" || method == "modelopt" {
            if let Some(fmt) = qc
                .get("fmt")
                .or_else(|| qc.get("format"))
                .and_then(|v| v.as_str())
            {
                let fmt = if fmt.starts_with("float8_") {
                    fmt.to_string()
                } else {
                    format!("float8_{fmt}")
                };
                return Ok(torch_dtype_to_safetensors(&fmt));
            }
            if let Some(scheme) = qc.get("kv_cache_scheme") {
                let bits = scheme.get("num_bits").and_then(|v| v.as_u64());
                let typ = scheme.get("type").and_then(|v| v.as_str());
                if bits == Some(8) && typ == Some("float") {
                    return Ok("F8_E4M3".into());
                }
            }
            // Most-used F8 dtype actually present in the weights.
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for d in observed_dtypes {
                if d == "F8_E5M2" || d == "F8_E4M3" {
                    *counts.entry(d.as_str()).or_default() += 1;
                }
            }
            if let Some((best, _)) = counts.iter().max_by_key(|(_, c)| **c) {
                return Ok(best.to_string());
            }
            return Err(NexoraError::Other(anyhow::anyhow!(
                "E_MEM_KV_DTYPE: fp8-quantized model has no F8 weights to infer the cache dtype (fix: pass --kv-cache-dtype=fp8)"
            )));
        }
        if method == "compressed-tensors" {
            // No kv_cache_scheme → fall through to torch_dtype/dtype below.
            if qc.get("kv_cache_scheme").is_some() {
                return Err(NexoraError::Other(anyhow::anyhow!(
                    "E_MEM_KV_DTYPE: compressed-tensors kv_cache_scheme unsupported (fix: pass an explicit --kv-cache-dtype)"
                )));
            }
        } else if !method.is_empty() {
            return Err(NexoraError::Other(anyhow::anyhow!(
                "E_MEM_KV_DTYPE: quant_method={method} unsupported (fix: pass an explicit --kv-cache-dtype)"
            )));
        }
    }
    if let Some(d) = config
        .get("torch_dtype")
        .or_else(|| config.get("dtype"))
        .and_then(|v| v.as_str())
    {
        return Ok(torch_dtype_to_safetensors(d));
    }
    Err(NexoraError::Other(anyhow::anyhow!(
        "E_MEM_KV_DTYPE: cannot resolve cache dtype — config lacks torch_dtype/dtype and no quantization_config (fix: pass --kv-cache-dtype)"
    )))
}

/// `2 × kv_tokens × kv_heads × head_dim × dtype_bytes × batch`, with the
/// hybrid-vs-pure sliding-window rule (upstream
/// `compute_safetensors_kv_cache_size`).
pub fn kv_cache_size(
    config: &serde_json::Value,
    cache_dtype: &str,
    max_model_len: u64,
    batch_size: u64,
) -> Result<u64> {
    let hidden = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            NexoraError::Other(anyhow::anyhow!("E_MEM_KV: config lacks hidden_size"))
        })?;
    let heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            NexoraError::Other(anyhow::anyhow!("E_MEM_KV: config lacks num_attention_heads"))
        })?;
    // MHA default; GQA/MQA set fewer key/value heads.
    let kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(heads);
    // Qwen3-style explicit head_dim beats hidden/heads.
    let head_dim = config
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| hidden / heads.max(1));
    let (full, sliding) = attention_layer_counts(config);
    let window = config
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(max_model_len);
    let hybrid = full > 0 && sliding > 0;
    let sliding_tokens = if hybrid {
        max_model_len
    } else {
        window.min(max_model_len)
    };
    let tokens = full.saturating_mul(max_model_len).saturating_add(sliding.saturating_mul(sliding_tokens));
    let bytes_each = dtype_bytes(cache_dtype)?;
    Ok(2u64
        .saturating_mul(tokens)
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(bytes_each)
        .saturating_mul(batch_size.max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn header() -> serde_json::Map<String, serde_json::Value> {
        json!({
            "__metadata__": {"format": "pt"},
            "w": {"dtype": "F16", "shape": [1024, 1024]},
            "b": {"dtype": "F32", "shape": [1024]},
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn accumulation_matches_hf_mem_math() {
        let mut comp = ComponentStat::default();
        accumulate_header(&mut comp, &header()).unwrap();
        // 1024*1024 F16 (2B) + 1024 F32 (4B).
        assert_eq!(comp.param_count, 1024 * 1024 + 1024);
        assert_eq!(comp.bytes_count, 1024 * 1024 * 2 + 1024 * 4);
        assert_eq!(comp.dtypes["F16"].bytes_count, 1024 * 1024 * 2);
    }

    #[test]
    fn kv_math_matches_llama_7b_fp16_ctx4k() {
        // 32 layers × 4096 tokens × 32 kv-heads × 128 head-dim × 2B × 2 (K+V).
        let config = json!({"hidden_size": 4096, "num_attention_heads": 32, "num_hidden_layers": 32});
        let bytes = kv_cache_size(&config, "F16", 4096, 1).unwrap();
        assert_eq!(bytes, 2 * 32 * 4096 * 32 * 128 * 2);
    }

    #[test]
    fn gqa_uses_kv_heads_and_head_dim_override() {
        let config = json!({
            "hidden_size": 4096, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 1,
        });
        let bytes = kv_cache_size(&config, "BF16", 128, 2).unwrap();
        assert_eq!(bytes, 2 * 128 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn torch_mapping_and_choices() {
        assert_eq!(torch_dtype_to_safetensors("torch.bfloat16"), "BF16");
        assert_eq!(torch_dtype_to_safetensors("float32"), "F32");
        assert!(dtype_bytes("Q4_K_M").is_err());
    }
}

// ---------------------------------------------------------------------------
// MoE (upstream `parse_moe_metadata`, compacted)
// ---------------------------------------------------------------------------

const EXPERT_TOKENS: &[&str] = &[
    "expert",
    "experts",
    "local_expert",
    "local_experts",
    "routed_expert",
    "routed_experts",
];

fn expert_id(name: &str) -> Option<u64> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if EXPERT_TOKENS.contains(part) && i + 1 < parts.len() && parts[i + 1].bytes().all(|b| b.is_ascii_digit()) {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoeStat {
    pub base_bytes: u64,
    pub base_params: u64,
    pub expert_count: u64,
    pub active_expert_count: Option<u64>,
    pub experts_total_bytes: u64,
    pub experts_total_params: u64,
}

fn config_int(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    // Upstream `_get_config_int`: only ints strictly greater than 1 count.
    keys.iter()
        .filter_map(|k| config.get(*k)?.as_u64())
        .find(|v| *v > 1)
}

/// Split expert tensors from the base model; validates count + uniformity
/// exactly like upstream (mismatch → coded error instead of panic).
pub fn parse_moe(
    headers: &HashMap<String, serde_json::Map<String, serde_json::Value>>,
    config: &serde_json::Value,
) -> Result<Option<MoeStat>> {
    let mut base = ComponentStat::default();
    let mut experts: HashMap<u64, ComponentStat> = HashMap::new();
    for header in headers.values() {
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = value.get("dtype").and_then(|v| v.as_str()).unwrap_or("F16");
            let shape: Vec<u64> = value
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|n| n.as_u64()).collect())
                .unwrap_or_default();
            let target = match expert_id(name) {
                Some(id) => experts.entry(id).or_default(),
                None => &mut base,
            };
            // Unknown dtypes surface here (same fail-loud policy as upstream).
            accumulate(target, dtype, &shape)?;
        }
    }
    if experts.is_empty() {
        return Ok(None);
    }
    let configured = config_int(
        config,
        &["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"],
    );
    let mut ids: Vec<u64> = experts.keys().copied().collect();
    ids.sort_unstable();
    if let Some(n) = configured {
        let expected: Vec<u64> = (0..n).collect();
        if ids != expected {
            return Err(NexoraError::Other(anyhow::anyhow!(
                "E_MEM_MOE: expert indices {ids:?} != config count {n} (fix: verify the repo is a single MoE checkpoint)"
            )));
        }
    }
    let template = experts[&ids[0]].clone();
    for id in ids.iter().skip(1) {
        let other = &experts[id];
        if other.param_count != template.param_count
            || other.bytes_count != template.bytes_count
        {
            return Err(NexoraError::Other(anyhow::anyhow!(
                "E_MEM_MOE: experts are not uniform (fix: estimate per-file instead)"
            )));
        }
    }
    Ok(Some(MoeStat {
        base_bytes: base.bytes_count,
        base_params: base.param_count,
        expert_count: configured.unwrap_or(ids.len() as u64),
        active_expert_count: config_int(
            config,
            &["num_experts_per_tok", "num_experts_per_token", "top_k"],
        ),
        experts_total_bytes: experts.values().map(|e| e.bytes_count).sum(),
        experts_total_params: experts.values().map(|e| e.param_count).sum(),
    }))
}
