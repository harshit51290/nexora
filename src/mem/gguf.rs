//! GGUF side of memory estimation.
//!
//! Ported from `alvarobartt/hf-mem` (MIT, © Álvaro Bartt {
//! https://github.com/alvarobartt/hf-mem }): header fetch with doubling
//! 1MB→100MB, magic/count parsing, KV dispatch table, tensor walk with
//! bits-per-weight, shard merging, suffix-matched KV-cache math.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::core::{NexoraError, Result};
use crate::mem::safetensors::DtypeStat;

/// Fetch starts at 1MB and doubles to a 100MB cap (upstream
/// `_INITIAL_FETCH_SIZE` / `_MAX_FETCH_SIZE`).
const INITIAL_FETCH_SIZE: u64 = 1_000_000;
const MAX_FETCH_SIZE: u64 = 100_000_000;

/// Bits per weight (upstream `GGUFDtypeBitsPerWeight`), keyed by raw
/// ggml type id so unknown ids fail loudly like upstream's enum cast.
pub fn bits_per_weight(type_id: u32) -> Result<f64> {
    match type_id {
        0 => Ok(32.0),   // F32
        1 => Ok(16.0),   // F16
        2 => Ok(4.5),    // Q4_0
        3 => Ok(5.0),    // Q4_1
        6 => Ok(5.5),    // Q5_0
        7 => Ok(6.0),    // Q5_1
        8 => Ok(8.5),    // Q8_0
        9 => Ok(9.0),    // Q8_1
        10 => Ok(2.625), // Q2_K
        11 => Ok(3.4375),// Q3_K
        12 => Ok(4.5),   // Q4_K
        13 => Ok(5.5),   // Q5_K
        14 => Ok(6.5625),// Q6_K
        15 => Ok(8.03125),// Q8_K
        16 => Ok(2.06),  // IQ2_XXS
        17 => Ok(2.31),  // IQ2_XS
        18 => Ok(3.06),  // IQ3_XXS
        19 => Ok(1.56),  // IQ1_S
        20 => Ok(4.5),   // IQ4_NL
        21 => Ok(3.44),  // IQ3_S
        22 => Ok(2.5),   // IQ2_S
        23 => Ok(4.25),  // IQ4_XS
        24 => Ok(8.0),   // I8
        25 => Ok(16.0),  // I16
        26 => Ok(32.0),  // I32
        27 => Ok(64.0),  // I64
        28 => Ok(64.0),  // F64
        29 => Ok(1.75),  // IQ1_M
        30 => Ok(16.0),  // BF16
        34 => Ok(1.6875),// TQ1_0
        35 => Ok(2.0625),// TQ2_0
        39 => Ok(4.25),  // MXFP4
        other => Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_GGUF_DTYPE: ggml type {other} unhandled (fix: update bits_per_weight from the GGUF spec)"
        ))),
    }
}

/// ggml type id for a KV-cache dtype name (mirrors the enum order in
/// `bits_per_weight` without depending on array positions).
pub fn kv_dtype_id(name: &str) -> Result<u32> {
    match name {
        "F32" => Ok(0),
        "F16" => Ok(1),
        "Q4_0" => Ok(2),
        "Q4_1" => Ok(3),
        "Q5_0" => Ok(6),
        "Q5_1" => Ok(7),
        "Q8_0" => Ok(8),
        "Q8_1" => Ok(9),
        "Q2_K" => Ok(10),
        "Q3_K" => Ok(11),
        "Q4_K" => Ok(12),
        "Q5_K" => Ok(13),
        "Q6_K" => Ok(14),
        "Q8_K" => Ok(15),
        "IQ2_XXS" => Ok(16),
        "IQ2_XS" => Ok(17),
        "IQ3_XXS" => Ok(18),
        "IQ1_S" => Ok(19),
        "IQ4_NL" => Ok(20),
        "IQ3_S" => Ok(21),
        "IQ2_S" => Ok(22),
        "IQ4_XS" => Ok(23),
        "I8" => Ok(24),
        "I16" => Ok(25),
        "I32" => Ok(26),
        "I64" => Ok(27),
        "F64" => Ok(28),
        "IQ1_M" => Ok(29),
        "BF16" => Ok(30),
        "TQ1_0" => Ok(34),
        "TQ2_0" => Ok(35),
        "MXFP4" => Ok(39),
        other => Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_KV_DTYPE: GGUF cache dtype {other} invalid (fix: one of {})",
            GGUF_KV_DTYPES.join(", ")
        ))),
    }
}
/// Human name for a ggml type id (display + dtype tables).
pub fn dtype_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => "UNKNOWN",
    }
}

/// Valid `--kv-cache-dtype` names for GGUF (upstream `GGUFDtype` members).
pub const GGUF_KV_DTYPES: &[&str] = &[
    "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K",
    "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",
    "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64",
    "IQ1_M", "BF16", "TQ1_0", "TQ2_0", "MXFP4",
];

/// KV metadata value (upstream `_PARSERS` output).
#[derive(Debug, Clone)]
pub enum KvValue {
    Int(i64),
    Uint(u64),
    Float(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<KvValue>),
}

impl KvValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            KvValue::Uint(v) => Some(*v),
            KvValue::Int(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
}

fn take(buf: &[u8], off: usize, n: usize) -> Result<&[u8]> {
    buf.get(off..off + n).ok_or_else(|| {
        NexoraError::Other(anyhow::anyhow!(
            "E_MEM_GGUF_TRUNCATED: need {n} bytes at offset {off}, buffer is {} (fix: fetch a larger range)",
            buf.len()
        ))
    })
}

fn read_u8(buf: &[u8], off: usize) -> Result<(u64, usize)> {
    Ok((take(buf, off, 1)?[0] as u64, off + 1))
}

fn read_u16(buf: &[u8], off: usize) -> Result<(u64, usize)> {
    Ok((u16::from_le_bytes(take(buf, off, 2)?.try_into().unwrap()) as u64, off + 2))
}

fn read_u32(buf: &[u8], off: usize) -> Result<(u64, usize)> {
    Ok((u32::from_le_bytes(take(buf, off, 4)?.try_into().unwrap()) as u64, off + 4))
}

fn read_u64(buf: &[u8], off: usize) -> Result<(u64, usize)> {
    Ok((u64::from_le_bytes(take(buf, off, 8)?.try_into().unwrap()), off + 8))
}

fn read_string(buf: &[u8], off: usize) -> Result<(String, usize)> {
    let (len, mut off) = read_u64(buf, off)?;
    let bytes = take(buf, off, len as usize)?;
    off += len as usize;
    std::str::from_utf8(bytes)
        .map(|s| (s.to_string(), off))
        .map_err(|_| {
            NexoraError::Other(anyhow::anyhow!(
                "E_MEM_GGUF_TRUNCATED: non-UTF8 GGUF string (fix: fetch a larger range)"
            ))
        })
}

/// Dispatch table over `GGUFMetadataDtype` ids (upstream `_PARSERS`).
fn read_value(buf: &[u8], off: usize, type_id: u64) -> Result<(KvValue, usize)> {
    match type_id {
        0 => read_u8(buf, off).map(|(v, o)| (KvValue::Uint(v), o)),
        1 => Ok((KvValue::Int(take(buf, off, 1)?[0] as i8 as i64), off + 1)),
        2 => read_u16(buf, off).map(|(v, o)| (KvValue::Uint(v), o)),
        3 => Ok((
            KvValue::Int(i16::from_le_bytes(take(buf, off, 2)?.try_into().unwrap()) as i64),
            off + 2,
        )),
        4 => read_u32(buf, off).map(|(v, o)| (KvValue::Uint(v), o)),
        5 => Ok((
            KvValue::Int(i32::from_le_bytes(take(buf, off, 4)?.try_into().unwrap()) as i64),
            off + 4,
        )),
        6 => Ok((
            KvValue::Float(f32::from_le_bytes(take(buf, off, 4)?.try_into().unwrap()) as f64),
            off + 4,
        )),
        7 => Ok((KvValue::Bool(take(buf, off, 1)?[0] != 0), off + 1)),
        8 => read_string(buf, off).map(|(v, o)| (KvValue::Str(v), o)),
        9 => {
            let (vt, mut off) = read_u32(buf, off)?;
            let (len, o) = read_u64(buf, off)?;
            off = o;
            let mut items = Vec::new();
            for _ in 0..len {
                let (v, o) = read_value(buf, off, vt)?;
                off = o;
                items.push(v);
            }
            Ok((KvValue::Arr(items), off))
        }
        10 => read_u64(buf, off).map(|(v, o)| (KvValue::Uint(v), o)),
        11 => Ok((
            KvValue::Int(i64::from_le_bytes(take(buf, off, 8)?.try_into().unwrap())),
            off + 8,
        )),
        12 => Ok((
            KvValue::Float(f64::from_le_bytes(take(buf, off, 8)?.try_into().unwrap())),
            off + 8,
        )),
        other => Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_GGUF_DTYPE: metadata value type {other} unknown"
        ))),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GgufStat {
    pub dtypes: HashMap<String, DtypeStat>,
    pub param_count: u64,
    pub bytes_count: u64,
    /// Suffix-matched KV fields (`block_count`, `head_count_kv`, …) for the
    /// experimental cache estimate; `None` unless requested.
    pub kv_fields: HashMap<String, u64>,
    /// Per-block bytes (`blk.N`) + `shared` (embeddings/output/norms) for
    /// the partial-offload planner (P2). Names only; no tensor data.
    pub layer_bytes: HashMap<String, u64>,
}

/// Block key for a tensor: `blk.N` for llama-style blocks, else `shared`
/// (embeddings, output head, norms — also offloadable, counted first).
pub fn layer_key(tensor_name: &str) -> String {
    let mut parts = tensor_name.split('.');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("blk"), Some(n), _) if n.bytes().all(|b| b.is_ascii_digit()) => {
            format!("blk.{n}")
        }
        _ => "shared".to_string(),
    }
}

/// Planned partial offload for a VRAM budget (P2): shared tensors first,
/// then blocks in order, stopping before exceeding `budget_bytes`.
/// Returns (blocks fitting, total blocks, bytes placed).
pub fn plan_offload(stat: &GgufStat, budget_bytes: u64) -> (usize, usize, u64) {
    let mut blocks: Vec<(u64, u64)> = stat
        .layer_bytes
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("blk.")
                .and_then(|n| n.parse::<u64>().ok())
                .map(|n| (n, *v))
        })
        .collect();
    blocks.sort_unstable();
    let total_blocks = blocks.len();
    let shared = stat.layer_bytes.get("shared").copied().unwrap_or(0);
    if shared > budget_bytes {
        return (0, total_blocks, 0);
    }
    let mut placed = shared;
    let mut fit = 0;
    for (_, bytes) in blocks {
        if placed + bytes > budget_bytes {
            break;
        }
        placed += bytes;
        fit += 1;
    }
    (fit, total_blocks, placed)
}

/// Parse fetched GGUF bytes: magic → KV map → tensor walk (upstream
/// `parse_gguf_metadata`). Set `want_kv` for the experimental estimate.
pub fn parse(buf: &[u8], want_kv: bool) -> Result<GgufStat> {
    if buf.get(0..4) != Some(b"GGUF".as_slice()) {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_GGUF_MAGIC: not a GGUF file (fix: check the URL points at weight bytes)"
        )));
    }
    let (tensor_count, _) = read_u64(buf, 8)?;
    let (kv_count, mut off) = read_u64(buf, 16)?;
    let mut kv: HashMap<String, KvValue> = HashMap::new();
    for _ in 0..kv_count {
        let (key, o) = read_string(buf, off)?;
        off = o;
        let (vt, o) = read_u32(buf, off)?;
        off = o;
        let (value, o) = read_value(buf, off, vt)?;
        off = o;
        kv.insert(key, value);
    }
    let mut stat = GgufStat::default();
    if want_kv {
        // Suffix-matched so every family works (`llama.block_count` →
        // `block_count`), exactly like upstream.
        for ending in [
            "block_count",
            "head_count_kv",
            "head_count",
            "embedding_length",
            "context_length",
        ] {
            if let Some((_, v)) = kv.iter().find(|(k, _)| k.ends_with(ending)) {
                if let Some(n) = v.as_u64() {
                    stat.kv_fields.insert(ending.to_string(), n);
                }
            }
        }
    }
    for _ in 0..tensor_count {
        let (name, o) = read_string(buf, off)?;
        off = o;
        let (n_dims, o) = read_u32(buf, off)?;
        off = o;
        let mut params: u64 = 1;
        for _ in 0..n_dims {
            let (d, o) = read_u64(buf, off)?;
            off = o;
            params = params.saturating_mul(d);
        }
        let (tensor_type, o) = read_u32(buf, off)?;
        off = o;
        let (_, o) = read_u64(buf, off)?; // data offset
        off = o;
        let bytes = ((bits_per_weight(tensor_type as u32)? / 8.0) * params as f64) as u64;
        let entry = stat
            .dtypes
            .entry(dtype_name(tensor_type as u32).to_string())
            .or_default();
        entry.param_count = entry.param_count.saturating_add(params);
        entry.bytes_count = entry.bytes_count.saturating_add(bytes);
        stat.param_count = stat.param_count.saturating_add(params);
        stat.bytes_count = stat.bytes_count.saturating_add(bytes);
        let slot = stat.layer_bytes.entry(layer_key(&name)).or_default();
        *slot = slot.saturating_add(bytes);
    }
    Ok(stat)
}

/// Merge shard stats (upstream `merge_shards`).
pub fn merge(a: GgufStat, b: &GgufStat) -> GgufStat {
    let mut out = a;
    for (dtype, m) in &b.dtypes {
        let entry = out.dtypes.entry(dtype.clone()).or_default();
        entry.param_count = entry.param_count.saturating_add(m.param_count);
        entry.bytes_count = entry.bytes_count.saturating_add(m.bytes_count);
    }
    out.param_count = out.param_count.saturating_add(b.param_count);
    out.bytes_count = out.bytes_count.saturating_add(b.bytes_count);
    for (layer, bytes) in &b.layer_bytes {
        let slot = out.layer_bytes.entry(layer.clone()).or_default();
        *slot = slot.saturating_add(*bytes);
    }
    if out.kv_fields.is_empty() {
        out.kv_fields = b.kv_fields.clone();
    }
    out
}

/// `blocks × 2 × kv_heads × (embed/heads) × ctx × batch × dtype_bits/8`
/// (upstream `compute_gguf_kv_cache_size`).
pub fn kv_cache_size(
    fields: &HashMap<String, u64>,
    kv_cache_dtype: &str,
    context_len_override: Option<u64>,
    batch_size: u64,
) -> Result<u64> {
    if !GGUF_KV_DTYPES.contains(&kv_cache_dtype) {
        return Err(NexoraError::Other(anyhow::anyhow!(
            "E_MEM_KV_DTYPE: GGUF cache dtype {kv_cache_dtype} invalid (fix: one of {})",
            GGUF_KV_DTYPES.join(", ")
        )));
    }
    let get = |k: &str| {
        fields.get(k).copied().ok_or_else(|| {
            NexoraError::Other(anyhow::anyhow!(
                "E_MEM_KV: GGUF metadata lacks {k} (fix: shard 1 usually carries it — estimate the first shard)"
            ))
        })
    };
    let blocks = get("block_count")?;
    let kv_heads = get("head_count_kv")?;
    let heads = get("head_count")?;
    let embed = get("embedding_length")?;
    let ctx = context_len_override.or_else(|| fields.get("context_length").copied());
    let ctx = ctx.ok_or_else(|| {
        NexoraError::Other(anyhow::anyhow!(
            "E_MEM_KV: no context_length (fix: pass --max-model-len)"
        ))
    })?;
    let head_dim = embed / heads.max(1);
    let bits = bits_per_weight(kv_dtype_id(kv_cache_dtype)?)?;
    Ok((blocks as f64 * 2.0 * kv_heads as f64 * head_dim as f64 * ctx as f64 * batch_size.max(1) as f64 * bits / 8.0) as u64)
}

/// Fetch with doubling 1MB→100MB, retrying only truncation (upstream
/// `fetch_gguf_metadata`).
pub async fn fetch(url: &str, want_kv: bool) -> Result<GgufStat> {
    let mut size = INITIAL_FETCH_SIZE;
    loop {
        let buf = super::range_get(url, 0, size).await?;
        match parse(&buf, want_kv) {
            Ok(stat) => return Ok(stat),
            Err(e) => {
                let msg = e.to_string();
                let truncated = msg.contains("E_MEM_GGUF_TRUNCATED");
                if !truncated || size >= MAX_FETCH_SIZE {
                    if truncated {
                        return Err(NexoraError::Other(anyhow::anyhow!(
                            "E_MEM_GGUF_META: metadata exceeds 100MB (fix: file an issue — this model is extraordinary)"
                        )));
                    }
                    return Err(e);
                }
                size = (size * 2).min(MAX_FETCH_SIZE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32le(v: u32, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64le(v: u64, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn gguf_str(s: &str, buf: &mut Vec<u8>) {
        u64le(s.len() as u64, buf);
        buf.extend_from_slice(s.as_bytes());
    }
    fn kv_u32(key: &str, v: u32, buf: &mut Vec<u8>) {
        gguf_str(key, buf);
        u32le(4, buf); // UINT32
        u32le(v, buf);
    }

    /// Minimal file: 5 KV fields + one 4x4 F16 tensor (16 params, 32 bytes).
    fn tiny_file() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        u32le(3, &mut b); // version (skipped by parser)
        u64le(1, &mut b); // tensor_count
        u64le(5, &mut b); // metadata_kv_count
        kv_u32("llama.block_count", 2, &mut b);
        kv_u32("llama.head_count_kv", 2, &mut b);
        kv_u32("llama.head_count", 4, &mut b);
        kv_u32("llama.embedding_length", 8, &mut b);
        kv_u32("llama.context_length", 16, &mut b);
        gguf_str("w", &mut b);
        u32le(2, &mut b); // n_dimensions
        u64le(4, &mut b);
        u64le(4, &mut b);
        u32le(1, &mut b); // F16
        u64le(0, &mut b); // data offset
        b
    }

    #[test]
    fn tensor_walk_matches_hf_mem_math() {
        let stat = parse(&tiny_file(), true).unwrap();
        assert_eq!(stat.param_count, 16);
        assert_eq!(stat.bytes_count, 32);
        // 2 blocks x 2 x 2 kv-heads x (8/4) headdim x 16 ctx x 1 batch x 2B.
        let kv = kv_cache_size(&stat.kv_fields, "F16", None, 1).unwrap();
        assert_eq!(kv, 2 * 2 * 2 * 2 * 16 * 2);
    }

    #[test]
    fn magic_is_validated() {
        let mut bad = tiny_file();
        bad[0] = b'X';
        assert!(parse(&bad, false).is_err());
    }

    #[test]
    fn bits_table_spot_checks() {
        assert_eq!(bits_per_weight(12).unwrap(), 4.5); // Q4_K
        assert_eq!(bits_per_weight(1).unwrap(), 16.0); // F16
        assert!(bits_per_weight(4).is_err()); // reserved Q4_2
    }

    #[test]
    fn shards_merge_additively() {
        let a = parse(&tiny_file(), false).unwrap();
        let merged = merge(a.clone(), &a);
        assert_eq!(merged.param_count, 32);
        assert_eq!(merged.bytes_count, 64);
    }

    #[test]
    fn layer_keys_split_blocks_from_shared() {
        assert_eq!(layer_key("blk.3.attn_q.weight"), "blk.3");
        assert_eq!(layer_key("blk.12.mlp.weight"), "blk.12");
        assert_eq!(layer_key("token_embd.weight"), "shared");
        assert_eq!(layer_key("output.weight"), "shared");
        assert_eq!(layer_key("blk.x.foo"), "shared");
    }

    #[test]
    fn offload_plan_is_greedy_in_order() {
        let mut stat = GgufStat::default();
        stat.layer_bytes.insert("shared".into(), 100);
        stat.layer_bytes.insert("blk.0".into(), 50);
        stat.layer_bytes.insert("blk.1".into(), 60);
        stat.layer_bytes.insert("blk.2".into(), 70);
        assert_eq!(plan_offload(&stat, 200), (1, 3, 150));
        assert_eq!(plan_offload(&stat, 90), (0, 3, 0));
        assert_eq!(plan_offload(&stat, 10_000), (3, 3, 280));
    }
}
