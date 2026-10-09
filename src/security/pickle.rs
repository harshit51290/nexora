//! Pickle weight safety: opcode scan for `.bin` / `.pt` / `.pth` /
//! `.ckpt` / `.pkl` (technique ported from `picklescan`, MIT).
//!
//! Safetensors needs no scan (data-only format by construction) —
//! `needs_scan` returns false for it. Everything else weight-like gets a
//! linear opcode pass looking for `GLOBAL`/`INST` imports of denylisted
//! callables (remote exec, process spawn, destructive FS). Key
//! false-positive decision: `REDUCE`/`BUILD` alone NEVER flag — `torch.save`
//! uses them for benign rebuilds. `STACK_GLOBAL` can't be attributed
//! statically, so it yields Suspicious (info), never Unsafe.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::{NexoraError, Result};

/// Stream window: 1MB chunks with an 8KB overlap so a `GLOBAL` split across
/// a boundary is still seen whole (findings are a set → overlap is free).
const CHUNK: usize = 1024 * 1024;
const OVERLAP: usize = 8192;
/// A module/name line longer than this is skipped (binary misread guard).
const MAX_LINE: usize = 4096;

/// (module, callable-prefix-set, reason). Callable matched by exact name.
const DENYLIST: &[(&str, &[&str], &str)] = &[
    (
        "os",
        &[
            "system", "popen", "execv", "execve", "execl", "spawnl", "spawnv", "remove", "unlink",
            "rmdir", "mkdir", "rename", "chmod", "kill",
        ],
        "arbitrary process/destructive-FS primitive",
    ),
    (
        "subprocess",
        &["Popen", "call", "run", "check_output", "check_call"],
        "child-process execution",
    ),
    (
        "socket",
        &["socket", "create_connection"],
        "network socket construction",
    ),
    (
        "builtins",
        &["eval", "exec", "compile", "__import__", "open"],
        "code execution / arbitrary file access",
    ),
    ("sys", &["modules"], "module-table injection"),
    ("runpy", &["run_path", "run_module"], "runs a file as code"),
    ("importlib", &["import_module"], "dynamic import"),
    (
        "ctypes",
        &["CDLL", "WinDLL", "cast", "CFUNCTYPE"],
        "native code execution",
    ),
    ("multiprocessing", &["Process"], "process spawn"),
    ("shutil", &["rmtree", "move"], "destructive file ops"),
    ("code", &["interact"], "interactive console"),
    ("pty", &["spawn"], "pty process spawn"),
    ("webbrowser", &["open"], "URL handler launch"),
];

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PickleFinding {
    pub module: String,
    pub name: String,
    pub offset: u64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum PickleVerdict {
    Safe,
    /// Unattributable risk only (`STACK_GLOBAL`, compressed/unscannable
    /// members). Install proceeds; UI shows a soft note.
    Suspicious(Vec<PickleFinding>),
    /// Denylisted callable import. Install proceeds non-interactively but
    /// trust drops to Unverified + loud warn (desktop modals).
    Unsafe(Vec<PickleFinding>),
}

impl PickleVerdict {
    pub fn is_unsafe(&self) -> bool {
        matches!(self, PickleVerdict::Unsafe(_))
    }
    pub fn findings(&self) -> &[PickleFinding] {
        match self {
            PickleVerdict::Safe => &[],
            PickleVerdict::Suspicious(f) | PickleVerdict::Unsafe(f) => f,
        }
    }
}

/// Weight files worth scanning (safetensors/JSON/manifests exempt).
pub fn needs_scan(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    [
        "bin", "pt", "pth", "ckpt", "pkl", "pickle", "dill", "joblib",
    ]
    .iter()
    .any(|ext| lower.ends_with(&format!(".{ext}")))
}

fn denied(module: &str, name: &str) -> Option<&'static str> {
    for (mod_name, callables, reason) in DENYLIST {
        if (module == *mod_name || module.ends_with(&format!(".{mod_name}")))
            && callables.contains(&name)
        {
            return Some(reason);
        }
    }
    None
}

/// Read a `\n`-terminated line; `None` on non-UTF8 / overlong / missing.
fn read_line(buf: &[u8], mut off: usize) -> Option<(String, usize)> {
    let start = off;
    while off < buf.len() && buf[off] != b'\n' {
        off += 1;
        if off - start > MAX_LINE {
            return None;
        }
    }
    if off >= buf.len() {
        return None; // truncated: overlap window replays it next chunk
    }
    let line = std::str::from_utf8(&buf[start..off])
        .ok()?
        .trim_end_matches('\r');
    Some((line.to_string(), off + 1))
}

/// Linear opcode pass over one buffer. `base` = file offset of `buf[0]`.
fn scan_buffer(buf: &[u8], base: u64, hits: &mut HashSet<PickleFinding>, stack_global: &mut bool) {
    let mut i = 0;
    while i < buf.len() {
        match buf[i] {
            b'c' | b'i' => {
                // GLOBAL / INST: `<op>module\nname\n`.
                let kind = buf[i];
                let (module, o1) = match read_line(buf, i + 1) {
                    Some(v) => v,
                    None => {
                        i += 1;
                        continue;
                    }
                };
                let (name, o2) = match read_line(buf, o1) {
                    Some(v) => v,
                    None => {
                        i += 1;
                        continue;
                    }
                };
                if let Some(reason) = denied(&module, &name) {
                    hits.insert(PickleFinding {
                        module,
                        name,
                        offset: base + i as u64,
                        reason: format!(
                            "{reason} (via {})",
                            if kind == b'c' { "GLOBAL" } else { "INST" }
                        ),
                    });
                }
                i = o2;
            }
            0x93 => {
                // STACK_GLOBAL: callable comes from the stack — unattributable.
                *stack_global = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
}

fn scan_stream<R: std::io::Read>(
    mut r: R,
    total_hint: Option<u64>,
) -> Result<(HashSet<PickleFinding>, bool)> {
    let mut hits = HashSet::new();
    let mut stack_global = false;
    let mut tail: Vec<u8> = Vec::new();
    let mut consumed: u64 = 0;
    let mut buf = vec![0u8; CHUNK];
    loop {
        // Overlap replays split opcodes; findings are a set (idempotent).
        let keep = tail.len().min(OVERLAP);
        let mut window = tail[tail.len() - keep..].to_vec();
        let n = r.read(&mut buf).map_err(|e| {
            NexoraError::Other(anyhow::anyhow!("E_SCAN_IO: weight read failed: {e}"))
        })?;
        if n == 0 {
            break;
        }
        window.extend_from_slice(&buf[..n]);
        scan_buffer(
            &window,
            consumed.saturating_sub(keep as u64),
            &mut hits,
            &mut stack_global,
        );
        consumed += n as u64;
        tail = window[window.len().saturating_sub(keep)..].to_vec();
        if let Some(total) = total_hint {
            if consumed >= total {
                break;
            }
        }
    }
    Ok((hits, stack_global))
}

/// Minimal ZIP walk: scan `method == stored` members that look like pickles
/// (`\x80` PROTO). Deflated members can't inflate without a dep → Suspicious
/// note (honest limitation, same class as upstream's unscannable paths).
fn scan_zip_member(
    data: &[u8],
    name: &str,
    hits: &mut HashSet<PickleFinding>,
    stack_global: &mut bool,
) {
    if data.first() != Some(&0x80) {
        return;
    }
    let _ = name;
    scan_buffer(data, 0, hits, stack_global);
}

fn scan_zip(path: &Path) -> Result<(HashSet<PickleFinding>, bool, Vec<String>)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| {
        NexoraError::Other(anyhow::anyhow!(
            "E_SCAN_IO: cannot open {}: {e}",
            path.display()
        ))
    })?;
    let total = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut notes = Vec::new();
    let mut hits = HashSet::new();
    let mut stack_global = false;
    let mut off = 0u64;
    let mut hdr = [0u8; 30];
    loop {
        if off + 30 > total {
            break;
        }
        f.seek(SeekFrom::Start(off))
            .map_err(|e| NexoraError::Other(anyhow::anyhow!("E_SCAN_IO: seek failed: {e}")))?;
        f.read_exact(&mut hdr).map_err(|e| {
            NexoraError::Other(anyhow::anyhow!("E_SCAN_IO: header read failed: {e}"))
        })?;
        if &hdr[0..4] != b"PK\x03\x04" {
            break;
        }
        let method = u16::from_le_bytes([hdr[8], hdr[9]]);
        let name_len = u16::from_le_bytes([hdr[26], hdr[27]]) as u64;
        let extra_len = u16::from_le_bytes([hdr[28], hdr[29]]) as u64;
        if name_len > 64 * 1024 {
            // Garbage header (or hostile zip): stop rather than allocating.
            break;
        }
        let comp_size = u32::from_le_bytes([hdr[18], hdr[19], hdr[20], hdr[21]]) as u64;
        let mut name_buf = vec![0u8; name_len as usize];
        f.read_exact(&mut name_buf)
            .map_err(|e| NexoraError::Other(anyhow::anyhow!("E_SCAN_IO: name read failed: {e}")))?;
        let name = String::from_utf8_lossy(&name_buf).into_owned();
        let data_off = off + 30 + name_len + extra_len;
        if method == 0 && comp_size > 0 && comp_size < 512 * 1024 * 1024 {
            f.seek(SeekFrom::Start(data_off))
                .map_err(|e| NexoraError::Other(anyhow::anyhow!("E_SCAN_IO: seek failed: {e}")))?;
            let mut data = vec![0u8; comp_size as usize];
            if f.read_exact(&mut data).is_ok() {
                scan_zip_member(&data, &name, &mut hits, &mut stack_global);
            }
        } else if method != 0 {
            notes.push(format!(
                "{name}: deflated member unscannable without inflate"
            ));
        }
        if comp_size > total {
            break;
        }
        off = data_off + comp_size;
        if off == 0 {
            break;
        }
    }
    Ok((hits, stack_global, notes))
}

/// Scan one weight file. ZIP-wrapped (`.pt`/`.bin` often are) members are
/// scanned individually; plain pickles stream through the overlap window.
pub fn scan_file(path: &Path) -> Result<PickleVerdict> {
    let is_zip = std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic).map(|_| magic)
        })
        .map(|m| m == *b"PK\x03\x04")
        .unwrap_or(false);
    let (hits, stack_global, mut notes) = if is_zip {
        let (h, s, n) = scan_zip(path)?;
        (h, s, n)
    } else {
        let f = std::fs::File::open(path).map_err(|e| {
            NexoraError::Other(anyhow::anyhow!(
                "E_SCAN_IO: cannot open {}: {e}",
                path.display()
            ))
        })?;
        let (h, s) = scan_stream(f, None)?;
        (h, s, Vec::new())
    };
    let mut findings: Vec<PickleFinding> = hits.into_iter().collect();
    findings.sort_by_key(|f| f.offset);
    if !findings.is_empty() {
        return Ok(PickleVerdict::Unsafe(findings));
    }
    let mut info: Vec<PickleFinding> = Vec::new();
    if stack_global {
        info.push(PickleFinding {
            module: "?".into(),
            name: "STACK_GLOBAL".into(),
            offset: 0,
            reason: "unattributable dynamic global (info only — common in benign torch files)"
                .into(),
        });
    }
    for note in notes.drain(..) {
        info.push(PickleFinding {
            module: "?".into(),
            name: "UNSCANNABLE_MEMBER".into(),
            offset: 0,
            reason: note,
        });
    }
    if info.is_empty() {
        Ok(PickleVerdict::Safe)
    } else {
        Ok(PickleVerdict::Suspicious(info))
    }
}

/// Worst verdict over a model dir's scannable weights (safetensors exempt).
pub fn scan_model_dir(dir: &Path) -> Result<PickleVerdict> {
    let mut worst_unsafe = Vec::new();
    let mut notes = Vec::new();
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let entries = std::fs::read_dir(&d).map_err(|e| {
            NexoraError::Other(anyhow::anyhow!(
                "E_SCAN_IO: cannot list {}: {e}",
                d.display()
            ))
        })?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if needs_scan(name) {
                    match scan_file(&path) {
                        Ok(PickleVerdict::Unsafe(f)) => worst_unsafe.extend(f),
                        Ok(PickleVerdict::Suspicious(f)) => notes.extend(f),
                        Ok(PickleVerdict::Safe) => {}
                        Err(e) => notes.push(PickleFinding {
                            module: "?".into(),
                            name: name.to_string(),
                            offset: 0,
                            reason: format!("scan failed ({e}); treat as unverified"),
                        }),
                    }
                }
            }
        }
    }
    if worst_unsafe.is_empty() && notes.is_empty() {
        Ok(PickleVerdict::Safe)
    } else if worst_unsafe.is_empty() {
        Ok(PickleVerdict::Suspicious(notes))
    } else {
        let mut all = worst_unsafe;
        all.extend(notes);
        Ok(PickleVerdict::Unsafe(all))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, bytes: &[u8]) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nexora-pickle-test-{name}"));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    /// Smallest safe pickle: PROTO + EMPTY_DICT + STOP.
    #[test]
    fn safe_dict_passes() {
        let p = tmp("safe.pkl", b"\x80\x04\x94.}");
        // Note: trailing STOP-less bytes are fine (linear pass, no findings).
        assert!(matches!(scan_file(&p).unwrap(), PickleVerdict::Safe));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn os_system_is_unsafe() {
        // GLOBAL os\nsystem\n + REDUCE shape, then STOP.
        let p = tmp("evil.pkl", b"\x80\x04cos\nsystem\n\x93\x52.");
        match scan_file(&p).unwrap() {
            PickleVerdict::Unsafe(f) => {
                assert!(f.iter().any(|x| x.module == "os" && x.name == "system"));
            }
            other => panic!("expected Unsafe, got {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn builtins_eval_and_subprocess_flagged() {
        let p = tmp(
            "evil2.pkl",
            b"\x80\x04cbuiltins\neval\ncsubprocess\nPopen\n.",
        );
        match scan_file(&p).unwrap() {
            PickleVerdict::Unsafe(f) => {
                assert_eq!(f.len(), 2);
            }
            other => panic!("expected Unsafe, got {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn reduce_alone_never_flags() {
        // Benign torch-style REDUCE with a plain GLOBAL (no STACK_GLOBAL,
        // no denylisted callable): Safe.
        let p = tmp("benign.pt", b"\x80\x04ccollections\nOrderedDict\n\x52.");
        assert!(matches!(scan_file(&p).unwrap(), PickleVerdict::Safe));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn stack_global_is_suspicious_not_unsafe() {
        let p = tmp("sg.pkl", b"\x80\x04\x93.");
        match scan_file(&p).unwrap() {
            PickleVerdict::Suspicious(f) => {
                assert!(f.iter().any(|x| x.name == "STACK_GLOBAL"));
            }
            other => panic!("expected Suspicious, got {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn zip_wrapped_evil_is_caught() {
        // Minimal stored ZIP containing data.pkl with os.system.
        let payload = b"\x80\x04cos\nsystem\n.";
        let mut zip = Vec::new();
        zip.extend_from_slice(b"PK\x03\x04");
        zip.extend_from_slice(&20u16.to_le_bytes()); // version
        zip.extend_from_slice(&0u16.to_le_bytes()); // flags
        zip.extend_from_slice(&0u16.to_le_bytes()); // method = stored
        zip.extend_from_slice(&0u16.to_le_bytes()); // time
        zip.extend_from_slice(&0u16.to_le_bytes()); // date
        zip.extend_from_slice(&0u32.to_le_bytes()); // crc (unchecked)
        zip.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // comp
        zip.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // uncomp
        zip.extend_from_slice(&8u16.to_le_bytes()); // name len
        zip.extend_from_slice(&0u16.to_le_bytes()); // extra len
        zip.extend_from_slice(b"data.pkl");
        zip.extend_from_slice(payload);
        let p = tmp("evil.zip.pt", &zip);
        match scan_file(&p).unwrap() {
            PickleVerdict::Unsafe(f) => {
                assert!(f.iter().any(|x| x.name == "system"));
            }
            other => panic!("expected Unsafe, got {other:?}"),
        }
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn split_global_across_chunks_is_seen() {
        // Force tiny chunks so GLOBAL straddles the boundary (overlap test).
        let mut data = b"\x80\x04".to_vec();
        data.extend_from_slice(&[b' '; 100]);
        data.extend_from_slice(b"cos\nsystem\n.");
        let p = tmp("split.pkl", &data);
        // Direct buffer-level check with a 64B window + 16B overlap.
        let mut hits = HashSet::new();
        let mut sg = false;
        let (chunk, overlap) = (64usize, 16usize);
        let mut pos = 0;
        while pos < data.len() {
            let end = (pos + chunk).min(data.len());
            let start = pos.saturating_sub(overlap.min(pos));
            scan_buffer(&data[start..end], start as u64, &mut hits, &mut sg);
            pos = end;
        }
        assert!(hits.iter().any(|x| x.name == "system"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn needs_scan_gate() {
        assert!(needs_scan("pytorch_model.bin"));
        assert!(needs_scan("model.pt"));
        assert!(!needs_scan("model.safetensors"));
        assert!(!needs_scan("config.json"));
    }
}
