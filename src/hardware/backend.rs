//! Hardware backends: `HardwareBackend` trait, CPU impl via `sysinfo`,
//! NVIDIA impl via `nvidia-smi` CSV parsing, plus the mandatory VRAM gate
//! that runs BEFORE every download/load (AGENTS.md rule 6).
//!
//! Never assumes NVIDIA forever — DirectML/AMD/Intel/Metal join as new
//! impls behind the same trait (docs/02-ARCHITECTURE.md §2.5).

use crate::core::{NexoraError, Result};
use serde::{Deserialize, Serialize};

/// Full machine snapshot for the dashboard (docs/07-HARDWARE.md §7.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_label: String,
    pub cpu_cores: usize,
    pub system_ram_mb: u64,
    pub gpu_label: Option<String>,
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub cuda_available: bool,
    pub driver_version: Option<String>,
    pub disk_free_bytes: u64,
    pub os: String,
    pub arch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub ram_total_mb: u64,
    pub ram_free_mb: u64,
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub cuda: bool,
    pub directml: bool,
    pub cpu_inference: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            cuda: false,
            directml: false,
            cpu_inference: true,
        }
    }
}

/// docs/02-ARCHITECTURE.md §2.5 — no NVIDIA hardcode at call sites.
pub trait HardwareBackend {
    fn detect() -> HardwareInfo;
    fn memory() -> MemoryInfo;
    fn capabilities() -> Capabilities;
}

/// Where the model will actually execute (docs/07-HARDWARE.md §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Gpu,
    Offload,
    Cpu,
}

/// Always-on baseline implemented with `sysinfo` (CPU/RAM/OS + disk).
pub struct CpuBackend;

impl HardwareBackend for CpuBackend {
    fn detect() -> HardwareInfo {
        use sysinfo::System;
        let mut sys = System::new_all();
        sys.refresh_all();
        let cpu_label = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Unknown CPU".into());
        let disk_free_bytes = sysinfo_free_disk_bytes();
        HardwareInfo {
            cpu_cores: sys.cpus().len(),
            system_ram_mb: sys.total_memory() / 1024 / 1024,
            cpu_label,
            gpu_label: None,
            vram_total_mb: None,
            vram_free_mb: None,
            cuda_available: false,
            driver_version: None,
            disk_free_bytes,
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }

    fn memory() -> MemoryInfo {
        use sysinfo::System;
        let mut sys = System::new_all();
        sys.refresh_memory();
        MemoryInfo {
            ram_total_mb: sys.total_memory() / 1024 / 1024,
            ram_free_mb: sys.available_memory() / 1024 / 1024,
            vram_total_mb: None,
            vram_free_mb: None,
        }
    }

    fn capabilities() -> Capabilities {
        Capabilities {
            cuda: false,
            directml: false,
            cpu_inference: true,
        }
    }
}

/// NVIDIA backend: shells out to `nvidia-smi` (child process) and parses
/// CSV; `None` when the tool/driver is absent (CPU-only machines).
pub struct NvidiaBackend;

impl NvidiaBackend {
    pub fn smi_snapshot() -> Option<(String, u64, u64, String)> {
        let out = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        parse_nvidia_smi_csv(&String::from_utf8_lossy(&out.stdout))
    }
}

impl HardwareBackend for NvidiaBackend {
    fn detect() -> HardwareInfo {
        let mut base = CpuBackend::detect();
        if let Some((gpu, total_mb, free_mb, driver)) = Self::smi_snapshot() {
            base.gpu_label = Some(gpu);
            base.vram_total_mb = Some(total_mb);
            base.vram_free_mb = Some(free_mb);
            base.cuda_available = true;
            base.driver_version = Some(driver);
        }
        base
    }

    fn memory() -> MemoryInfo {
        let mut mem = CpuBackend::memory();
        if let Some((_, total_mb, free_mb, _)) = Self::smi_snapshot() {
            mem.vram_total_mb = Some(total_mb);
            mem.vram_free_mb = Some(free_mb);
        }
        mem
    }

    fn capabilities() -> Capabilities {
        Capabilities {
            cuda: Self::smi_snapshot().is_some(),
            directml: false,
            cpu_inference: true,
        }
    }
}

/// Parse one `nvidia-smi --format=csv,noheader,nounits` line:
/// `NVIDIA GeForce GTX 1050, 4096, 3600, 537.13`.
pub fn parse_nvidia_smi_csv(csv: &str) -> Option<(String, u64, u64, String)> {
    let line = csv.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 4 {
        return None;
    }
    Some((
        parts[0].to_string(),
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
        parts[3].to_string(),
    ))
}

/// Mandatory gate before download/load: `Required -> Available -> fits?
/// GPU : offload -> still impossible? CPU mode` (docs/07 §7.3).
pub fn vram_fits(required_mb: u64, available_mb: Option<u64>) -> Result<()> {
    match available_mb {
        // No discrete GPU: CPU path handles it (possibly slowly); not an error.
        None => Ok(()),
        Some(free) if required_mb <= free => Ok(()),
        Some(free) => Err(NexoraError::VramShort {
            required_mb,
            available_mb: free,
        }),
    }
}

/// Pick GPU / offload / CPU placement for a load request.
pub fn select_execution_mode(required_mb: u64, free_mb: Option<u64>) -> ExecutionMode {
    match free_mb {
        Some(free) if required_mb <= free => ExecutionMode::Gpu,
        Some(free) if required_mb <= free + 8 * 1024 => ExecutionMode::Offload,
        _ => ExecutionMode::Cpu,
    }
}

fn sysinfo_free_disk_bytes() -> u64 {
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .map(|d| d.available_space())
        .max()
        .unwrap_or(0)
}
