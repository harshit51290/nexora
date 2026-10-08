//! Hardware detection + VRAM gating (docs/07-HARDWARE.md).

pub mod backend;

pub use backend::{
    select_execution_mode, vram_fits, Capabilities, CpuBackend, ExecutionMode, HardwareBackend,
    HardwareInfo, MemoryInfo, NvidiaBackend,
};
