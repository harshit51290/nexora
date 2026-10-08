//! UI state: navigation sections, modes, seed library, analysis result.
//! All signals are created in `App` and passed down as args (plain helper
//! fns — no cross-component hook or Props machinery).

use dioxus::prelude::*;
use nexora::hardware::HardwareInfo;

/// The 10 app sections (docs/11-UI-UX.md §11.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Home,
    Discover,
    Models,
    Workflows,
    Generate,
    Downloads,
    Environments,
    Hardware,
    Runtime,
    Settings,
}

impl Section {
    pub fn all() -> [Section; 10] {
        use Section::*;
        [
            Home,
            Discover,
            Models,
            Workflows,
            Generate,
            Downloads,
            Environments,
            Hardware,
            Runtime,
            Settings,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            Section::Home => "Home",
            Section::Discover => "Discover",
            Section::Models => "Models",
            Section::Workflows => "Workflows",
            Section::Generate => "Generate",
            Section::Downloads => "Downloads",
            Section::Environments => "Environments",
            Section::Hardware => "Hardware",
            Section::Runtime => "Runtime",
            Section::Settings => "Settings",
        }
    }
}

/// Beginner (default) hides runtime detail; Advanced/Developer expose it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Beginner,
    Advanced,
    Developer,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Beginner => "Beginner",
            Mode::Advanced => "Advanced",
            Mode::Developer => "Developer",
        }
    }
}

/// Library row (mirrors the TS `models` store seeds).
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub task: String,
    pub runtime: String,
    pub vram_note: String,
    pub trust: String,
}

pub fn seed_models() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            id: "qwen".into(),
            name: "Qwen".into(),
            task: "text-generation".into(),
            runtime: "llama.cpp".into(),
            vram_note: "VRAM ~4.8 GB (Q4)".into(),
            trust: "Trusted".into(),
        },
        ModelEntry {
            id: "sd15".into(),
            name: "Stable Diffusion 1.5".into(),
            task: "text-to-image".into(),
            runtime: "diffusers".into(),
            vram_note: "VRAM ~3.5 GB (FP16)".into(),
            trust: "Trusted".into(),
        },
        ModelEntry {
            id: "whisper".into(),
            name: "Whisper".into(),
            task: "speech-to-text".into(),
            runtime: "audio".into(),
            vram_note: "VRAM ~1 GB".into(),
            trust: "Trusted".into(),
        },
        ModelEntry {
            id: "kokoro".into(),
            name: "Kokoro".into(),
            task: "text-to-speech".into(),
            runtime: "audio".into(),
            vram_note: "VRAM ~1 GB".into(),
            trust: "Community".into(),
        },
    ]
}

/// Analyzer verdict for a pasted HF URL.
#[derive(Debug, Clone)]
pub struct Analyzed {
    pub id: String,
    pub task: String,
    pub runtime: String,
    pub notes: String,
}

/// Everything `App` owns. Plain struct of signals, passed by value
/// (`Signal` is `Copy`) into render helpers.
#[derive(Clone, Copy)]
pub struct Ui {
    pub section: Signal<Section>,
    pub mode: Signal<Mode>,
    pub models: Signal<Vec<ModelEntry>>,
    pub hardware: Signal<Option<HardwareInfo>>,
    pub hw_error: Signal<Option<String>>,
    pub analysis: Signal<Option<Analyzed>>,
    pub analysis_error: Signal<Option<String>>,
    pub status_line: Signal<String>,
}

pub fn fmt_gb(mb: u64) -> String {
    format!("{:.1} GB", mb as f64 / 1024.0)
}

pub fn fmt_vram(mb: Option<u64>) -> String {
    mb.map(fmt_gb).unwrap_or_else(|| "n/a".into())
}
