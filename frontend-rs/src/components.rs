//! Render helpers (plain fns — all signals are created in `App` or locally
//! here as custom-hook style locals, then passed by value).

use dioxus::prelude::*;

use crate::backend;
use crate::state::{fmt_vram, Analyzed, Mode, ModelEntry, Section, Ui};

pub const SHELL: &str =
    "display:flex;min-height:100vh;background:#0b0e14;color:#e6e9f0;font-family:sans-serif";
pub const SIDEBAR: &str =
    "width:220px;padding:16px;background:#11151d;border-right:1px solid #232a38";
pub const MAIN: &str = "flex:1;padding:24px;max-width:960px";
pub const CARD: &str =
    "background:#141a25;border:1px solid #232a38;border-radius:10px;padding:16px;margin:12px 0";
pub const BTN: &str = "background:#2f81f7;color:white;border:none;border-radius:8px;padding:8px 16px;cursor:pointer;margin:4px 4px 4px 0";
pub const BTN_GHOST: &str = "background:transparent;color:#9fb3c8;border:1px solid #2b3648;border-radius:8px;padding:8px 16px;cursor:pointer;margin:4px 4px 4px 0";
pub const INPUT: &str = "background:#0b0e14;color:#e6e9f0;border:1px solid #2b3648;border-radius:8px;padding:8px 12px;width:100%;box-sizing:border-box;margin:4px 0";
pub const LABEL: &str = "display:block;color:#9fb3c8;font-size:12px;margin:8px 0 2px";
pub const H1: &str = "font-size:24px;margin:0 0 4px";
pub const SUB: &str = "color:#9fb3c8;margin:0 0 16px";
pub const BAR_BG: &str = "background:#232a38;border-radius:6px;height:10px;margin:2px 0 8px";
pub const NAV_ACTIVE: &str = "display:block;width:100%;text-align:left;background:#1c2534;color:white;border:1px solid #2f81f7;border-radius:8px;padding:8px 12px;margin:2px 0;cursor:pointer";
pub const NAV: &str = "display:block;width:100%;text-align:left;background:transparent;color:#9fb3c8;border:1px solid transparent;border-radius:8px;padding:8px 12px;margin:2px 0;cursor:pointer";

pub fn sidebar(ui: Ui) -> Element {
    let current = ui.section();
    rsx! {
        div { style: SIDEBAR,
            div { style: "font-size:20px;font-weight:bold;margin-bottom:4px", "Nexora" }
            div { style: SUB, "Local AI runtime manager" }
            for s in Section::all() {
                button {
                    key: "{s.label()}",
                    style: if s == current { NAV_ACTIVE } else { NAV },
                    onclick: move |_| ui.section.set(s),
                    "{s.label()}"
                }
            }
            div { style: "margin-top:16px;color:#9fb3c8;font-size:12px", "Mode" }
            for m in [Mode::Beginner, Mode::Advanced, Mode::Developer] {
                button {
                    key: "{m.label()}",
                    style: if m == ui.mode() { NAV_ACTIVE } else { NAV },
                    onclick: move |_| ui.mode.set(m),
                    "{m.label()}"
                }
            }
        }
    }
}

pub fn error_banner(err: Signal<Option<String>>) -> Element {
    match err().clone() {
        None => rsx! {},
        Some(msg) => rsx! {
            div { style: "background:#3d1a1a;border:1px solid #8c3535;border-radius:10px;padding:12px 16px;margin:12px 0",
                div { style: "font-weight:bold;color:#ff9d9d", "Something needs attention" }
                div { "{msg}" }
                button { style: BTN_GHOST, onclick: move |_| err.set(None), "Dismiss" }
            }
        },
    }
}

pub fn hardware_card(ui: Ui) -> Element {
    let button_label = if ui.hardware().is_some() {
        "Re-detect"
    } else {
        "Detect hardware"
    };
    rsx! {
        div { style: CARD,
            div { style: "font-weight:bold;margin-bottom:8px", "Your Hardware" }
            {
                match ui.hardware().clone() {
                    None => rsx! { div { style: SUB, "Not detected yet." } },
                    Some(h) => rsx! {
                        div { "GPU: {h.gpu_label.clone().unwrap_or_else(|| \"CPU only\".into())} ({fmt_vram(h.vram_total_mb)})" }
                        div { "RAM: {crate::state::fmt_gb(h.system_ram_mb)}" }
                        div { "CPU: {h.cpu_label} ({h.cpu_cores} cores)" }
                        div { "CUDA: {h.cuda_available} · {h.os}/{h.arch} · disk free {h.disk_free_bytes / 1024 / 1024 / 1024} GB" }
                    },
                }
            }
            {error_banner(ui.hw_error)}
            button {
                style: BTN,
                onclick: move |_| {
                    ui.hardware.set(Some(backend::detect_hardware()));
                    ui.hw_error.set(None);
                },
                "{button_label}"
            }
        }
    }
}

pub fn analyze_bar(ui: Ui) -> Element {
    let mut url = use_signal(String::new);
    rsx! {
        div { style: CARD,
            div { style: "font-weight:bold;margin-bottom:4px", "Run AI models locally." }
            div { style: SUB, "Paste a Hugging Face model URL" }
            input {
                r#type: "text",
                style: INPUT,
                placeholder: "https://huggingface.co/...",
                value: "{url}",
                oninput: move |e| url.set(e.value()),
            }
            button {
                style: BTN,
                onclick: move |_| {
                    match backend::analyze_url(&url()) {
                        Ok(a) => {
                            ui.analysis.set(Some(a));
                            ui.analysis_error.set(None);
                        }
                        Err(msg) => {
                            ui.analysis.set(None);
                            ui.analysis_error.set(Some(msg));
                        }
                    }
                },
                "Analyze Model"
            }
            {error_banner(ui.analysis_error)}
            {
                match ui.analysis().clone() {
                    None => rsx! {},
                    Some(a) => rsx! {
                        div { style: "margin-top:12px",
                            div { "Model: {a.id}" }
                            div { "Task: {a.task} · Runtime: {a.runtime}" }
                            div { style: SUB, "{a.notes}" }
                        }
                    },
                }
            }
        }
    }
}

fn bar(label: &str, pct: u8) -> Element {
    rsx! {
        div {
            div { style: "font-size:12px;color:#9fb3c8", "{label} {pct}%" }
            div { style: BAR_BG,
                div { style: "background:#2f81f7;border-radius:6px;height:10px;width:{pct}%", "" }
            }
        }
    }
}

pub fn compat_bars(runtime: &str) -> Element {
    let (overall, gpu) = match runtime {
        "llama.cpp" => (95, 85),
        "diffusers" => (90, 75),
        "transformers" => (92, 80),
        "audio" => (93, 90),
        _ => (50, 40),
    };
    rsx! {
        div {
            {bar("Compatibility", overall)}
            {bar("GPU", gpu)}
            {bar("RAM", 100)}
            {bar("Runtime", 100)}
        }
    }
}

pub fn model_card(m: &ModelEntry, advanced: bool) -> Element {
    rsx! {
        div { style: CARD,
            div { style: "font-weight:bold;font-size:16px", "{m.name}" }
            div { style: SUB, "{m.task} · {m.vram_note} · {m.trust}" }
            if advanced {
                div { style: SUB, "Runtime: {m.runtime}" }
            }
            {compat_bars(&m.runtime)}
            button { style: BTN, onclick: move |_| {}, "Run" }
            button { style: BTN_GHOST, onclick: move |_| {}, "Settings" }
        }
    }
}

pub fn generate_form(task: &str, status: Signal<String>) -> Element {
    let mut prompt = use_signal(String::new);
    let mut extra = use_signal(String::new);
    let fields = match task {
        t if t.contains("image") => vec![
            ("Prompt", "cyberpunk warrior"),
            ("Negative prompt", "blurry, low quality"),
            ("Steps", "25"),
            ("CFG", "7"),
        ],
        t if t.contains("speech") => vec![
            ("Text", "Hello from Nexora"),
            ("Voice", "default"),
            ("Speed", "1.0"),
        ],
        _ => vec![
            ("Prompt", "Explain VRAM in one paragraph"),
            ("Temperature", "0.7"),
            ("Max tokens", "2048"),
        ],
    };
    rsx! {
        div { style: CARD,
            div { style: "font-weight:bold;margin-bottom:8px", "Generate · {task}" }
            for (label, hint) in fields {
                div {
                    key: "{label}",
                    label { style: LABEL, "{label}" }
                    input {
                        r#type: "text",
                        style: INPUT,
                        placeholder: "{hint}",
                        oninput: move |e| {
                            if label == "Prompt" || label == "Text" {
                                prompt.set(e.value());
                            } else {
                                extra.set(e.value());
                            }
                        },
                    }
                }
            }
            button {
                style: BTN,
                onclick: move |_| {
                    let _ = (prompt(), extra());
                    status.set(backend::generate_not_wired());
                },
                "Generate"
            }
            if !status().is_empty() {
                div { style: SUB, "{status()}" }
            }
        }
    }
}

pub fn log_viewer(lines: &[&str]) -> Element {
    rsx! {
        div { style: CARD,
            div { style: "font-weight:bold;margin-bottom:8px", "Logs" }
            pre { style: "background:#0b0e14;border-radius:8px;padding:12px;overflow:auto;font-size:12px",
                for (i, line) in lines.iter().enumerate() {
                    div { key: "{i}", "{line}" }
                }
            }
        }
    }
}

pub fn trust_badge(trust: &str) -> String {
    match trust {
        "Trusted" => "✓ Trusted".into(),
        "Community" => "• Community".into(),
        "Unverified" => "⚠ Unverified — custom code, review before run".into(),
        _ => "⛔ Blocked".into(),
    }
}

pub fn _compat_for_analysis(_a: &Analyzed) -> Element {
    rsx! {}
}
