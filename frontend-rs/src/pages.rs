//! One render fn per section (docs/11-UI-UX.md). Home is fully live
//! (real hardware detect + real URL parse/classify); the rest render real
//! seed/library state with execution behind TODO-WIRE-UI stubs.

use dioxus::prelude::*;

use crate::components::{
    analyze_bar, generate_form, hardware_card, model_card, BTN_GHOST, CARD, H1, MAIN, SUB,
};
use crate::state::{Mode, Section, Ui};

pub fn render_section(ui: Ui) -> Element {
    match ui.section() {
        Section::Home => page_home(ui),
        Section::Discover => page_simple(
            "Discover",
            "Model explorer with task/size/license/VRAM filters lands here (docs/05 §5.7).",
        ),
        Section::Models => page_models(ui),
        Section::Workflows => page_simple(
            "Workflows",
            "Node graphs (Prompt → LLM → Image → Upscaler) execute via the scheduler workflow runner.",
        ),
        Section::Generate => page_generate(ui),
        Section::Downloads => page_simple(
            "Downloads",
            "Pause/resume/cancel, speed + ETA, checksum verification (docs/08 §8.2).",
        ),
        Section::Environments => page_simple(
            "Environments",
            "Isolated per-runtime envs with pin records + rollback (docs/09 §9.1–9.2).",
        ),
        Section::Hardware => page_hardware(ui),
        Section::Runtime => page_simple(
            "Runtime",
            "Adapter states + env health + low-VRAM flags (docs/04 §4.3).",
        ),
        Section::Settings => page_simple(
            "Settings",
            "Data-root relocation, auto-unload toggle, HF token (OS store only).",
        ),
    }
}

fn page_simple(title: &str, body: &str) -> Element {
    rsx! {
        div { style: MAIN,
            h1 { style: H1, "{title}" }
            p { style: SUB, "{body}" }
        }
    }
}

fn page_home(ui: Ui) -> Element {
    let names: Vec<String> = ui.models().iter().map(|m| m.name.clone()).collect();
    rsx! {
        div { style: MAIN,
            h1 { style: H1, "Nexora" }
            p { style: SUB, "Paste a Hugging Face URL. Get a running model." }
            {analyze_bar(ui)}
            {hardware_card(ui)}
            div { style: CARD,
                div { style: "font-weight:bold;margin-bottom:8px", "Recently Used" }
                for n in names {
                    div { key: "{n}", "• {n}" }
                }
            }
        }
    }
}

fn page_models(ui: Ui) -> Element {
    let advanced = ui.mode() != Mode::Beginner;
    let models = ui.models().clone();
    rsx! {
        div { style: MAIN,
            h1 { style: H1, "My Models" }
            p { style: SUB, "All · Text · Image · Audio · Video · Vision · Installed · Favorites" }
            for m in &models {
                div { key: "{m.id}", {model_card(m, advanced)} }
            }
        }
    }
}

fn page_generate(_ui: Ui) -> Element {
    let mut task = use_signal(|| "text-to-image".to_string());
    let mut status = use_signal(String::new);
    rsx! {
        div { style: MAIN,
            h1 { style: H1, "Generate" }
            for t in ["text-generation", "text-to-image", "text-to-speech"] {
                button {
                    key: "{t}",
                    style: BTN_GHOST,
                    onclick: move |_| task.set(t.to_string()),
                    "{t}"
                }
            }
            {generate_form(&task(), status)}
        }
    }
}

fn page_hardware(ui: Ui) -> Element {
    rsx! {
        div { style: MAIN,
            h1 { style: H1, "Hardware" }
            p { style: SUB, "Profiles: <4GB ultra-low · 4–6 low · 6–12 medium · 12–24 high · 24GB+ extreme." }
            {hardware_card(ui)}
            div { style: SUB, "VRAM gate runs before every download/load (AGENTS.md rule 6)." }
        }
    }
}
