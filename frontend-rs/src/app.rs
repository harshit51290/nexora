//! App shell: owns every signal, renders sidebar + active section.
//! `#[component]` is used only here (no params — zero Props risk); every
//! other render unit is a plain helper fn called during this render.

use dioxus::prelude::*;

use crate::components::{sidebar, SHELL};
use crate::pages::render_section;
use crate::state::{seed_models, Mode, Section, Ui};

#[component]
pub fn App() -> Element {
    let section = use_signal(|| Section::Home);
    let mode = use_signal(|| Mode::Beginner);
    let models = use_signal(seed_models);
    let hardware = use_signal(|| None);
    let hw_error = use_signal(|| None);
    let analysis = use_signal(|| None);
    let analysis_error = use_signal(|| None);
    let status_line = use_signal(String::new);

    let ui = Ui {
        section,
        mode,
        models,
        hardware,
        hw_error,
        analysis,
        analysis_error,
        status_line,
    };

    rsx! {
        div { style: SHELL,
            {sidebar(ui)}
            {render_section(ui)}
        }
    }
}
