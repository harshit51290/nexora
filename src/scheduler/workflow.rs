//! Node-graph workflow runner (docs/10 §10.5).
//!
//! Executes templates from `workflows/templates/*.json`:
//! validate the DAG, then run nodes in topological order, each node as
//! sequential `prepare -> run`, passing predecessor outputs as the next node's
//! inputs. Every failure is a [`WorkflowError`] carrying the failing
//! `node_id`, a machine-readable `code`, and a human `hint` (AGENTS.md rule 7).
//!
//! Orchestration-only (AGENTS.md rule 1): the default executor
//! ([`stub_execute`]) records what *would* run so graphs, ordering, and
//! input-passing are testable without models. Production executors dispatch
//! each node to the matching runtime adapter (`prepare -> run` over a
//! supervised child process) and are injected via [`Workflow::run_with`].

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Node types shipped in `workflows/templates/*.json` (docs/10 §10.5).
pub const KNOWN_NODE_TYPES: &[&str] = &[
    "prompt-input",
    "text-input",
    "image-input",
    "audio-input",
    "llm",
    "prompt-enhance",
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
    "upscaler",
    "tts",
    "stt",
    "save-output",
];

/// One graph node: `{ "id": ..., "type": ..., "params": {...} }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// One directed edge: `{ "from": ..., "to": ... }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEdge {
    pub from: String,
    pub to: String,
}

/// Parsed `workflows/templates/*.json` document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub nodes: Vec<WorkflowNode>,
    #[serde(default)]
    pub edges: Vec<WorkflowEdge>,
}

/// Coded workflow failure, always naming the node at fault when known.
#[derive(Debug, Clone)]
pub struct WorkflowError {
    pub node: Option<String>,
    pub code: &'static str,
    pub message: String,
    pub hint: String,
}

impl WorkflowError {
    pub fn new(
        node: Option<&str>,
        code: &'static str,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self {
            node: node.map(|s| s.to_string()),
            code,
            message: message.into(),
            hint: hint.into(),
        }
    }

    fn structural(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            None,
            code,
            message,
            "Fix the workflow JSON (or pick another template); no model or runtime is involved.",
        )
    }
}

impl std::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.node {
            Some(n) => write!(f, "[{}] node '{n}': {} Fix: {}", self.code, self.message, self.hint),
            None => write!(f, "[{}] {} Fix: {}", self.code, self.message, self.hint),
        }
    }
}

impl std::error::Error for WorkflowError {}

/// Executor for one node: `prepare` already validated the type; `input` is
/// `{ "inputs": { <pred-id>: <pred-output> }, "params": { ...node params } }`.
/// Return the node's output JSON, which is passed on to successors.
pub type NodeFn =
    dyn Fn(&WorkflowNode, &serde_json::Value) -> Result<serde_json::Value, NodeFailure>;

/// Executor failure: machine code + detail for the failing node.
#[derive(Debug, Clone)]
pub struct NodeFailure {
    pub code: String,
    pub message: String,
}

impl NodeFailure {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Per-node outputs in execution order.
#[derive(Debug, Clone, Default)]
pub struct WorkflowRun {
    /// `(node_id, output)` in the order nodes executed.
    pub steps: Vec<(String, serde_json::Value)>,
}

impl WorkflowRun {
    pub fn output_of(&self, node_id: &str) -> Option<&serde_json::Value> {
        self.steps
            .iter()
            .find(|(id, _)| id == node_id)
            .map(|(_, out)| out)
    }
}

impl WorkflowTemplate {
    /// Parse + structurally validate a template document.
    pub fn parse(json: &str) -> Result<Self, WorkflowError> {
        let tpl: Self = serde_json::from_str(json).map_err(|e| {
            WorkflowError::structural(
                "E-WORKFLOW-PARSE",
                format!("template is not valid JSON: {e}"),
            )
        })?;
        tpl.validate()?;
        Ok(tpl)
    }

    /// Load a template file (e.g. `workflows/templates/text-to-image.json`).
    pub fn load_file(path: &std::path::Path) -> Result<Self, WorkflowError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            WorkflowError::structural(
                "E-WORKFLOW-IO",
                format!("cannot read {}: {e}", path.display()),
            )
        })?;
        Self::parse(&text)
    }

    /// Structural validation: non-empty graph, unique node ids, edges pointing
    /// at known nodes, no self-edges, acyclic (Kahn check).
    pub fn validate(&self) -> Result<(), WorkflowError> {
        if self.nodes.is_empty() {
            return Err(WorkflowError::structural(
                "E-WORKFLOW-EMPTY",
                format!("workflow '{}' has no nodes", self.name),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for node in &self.nodes {
            if node.id.trim().is_empty() {
                return Err(WorkflowError::structural(
                    "E-WORKFLOW-BAD-NODE",
                    "a node has an empty id",
                ));
            }
            if !seen.insert(node.id.as_str()) {
                return Err(WorkflowError::structural(
                    "E-WORKFLOW-DUPLICATE-NODE",
                    format!("duplicate node id '{}'", node.id),
                ));
            }
        }
        for edge in &self.edges {
            if !seen.contains(edge.from.as_str()) {
                return Err(WorkflowError::structural(
                    "E-WORKFLOW-UNKNOWN-NODE",
                    format!("edge from unknown node '{}'", edge.from),
                ));
            }
            if !seen.contains(edge.to.as_str()) {
                return Err(WorkflowError::structural(
                    "E-WORKFLOW-UNKNOWN-NODE",
                    format!("edge to unknown node '{}'", edge.to),
                ));
            }
            if edge.from == edge.to {
                return Err(WorkflowError::structural(
                    "E-WORKFLOW-SELF-EDGE",
                    format!("node '{}' has an edge to itself", edge.from),
                ));
            }
        }
        // Acyclicity: Kahn's algorithm must drain every node.
        if self.execution_order_ids().len() != self.nodes.len() {
            return Err(WorkflowError::structural(
                "E-WORKFLOW-CYCLE",
                format!("workflow '{}' contains a cycle", self.name),
            ));
        }
        Ok(())
    }

    /// Topological node order (Kahn). Deterministic: ties break by template
    /// order. May be partial when the graph has a cycle — [`validate`](Self::validate)
    /// rejects those before any run.
    pub fn execution_order_ids(&self) -> Vec<String> {
        let mut indegree: HashMap<&str, usize> =
            self.nodes.iter().map(|n| (n.id.as_str(), 0)).collect();
        let mut outgoing: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &self.edges {
            // Validation guarantees both endpoints exist; tolerate unknown
            // here so ordering stays total on hand-built graphs.
            if indegree.contains_key(edge.to.as_str()) {
                *indegree.entry(edge.to.as_str()).or_insert(0) += 1;
                outgoing.entry(edge.from.as_str()).or_default().push(edge.to.as_str());
            }
        }
        let mut ready: VecDeque<&str> = self
            .nodes
            .iter()
            .filter(|n| indegree[n.id.as_str()] == 0)
            .map(|n| n.id.as_str())
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_front() {
            order.push(id.to_string());
            if let Some(nexts) = outgoing.get(id) {
                for next in nexts {
                    if let Some(deg) = indegree.get_mut(next) {
                        *deg -= 1;
                        if *deg == 0 {
                            ready.push_back(*next);
                        }
                    }
                }
            }
        }
        order
    }

    /// `prepare` stage for one node: the type must be a shipped template type.
    /// Unknown types fail here — before any successor consumes their output.
    pub fn prepare(&self, node: &WorkflowNode) -> Result<(), WorkflowError> {
        if KNOWN_NODE_TYPES.contains(&node.node_type.as_str()) {
            Ok(())
        } else {
            Err(WorkflowError::new(
                Some(&node.id),
                "E-WORKFLOW-UNKNOWN-TYPE",
                format!("unknown node type '{}'", node.node_type),
                format!(
                    "Use one of: {}. Custom types arrive with plugins (docs/06 §6.9).",
                    KNOWN_NODE_TYPES.join(", ")
                ),
            ))
        }
    }

    /// Run the graph with the orchestration-only stub executor.
    pub fn run(&self, initial_input: serde_json::Value) -> Result<WorkflowRun, WorkflowError> {
        self.run_with(initial_input, &stub_execute)
    }

    /// Run the graph: validate DAG, then per node in topological order run
    /// sequential `prepare -> execute`, threading predecessor outputs in as
    /// the next node's `inputs`.
    pub fn run_with(
        &self,
        initial_input: serde_json::Value,
        execute: &NodeFn,
    ) -> Result<WorkflowRun, WorkflowError> {
        self.validate()?;
        let by_id: HashMap<&str, &WorkflowNode> =
            self.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
        let mut preds: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &self.edges {
            preds.entry(edge.to.as_str()).or_default().push(edge.from.as_str());
        }
        let order = self.execution_order_ids();
        let mut outputs: HashMap<String, serde_json::Value> = HashMap::new();
        let mut run = WorkflowRun::default();
        for id in &order {
            let node = by_id[id.as_str()];
            // prepare stage: type check before anything consumes this node.
            self.prepare(node)?;
            // run stage: predecessor outputs become this node's inputs;
            // source nodes (no predecessors) receive the run's initial input.
            let inputs = if preds.get(id.as_str()).map_or(true, |p| p.is_empty()) {
                let mut map = serde_json::Map::new();
                map.insert("_initial".to_string(), initial_input.clone());
                serde_json::Value::Object(map)
            } else {
                let mut map = serde_json::Map::new();
                for pred in &preds[id.as_str()] {
                    let out = outputs.get(*pred).cloned().unwrap_or(serde_json::Value::Null);
                    map.insert((*pred).to_string(), out);
                }
                serde_json::Value::Object(map)
            };
            let input = serde_json::json!({ "inputs": inputs, "params": node.params });
            let output = execute(node, &input).map_err(|f| {
                WorkflowError::new(
                    Some(id.as_str()),
                    "E-WORKFLOW-NODE-FAILED",
                    format!("[{}] {}", f.code, f.message),
                    "Check Logs for the node detail, fix its params or upstream input, then re-run the workflow.",
                )
            })?;
            outputs.insert(id.clone(), output.clone());
            run.steps.push((id.clone(), output));
        }
        Ok(run)
    }
}

/// Orchestration-only stub executor: echoes what *would* run so graph shape,
/// ordering, and input-passing are testable without models loaded.
/// Production executors dispatch on `node.node_type` to the matching runtime
/// adapter (`prepare -> run` over a supervised child process).
pub fn stub_execute(
    node: &WorkflowNode,
    input: &serde_json::Value,
) -> Result<serde_json::Value, NodeFailure> {
    Ok(serde_json::json!({
        "node": node.id,
        "type": node.node_type,
        "echo_input": input,
        "stub": true,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT_TO_IMAGE: &str = include_str!("../../workflows/templates/text-to-image.json");
    const LLM_TO_IMAGE: &str = include_str!("../../workflows/templates/llm-to-image.json");

    #[test]
    fn shipped_templates_validate() {
        for file in [
            "text-to-image",
            "image-to-image",
            "text-to-speech",
            "speech-to-text",
            "text-to-video",
            "image-to-video",
            "llm-to-image",
            "llm-to-tts",
        ] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("workflows/templates")
                .join(format!("{file}.json"));
            let tpl = WorkflowTemplate::load_file(&path)
                .unwrap_or_else(|e| panic!("{file}: {e}"));
            assert!(!tpl.execution_order_ids().is_empty(), "{file}");
        }
    }

    #[test]
    fn chain_passes_outputs_as_next_inputs() {
        let tpl = WorkflowTemplate::parse(TEXT_TO_IMAGE).unwrap();
        assert_eq!(
            tpl.execution_order_ids(),
            ["prompt", "generate", "upscale", "save"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );
        let run = tpl
            .run_with(serde_json::json!("a cyberpunk warrior"), &|node, input| {
                // Each non-source node must see its predecessor's output.
                let inputs = &input["inputs"];
                let out = match node.id.as_str() {
                    "prompt" => serde_json::json!("expanded prompt"),
                    "generate" => {
                        assert!(inputs["prompt"].is_string(), "generate misses prompt output");
                        serde_json::json!("image-bytes-ref")
                    }
                    "upscale" => {
                        assert_eq!(inputs["generate"], serde_json::json!("image-bytes-ref"));
                        serde_json::json!("image-hires-ref")
                    }
                    "save" => {
                        assert_eq!(inputs["upscale"], serde_json::json!("image-hires-ref"));
                        serde_json::json!({"saved": "outputs/Images/x.png"})
                    }
                    other => return Err(NodeFailure::new("E_TEST", other)),
                };
                Ok(out)
            })
            .unwrap();
        assert_eq!(run.steps.len(), 4);
        assert_eq!(
            run.output_of("save").unwrap()["saved"],
            serde_json::json!("outputs/Images/x.png")
        );
        // llm->image chain from docs/10 §10.5 also runs end to end.
        let tpl = WorkflowTemplate::parse(LLM_TO_IMAGE).unwrap();
        let run = tpl.run(serde_json::json!("rough idea")).unwrap();
        assert_eq!(run.steps.len(), 5);
    }

    #[test]
    fn cycle_unknown_and_bad_type_are_coded() {
        let cycled = serde_json::json!({
            "name": "cycled",
            "nodes": [
                {"id": "a", "type": "llm", "params": {}},
                {"id": "b", "type": "llm", "params": {}},
            ],
            "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
        });
        let err = WorkflowTemplate::parse(&cycled.to_string()).unwrap_err();
        assert_eq!(err.code, "E-WORKFLOW-CYCLE");

        let dangling = serde_json::json!({
            "name": "dangling",
            "nodes": [{"id": "a", "type": "llm", "params": {}}],
            "edges": [{"from": "a", "to": "ghost"}],
        });
        let err = WorkflowTemplate::parse(&dangling.to_string()).unwrap_err();
        assert_eq!(err.code, "E-WORKFLOW-UNKNOWN-NODE");

        let bad_type = serde_json::json!({
            "name": "bad-type",
            "nodes": [{"id": "a", "type": "teleport", "params": {}}],
            "edges": [],
        });
        let tpl = WorkflowTemplate::parse(&bad_type.to_string()).unwrap();
        let err = tpl.run(serde_json::json!(null)).unwrap_err();
        assert_eq!(err.code, "E-WORKFLOW-UNKNOWN-TYPE");
        assert_eq!(err.node.as_deref(), Some("a"));
    }

    #[test]
    fn node_failure_names_the_node() {
        let tpl = WorkflowTemplate::parse(TEXT_TO_IMAGE).unwrap();
        let err = tpl
            .run_with(serde_json::json!(null), &|node, _| {
                if node.id == "generate" {
                    return Err(NodeFailure::new("E_VRAM_LOW", "needs 6GB, have 4GB"));
                }
                Ok(serde_json::Value::Null)
            })
            .unwrap_err();
        assert_eq!(err.code, "E-WORKFLOW-NODE-FAILED");
        assert_eq!(err.node.as_deref(), Some("generate"));
        assert!(err.to_string().contains("E_VRAM_LOW"));
    }
}
