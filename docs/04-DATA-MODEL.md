# 04 — Data Model, States, Capabilities

## 4.1 SQLite tables (SQLx + migrations/)
`models, model_files, runtimes, environments, generations, downloads, workflows, plugins, hardware_profiles, settings, logs`.
```sql
-- models (min)
CREATE TABLE models(id TEXT PRIMARY KEY, name TEXT NOT NULL, repository TEXT NOT NULL, revision TEXT,
task TEXT, runtime TEXT, size_bytes INTEGER, status TEXT NOT NULL, installed_at TEXT,
capabilities TEXT NOT NULL DEFAULT '[]', license TEXT, trust_level TEXT);
-- model_files(id, model_id, path, bytes, sha256, dedup_hash)
-- runtimes(id, kind, version, status, env_id, entrypoint)
-- environments(id, python TEXT, packages TEXT, cuda TEXT, model_rev TEXT)
-- generations(id, model_id, prompt TEXT, params TEXT, seed INTEGER, output_path TEXT, created_at TEXT, runtime TEXT)
-- downloads(model_id, bytes_done, bytes_total, speed, eta, state)
-- workflows(id, name, graph TEXT); plugins(name, version, entrypoint, supported_models TEXT);
-- hardware_profiles(id, label, gpu_vram_gb REAL, system_ram_gb REAL, notes)
-- settings(key PRIMARY KEY, value); logs(ts, scope, level, msg)
```

## 4.2 Model state machine
`DISCOVERED -> ANALYZING -> SUPPORTED -> DOWNLOADING -> INSTALLED -> VALIDATING -> READY -> LOADED -> RUNNING`, failure -> `ERROR`. Persist every transition; ERROR must carry machine-readable code + human recommendation (see `11-UI-UX.md`).

## 4.3 Runtime state machine
`NOT_INSTALLED -> INSTALLING -> READY -> STARTING -> RUNNING -> STOPPING -> STOPPED`. Child-process supervised; crash -> log + ERROR without killing app.

## 4.4 Capability system (drives UI)
```json
{"capabilities":["text-generation","chat"]}
{"capabilities":["text-to-image","image-to-image"]}
{"capabilities":["text-to-speech"]}
```
Allowed tokens: `text-generation,chat,embedding,classification,image-to-image,text-to-image,image-editing,video-generation,speech-to-text,text-to-speech,audio-generation,segmentation,detection,ocr`. UI renders form per capability; API validates against it.

## 4.5 Storage layout
`models/ runtimes/ environments/ cache/ outputs/{Images,Audio,Video,Text,Other}/ logs/ plugins/`. Root user-relocatable. Each output stores sidecar JSON: model, prompt, seed, params, timestamp, runtime (reproducibility; powers history Reuse/Regenerate).
