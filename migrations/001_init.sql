-- 001_init.sql — SQLite schema (docs/04-DATA-MODEL.md §4.1).
-- State columns persist ModelState/RuntimeState/JobState strings;
-- every ERROR row must pair with a code + human fix in logs.

CREATE TABLE IF NOT EXISTS models(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  repository TEXT NOT NULL,
  revision TEXT,
  task TEXT,
  runtime TEXT,
  size_bytes INTEGER,
  status TEXT NOT NULL,
  installed_at TEXT,
  capabilities TEXT NOT NULL DEFAULT '[]',
  license TEXT,
  trust_level TEXT
);

CREATE TABLE IF NOT EXISTS model_files(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  bytes INTEGER,
  sha256 TEXT,
  dedup_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_files_model ON model_files(model_id);
CREATE INDEX IF NOT EXISTS idx_model_files_dedup ON model_files(dedup_hash);

CREATE TABLE IF NOT EXISTS runtimes(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  version TEXT,
  status TEXT NOT NULL,
  env_id TEXT REFERENCES environments(id),
  entrypoint TEXT
);

CREATE TABLE IF NOT EXISTS environments(
  id TEXT PRIMARY KEY,
  python TEXT,
  packages TEXT,
  cuda TEXT,
  model_rev TEXT
);

CREATE TABLE IF NOT EXISTS generations(
  id TEXT PRIMARY KEY,
  model_id TEXT REFERENCES models(id),
  prompt TEXT,
  params TEXT,
  seed INTEGER,
  output_path TEXT,
  created_at TEXT,
  runtime TEXT
);

CREATE TABLE IF NOT EXISTS downloads(
  model_id TEXT PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
  bytes_done INTEGER NOT NULL DEFAULT 0,
  bytes_total INTEGER,
  speed REAL,
  eta INTEGER,
  state TEXT NOT NULL DEFAULT 'Waiting'
);

CREATE TABLE IF NOT EXISTS workflows(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  graph TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS plugins(
  name TEXT PRIMARY KEY,
  version TEXT,
  entrypoint TEXT,
  supported_models TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS hardware_profiles(
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  gpu_vram_gb REAL,
  system_ram_gb REAL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS settings(
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS logs(
  ts TEXT NOT NULL,
  scope TEXT NOT NULL,
  level TEXT NOT NULL,
  msg TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts);
