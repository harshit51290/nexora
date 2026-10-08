-- 002_jobs.sql — persisted job queue (docs/10-API-CLI.md §10.4).
-- Minimal queue truth: id, kind, status, model, prompt, created_at.
-- Terminal artifacts stay in `generations` (no schema change there).
-- Per-run output paths and VRAM estimates are transient (rebuilt in memory).
-- Status spelling is capitalized (`Waiting` etc, matching JobState::as_str
-- and docs/10 §10.4). Kind is lowercase (`image` etc, matching the jobs API
-- serde spelling).
--
-- DOWN-MIGRATION (manual rollback): DROP TABLE IF EXISTS jobs
-- (the idx_jobs_status index drops implicitly with the table).

CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL DEFAULT 'llm',
  status TEXT NOT NULL DEFAULT 'Waiting',
  model TEXT,
  prompt TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
