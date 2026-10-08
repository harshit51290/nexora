-- 003_mem_estimate.sql — hf-mem port persistence (docs/05 §5.8).
-- Real weight/KV/total bytes measured pre-download; NULL until the first
-- estimate runs (gate falls back to the size×1.2 heuristic, docs/07 §7.4).

ALTER TABLE models ADD COLUMN est_weights_bytes INTEGER;
ALTER TABLE models ADD COLUMN est_kv_bytes INTEGER;
ALTER TABLE models ADD COLUMN est_total_bytes INTEGER;
ALTER TABLE models ADD COLUMN est_at TEXT;
