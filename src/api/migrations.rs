//! Versioned migration runner (replaces the hand-rolled `001` replay).
//!
//! `sqlx::migrate!` needs the `sqlx` **`"migrate"` feature**, which is not
//! enabled in `Cargo.toml` — and `Cargo.toml` is outside this slice's scope
//! (see NEED-CARGO below). So this module is the runner until the feature
//! lands: each migration is a `(version, sql)` pair compiled in from
//! `migrations/*.sql`, applied once, and recorded in `_schema_migrations`.
//! Adding `migrations/002_*.sql` later means appending one line to
//! [`MIGRATIONS`]; no replay-by-hand.
//!
//! NEED-CARGO: add `"migrate"` to the `sqlx` features in `Cargo.toml`, then
//! replace [`apply_migrations`] with `sqlx::migrate!("./migrations").run(&pool)`.

use sqlx::SqlitePool;

use super::core_stub::CoreStubError;

/// Compile-time copies of the canonical migration files (single source stays
/// `migrations/*.sql`). `002_jobs` is registered here too so bootstrap
/// applies the whole chain in order — its `IF NOT EXISTS` statements make
/// this idempotent with `jobs::apply_jobs_schema` (which owns that file and
/// stays untouched outside this slice's scope).
const MIGRATIONS: &[(&str, &str)] = &[
    ("001_init", include_str!("../../migrations/001_init.sql")),
    ("002_jobs", include_str!("../../migrations/002_jobs.sql")),
    ("003_mem_estimate", include_str!("../../migrations/003_mem_estimate.sql")),
];

/// Apply every pending migration exactly once, in order.
pub async fn apply_migrations(pool: &SqlitePool) -> Result<(), CoreStubError> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS _schema_migrations(
           version TEXT PRIMARY KEY,
           applied_at TEXT NOT NULL
         )",
    )
    .execute(pool)
    .await?;
    for (version, sql) in MIGRATIONS {
        let done: bool =
            sqlx::query_scalar("SELECT COUNT(*) FROM _schema_migrations WHERE version = ?")
                .bind(*version)
                .fetch_one(pool)
                .await
                .map(|n: i64| n > 0)?;
        if done {
            continue;
        }
        // Strip full-line `--` comments BEFORE splitting: a semicolon
        // inside a comment would otherwise split mid-comment and execute a
        // fragment (this bit us in 003: "pre-download; NULL ..." failed
        // with `near "NULL": syntax error`). Convention: migration comments
        // are always full lines starting with `--`.
        let stripped: String = sql
            .lines()
            .filter(|l| !l.trim_start().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n");
        for stmt in stripped.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }
            sqlx::query(stmt).execute(pool).await?;
        }
        sqlx::query("INSERT INTO _schema_migrations(version, applied_at) VALUES(?,?)")
            .bind(*version)
            .bind(chrono::Utc::now().to_rfc3339())
            .execute(pool)
            .await?;
    }
    Ok(())
}
