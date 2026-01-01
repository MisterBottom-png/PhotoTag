use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Connection;
use std::path::PathBuf;
use tauri::api::path::app_data_dir;
use tauri::Config;
use crate::error::Result;
use crate::schema;

pub type DbPool = r2d2::Pool<SqliteConnectionManager>;
pub type DbConnection = r2d2::PooledConnection<SqliteConnectionManager>;

/// Initializes the database connection pool and runs migrations.
pub fn init_database() -> Result<DbPool> {
    let db_path = get_database_path()?;
    log::info!("Database path: {}", db_path.display());

    // Ensure the parent directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let manager = SqliteConnectionManager::file(db_path);
    let pool = r2d2::Pool::new(manager)?;

    run_migrations(&pool.get()?)?;

    Ok(pool)
}

/// Gets the platform-specific path to the application's data directory.
fn get_database_path() -> Result<PathBuf> {
    let app_data_path = app_data_dir(&Config::default())
        .ok_or_else(|| crate::error::Error::Path("Failed to get app data dir".to_string()))?;
    
    let app_dir = app_data_path.join("PhotoCatalogApp");
    Ok(app_dir.join("library.db"))
}

/// Applies all pending database migrations.
fn run_migrations(connection: &DbConnection) -> Result<()> {
    // `DbConnection` dereferences to the underlying rusqlite `Connection`,
    // allowing us to call the rusqlite APIs directly.
    let connection: &Connection = &*connection;


    log::info!("Running database migrations...");
    
    // Migration 0001: Initial Schema
    connection.execute_batch(schema::MIGRATION_0001)?;

    log::info!("Migrations applied successfully.");
    Ok(())
}