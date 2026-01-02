use crate::config::AppPaths;
use crate::error::Result;
use crate::models::{
    CsvExportRow, ExifMetadata, PhotoRecord, PhotoWithTags, QueryFilters, SmartViewCounts,
    TagRecord, TaggingResult,
};
use crate::schema;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, types::Value, Connection, OptionalExtension};
use std::collections::HashSet;

pub type DbPool = r2d2::Pool<SqliteConnectionManager>;
pub type DbConnection = r2d2::PooledConnection<SqliteConnectionManager>;

/// Initializes the database connection pool and runs migrations.
pub fn init_database(paths: &AppPaths) -> Result<DbPool> {
    let db_path = &paths.db_path;
    log::info!("Database path: {}", db_path.display());

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let manager = SqliteConnectionManager::file(db_path);
    let pool = r2d2::Pool::new(manager)?;
    let conn = pool.get()?;
    run_migrations(&conn)?;

    Ok(pool)
}

/// Applies all pending database migrations.
fn run_migrations(connection: &DbConnection) -> Result<()> {
    let connection: &Connection = &*connection;

    log::info!("Running database migrations...");
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );",
    )?;

    let mut applied = HashSet::new();
    let mut stmt = connection.prepare("SELECT version FROM schema_migrations")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        applied.insert(row?);
    }

    let migrations = [
        ("0001", schema::MIGRATION_0001),
        ("0002", schema::MIGRATION_0002),
        ("0003", schema::MIGRATION_0003),
    ];

    for (version, migration) in migrations {
        if !applied.contains(version) {
            log::info!("Applying migration {version}...");
            if version == "0003" {
                apply_migration_0003(connection)?;
            } else {
                connection.execute_batch(migration)?;
            }
            connection.execute(
                "INSERT INTO schema_migrations (version) VALUES (?1)",
                params![version],
            )?;
        }
    }
    log::info!("Migrations applied successfully.");
    Ok(())
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for col in rows {
        if col? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn apply_migration_0003(conn: &Connection) -> Result<()> {
    let columns = [
        ("rating", "ALTER TABLE photos ADD COLUMN rating INTEGER"),
        (
            "picked",
            "ALTER TABLE photos ADD COLUMN picked INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "rejected",
            "ALTER TABLE photos ADD COLUMN rejected INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "last_modified",
            "ALTER TABLE photos ADD COLUMN last_modified INTEGER",
        ),
        (
            "import_batch_id",
            "ALTER TABLE photos ADD COLUMN import_batch_id TEXT",
        ),
    ];

    for (name, sql) in columns {
        if !column_exists(conn, "photos", name)? {
            conn.execute(sql, [])?;
        }
    }

    conn.execute(
        "UPDATE photos SET last_modified = strftime('%s','now') WHERE last_modified IS NULL",
        [],
    )?;

    // Indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_rating ON photos (rating)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_picked ON photos (picked)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_rejected ON photos (rejected)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_import_batch_id ON photos (import_batch_id)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_photos_cull_state ON photos (picked, rejected, rating)",
        [],
    )?;

    Ok(())
}

pub fn upsert_photo(conn: &DbConnection, photo: &PhotoRecord) -> Result<i64> {
    // Check existing record
    let existing: Option<(i64, i64, i64)> = conn
        .query_row(
            "SELECT id, mtime, size FROM photos WHERE path = ?1",
            params![photo.path],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;

    if let Some((id, mtime, size)) = existing {
        if mtime == photo.mtime && size == photo.size {
            // Update timestamps only
            conn.execute(
                "UPDATE photos SET updated_at = strftime('%s','now') WHERE id = ?1",
                params![id],
            )?;
            return Ok(id);
        }
    }

    conn.execute(
        "INSERT INTO photos (
            path,
            hash,
            file_name,
            ext,
            size,
            mtime,
            width,
            height,
            make,
            model,
            lens,
            date_taken,
            iso,
            fnumber,
            focal_length,
            exposure_time,
            exposure_comp,
            gps_lat,
            gps_lng,
            thumb_path,
            preview_path,
            import_batch_id,
            created_at,
            updated_at,
            last_modified
        )
        VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22,
            strftime('%s','now'),
            strftime('%s','now'),
            strftime('%s','now')
        )
        ON CONFLICT(path) DO UPDATE SET
            hash = excluded.hash,
            file_name = excluded.file_name,
            ext = excluded.ext,
            size = excluded.size,
            mtime = excluded.mtime,
            width = excluded.width,
            height = excluded.height,
            make = excluded.make,
            model = excluded.model,
            lens = excluded.lens,
            date_taken = excluded.date_taken,
            iso = excluded.iso,
            fnumber = excluded.fnumber,
            focal_length = excluded.focal_length,
            exposure_time = excluded.exposure_time,
            exposure_comp = excluded.exposure_comp,
            gps_lat = excluded.gps_lat,
            gps_lng = excluded.gps_lng,
            thumb_path = excluded.thumb_path,
            preview_path = excluded.preview_path,
            updated_at = strftime('%s','now'),
            last_modified = strftime('%s','now')",
        params![
            photo.path,
            photo.hash,
            photo.file_name,
            photo.ext,
            photo.size,
            photo.mtime,
            photo.width,
            photo.height,
            photo.make,
            photo.model,
            photo.lens,
            photo.date_taken,
            photo.iso,
            photo.fnumber,
            photo.focal_length,
            photo.exposure_time,
            photo.exposure_comp,
            photo.gps_lat,
            photo.gps_lng,
            photo.thumb_path,
            photo.preview_path,
            photo.import_batch_id,
        ],
    )?;

    let id = conn.query_row(
        "SELECT id FROM photos WHERE path = ?1",
        params![photo.path],
        |row| row.get(0),
    )?;
    Ok(id)
}

pub fn replace_auto_tags(
    conn: &DbConnection,
    photo_id: i64,
    tagging: TaggingResult,
    _exif: &ExifMetadata,
) -> Result<()> {
    if tagging.tags.is_empty() {
        return Ok(());
    }
    conn.execute(
        "DELETE FROM tags WHERE photo_id = ?1 AND source = 'auto' AND locked = 0",
        params![photo_id],
    )?;

    for (tag, confidence) in tagging.tags {
        conn.execute(
            "INSERT OR IGNORE INTO tags (photo_id, tag, confidence, source, locked, created_at) VALUES (?1, ?2, ?3, 'auto', 0, strftime('%s','now'))",
            params![photo_id, tag, confidence],
        )?;
    }
    Ok(())
}

pub fn get_photo_status(conn: &DbConnection, path: &str) -> Result<Option<(i64, i64)>> {
    conn.query_row(
        "SELECT mtime, size FROM photos WHERE path = ?1",
        params![path],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .optional()
    .map_err(Into::into)
}

pub fn list_paths_with_prefix(conn: &DbConnection, root: &str) -> Result<HashSet<String>> {
    let like = format!("{}%", root.replace('%', "\\%").replace('_', "\\_"));
    let mut stmt = conn.prepare("SELECT path FROM photos WHERE path LIKE ?1 ESCAPE '\\\\'")?;
    let rows = stmt.query_map(params![like], |row| row.get::<_, String>(0))?;
    let mut paths = HashSet::new();
    for row in rows {
        paths.insert(row?);
    }
    Ok(paths)
}

fn resolve_sort_column(sort_by: Option<&str>) -> &'static str {
    match sort_by {
        Some("date_taken") => "date_taken",
        Some("created_at") => "created_at",
        Some("file_name") => "file_name",
        Some("iso") => "iso",
        Some("fnumber") => "fnumber",
        Some("focal_length") => "focal_length",
        Some("exposure_time") => "exposure_time",
        Some("rating") => "rating",
        Some("picked") => "picked",
        Some("rejected") => "rejected",
        Some("last_modified") => "last_modified",
        Some("import_batch_id") => "import_batch_id",
        _ => "date_taken",
    }
}

fn resolve_sort_dir(sort_dir: Option<&str>) -> &'static str {
    match sort_dir {
        Some("ASC") | Some("asc") => "ASC",
        _ => "DESC",
    }
}

fn latest_import_batch_id(conn: &DbConnection) -> Result<Option<String>> {
    conn.query_row(
        "SELECT import_batch_id FROM photos WHERE import_batch_id IS NOT NULL ORDER BY created_at DESC LIMIT 1",
        [],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

pub fn add_manual_tag(conn: &DbConnection, photo_id: i64, tag: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO tags (id, photo_id, tag, confidence, source, locked, created_at) VALUES ((SELECT id FROM tags WHERE photo_id = ?1 AND tag = ?2), ?1, ?2, 1.0, 'manual', 1, strftime('%s','now'))",
        params![photo_id, tag],
    )?;
    Ok(())
}

pub fn remove_tag(conn: &DbConnection, photo_id: i64, tag: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM tags WHERE photo_id = ?1 AND tag = ?2 AND source = 'manual'",
        params![photo_id, tag],
    )?;
    Ok(())
}

pub fn set_rating(conn: &DbConnection, photo_id: i64, rating: Option<i64>) -> Result<()> {
    conn.execute(
        "UPDATE photos SET rating = ?1, last_modified = strftime('%s','now') WHERE id = ?2",
        params![rating, photo_id],
    )?;
    Ok(())
}

pub fn set_picked(conn: &DbConnection, photo_id: i64, picked: bool) -> Result<()> {
    conn.execute(
        "UPDATE photos SET picked = ?1, last_modified = strftime('%s','now') WHERE id = ?2",
        params![picked as i64, photo_id],
    )?;
    Ok(())
}

pub fn set_rejected(conn: &DbConnection, photo_id: i64, rejected: bool) -> Result<()> {
    conn.execute(
        "UPDATE photos SET rejected = ?1, last_modified = strftime('%s','now') WHERE id = ?2",
        params![rejected as i64, photo_id],
    )?;
    Ok(())
}

pub fn batch_update_cull(
    conn: &DbConnection,
    photo_ids: &[i64],
    rating: Option<Option<i64>>,
    picked: Option<bool>,
    rejected: Option<bool>,
) -> Result<usize> {
    if photo_ids.is_empty() {
        return Ok(0);
    }

    let mut sets: Vec<String> = Vec::new();
    let mut params: Vec<Value> = Vec::new();

    if let Some(rating_value) = rating {
        sets.push("rating = ?".into());
        match rating_value {
            Some(v) => params.push(v.into()),
            None => params.push(Value::Null),
        }
    }
    if let Some(p) = picked {
        sets.push("picked = ?".into());
        params.push((p as i64).into());
    }
    if let Some(r) = rejected {
        sets.push("rejected = ?".into());
        params.push((r as i64).into());
    }

    if sets.is_empty() {
        return Ok(0);
    }

    sets.push("last_modified = strftime('%s','now')".into());

    let mut sql = format!("UPDATE photos SET {} WHERE id IN (", sets.join(", "));
    for (idx, _) in photo_ids.iter().enumerate() {
        if idx > 0 {
            sql.push(',');
        }
        sql.push('?');
    }
    sql.push(')');

    for id in photo_ids {
        params.push((*id).into());
    }

    let updated = conn.execute(&sql, rusqlite::params_from_iter(params))?;
    Ok(updated)
}

pub fn get_smart_view_counts(conn: &DbConnection) -> Result<SmartViewCounts> {
    let unsorted = conn.query_row(
        "SELECT COUNT(*) FROM photos WHERE rating IS NULL AND picked = 0 AND rejected = 0",
        [],
        |row| row.get(0),
    )?;
    let picks = conn.query_row(
        "SELECT COUNT(*) FROM photos WHERE picked = 1 AND rejected = 0",
        [],
        |row| row.get(0),
    )?;
    let rejects = conn.query_row(
        "SELECT COUNT(*) FROM photos WHERE rejected = 1",
        [],
        |row| row.get(0),
    )?;

    let mut last_import = 0;
    if let Some(batch_id) = latest_import_batch_id(conn)? {
        last_import = conn.query_row(
            "SELECT COUNT(*) FROM photos WHERE import_batch_id = ?1",
            params![batch_id],
            |row| row.get(0),
        )?;
    }

    let all = conn.query_row("SELECT COUNT(*) FROM photos", [], |row| row.get(0))?;

    Ok(SmartViewCounts {
        unsorted,
        picks,
        rejects,
        last_import,
        all,
    })
}

pub fn query_photos(conn: &DbConnection, filters: QueryFilters) -> Result<Vec<PhotoWithTags>> {
    let mut sql = "SELECT * FROM photos WHERE 1=1".to_string();
    let mut params: Vec<rusqlite::types::Value> = Vec::new();

    if let Some(search) = filters.search {
        sql.push_str(" AND (file_name LIKE ? OR make LIKE ? OR model LIKE ? OR lens LIKE ?)");
        let pattern = format!("%{}%", search);
        for _ in 0..4 {
            params.push(pattern.clone().into());
        }
    }
    if let Some(make) = filters.camera_make {
        sql.push_str(" AND make = ?");
        params.push(make.into());
    }
    if let Some(model) = filters.camera_model {
        sql.push_str(" AND model = ?");
        params.push(model.into());
    }
    if let Some(lens) = filters.lens {
        sql.push_str(" AND lens = ?");
        params.push(lens.into());
    }
    if let Some(min_iso) = filters.iso_min {
        sql.push_str(" AND iso >= ?");
        params.push(min_iso.into());
    }
    if let Some(max_iso) = filters.iso_max {
        sql.push_str(" AND iso <= ?");
        params.push(max_iso.into());
    }
    if let Some(min_ap) = filters.aperture_min {
        sql.push_str(" AND fnumber >= ?");
        params.push(min_ap.into());
    }
    if let Some(max_ap) = filters.aperture_max {
        sql.push_str(" AND fnumber <= ?");
        params.push(max_ap.into());
    }
    if let Some(min_focal) = filters.focal_min {
        sql.push_str(" AND focal_length >= ?");
        params.push(min_focal.into());
    }
    if let Some(max_focal) = filters.focal_max {
        sql.push_str(" AND focal_length <= ?");
        params.push(max_focal.into());
    }
    if let Some(date_from) = filters.date_from {
        sql.push_str(" AND date_taken >= ?");
        params.push(date_from.into());
    }
    if let Some(date_to) = filters.date_to {
        sql.push_str(" AND date_taken <= ?");
        params.push(date_to.into());
    }
    if let Some(has_gps) = filters.has_gps {
        if has_gps {
            sql.push_str(" AND gps_lat IS NOT NULL AND gps_lng IS NOT NULL");
        } else {
            sql.push_str(" AND (gps_lat IS NULL OR gps_lng IS NULL)");
        }
    }

    if !filters.tags.is_empty() {
        sql.push_str(" AND id IN (SELECT photo_id FROM tags WHERE tag IN (");
        for (i, tag) in filters.tags.iter().enumerate() {
            if i > 0 {
                sql.push(',');
            }
            sql.push('?');
            params.push(tag.clone().into());
        }
        sql.push_str("))");
    }

    if let Some(smart_view) = filters.smart_view.as_deref() {
        match smart_view {
            "UNSORTED" => {
                sql.push_str(" AND rating IS NULL AND picked = 0 AND rejected = 0");
            }
            "PICKS" => {
                sql.push_str(" AND picked = 1 AND rejected = 0");
            }
            "REJECTS" => {
                sql.push_str(" AND rejected = 1");
            }
            "LAST_IMPORT" => {
                if let Some(batch_id) = latest_import_batch_id(conn)? {
                    sql.push_str(" AND import_batch_id = ?");
                    params.push(batch_id.into());
                } else {
                    sql.push_str(" AND 0");
                }
            }
            _ => {}
        }
    }

    let sort_by = if filters.sort_by.is_none() {
        if matches!(filters.mode.as_deref(), Some(mode) if mode.eq_ignore_ascii_case("cull")) {
            "last_modified"
        } else {
            "date_taken"
        }
    } else {
        resolve_sort_column(filters.sort_by.as_deref())
    };
    let sort_dir = resolve_sort_dir(filters.sort_dir.as_deref());
    sql.push_str(&format!(" ORDER BY {} {}", sort_by, sort_dir));
    if let Some(limit) = filters.limit {
        sql.push_str(&format!(" LIMIT {}", limit));
    }
    if let Some(offset) = filters.offset {
        sql.push_str(&format!(" OFFSET {}", offset));
    }

    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query(rusqlite::params_from_iter(params))?;
    let mut results = Vec::new();
    while let Some(row) = rows.next()? {
        let photo = PhotoRecord {
            id: Some(row.get("id")?),
            path: row.get("path")?,
            hash: row.get("hash")?,
            file_name: row.get("file_name")?,
            ext: row.get("ext")?,
            size: row.get("size")?,
            mtime: row.get("mtime")?,
            width: row.get("width")?,
            height: row.get("height")?,
            make: row.get("make")?,
            model: row.get("model")?,
            lens: row.get("lens")?,
            date_taken: row.get("date_taken")?,
            iso: row.get("iso")?,
            fnumber: row.get("fnumber")?,
            focal_length: row.get("focal_length")?,
            exposure_time: row.get("exposure_time")?,
            exposure_comp: row.get("exposure_comp")?,
            gps_lat: row.get("gps_lat")?,
            gps_lng: row.get("gps_lng")?,
            thumb_path: row.get("thumb_path")?,
            preview_path: row.get("preview_path")?,
            rating: row.get("rating")?,
            picked: row.get::<_, i64>("picked")? == 1,
            rejected: row.get::<_, i64>("rejected")? == 1,
            last_modified: row.get("last_modified")?,
            import_batch_id: row.get("import_batch_id")?,
            created_at: row.get("created_at")?,
            updated_at: row.get("updated_at")?,
        };
        let tags = query_tags(conn, photo.id.unwrap())?;
        results.push(PhotoWithTags { photo, tags });
    }

    Ok(results)
}

pub fn query_tags(conn: &DbConnection, photo_id: i64) -> Result<Vec<TagRecord>> {
    let mut stmt = conn.prepare("SELECT * FROM tags WHERE photo_id = ?1")?;
    let mut rows = stmt.query(params![photo_id])?;
    let mut tags = Vec::new();
    while let Some(row) = rows.next()? {
        tags.push(TagRecord {
            id: Some(row.get("id")?),
            photo_id,
            tag: row.get("tag")?,
            confidence: row.get("confidence")?,
            source: row.get("source")?,
            locked: row.get::<_, i64>("locked")? == 1,
            created_at: row.get("created_at")?,
        });
    }
    Ok(tags)
}

pub fn get_photo(conn: &DbConnection, photo_id: i64) -> Result<Option<PhotoWithTags>> {
    let mut stmt = conn.prepare("SELECT * FROM photos WHERE id = ?1")?;
    let mut rows = stmt.query(params![photo_id])?;
    if let Some(row) = rows.next()? {
        let photo = PhotoRecord {
            id: Some(row.get("id")?),
            path: row.get("path")?,
            hash: row.get("hash")?,
            file_name: row.get("file_name")?,
            ext: row.get("ext")?,
            size: row.get("size")?,
            mtime: row.get("mtime")?,
            width: row.get("width")?,
            height: row.get("height")?,
            make: row.get("make")?,
            model: row.get("model")?,
            lens: row.get("lens")?,
            date_taken: row.get("date_taken")?,
            iso: row.get("iso")?,
            fnumber: row.get("fnumber")?,
            focal_length: row.get("focal_length")?,
            exposure_time: row.get("exposure_time")?,
            exposure_comp: row.get("exposure_comp")?,
            gps_lat: row.get("gps_lat")?,
            gps_lng: row.get("gps_lng")?,
            thumb_path: row.get("thumb_path")?,
            preview_path: row.get("preview_path")?,
            rating: row.get("rating")?,
            picked: row.get::<_, i64>("picked")? == 1,
            rejected: row.get::<_, i64>("rejected")? == 1,
            last_modified: row.get("last_modified")?,
            import_batch_id: row.get("import_batch_id")?,
            created_at: row.get("created_at")?,
            updated_at: row.get("updated_at")?,
        };
        let tags = query_tags(conn, photo_id)?;
        Ok(Some(PhotoWithTags { photo, tags }))
    } else {
        Ok(None)
    }
}

pub fn export_csv(conn: &DbConnection, filters: QueryFilters) -> Result<Vec<CsvExportRow>> {
    let photos = query_photos(conn, filters)?;
    let rows = photos
        .into_iter()
        .map(|p| CsvExportRow {
            filename: p.photo.file_name.clone(),
            path: p.photo.path.clone(),
            camera: p.photo.make.clone(),
            lens: p.photo.lens.clone(),
            date: p.photo.date_taken,
            iso: p.photo.iso,
            fnumber: p.photo.fnumber,
            focal: p.photo.focal_length,
            shutter: p.photo.exposure_time,
            tags: p.tags.iter().map(|t| t.tag.clone()).collect(),
        })
        .collect();
    Ok(rows)
}
