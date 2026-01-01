use crate::config::AppPaths;
use crate::error::Result;
use crate::models::{CsvExportRow, ExifMetadata, PhotoRecord, PhotoWithTags, QueryFilters, TagRecord, TaggingResult};
use crate::schema;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, Connection, OptionalExtension};
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
    connection.execute_batch(schema::MIGRATION_0001)?;
    connection.execute_batch(schema::MIGRATION_0002)?;
    log::info!("Migrations applied successfully.");
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
        "REPLACE INTO photos (id, path, hash, file_name, ext, size, mtime, width, height, make, model, lens, date_taken, iso, fnumber, focal_length, exposure_time, exposure_comp, gps_lat, gps_lng, thumb_path, preview_path, created_at, updated_at)
        VALUES ((SELECT id FROM photos WHERE path = ?1), ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, strftime('%s','now'), strftime('%s','now'))",
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
        ],
    )?;

    let id = conn.query_row("SELECT id FROM photos WHERE path = ?1", params![photo.path], |row| row.get(0))?;
    Ok(id)
}

pub fn replace_auto_tags(conn: &DbConnection, photo_id: i64, tagging: TaggingResult, _exif: &ExifMetadata) -> Result<()> {
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
            sql.push_str(&format!("'{}'", tag.replace('"', "")));
        }
        sql.push_str(")")
    }

    let sort_by = filters.sort_by.unwrap_or_else(|| "date_taken".into());
    let sort_dir = filters.sort_dir.unwrap_or_else(|| "DESC".into());
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
