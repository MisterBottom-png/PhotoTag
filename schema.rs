/// MIGRATION 0001: Initial database schema.
pub const MIGRATION_0001: &str = r#"
-- Photos Table: Stores information about each imported photo.
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    file_name TEXT NOT NULL,
    ext TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    make TEXT,
    model TEXT,
    lens TEXT,
    date_taken INTEGER, -- Stored as Unix timestamp
    iso INTEGER,
    fnumber REAL,
    focal_length REAL,
    exposure_time REAL,
    exposure_comp REAL,
    gps_lat REAL,
    gps_lng REAL,
    thumb_path TEXT,
    preview_path TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- Tags Table: Associates tags with photos.
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    confidence REAL, -- 0.0 to 1.0
    source TEXT NOT NULL, -- 'auto' or 'manual'
    locked BOOLEAN NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
    UNIQUE (photo_id, tag)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_photos_path ON photos (path);
CREATE INDEX IF NOT EXISTS idx_photos_hash ON photos (hash);
CREATE INDEX IF NOT EXISTS idx_photos_date_taken ON photos (date_taken);
CREATE INDEX IF NOT EXISTS idx_tags_photo_id ON tags (photo_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags (tag);
CREATE INDEX IF NOT EXISTS idx_tags_source ON tags (source);
"#;

pub const MIGRATION_0002: &str = r#"
-- Import roots to support incremental scanning
CREATE TABLE IF NOT EXISTS import_roots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path TEXT NOT NULL UNIQUE,
    last_scanned_at INTEGER
);
"#;

pub const MIGRATION_0003: &str = r#"
-- Cull workflow fields
ALTER TABLE photos ADD COLUMN rating INTEGER;
ALTER TABLE photos ADD COLUMN picked INTEGER NOT NULL DEFAULT 0;
ALTER TABLE photos ADD COLUMN rejected INTEGER NOT NULL DEFAULT 0;
ALTER TABLE photos ADD COLUMN last_modified INTEGER;
ALTER TABLE photos ADD COLUMN import_batch_id TEXT;

-- Backfill last_modified for existing rows (will be set on insert/update in code)
UPDATE photos SET last_modified = strftime('%s', 'now') WHERE last_modified IS NULL;

-- Cull workflow indexes
CREATE INDEX IF NOT EXISTS idx_photos_rating ON photos (rating);
CREATE INDEX IF NOT EXISTS idx_photos_picked ON photos (picked);
CREATE INDEX IF NOT EXISTS idx_photos_rejected ON photos (rejected);
CREATE INDEX IF NOT EXISTS idx_photos_import_batch_id ON photos (import_batch_id);
CREATE INDEX IF NOT EXISTS idx_photos_cull_state ON photos (picked, rejected, rating);
"#;
