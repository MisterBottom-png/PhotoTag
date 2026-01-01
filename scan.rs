use crate::config::{AppPaths, TaggingConfig};
use crate::db::{self, DbPool};
use crate::error::Result;
use crate::exiftool;
use crate::models::{ImportProgressEvent, PhotoRecord};
use crate::tagging::TaggingEngine;
use crate::thumbnails;
use std::fs;
use std::path::{Path, PathBuf};
use tauri::Manager;
use uuid::Uuid;
use walkdir::WalkDir;
use xxhash_rust::xxh3::xxh3_128;

const SUPPORTED_EXT: &[&str] = &[
    "jpg", "jpeg", "png", "tiff", "tif", "cr2", "nef", "arw", "dng", "raf",
];

pub fn scan_folder(
    app: tauri::AppHandle,
    root: PathBuf,
    pool: DbPool,
    paths: AppPaths,
    tagging: TaggingConfig,
) -> Result<()> {
    let root_str = root.to_string_lossy().to_string();
    let existing_paths = {
        let conn = pool.get()?;
        db::list_paths_with_prefix(&conn, &root_str).unwrap_or_default()
    };
    let discovered: Vec<PathBuf> = WalkDir::new(&root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| SUPPORTED_EXT.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .filter(|e| !existing_paths.contains(e.path().to_string_lossy().as_ref()))
        .map(|e| e.into_path())
        .collect();

    let import_batch_id = Uuid::new_v4().to_string();
    let total = discovered.len();
    let emitter = app.clone();
    let mut tagging_engine = TaggingEngine::new(tagging)?;
    for (idx, path) in discovered.iter().enumerate() {
        emit_progress(&emitter, total, idx, path);
        process_file(path, &pool, &paths, &mut tagging_engine, &import_batch_id)?;
    }
    emit_progress(&emitter, total, total, &root);
    Ok(())
}

fn emit_progress(app: &tauri::AppHandle, total: usize, processed: usize, path: &Path) {
    let _ = app.emit_all(
        "import-progress",
        ImportProgressEvent {
            discovered: total,
            processed,
            current_file: path.to_str().map(|s| s.to_string()),
        },
    );
}

fn compute_hash(path: &Path) -> Result<String> {
    let data = fs::read(path)?;
    let digest = xxh3_128(&data);
    Ok(format!("{:x}", digest))
}

fn process_file(
    path: &Path,
    pool: &DbPool,
    paths: &AppPaths,
    engine: &mut TaggingEngine,
    import_batch_id: &str,
) -> Result<()> {
    let metadata = fs::metadata(path)?;
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let size = metadata.len() as i64;
    let path_str = path.to_string_lossy().to_string();
    let conn = pool.get()?;
    if let Some((existing_mtime, existing_size)) = db::get_photo_status(&conn, &path_str)? {
        if existing_mtime == mtime && existing_size == size {
            return Ok(());
        }
    }

    let hash = compute_hash(path)?;
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_lowercase();

    let exif = exiftool::read_metadata(paths, path).unwrap_or_default();

    let preview_output = paths.previews_dir.join(format!("{}_preview.jpg", hash));

    let has_preview = exiftool::extract_preview(paths, path, &preview_output).unwrap_or(false);
    let final_preview = if has_preview {
        preview_output.clone()
    } else {
        thumbnails::build_preview(path, &paths.previews_dir).unwrap_or(preview_output.clone())
    };
    let thumb_path = thumbnails::build_thumbnail(&final_preview, &paths.thumbs_dir)
        .unwrap_or(paths.thumbs_dir.join(format!("{}_thumb.jpg", hash)));

    let mut photo = PhotoRecord {
        id: None,
        path: path.to_string_lossy().to_string(),
        hash: hash.clone(),
        file_name,
        ext,
        size,
        mtime,
        width: exif.width,
        height: exif.height,
        make: exif.make.clone(),
        model: exif.model.clone(),
        lens: exif.lens.clone(),
        date_taken: exif.datetime_original,
        iso: exif.iso,
        fnumber: exif.fnumber,
        focal_length: exif.focal_length,
        exposure_time: exif.exposure_time,
        exposure_comp: exif.exposure_comp,
        gps_lat: exif.gps_lat,
        gps_lng: exif.gps_lng,
        thumb_path: Some(thumb_path.to_string_lossy().to_string()),
        preview_path: Some(final_preview.to_string_lossy().to_string()),
        rating: None,
        picked: false,
        rejected: false,
        last_modified: None,
        import_batch_id: Some(import_batch_id.to_string()),
        created_at: None,
        updated_at: None,
    };

    let photo_id = db::upsert_photo(&conn, &photo)?;
    photo.id = Some(photo_id);

    let tagging = engine.classify(&preview_output, &exif).unwrap_or_default();
    db::replace_auto_tags(&conn, photo_id, tagging, &exif)?;

    Ok(())
}
