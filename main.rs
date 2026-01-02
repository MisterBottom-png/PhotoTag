// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod db;
mod error;
mod embedding;
mod exiftool;
mod jobs;
mod models;
mod schema;
mod tagging;
mod thumbnails;

use crate::config::{AppPaths, InferenceDevicePreference, TaggingConfig};
use crate::db::DbPool;
use crate::error::Error;
use crate::jobs::JobManager;
use crate::models::{InferenceStatus, PhotoWithTags, QueryFilters, SmartViewCounts};
use crate::tagging::TaggingEngine;
use std::env;
use std::path::{Path, PathBuf};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::Command;
use std::sync::{Arc, Mutex};

type InvokeResult<T> = std::result::Result<T, String>;

pub struct AppState {
    db: DbPool,
    paths: AppPaths,
    tagging: Arc<Mutex<TaggingConfig>>,
    jobs: JobManager,
}

fn resolve_model_path(
    paths: &AppPaths,
    models_dir_override: Option<&Path>,
    default_name: &Path,
    env_key: &str,
) -> PathBuf {
    if let Some(override_path) = env::var_os(env_key) {
        return PathBuf::from(override_path);
    }
    if let Some(models_dir) = models_dir_override {
        if default_name.is_absolute() {
            default_name.to_path_buf()
        } else {
            models_dir.join(default_name)
        }
    } else {
        paths.resolve_model(default_name)
    }
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn query_photos(
    state: tauri::State<AppState>,
    filters: QueryFilters,
) -> InvokeResult<Vec<PhotoWithTags>> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::query_photos(&conn, filters).map_err(|e| e.to_string())
}

#[tauri::command]
fn add_manual_tag(state: tauri::State<AppState>, photo_id: i64, tag: String) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::add_manual_tag(&conn, photo_id, &tag).map_err(|e| e.to_string())
}

#[tauri::command]
fn remove_manual_tag(
    state: tauri::State<AppState>,
    photo_id: i64,
    tag: String,
) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::remove_tag(&conn, photo_id, &tag).map_err(|e| e.to_string())
}

#[tauri::command]
async fn rerun_auto(state: tauri::State<'_, AppState>, photo_id: i64) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    let photo = db::get_photo(&conn, photo_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| Error::Init("Photo not found".into()).to_string())?;
    let preview = photo.photo.preview_path.clone();
    let exif = crate::models::ExifMetadata {
        make: photo.photo.make.clone(),
        model: photo.photo.model.clone(),
        lens: photo.photo.lens.clone(),
        body_serial: None,
        datetime_original: photo.photo.date_taken,
        iso: photo.photo.iso,
        fnumber: photo.photo.fnumber,
        focal_length: photo.photo.focal_length,
        exposure_time: photo.photo.exposure_time,
        exposure_comp: photo.photo.exposure_comp,
        gps_lat: photo.photo.gps_lat,
        gps_lng: photo.photo.gps_lng,
        width: photo.photo.width,
        height: photo.photo.height,
    };
    let config = state.tagging.lock().unwrap().clone();
    let pool = state.db.clone();

    tauri::async_runtime::spawn_blocking(move || -> InvokeResult<()> {
        let mut engine = TaggingEngine::new(config).map_err(|e| e.to_string())?;
        let Some(preview) = preview.as_ref() else {
            return Ok(());
        };
        let conn = pool.get().map_err(|e| e.to_string())?;
        match catch_unwind(AssertUnwindSafe(|| {
            engine.classify(std::path::Path::new(preview), &exif)
        })) {
            Ok(Ok(tagging)) => {
                db::replace_auto_tags(&conn, photo_id, tagging, &exif)
                    .map_err(|e| e.to_string())?;
            }
            Ok(Err(err)) => {
                log::warn!("Auto tagging failed for {}: {}", preview, err);
            }
            Err(_) => {
                log::warn!(
                    "ONNX runtime panicked while tagging {}; skipping auto tags",
                    preview
                );
            }
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn get_inference_status(state: tauri::State<AppState>) -> InvokeResult<InferenceStatus> {
    let config = state.tagging.lock().unwrap().clone();
    Ok(tagging::inference_status(&config))
}

#[tauri::command]
fn set_inference_device(
    state: tauri::State<AppState>,
    device: InferenceDevicePreference,
) -> InvokeResult<InferenceStatus> {
    {
        let mut config = state.tagging.lock().unwrap();
        config.inference_device = device;
    }
    tagging::clear_session_cache();
    let config = state.tagging.lock().unwrap().clone();
    Ok(tagging::inference_status(&config))
}

#[tauri::command]
fn test_inference(
    state: tauri::State<AppState>,
    count: Option<u32>,
) -> InvokeResult<()> {
    if !cfg!(debug_assertions) {
        return Ok(());
    }
    let limit = count.unwrap_or(12).clamp(1, 200);
    let config = state.tagging.lock().unwrap().clone();
    let pool = state.db.clone();
    std::thread::spawn(move || {
        let conn = match pool.get() {
            Ok(conn) => conn,
            Err(err) => {
                log::warn!("Test inference: failed to get DB connection: {err}");
                return;
            }
        };
        let mut filters = QueryFilters::default();
        filters.limit = Some(limit as i64);
        let photos = match db::query_photos(&conn, filters) {
            Ok(photos) => photos,
            Err(err) => {
                log::warn!("Test inference: query failed: {err}");
                return;
            }
        };
        let mut engine = match TaggingEngine::new(config) {
            Ok(engine) => engine,
            Err(err) => {
                log::warn!("Test inference: tagging engine init failed: {err}");
                return;
            }
        };
        let mut processed = 0usize;
        for photo in photos {
            let Some(preview) = photo.photo.preview_path.as_deref() else {
                continue;
            };
            let exif = crate::models::ExifMetadata {
                make: photo.photo.make.clone(),
                model: photo.photo.model.clone(),
                lens: photo.photo.lens.clone(),
                body_serial: None,
                datetime_original: photo.photo.date_taken,
                iso: photo.photo.iso,
                fnumber: photo.photo.fnumber,
                focal_length: photo.photo.focal_length,
                exposure_time: photo.photo.exposure_time,
                exposure_comp: photo.photo.exposure_comp,
                gps_lat: photo.photo.gps_lat,
                gps_lng: photo.photo.gps_lng,
                width: photo.photo.width,
                height: photo.photo.height,
            };
            let start = std::time::Instant::now();
            let _ = engine.classify(std::path::Path::new(preview), &exif);
            let total = start.elapsed();
            log::info!(
                "Test inference: {} in {}ms",
                preview,
                total.as_millis()
            );
            processed += 1;
            if processed >= limit as usize {
                break;
            }
        }
        log::info!("Test inference complete: {processed} image(s)");
    });
    Ok(())
}

#[tauri::command]
fn set_rating(
    state: tauri::State<AppState>,
    photo_id: i64,
    rating: Option<i64>,
) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::set_rating(&conn, photo_id, rating).map_err(|e| e.to_string())
}

#[tauri::command]
fn toggle_picked(state: tauri::State<AppState>, photo_id: i64, value: bool) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::set_picked(&conn, photo_id, value).map_err(|e| e.to_string())
}

#[tauri::command]
fn toggle_rejected(state: tauri::State<AppState>, photo_id: i64, value: bool) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::set_rejected(&conn, photo_id, value).map_err(|e| e.to_string())
}

#[tauri::command]
fn batch_update_cull(
    state: tauri::State<AppState>,
    photo_ids: Vec<i64>,
    rating: Option<Option<i64>>,
    picked: Option<bool>,
    rejected: Option<bool>,
) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::batch_update_cull(&conn, &photo_ids, rating, picked, rejected)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_smart_views_counts(state: tauri::State<AppState>) -> InvokeResult<SmartViewCounts> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::get_smart_view_counts(&conn).map_err(|e| e.to_string())
}

#[tauri::command]
fn export_csv(
    state: tauri::State<AppState>,
    filters: QueryFilters,
) -> InvokeResult<Vec<crate::models::CsvExportRow>> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    db::export_csv(&conn, filters).map_err(|e| e.to_string())
}

#[tauri::command]
fn find_duplicates(
    state: tauri::State<AppState>,
    threshold: Option<u32>,
) -> InvokeResult<Vec<crate::models::DuplicateGroup>> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    let threshold = threshold.unwrap_or(8).min(20);
    db::find_duplicates(&conn, threshold).map_err(|e| e.to_string())
}

#[tauri::command]
fn find_similar(
    state: tauri::State<AppState>,
    photo_id: i64,
    limit: Option<i64>,
) -> InvokeResult<Vec<crate::models::SimilarPhoto>> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    let limit = limit.unwrap_or(12).clamp(1, 50);
    db::find_similar(&conn, photo_id, limit).map_err(|e| e.to_string())
}

#[tauri::command]
async fn import_folder(
    path: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> InvokeResult<String> {
    state
        .jobs
        .start_import(
            app,
            std::path::PathBuf::from(path),
            state.db.clone(),
            state.paths.clone(),
            state.tagging.lock().unwrap().clone(),
        )
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel_import(state: tauri::State<AppState>) -> InvokeResult<()> {
    state.jobs.cancel_current().map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel_import_file(state: tauri::State<AppState>, path: String) -> InvokeResult<()> {
    state.jobs.cancel_file(path).map_err(|e| e.to_string())
}

#[tauri::command]
fn is_importing(state: tauri::State<AppState>) -> InvokeResult<bool> {
    Ok(state.jobs.is_importing())
}

#[tauri::command]
fn is_directory(path: String) -> InvokeResult<bool> {
    std::fs::metadata(path)
        .map(|meta| meta.is_dir())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn show_in_folder(path: String) -> InvokeResult<()> {
    if path.trim().is_empty() {
        return Err("No file path provided".into());
    }
    Command::new("explorer")
        .arg(format!("/select,{}", path))
        .spawn()
        .map_err(|e| format!("Failed to open Explorer: {e}"))?;
    Ok(())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let context = tauri::generate_context!();
    let paths = AppPaths::discover(context.config()).expect("Failed to discover app paths");
    let models_dir_override = env::var_os("PHOTO_TAGGER_MODELS_DIR").map(PathBuf::from);
    let mut tagging = TaggingConfig::default();
    tagging.scene_model_path = resolve_model_path(
        &paths,
        models_dir_override.as_deref(),
        &tagging.scene_model_path,
        "PHOTO_TAGGER_SCENE_MODEL",
    );
    tagging.detection_model_path = resolve_model_path(
        &paths,
        models_dir_override.as_deref(),
        &tagging.detection_model_path,
        "PHOTO_TAGGER_DETECTION_MODEL",
    );
    tagging.face_model_path = resolve_model_path(
        &paths,
        models_dir_override.as_deref(),
        &tagging.face_model_path,
        "PHOTO_TAGGER_FACE_MODEL",
    );
    let db_pool = db::init_database(&paths).expect("Failed to initialize database");

    tauri::Builder::default()
        .manage(AppState {
            db: db_pool,
            paths,
            tagging: Arc::new(Mutex::new(tagging)),
            jobs: JobManager::default(),
        })
        .setup(|_app| Ok(()))
        .invoke_handler(tauri::generate_handler![
            greet,
            import_folder,
            cancel_import,
            cancel_import_file,
            is_importing,
            is_directory,
            show_in_folder,
            query_photos,
            add_manual_tag,
            remove_manual_tag,
            rerun_auto,
            export_csv,
            set_rating,
            toggle_picked,
            toggle_rejected,
            batch_update_cull,
            get_smart_views_counts,
            find_duplicates,
            find_similar,
            get_inference_status,
            set_inference_device,
            test_inference
        ])
        .run(context)
        .expect("error while running tauri application");
}
