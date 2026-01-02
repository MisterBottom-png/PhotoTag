// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod db;
mod error;
mod exiftool;
mod models;
mod scan;
mod schema;
mod tagging;
mod thumbnails;

use crate::config::{AppPaths, TaggingConfig};
use crate::db::DbPool;
use crate::error::Error;
use crate::models::{PhotoWithTags, QueryFilters, SmartViewCounts};
use crate::scan::scan_folder;
use crate::tagging::TaggingEngine;
use std::panic::{catch_unwind, AssertUnwindSafe};

type InvokeResult<T> = std::result::Result<T, String>;

pub struct AppState {
    db: DbPool,
    paths: AppPaths,
    tagging: TaggingConfig,
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
fn rerun_auto(state: tauri::State<AppState>, photo_id: i64) -> InvokeResult<()> {
    let conn = state.db.get().map_err(|e| e.to_string())?;
    let photo = db::get_photo(&conn, photo_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| Error::Init("Photo not found".into()).to_string())?;
    let mut engine = TaggingEngine::new(state.tagging.clone()).map_err(|e| e.to_string())?;
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
    if let Some(preview) = &photo.photo.preview_path {
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
    }
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
async fn import_folder(
    path: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> InvokeResult<()> {
    let db = state.db.clone();
    let paths = state.paths.clone();
    let tagging = state.tagging.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        scan_folder(app, std::path::PathBuf::from(path), db, paths, tagging)
    })
    .await
    .map_err(|e| format!("Task join error: {e}"))?;
    result.map_err(|e| e.to_string())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let context = tauri::generate_context!();
    let paths = AppPaths::discover(context.config()).expect("Failed to discover app paths");
    let mut tagging = TaggingConfig::default();
    tagging.scene_model_path = paths.resolve_model(&tagging.scene_model_path);
    tagging.detection_model_path = paths.resolve_model(&tagging.detection_model_path);
    tagging.face_model_path = paths.resolve_model(&tagging.face_model_path);
    let db_pool = db::init_database(&paths).expect("Failed to initialize database");

    tauri::Builder::default()
        .manage(AppState {
            db: db_pool,
            paths,
            tagging,
        })
        .setup(|_app| Ok(()))
        .invoke_handler(tauri::generate_handler![
            greet,
            import_folder,
            query_photos,
            add_manual_tag,
            remove_manual_tag,
            rerun_auto,
            export_csv,
            set_rating,
            toggle_picked,
            toggle_rejected,
            batch_update_cull,
            get_smart_views_counts
        ])
        .run(context)
        .expect("error while running tauri application");
}
