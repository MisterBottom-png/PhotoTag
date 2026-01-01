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
use crate::error::{Error, Result};
use crate::models::{PhotoWithTags, QueryFilters};
use crate::scan::scan_folder;
use crate::tagging::TaggingEngine;
use tauri::Manager;

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
fn query_photos(state: tauri::State<AppState>, filters: QueryFilters) -> Result<Vec<PhotoWithTags>> {
    let conn = state.db.get()?;
    db::query_photos(&conn, filters)
}

#[tauri::command]
fn add_manual_tag(state: tauri::State<AppState>, photo_id: i64, tag: String) -> Result<()> {
    let conn = state.db.get()?;
    db::add_manual_tag(&conn, photo_id, &tag)
}

#[tauri::command]
fn remove_manual_tag(state: tauri::State<AppState>, photo_id: i64, tag: String) -> Result<()> {
    let conn = state.db.get()?;
    db::remove_tag(&conn, photo_id, &tag)
}

#[tauri::command]
fn rerun_auto(state: tauri::State<AppState>, photo_id: i64) -> Result<()> {
    let conn = state.db.get()?;
    let photo = db::get_photo(&conn, photo_id)?.ok_or_else(|| Error::Init("Photo not found".into()))?;
    let engine = TaggingEngine::new(state.tagging.clone())?;
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
        let tagging = engine.classify(std::path::Path::new(preview), &exif)?;
        db::replace_auto_tags(&conn, photo_id, tagging, &exif)?;
    }
    Ok(())
}

#[tauri::command]
fn export_csv(state: tauri::State<AppState>, filters: QueryFilters) -> Result<Vec<crate::models::CsvExportRow>> {
    let conn = state.db.get()?;
    db::export_csv(&conn, filters)
}

#[tauri::command]
async fn import_folder(path: String, app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<()> {
    let db = state.db.clone();
    let paths = state.paths.clone();
    let tagging = state.tagging.clone();
    tauri::async_runtime::spawn_blocking(move || scan_folder(app, std::path::PathBuf::from(path), db, paths, tagging))
        .await
        .map_err(|e| Error::Init(format!("Task join error: {e}")))??;
    Ok(())
}

fn main() {
    env_logger::init();

    let paths = AppPaths::discover().expect("Failed to discover app paths");
    let tagging = TaggingConfig::default();
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
            export_csv
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
