// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod db;
mod error;
mod schema;

use tauri::Manager;
use db::DbPool;

pub struct AppState {
    db: DbPool,
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

fn main() {
    // Initialize logging
    env_logger::init();

    // Initialize the database
    let db_pool = db::init_database().expect("Failed to initialize database");

    tauri::Builder::default()
        .manage(AppState { db: db_pool })
        .setup(|app| {
            // Any additional setup can go here
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}