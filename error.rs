use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database Pool Error: {0}")]
    DbPool(#[from] r2d2::Error),

    #[error("Database Error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Tauri API Error: {0}")]
    Tauri(#[from] tauri::Error),

    #[error("Json Error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Image Error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Path Error: {0}")]
    Path(String),

    #[error("Initialization Failed: {0}")]
    Init(String),
}
