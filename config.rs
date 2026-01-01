use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::api::path::app_data_dir;
use tauri::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggingConfig {
    pub scene_model_path: PathBuf,
    pub detection_model_path: PathBuf,
    pub confidence_threshold: f32,
    pub suggestion_threshold: f32,
    pub portrait_min_area_ratio: f32,
}

impl Default for TaggingConfig {
    fn default() -> Self {
        Self {
            scene_model_path: PathBuf::from("models/scene_classifier.onnx"),
            detection_model_path: PathBuf::from("models/face_detector.onnx"),
            confidence_threshold: 0.65,
            suggestion_threshold: 0.45,
            portrait_min_area_ratio: 0.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppPaths {
    pub root: PathBuf,
    pub db_path: PathBuf,
    pub thumbs_dir: PathBuf,
    pub previews_dir: PathBuf,
    pub models_dir: PathBuf,
    pub bin_dir: PathBuf,
}

impl AppPaths {
    pub fn discover() -> Result<Self, crate::error::Error> {
        let app_root = app_data_dir(&Config::default())
            .ok_or_else(|| crate::error::Error::Path("Failed to get app data dir".to_string()))?
            .join("PhotoCatalogApp");

        let db_path = app_root.join("library.db");
        let thumbs_dir = app_root.join("thumbs");
        let previews_dir = app_root.join("previews");
        let models_dir = app_root.join("models");
        let bin_dir = app_root.join("bin");

        std::fs::create_dir_all(&thumbs_dir)?;
        std::fs::create_dir_all(&previews_dir)?;
        std::fs::create_dir_all(&models_dir)?;
        std::fs::create_dir_all(&bin_dir)?;

        // Copy bundled binaries/models from the working directory into the app data dir for portability during dev.
        if let Ok(entries) = std::fs::read_dir("./bin") {
            for entry in entries.flatten() {
                let dest = bin_dir.join(entry.file_name());
                let _ = std::fs::copy(entry.path(), dest);
            }
        }
        if let Ok(entries) = std::fs::read_dir("./models") {
            for entry in entries.flatten() {
                let dest = models_dir.join(entry.file_name());
                let _ = std::fs::copy(entry.path(), dest);
            }
        }

        Ok(Self {
            root: app_root,
            db_path,
            thumbs_dir,
            previews_dir,
            models_dir,
            bin_dir,
        })
    }

    pub fn resolve_bin(&self, name: &str) -> PathBuf {
        self.bin_dir.join(name)
    }

    pub fn resolve_model(&self, name: &Path) -> PathBuf {
        if name.is_absolute() {
            name.to_path_buf()
        } else {
            self.models_dir.join(name)
        }
    }

    pub fn ensure_subdir(&self, dir: &Path) -> Result<PathBuf, crate::error::Error> {
        std::fs::create_dir_all(dir)?;
        Ok(dir.to_path_buf())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub tagging: TaggingConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tagging: TaggingConfig::default(),
        }
    }
}
