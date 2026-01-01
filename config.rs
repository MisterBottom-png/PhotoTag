use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::api::path::{app_data_dir, resource_dir};
use tauri::{Config, Env, PackageInfo};

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

        // Copy bundled resources or fall back to local dev folders.
        let package_info = PackageInfo {
            name: "PhotoTagger".to_string(),
            version: env!("CARGO_PKG_VERSION").parse().unwrap(),
            authors: env!("CARGO_PKG_AUTHORS"),
            description: env!("CARGO_PKG_DESCRIPTION"),
        };
        let env = Env::default();
        if let Some(resource_root) = resource_dir(&package_info, &env) {
            let _ = copy_dir_recursive(&resource_root.join("bin"), &bin_dir);
            let _ = copy_dir_recursive(&resource_root.join("models"), &models_dir);
        } else {
            let _ = copy_dir_recursive(Path::new("./bin"), &bin_dir);
            let _ = copy_dir_recursive(Path::new("./models"), &models_dir);
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

fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            std::fs::copy(&src_path, &dest_path)?;
        }
    }
    Ok(())
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
