use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashMap;
use std::path::PathBuf;

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhotoRecord {
    pub id: Option<i64>,
    pub path: String,
    pub hash: String,
    pub file_name: String,
    pub ext: String,
    pub size: i64,
    pub mtime: i64,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    pub date_taken: Option<i64>,
    pub iso: Option<i64>,
    pub fnumber: Option<f64>,
    pub focal_length: Option<f64>,
    pub exposure_time: Option<f64>,
    pub exposure_comp: Option<f64>,
    pub gps_lat: Option<f64>,
    pub gps_lng: Option<f64>,
    pub thumb_path: Option<String>,
    pub preview_path: Option<String>,
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRecord {
    pub id: Option<i64>,
    pub photo_id: i64,
    pub tag: String,
    pub confidence: Option<f32>,
    pub source: String,
    pub locked: bool,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryFilters {
    pub search: Option<String>,
    pub tags: Vec<String>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub lens: Option<String>,
    pub iso_min: Option<i64>,
    pub iso_max: Option<i64>,
    pub aperture_min: Option<f64>,
    pub aperture_max: Option<f64>,
    pub focal_min: Option<f64>,
    pub focal_max: Option<f64>,
    pub date_from: Option<i64>,
    pub date_to: Option<i64>,
    pub has_gps: Option<bool>,
    pub sort_by: Option<String>,
    pub sort_dir: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhotoWithTags {
    pub photo: PhotoRecord,
    pub tags: Vec<TagRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImportProgressEvent {
    pub discovered: usize,
    pub processed: usize,
    pub current_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExifMetadata {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    pub body_serial: Option<String>,
    pub datetime_original: Option<i64>,
    pub iso: Option<i64>,
    pub fnumber: Option<f64>,
    pub focal_length: Option<f64>,
    pub exposure_time: Option<f64>,
    pub exposure_comp: Option<f64>,
    pub gps_lat: Option<f64>,
    pub gps_lng: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaggingResult {
    pub tags: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvExportRow {
    pub filename: String,
    pub path: String,
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub date: Option<i64>,
    pub iso: Option<i64>,
    pub fnumber: Option<f64>,
    pub focal: Option<f64>,
    pub shutter: Option<f64>,
    pub tags: Vec<String>,
}
