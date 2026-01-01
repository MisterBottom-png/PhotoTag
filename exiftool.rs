use crate::config::AppPaths;
use crate::error::{Error, Result};
use crate::models::ExifMetadata;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Deserialize, Clone)]
struct ExifToolEntry {
    #[serde(rename = "Make")]
    make: Option<String>,
    #[serde(rename = "Model")]
    model: Option<String>,
    #[serde(rename = "BodySerialNumber")]
    body_serial: Option<String>,
    #[serde(rename = "LensModel")]
    lens_model: Option<String>,
    #[serde(rename = "Lens")]
    lens: Option<String>,
    #[serde(rename = "LensInfo")]
    lens_info: Option<String>,
    #[serde(rename = "LensMake")]
    lens_make: Option<String>,
    #[serde(rename = "DateTimeOriginal")]
    date_time_original: Option<String>,
    #[serde(rename = "CreateDate")]
    create_date: Option<String>,
    #[serde(rename = "ModifyDate")]
    modify_date: Option<String>,
    #[serde(rename = "ISO")]
    iso: Option<i64>,
    #[serde(rename = "FNumber")]
    fnumber: Option<f64>,
    #[serde(rename = "FocalLength")]
    focal_length: Option<f64>,
    #[serde(rename = "ExposureTime")]
    exposure_time: Option<f64>,
    #[serde(rename = "ExposureCompensation")]
    exposure_comp: Option<f64>,
    #[serde(rename = "GPSLatitude")]
    gps_lat: Option<f64>,
    #[serde(rename = "GPSLongitude")]
    gps_lng: Option<f64>,
    #[serde(rename = "ImageWidth")]
    width: Option<i64>,
    #[serde(rename = "ImageHeight")]
    height: Option<i64>,
}

fn parse_datetime(value: &Option<String>) -> Option<i64> {
    value.as_ref().and_then(|s| {
        chrono::NaiveDateTime::parse_from_str(s, "%Y:%m:%d %H:%M:%S")
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
            .ok()
            .map(|dt| chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc).timestamp())
    })
}

pub fn read_metadata(paths: &AppPaths, file_path: &Path) -> Result<ExifMetadata> {
    let exe = paths.resolve_bin("exiftool.exe");
    let output = Command::new(exe)
        .arg("-json")
        .arg(file_path)
        .output()
        .map_err(|e| Error::Init(format!("Failed to execute ExifTool: {e}")))?;

    if !output.status.success() {
        return Err(Error::Init(format!("ExifTool returned non-zero status for {:?}", file_path)));
    }

    let entries: Vec<ExifToolEntry> = serde_json::from_slice(&output.stdout)?;
    let entry = entries.get(0).cloned().unwrap_or(ExifToolEntry {
        make: None,
        model: None,
        body_serial: None,
        lens_model: None,
        lens: None,
        lens_info: None,
        lens_make: None,
        date_time_original: None,
        create_date: None,
        modify_date: None,
        iso: None,
        fnumber: None,
        focal_length: None,
        exposure_time: None,
        exposure_comp: None,
        gps_lat: None,
        gps_lng: None,
        width: None,
        height: None,
    });

    let lens_value = entry
        .lens_model
        .or(entry.lens)
        .or(entry.lens_info)
        .or(entry.lens_make);

    Ok(ExifMetadata {
        make: entry.make,
        model: entry.model,
        lens: lens_value,
        body_serial: entry.body_serial,
        datetime_original: parse_datetime(&entry.date_time_original)
            .or_else(|| parse_datetime(&entry.create_date))
            .or_else(|| parse_datetime(&entry.modify_date)),
        iso: entry.iso,
        fnumber: entry.fnumber,
        focal_length: entry.focal_length,
        exposure_time: entry.exposure_time,
        exposure_comp: entry.exposure_comp,
        gps_lat: entry.gps_lat,
        gps_lng: entry.gps_lng,
        width: entry.width,
        height: entry.height,
    })
}

pub fn extract_preview(paths: &AppPaths, file_path: &Path, out_path: &Path) -> Result<bool> {
    let ext = file_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "jpg" || ext == "jpeg" || ext == "png" {
        return Ok(false);
    }

    let exe = paths.resolve_bin("exiftool.exe");
    let output = Command::new(exe)
        .args(["-b", "-PreviewImage", "-JpgFromRaw", "-BigImage"])
        .arg(file_path)
        .output()
        .map_err(|e| Error::Init(format!("Failed to execute ExifTool: {e}")))?;

    if !output.status.success() || output.stdout.is_empty() {
        return Ok(false);
    }

    std::fs::write(out_path, &output.stdout)?;
    Ok(out_path.exists())
}
