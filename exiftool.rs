use crate::config::AppPaths;
use crate::error::{Error, Result};
use crate::models::ExifMetadata;
use serde_json::Value;
use std::path::Path;
use std::process::Command;

fn parse_datetime(value: &Option<String>) -> Option<i64> {
    value.as_ref().and_then(|s| {
        chrono::NaiveDateTime::parse_from_str(s, "%Y:%m:%d %H:%M:%S")
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
            .ok()
            .map(|dt| {
                chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc)
                    .timestamp()
            })
    })
}

pub fn read_metadata(paths: &AppPaths, file_path: &Path) -> Result<ExifMetadata> {
    let exe = paths.resolve_bin("exiftool.exe");
    let output = Command::new(exe)
        .args(["-json", "-n"])
        .arg(file_path)
        .output()
        .map_err(|e| Error::Init(format!("Failed to execute ExifTool: {e}")))?;

    if !output.status.success() {
        return Err(Error::Init(format!(
            "ExifTool returned non-zero status for {:?}",
            file_path
        )));
    }

    let entries: Vec<Value> = serde_json::from_slice(&output.stdout)?;
    let entry = entries.get(0).cloned().unwrap_or(Value::Null);

    let lens_value = get_string(&entry, "LensModel")
        .or_else(|| get_string(&entry, "Lens"))
        .or_else(|| get_string(&entry, "LensInfo"))
        .or_else(|| get_string(&entry, "LensMake"));

    let make = get_string(&entry, "Make");
    let model = normalize_model(&make, get_string(&entry, "Model"));

    Ok(ExifMetadata {
        make,
        model,
        lens: lens_value,
        body_serial: get_string(&entry, "BodySerialNumber"),
        datetime_original: parse_datetime_value(&entry, "DateTimeOriginal")
            .or_else(|| parse_datetime_value(&entry, "CreateDate"))
            .or_else(|| parse_datetime_value(&entry, "ModifyDate")),
        iso: get_i64(&entry, "ISO"),
        fnumber: get_f64(&entry, "FNumber"),
        focal_length: get_f64(&entry, "FocalLength"),
        exposure_time: get_f64(&entry, "ExposureTime"),
        exposure_comp: get_f64(&entry, "ExposureCompensation"),
        gps_lat: get_f64(&entry, "GPSLatitude"),
        gps_lng: get_f64(&entry, "GPSLongitude"),
        width: get_i64(&entry, "ImageWidth"),
        height: get_i64(&entry, "ImageHeight"),
    })
}

fn normalize_model(make: &Option<String>, model: Option<String>) -> Option<String> {
    let model = model?;
    let model_trim = model.trim();
    if model_trim.is_empty() {
        return None;
    }
    let Some(make) = make.as_ref() else {
        return Some(model_trim.to_string());
    };
    let make_trim = make.trim();
    if make_trim.is_empty() {
        return Some(model_trim.to_string());
    }

    let make_lc = make_trim.to_ascii_lowercase();
    let model_lc = model_trim.to_ascii_lowercase();
    if model_lc.starts_with(&make_lc) {
        let remainder = &model_trim[make_trim.len()..];
        let mut chars = remainder.chars();
        match chars.next() {
            None => return Some(model_trim.to_string()),
            Some(c) if c.is_whitespace() || c == '-' || c == '_' || c == '/' => {
                let cleaned = remainder
                    .trim_start_matches(|c: char| c.is_whitespace() || c == '-' || c == '_' || c == '/');
                if cleaned.is_empty() {
                    return Some(model_trim.to_string());
                }
                return Some(cleaned.to_string());
            }
            _ => {}
        }
    }

    Some(model_trim.to_string())
}

fn parse_datetime_value(entry: &Value, key: &str) -> Option<i64> {
    let value = get_string(entry, key);
    parse_datetime(&value)
}

fn get_string(entry: &Value, key: &str) -> Option<String> {
    entry.get(key).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    })
}

fn get_i64(entry: &Value, key: &str) -> Option<i64> {
    entry.get(key).and_then(|v| match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    })
}

fn get_f64(entry: &Value, key: &str) -> Option<f64> {
    entry.get(key).and_then(|v| match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
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
