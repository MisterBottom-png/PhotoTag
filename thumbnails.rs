use crate::error::Result;
use image::imageops::FilterType;
use std::path::{Path, PathBuf};

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "tiff", "tif", "bmp", "gif", "webp"];

pub fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn resize_image(input: &Path, output: &Path, max_dim: u32) -> Result<()> {
    let img = image::open(input)?;
    let resized = img.resize(max_dim, max_dim, FilterType::CatmullRom);
    resized.save(output)?;
    Ok(())
}

pub fn build_thumbnail(preview: &Path, dest_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;
    let filename = preview
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("thumb.jpg");
    let output = dest_dir.join(filename);
    resize_image(preview, &output, 320)?;
    Ok(output)
}

pub fn build_preview(original_or_preview: &Path, dest_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;
    let filename = original_or_preview
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("preview.jpg");
    let output = dest_dir.join(filename);
    resize_image(original_or_preview, &output, 1600)?;
    Ok(output)
}
