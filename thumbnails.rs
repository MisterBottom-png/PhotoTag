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

fn resize_dims(width: u32, height: u32, max_dim: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (max_dim.max(1), max_dim.max(1));
    }
    let scale_w = max_dim as f32 / width as f32;
    let scale_h = max_dim as f32 / height as f32;
    let scale = scale_w.min(scale_h);
    let new_w = (width as f32 * scale).round().max(1.0) as u32;
    let new_h = (height as f32 * scale).round().max(1.0) as u32;
    (new_w, new_h)
}

fn resize_image(input: &Path, output: &Path, max_dim: u32) -> Result<()> {
    let img = image::open(input)?;
    let (dst_w, dst_h) = resize_dims(img.width(), img.height(), max_dim);
    let mut used_gpu = false;
    #[cfg(target_os = "windows")]
    {
        if let Ok(gpu_resized) = crate::gpu::resize_rgba8(&img.to_rgba8(), dst_w, dst_h) {
            gpu_resized.save(output)?;
            used_gpu = true;
        }
    }
    if !used_gpu {
        let resized = img.resize(max_dim, max_dim, FilterType::CatmullRom);
        resized.save(output)?;
    }
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
