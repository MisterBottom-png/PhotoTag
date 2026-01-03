use crate::error::Result;
use image::imageops::FilterType;
use std::path::Path;

pub fn compute_embedding(path: &Path) -> Result<Vec<f32>> {
    let img = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(&img, 64, 64, FilterType::Triangle);
    #[cfg(target_os = "windows")]
    if crate::gpu::gpu_preprocess_enabled() {
        if let Ok(hist) = crate::gpu::histogram_embedding(&resized) {
        return Ok(hist);
        }
    }
    let bins = 16usize;
    let mut hist = vec![0f32; bins * 3];
    for pixel in resized.pixels() {
        let r = (pixel[0] as usize * bins) / 256;
        let g = (pixel[1] as usize * bins) / 256;
        let b = (pixel[2] as usize * bins) / 256;
        hist[r] += 1.0;
        hist[bins + g] += 1.0;
        hist[2 * bins + b] += 1.0;
    }
    Ok(hist)
}

pub fn normalize_embedding(vec: &[f32]) -> (Vec<f32>, f32) {
    let mut norm = 0.0f32;
    for v in vec {
        norm += v * v;
    }
    norm = norm.sqrt().max(1e-6);
    let normalized = vec.iter().map(|v| v / norm).collect();
    (normalized, norm)
}

pub fn serialize_embedding(vec: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vec.len() * 4);
    for v in vec {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn deserialize_embedding(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
