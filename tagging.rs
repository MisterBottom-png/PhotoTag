use crate::config::TaggingConfig;
use crate::error::{Error, Result};
use crate::models::{ExifMetadata, TaggingResult};
use image::imageops::FilterType;
use lazy_static::lazy_static;
use onnxruntime::{
    environment::Environment,
    ndarray::Array,
    session::Session,
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel, LoggingLevel,
};
use std::collections::HashMap;
use std::env;
use std::path::Path;

lazy_static! {
    static ref ORT_ENV: Option<Environment> = Environment::builder()
        .with_name("photo-tagging")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .ok();
}

pub struct TaggingEngine {
    scene_session: Option<Session<'static>>,
    detection_session: Option<Session<'static>>,
    config: TaggingConfig,
    onnx_enabled: bool,
}

impl TaggingEngine {
    pub fn new(config: TaggingConfig) -> Result<Self> {
        let enable_onnx = env::var("PHOTO_TAGGER_ENABLE_ONNX").ok().as_deref() == Some("1");
        if !enable_onnx {
            log::warn!("ONNX inference disabled; set PHOTO_TAGGER_ENABLE_ONNX=1 to enable.");
            return Ok(Self {
                scene_session: None,
                detection_session: None,
                config,
                onnx_enabled: false,
            });
        }

        let scene_path = config.scene_model_path.clone();
        let detect_path = config.detection_model_path.clone();
        if !scene_path.exists() {
            log::warn!("Scene model not found: {}", scene_path.display());
        }
        if !detect_path.exists() {
            log::warn!("Detection model not found: {}", detect_path.display());
        }

        let scene_model_path: &'static Path =
            Box::leak(config.scene_model_path.clone().into_boxed_path());
        let detection_model_path: &'static Path =
            Box::leak(config.detection_model_path.clone().into_boxed_path());

        let scene_session = ORT_ENV.as_ref().and_then(|env| {
            env.new_session_builder()
                .ok()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Basic).ok())
                .and_then(|b| b.with_model_from_file(scene_model_path).ok())
        });

        let detection_session = ORT_ENV.as_ref().and_then(|env| {
            env.new_session_builder()
                .ok()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Basic).ok())
                .and_then(|b| b.with_model_from_file(detection_model_path).ok())
        });
        if scene_session.is_some() {
            log::info!("Loaded scene model: {}", scene_path.display());
        } else {
            log::warn!("Failed to load scene model: {}", scene_path.display());
        }
        if detection_session.is_some() {
            log::info!("Loaded detection model: {}", detect_path.display());
        } else {
            log::warn!("Failed to load detection model: {}", detect_path.display());
        }

        Ok(Self {
            scene_session,
            detection_session,
            config,
            onnx_enabled: true,
        })
    }

    pub fn classify(&mut self, preview_path: &Path, exif: &ExifMetadata) -> Result<TaggingResult> {
        let mut tags: HashMap<String, f32> = HashMap::new();
        let scene_probs = match self.run_scene(preview_path) {
            Ok(map) => map,
            Err(err) => {
                log::warn!("Scene model failed for {}: {}", preview_path.display(), err);
                HashMap::new()
            }
        };
        let portrait_score = match self.run_portrait(preview_path, exif) {
            Ok(score) => score,
            Err(err) => {
                log::warn!("Detection model failed for {}: {}", preview_path.display(), err);
                0.0
            }
        };

        if let Some(street) = scene_probs.get("street") {
            tags.insert("street".into(), *street);
        }
        if let Some(landscape) = scene_probs.get("landscape") {
            tags.insert("landscape".into(), *landscape);
        }
        if let Some(nature) = scene_probs.get("nature") {
            tags.insert("nature".into(), *nature);
        }
        if portrait_score > 0.0 {
            tags.insert("portrait".into(), portrait_score);
        }
        if tags.is_empty() && !self.onnx_enabled {
            tags.extend(self.heuristic_tags(preview_path, exif));
        }
        if tags.is_empty() {
            log::info!("No tags produced for {}", preview_path.display());
        }

        Ok(TaggingResult { tags })
    }

    fn run_scene(&mut self, preview_path: &Path) -> Result<HashMap<String, f32>> {
        if self.scene_session.is_none() {
            return Ok(HashMap::new());
        }

        let session = self.scene_session.as_mut().unwrap();
        let (w, h) = model_input_hw(session, 224, 224);
        let img = image::open(preview_path)?;
        let resized = img.resize_exact(w, h, FilterType::Triangle).to_rgb32f();
        let nchw = model_expects_nchw(session);
        let input = if nchw {
            rgb32f_to_nchw(&resized, w, h)
        } else {
            rgb32f_to_nhwc(&resized)
        };
        let input_tensor = if nchw {
            Array::from_shape_vec((1, 3, h as usize, w as usize), input)
        } else {
            Array::from_shape_vec((1, h as usize, w as usize, 3), input)
        }
        .map_err(|e| Error::Init(format!("Invalid scene tensor shape: {e}")))?;
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session
            .run(vec![input_tensor])
            .map_err(|e| Error::Init(format!("Failed to run scene model: {e}")))?;

        if outputs.is_empty() {
            log::warn!("Scene model returned no outputs for {}", preview_path.display());
        }
        let logits = outputs.get(0).and_then(|t| t.as_slice()).unwrap_or(&[]);
        // simple mapping: assume first few indices map to our labels; in practice user should update mapping
        let mut map = HashMap::new();
        if !logits.is_empty() {
            let probs = softmax_first_n(logits, 3);
            map.insert("street".into(), probs.get(0).copied().unwrap_or(0.0));
            map.insert("landscape".into(), probs.get(1).copied().unwrap_or(0.0));
            map.insert("nature".into(), probs.get(2).copied().unwrap_or(0.0));
        }
        Ok(map)
    }

    fn run_portrait(&mut self, preview_path: &Path, exif: &ExifMetadata) -> Result<f32> {
        if self.detection_session.is_none() {
            return Ok(0.0);
        }
        let session = self.detection_session.as_mut().unwrap();
        let (w, h) = model_input_hw(session, 224, 224);
        let img = image::open(preview_path)?;
        let resized = img.resize_exact(w, h, FilterType::Triangle).to_rgb8();
        let nchw = model_expects_nchw(session);
        let input = if nchw {
            rgb8_to_nchw(&resized, w, h)
        } else {
            rgb8_to_nhwc(&resized)
        };
        let input_tensor = if nchw {
            Array::from_shape_vec((1, 3, h as usize, w as usize), input)
        } else {
            Array::from_shape_vec((1, h as usize, w as usize, 3), input)
        }
        .map_err(|e| Error::Init(format!("Invalid detector tensor shape: {e}")))?;
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session
            .run(vec![input_tensor])
            .map_err(|e| Error::Init(format!("Failed to run detector: {e}")))?;
        if outputs.is_empty() {
            log::warn!("Detection model returned no outputs for {}", preview_path.display());
        }
        let scores = outputs.get(0).and_then(|t| t.as_slice()).unwrap_or(&[]);
        let max_score = scores
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max)
            .max(0.0)
            .min(1.0);
        log::info!(
            "Detection score for {}: {:.4} (threshold {:.4})",
            preview_path.display(),
            max_score,
            self.config.portrait_min_area_ratio
        );

        let focal_boost = exif
            .focal_length
            .map(|f| if f > 70.0 { 0.1 } else { 0.0 })
            .unwrap_or(0.0);
        let score = (max_score + focal_boost).min(1.0).max(0.0);
        if score >= self.config.portrait_min_area_ratio {
            Ok(score)
        } else {
            Ok(0.0)
        }
    }

    fn heuristic_tags(&self, preview_path: &Path, exif: &ExifMetadata) -> HashMap<String, f32> {
        let mut tags = HashMap::new();
        let name = preview_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        if name.contains("street") {
            tags.insert("street".into(), 0.6);
        }
        if let (Some(w), Some(h)) = (exif.width, exif.height) {
            if w > h + (h / 5) {
                tags.insert("landscape".into(), 0.5);
            } else if h > w + (w / 5) {
                tags.insert("portrait".into(), 0.5);
            }
        }
        if let Some(focal) = exif.focal_length {
            if focal >= 70.0 {
                tags.insert("portrait".into(), 0.6);
            }
        }
        if exif.gps_lat.is_some() || exif.gps_lng.is_some() {
            tags.insert("nature".into(), 0.4);
        }
        tags
    }
}

fn model_expects_nchw(session: &Session<'_>) -> bool {
    let dims: Vec<Option<u32>> = session.inputs.get(0).map(|i| i.dimensions.clone()).unwrap_or_default();
    if dims.len() == 4 {
        match (dims[1], dims[2], dims[3]) {
            (Some(3), Some(_), Some(_)) => return true,
            (Some(_), Some(_), Some(3)) => return false,
            _ => {}
        }
    }
    true
}

fn model_input_hw(session: &Session<'_>, default_w: u32, default_h: u32) -> (u32, u32) {
    let dims: Vec<Option<u32>> = session.inputs.get(0).map(|i| i.dimensions.clone()).unwrap_or_default();
    if dims.len() == 4 {
        if let (Some(h), Some(w)) = (dims[2], dims[3]) {
            return (w, h);
        }
        if let (Some(h), Some(w)) = (dims[1], dims[2]) {
            return (w, h);
        }
    }
    (default_w, default_h)
}

fn softmax_first_n(values: &[f32], n: usize) -> Vec<f32> {
    if values.is_empty() || n == 0 {
        return Vec::new();
    }
    let slice = &values[..values.len().min(n)];
    let max_val = slice
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(slice.len());
    let mut sum = 0.0;
    for v in slice {
        let e = (v - max_val).exp();
        exps.push(e);
        sum += e;
    }
    if sum <= 0.0 {
        return vec![0.0; slice.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

fn rgb32f_to_nhwc(img: &image::Rgb32FImage) -> Vec<f32> {
    let mut input: Vec<f32> = Vec::with_capacity((img.width() * img.height() * 3) as usize);
    for pixel in img.pixels() {
        input.extend_from_slice(&[pixel[0], pixel[1], pixel[2]]);
    }
    input
}

fn rgb32f_to_nchw(img: &image::Rgb32FImage, w: u32, h: u32) -> Vec<f32> {
    let plane = (w * h) as usize;
    let mut input = vec![0.0; plane * 3];
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        input[idx] = pixel[0];
        input[idx + plane] = pixel[1];
        input[idx + plane * 2] = pixel[2];
    }
    input
}

fn rgb8_to_nhwc(img: &image::RgbImage) -> Vec<f32> {
    let mut input: Vec<f32> = Vec::with_capacity((img.width() * img.height() * 3) as usize);
    for pixel in img.pixels() {
        input.extend_from_slice(&[
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        ]);
    }
    input
}

fn rgb8_to_nchw(img: &image::RgbImage, w: u32, h: u32) -> Vec<f32> {
    let plane = (w * h) as usize;
    let mut input = vec![0.0; plane * 3];
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        input[idx] = pixel[0] as f32 / 255.0;
        input[idx + plane] = pixel[1] as f32 / 255.0;
        input[idx + plane * 2] = pixel[2] as f32 / 255.0;
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn fallback_scene_uses_filename() {
        let mut engine = TaggingEngine::new(TaggingConfig::default()).unwrap();
        let dummy = PathBuf::from("street_sample.jpg");
        let res = engine.run_scene(&dummy).unwrap();
        assert!(res.get("street").copied().unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn portrait_requires_detector() {
        let mut engine = TaggingEngine::new(TaggingConfig::default()).unwrap();
        let score = engine
            .run_portrait(Path::new("portrait.jpg"), &ExifMetadata::default())
            .unwrap();
        assert_eq!(score, 0.0);
    }
}
