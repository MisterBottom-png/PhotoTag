use crate::config::TaggingConfig;
use crate::error::{Error, Result};
use crate::models::{ExifMetadata, TaggingResult};
use image::imageops::FilterType;
use lazy_static::lazy_static;
use onnxruntime::{
    environment::Environment,
    ndarray::{Array, IxDyn},
    session::Session,
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel, LoggingLevel,
};
use std::collections::HashMap;
use std::env;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

lazy_static! {
    static ref ORT_ENV: Option<&'static Environment> = Environment::builder()
        .with_name("photo-tagging")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .ok()
        .map(|env| Box::leak(Box::new(env)))
        .map(|env_ref| env_ref as &'static Environment);
}

pub struct TaggingEngine {
    scene_session: Option<Session<'static>>,
    detection_session: Option<Session<'static>>,
    face_session: Option<Session<'static>>,
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
                face_session: None,
                config,
                onnx_enabled: false,
            });
        }

        let scene_path = config.scene_model_path.clone();
        let detect_path = config.detection_model_path.clone();
        let face_path = config.face_model_path.clone();
        if !scene_path.exists() {
            log::warn!("Scene model not found: {}", scene_path.display());
        }
        if !detect_path.exists() {
            log::warn!("Detection model not found: {}", detect_path.display());
        }
        if !face_path.exists() {
            log::warn!("Face model not found: {}", face_path.display());
        }

        let scene_model_path: &'static Path =
            Box::leak(config.scene_model_path.clone().into_boxed_path());
        let detection_model_path: &'static Path =
            Box::leak(config.detection_model_path.clone().into_boxed_path());
        let face_model_path: &'static Path =
            Box::leak(config.face_model_path.clone().into_boxed_path());

        let scene_session = ORT_ENV
            .as_ref()
            .copied()
            .and_then(|env| safe_session(env, scene_model_path, "scene"));

        let detection_session = ORT_ENV
            .as_ref()
            .copied()
            .and_then(|env| safe_session(env, detection_model_path, "detection"));
        let face_session = ORT_ENV
            .as_ref()
            .copied()
            .and_then(|env| safe_session(env, face_model_path, "face"));
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
        if face_session.is_some() {
            log::info!("Loaded face model: {}", face_path.display());
        } else {
            log::warn!("Failed to load face model: {}", face_path.display());
        }

        let onnx_enabled =
            scene_session.is_some() || detection_session.is_some() || face_session.is_some();
        Ok(Self {
            scene_session,
            detection_session,
            face_session,
            config,
            onnx_enabled,
        })
    }

    pub fn disable_onnx(&mut self) {
        self.scene_session = None;
        self.detection_session = None;
        self.face_session = None;
        self.onnx_enabled = false;
        log::warn!("ONNX disabled after runtime failure; continuing with heuristics only.");
    }

    pub fn classify(&mut self, preview_path: &Path, exif: &ExifMetadata) -> Result<TaggingResult> {
        let mut tags: HashMap<String, f32> = HashMap::new();
        let scene_probs = match safe_run(|| self.run_scene(preview_path)) {
            Ok(map) => map,
            Err(err) => {
                log::warn!("Scene model failed for {}: {}", preview_path.display(), err);
                HashMap::new()
            }
        };
        let portrait_score = match safe_run(|| self.run_portrait(preview_path, exif)) {
            Ok(score) => score,
            Err(err) => {
                log::warn!(
                    "Face model failed for {}: {}",
                    preview_path.display(),
                    err
                );
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
            log::warn!(
                "Scene model returned no outputs for {}",
                preview_path.display()
            );
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
        let face_score = self.run_face(preview_path)?;
        if face_score <= 0.0 {
            return Ok(0.0);
        }
        let focal_boost = exif
            .focal_length
            .map(|f| if f > 70.0 { 0.1 } else { 0.0 })
            .unwrap_or(0.0);
        let score = (face_score + focal_boost).min(1.0).max(0.0);
        Ok(score)
    }

    fn run_face(&mut self, preview_path: &Path) -> Result<f32> {
        if self.face_session.is_none() {
            return Ok(0.0);
        }
        let session = self.face_session.as_mut().unwrap();
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
            .map_err(|e| Error::Init(format!("Failed to run face detector: {e}")))?;
        if outputs.is_empty() {
            log::warn!(
                "Face model returned no outputs for {}",
                preview_path.display()
            );
        }
        let max_score = max_face_score(&outputs).unwrap_or(0.0).max(0.0).min(1.0);
        log::info!(
            "Face score for {}: {:.4} (threshold {:.4})",
            preview_path.display(),
            max_score,
            self.config.face_min_score
        );
        if max_score >= self.config.face_min_score {
            Ok(max_score)
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
    let dims: Vec<Option<u32>> = session
        .inputs
        .get(0)
        .map(|i| i.dimensions.clone())
        .unwrap_or_default();
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
    let dims: Vec<Option<u32>> = session
        .inputs
        .get(0)
        .map(|i| i.dimensions.clone())
        .unwrap_or_default();
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
    let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

fn max_face_score(outputs: &[OrtOwnedTensor<f32, IxDyn>]) -> Option<f32> {
    let mut best = None;
    for output in outputs {
        let shape = output.shape();
        if shape.len() >= 2 && shape[shape.len() - 1] == 2 {
            if let Some(slice) = output.as_slice() {
                for pair in slice.chunks_exact(2) {
                    let score = pair[1];
                    if !score.is_finite() {
                        continue;
                    }
                    best = Some(best.map_or(score, |b: f32| b.max(score)));
                }
            }
        }
    }
    best
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

fn safe_session(
    env: &'static Environment,
    model_path: &'static Path,
    label: &str,
) -> Option<Session<'static>> {
    match catch_unwind(AssertUnwindSafe(|| {
        env.new_session_builder()
            .ok()
            .and_then(|b| {
                b.with_optimization_level(GraphOptimizationLevel::Basic)
                    .ok()
            })
            .and_then(|b| b.with_model_from_file(model_path).ok())
    })) {
        Ok(session) => session,
        Err(_) => {
            log::warn!(
                "ONNX session panicked while loading {label} model: {}",
                model_path.display()
            );
            None
        }
    }
}

fn safe_run<T, F>(f: F) -> std::result::Result<T, String>
where
    F: FnOnce() -> std::result::Result<T, Error>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(res) => res.map_err(|e| format!("{e}")),
        Err(_) => Err("ONNX runtime panic".into()),
    }
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
