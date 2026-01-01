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
}

impl TaggingEngine {
    pub fn new(config: TaggingConfig) -> Result<Self> {
        let scene_session = ORT_ENV.as_ref().and_then(|env| {
            env.new_session_builder()
                .ok()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Basic).ok())
                .and_then(|b| b.with_model_from_file(&config.scene_model_path).ok())
        });

        let detection_session = ORT_ENV.as_ref().and_then(|env| {
            env.new_session_builder()
                .ok()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Basic).ok())
                .and_then(|b| b.with_model_from_file(&config.detection_model_path).ok())
        });

        Ok(Self {
            scene_session,
            detection_session,
            config,
        })
    }

    pub fn classify(&mut self, preview_path: &Path, exif: &ExifMetadata) -> Result<TaggingResult> {
        let mut tags: HashMap<String, f32> = HashMap::new();
        let scene_probs = self.run_scene(preview_path).unwrap_or_default();
        let portrait_score = self.run_portrait(preview_path, exif).unwrap_or(0.0);

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

        Ok(TaggingResult { tags })
    }

    fn run_scene(&mut self, preview_path: &Path) -> Result<HashMap<String, f32>> {
        if self.scene_session.is_none() {
            // simple heuristic fallback based on filename
            let name = preview_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();
            let mut map = HashMap::new();
            if name.contains("street") {
                map.insert("street".into(), 0.7);
            }
            return Ok(map);
        }

        let session = self.scene_session.as_mut().unwrap();
        let img = image::open(preview_path)?;
        let resized = img.resize(224, 224, FilterType::Triangle).to_rgb32f();
        let mut input: Vec<f32> = Vec::with_capacity(224 * 224 * 3);
        for pixel in resized.pixels() {
            input.extend_from_slice(&[pixel[0], pixel[1], pixel[2]]);
        }
        let input_tensor = Array::from_shape_vec((1, 224, 224, 3), input)
            .map_err(|e| Error::Init(format!("Invalid scene tensor shape: {e}")))?;
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session
            .run(vec![input_tensor])
            .map_err(|e| Error::Init(format!("Failed to run scene model: {e}")))?;

        let logits = outputs[0].as_slice().unwrap_or(&[]);
        // simple mapping: assume first few indices map to our labels; in practice user should update mapping
        let mut map = HashMap::new();
        if !logits.is_empty() {
            map.insert("street".into(), logits.get(0).copied().unwrap_or(0.0));
            map.insert("landscape".into(), logits.get(1).copied().unwrap_or(0.0));
            map.insert("nature".into(), logits.get(2).copied().unwrap_or(0.0));
        }
        Ok(map)
    }

    fn run_portrait(&mut self, preview_path: &Path, exif: &ExifMetadata) -> Result<f32> {
        if self.detection_session.is_none() {
            return Ok(0.0);
        }
        let session = self.detection_session.as_mut().unwrap();
        let img = image::open(preview_path)?;
        let resized = img.resize(320, 320, FilterType::Triangle).to_rgb8();
        let mut input: Vec<f32> = Vec::with_capacity((320 * 320 * 3) as usize);
        for p in resized.pixels() {
            input.extend_from_slice(&[p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]);
        }
        let input_tensor = Array::from_shape_vec((1, 320, 320, 3), input)
            .map_err(|e| Error::Init(format!("Invalid detector tensor shape: {e}")))?;
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session
            .run(vec![input_tensor])
            .map_err(|e| Error::Init(format!("Failed to run detector: {e}")))?;
        let scores = outputs[0].as_slice().unwrap_or(&[]);
        let max_score = scores.iter().cloned().fold(0.0, f32::max);

        let focal_boost = exif
            .focal_length
            .map(|f| if f > 70.0 { 0.1 } else { 0.0 })
            .unwrap_or(0.0);
        let score = (max_score + focal_boost).min(1.0);
        if score >= self.config.portrait_min_area_ratio {
            Ok(score)
        } else {
            Ok(0.0)
        }
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
