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
use std::collections::{HashMap, HashSet};
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
    scene_labels: Vec<String>,
    scene_label_map: HashMap<String, Vec<String>>,
    detection_labels: Vec<String>,
    detection_label_map: HashMap<String, Vec<String>>,
}

impl TaggingEngine {
    pub fn new(config: TaggingConfig) -> Result<Self> {
        let enable_onnx = match env::var("PHOTO_TAGGER_ENABLE_ONNX")
            .ok()
            .as_deref()
            .map(|v| v.to_ascii_lowercase())
        {
            Some(v) if v == "0" || v == "false" => false,
            Some(v) if v == "1" || v == "true" => true,
            Some(_) => true,
            None => true,
        };
        if !enable_onnx {
            log::warn!("ONNX inference disabled; set PHOTO_TAGGER_ENABLE_ONNX=1 to enable.");
            return Ok(Self {
                scene_session: None,
                detection_session: None,
                face_session: None,
                config,
                onnx_enabled: false,
                scene_labels: Vec::new(),
                scene_label_map: HashMap::new(),
                detection_labels: Vec::new(),
                detection_label_map: HashMap::new(),
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
        let scene_labels = load_labels_from_model(&scene_path);
        let scene_label_map = load_label_map(&scene_path);
        let detection_labels = load_labels_from_model(&detect_path);
        let detection_label_map = load_label_map(&detect_path);

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
            scene_labels,
            scene_label_map,
            detection_labels,
            detection_label_map,
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
        let scene_probs = match safe_run(|| self.run_scene(preview_path)) {
            Ok(map) => map,
            Err(err) => {
                log::warn!("Scene model failed for {}: {}", preview_path.display(), err);
                HashMap::new()
            }
        };
        let detection_probs = match safe_run(|| self.run_detection(preview_path)) {
            Ok(map) => map,
            Err(err) => {
                log::warn!(
                    "Detection model failed for {}: {}",
                    preview_path.display(),
                    err
                );
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

        let mut tags: HashMap<String, f32> = HashMap::new();
        let detection_set: HashSet<String> = detection_probs.keys().cloned().collect();
        for (tag, score) in scene_probs {
            if !detection_set.is_empty()
                && DETECTION_REQUIRED_TAGS.contains(&tag.as_str())
                && !detection_set.contains(&tag)
            {
                continue;
            }
            let mut adjusted = score;
            if !detection_set.is_empty() && !detection_set.contains(&tag) {
                adjusted *= SCENE_UNRELATED_PENALTY;
            }
            tags.insert(tag, adjusted);
        }
        for (tag, score) in detection_probs {
            let boosted = (score + DETECTION_TAG_BOOST).min(1.0);
            let entry = tags.entry(tag).or_insert(0.0);
            if boosted > *entry {
                *entry = boosted;
            }
        }
        if portrait_score > 0.0 {
            let entry = tags.entry("portrait".into()).or_insert(0.0);
            if portrait_score > *entry {
                *entry = portrait_score;
            }
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
        let mut best_mode = ScenePreprocess::Imagenet;
        let mut logits = run_scene_logits(session, &resized, nchw, w, h, best_mode)?;
        let mut best_top1 = top1_prob(&logits);
        for mode in [ScenePreprocess::Raw01, ScenePreprocess::TfMinus1] {
            let candidate = run_scene_logits(session, &resized, nchw, w, h, mode)?;
            let top1 = top1_prob(&candidate);
            if top1 > best_top1 {
                best_top1 = top1;
                best_mode = mode;
                logits = candidate;
            }
        }
        if best_mode != ScenePreprocess::Imagenet {
            log::info!(
                "Scene model used {best_mode:?} input (top1 {:.2}) for {}",
                best_top1,
                preview_path.display()
            );
        }
        let mut map = HashMap::new();
        if !logits.is_empty() {
            if !self.scene_labels.is_empty() {
                let max_labels = logits.len().min(self.scene_labels.len());
                let probs = softmax(&logits[..max_labels]);
                let scored: Vec<(String, f32)> = self.scene_labels[..max_labels]
                    .iter()
                    .cloned()
                    .zip(probs.into_iter())
                    .collect();

                if !self.scene_label_map.is_empty() {
                    let mut group_scores: HashMap<String, f32> = HashMap::new();
                    let mut group_counts: HashMap<String, usize> = HashMap::new();
                    let mut topk: Vec<(String, f32)> = scored.clone();
                    topk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    if topk.len() > SCENE_GROUP_TOPK {
                        topk.truncate(SCENE_GROUP_TOPK);
                    }
                    for (label, _) in topk.iter() {
                        if let Some(tags) = self.scene_label_map.get(label) {
                            for tag in tags {
                                *group_counts.entry(tag.clone()).or_insert(0) += 1;
                            }
                        }
                    }
                    for (label, prob) in scored.iter() {
                        if let Some(tags) = self.scene_label_map.get(label) {
                            for tag in tags {
                                let entry = group_scores.entry(tag.clone()).or_insert(0.0);
                                *entry += *prob;
                            }
                        }
                    }
                    group_scores.retain(|tag, _| {
                        group_counts
                            .get(tag)
                            .copied()
                            .unwrap_or(0)
                            >= SCENE_GROUP_MIN_LABELS
                    });
                    let mut grouped: Vec<(String, f32)> = group_scores.into_iter().collect();
                    grouped.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let mut added = 0usize;
                    for (tag, score) in grouped.iter() {
                        if *score >= self.config.confidence_threshold {
                            map.insert(tag.clone(), *score);
                            added += 1;
                            if added >= MAX_SCENE_TAGS {
                                break;
                            }
                        }
                    }
                    if added < MAX_SCENE_TAGS {
                        for (tag, score) in grouped.iter() {
                            if map.contains_key(tag) {
                                continue;
                            }
                            if *score >= self.config.suggestion_threshold {
                                map.insert(tag.clone(), *score);
                                added += 1;
                                if added >= MAX_SCENE_TAGS {
                                    break;
                                }
                            }
                        }
                    }
                    if added == 0 {
                        for (tag, score) in grouped.iter().take(MAX_SCENE_TAGS) {
                            map.insert(tag.clone(), *score);
                        }
                    }
                } else {
                    let mut scored = scored;
                    scored.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut processed = std::collections::HashSet::new();
                    let mut added = 0usize;
                    for (label, prob) in scored.iter() {
                        if *prob >= self.config.confidence_threshold {
                            processed.insert(label.clone());
                            added += apply_scene_label(&mut map, &self.scene_label_map, label, *prob);
                            if added >= MAX_SCENE_TAGS {
                                break;
                            }
                        }
                    }
                    if added < MAX_SCENE_TAGS {
                        for (label, prob) in scored.iter() {
                            if processed.contains(label) {
                                continue;
                            }
                            if *prob >= self.config.suggestion_threshold {
                                processed.insert(label.clone());
                                added += apply_scene_label(
                                    &mut map,
                                    &self.scene_label_map,
                                    label,
                                    *prob,
                                );
                                if added >= MAX_SCENE_TAGS {
                                    break;
                                }
                            }
                        }
                    }
                    if added == 0 {
                        for (label, prob) in scored.iter().take(MAX_SCENE_TAGS) {
                            added += apply_scene_label(&mut map, &self.scene_label_map, label, *prob);
                        }
                    }
                }
            } else {
                // Fallback for legacy fixed tags when no labels sidecar exists.
                let probs = softmax_first_n(&logits, 3);
                map.insert("street".into(), probs.get(0).copied().unwrap_or(0.0));
                map.insert("landscape".into(), probs.get(1).copied().unwrap_or(0.0));
                map.insert("nature".into(), probs.get(2).copied().unwrap_or(0.0));
            }
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

    fn run_detection(&mut self, preview_path: &Path) -> Result<HashMap<String, f32>> {
        if self.detection_session.is_none() {
            return Ok(HashMap::new());
        }
        let session = self.detection_session.as_mut().unwrap();
        let (w, h) = model_input_hw(session, 640, 640);
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
        .map_err(|e| Error::Init(format!("Invalid detection tensor shape: {e}")))?;
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session
            .run(vec![input_tensor])
            .map_err(|e| Error::Init(format!("Failed to run detection model: {e}")))?;
        if !outputs.is_empty() {
            let shapes = outputs
                .iter()
                .map(|o| format!("{:?}", o.shape()))
                .collect::<Vec<_>>()
                .join(", ");
            log::info!("Detection outputs: {shapes}");
        }
        if detection_outputs_pair(&outputs) && self.detection_labels.len() != 2 {
            log::warn!(
                "Detection labels count ({}) does not match 2-class detector; update person_detector.labels.txt",
                self.detection_labels.len()
            );
            return Ok(HashMap::new());
        }
        let class_scores = detection_class_scores(&outputs);
        if class_scores.is_empty() {
            log::warn!(
                "Detection model returned no class scores for {}",
                preview_path.display()
            );
            return Ok(HashMap::new());
        }
        if let Some((best_id, best_score)) = class_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            if let Some(label) = self.detection_labels.get(*best_id) {
                log::info!(
                    "Top detection for {}: {} ({:.2})",
                    preview_path.display(),
                    label,
                    best_score
                );
            }
        }
        let mut tags = HashMap::new();
        for (class_id, score) in class_scores {
            let label = match self.detection_labels.get(class_id) {
                Some(name) => name.as_str(),
                None => continue,
            };
            if !self.detection_label_map.is_empty() {
                apply_detection_label(&mut tags, &self.detection_label_map, label, score);
            } else if let Some(tag) = default_detection_tag(label) {
                let entry = tags.entry(tag.to_string()).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            }
        }
        Ok(tags)
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

fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    for v in values {
        let e = (v - max_val).exp();
        exps.push(e);
        sum += e;
    }
    if sum <= 0.0 {
        return vec![0.0; values.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

fn resolve_labels_path(model_path: &Path) -> Option<std::path::PathBuf> {
    let labels_path = model_path.with_extension("labels.txt");
    if labels_path.exists() {
        return Some(labels_path);
    }
    if let Some(stem) = model_path.file_stem().and_then(|s| s.to_str()) {
        let fallback = Path::new("models").join(format!("{stem}.labels.txt"));
        if fallback.exists() {
            return Some(fallback);
        }
    }
    None
}

fn load_labels_from_model(model_path: &Path) -> Vec<String> {
    let labels_path = match resolve_labels_path(model_path) {
        Some(path) => path,
        None => {
            log::warn!(
                "No labels sidecar found for scene model: {}",
                model_path.display()
            );
            return Vec::new();
        }
    };
    let contents = match std::fs::read_to_string(&labels_path) {
        Ok(data) => data,
        Err(err) => {
            log::warn!(
                "Failed to read labels from {}: {}",
                labels_path.display(),
                err
            );
            return Vec::new();
        }
    };
    let mut labels = Vec::new();
    for line in contents.lines() {
        if let Some(label) = normalize_label(line) {
            labels.push(label);
        }
    }
    if labels.is_empty() {
        log::warn!(
            "Labels file is empty or invalid: {}",
            labels_path.display()
        );
    }
    labels
}

fn load_label_map(model_path: &Path) -> HashMap<String, Vec<String>> {
    let mut tried: Vec<std::path::PathBuf> = Vec::new();
    let mut map_path = model_path.with_extension("tags.txt");
    tried.push(map_path.clone());
    if !map_path.exists() {
        if let Some(labels_path) = resolve_labels_path(model_path) {
            if let Some(name) = labels_path.file_name().and_then(|n| n.to_str()) {
                if let Some(stem) = name.strip_suffix(".labels.txt") {
                    let candidate =
                        labels_path.with_file_name(format!("{stem}.tags.txt"));
                    tried.push(candidate.clone());
                    if candidate.exists() {
                        map_path = candidate;
                    }
                }
            }
        }
    }
    if !map_path.exists() {
        if let Some(stem) = model_path.file_stem().and_then(|s| s.to_str()) {
            let fallback = Path::new("models").join(format!("{stem}.tags.txt"));
            tried.push(fallback.clone());
            if fallback.exists() {
                map_path = fallback;
            }
        }
    }
    if !map_path.exists() {
        if !tried.is_empty() {
            let tried_list = tried
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join("; ");
            log::warn!("No tag map found; tried: {tried_list}");
        }
        return HashMap::new();
    }
    let contents = match std::fs::read_to_string(&map_path) {
        Ok(data) => data,
        Err(err) => {
            log::warn!(
                "Failed to read tag map from {}: {}",
                map_path.display(),
                err
            );
            return HashMap::new();
        }
    };
    let mut map = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (tag, rest) = match line.split_once(':') {
            Some(pair) => pair,
            None => match line.split_once('=') {
                Some(pair) => pair,
                None => continue,
            },
        };
        let tag = tag.trim().to_lowercase();
        if tag.is_empty() {
            continue;
        }
        for raw in rest.split(',') {
            if let Some(label) = normalize_label(raw) {
                map.entry(label).or_insert_with(Vec::new).push(tag.clone());
            }
        }
    }
    map
}

fn normalize_label(line: &str) -> Option<String> {
    let mut label = line.trim();
    if label.is_empty() {
        return None;
    }
    if let Some((prefix, rest)) = label.split_once(':') {
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            label = rest.trim();
        }
    } else {
        let mut parts = label.splitn(2, char::is_whitespace);
        let first = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("");
        if !first.is_empty() && first.chars().all(|c| c.is_ascii_digit()) && !rest.is_empty() {
            label = rest.trim();
        }
    }
    if let Some((head, _)) = label.split_once(',') {
        label = head.trim();
    }
    label = label.trim_matches('"').trim_matches('\'');
    if label.is_empty() {
        return None;
    }
    Some(label.to_lowercase())
}

const MAX_SCENE_TAGS: usize = 5;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCENE_GROUP_TOPK: usize = 10;
const SCENE_GROUP_MIN_LABELS: usize = 2;
const SCENE_UNRELATED_PENALTY: f32 = 0.6;
const DETECTION_MIN_SCORE: f32 = 0.25;
const DETECTION_PAIR_FOREGROUND_INDEX: usize = 1;
const DETECTION_TAG_BOOST: f32 = 0.30;
const DETECTION_REQUIRED_TAGS: &[&str] = &[
    "amphibian",
    "bird",
    "cat",
    "clothing",
    "dog",
    "electronic",
    "fish",
    "food",
    "furniture",
    "insect_invertebrate",
    "instrument",
    "mammal_other",
    "reptile",
    "sport",
    "tool",
    "vehicle",
];

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ScenePreprocess {
    Imagenet,
    Raw01,
    TfMinus1,
}

fn apply_scene_label(
    map: &mut HashMap<String, f32>,
    label_map: &HashMap<String, Vec<String>>,
    label: &str,
    score: f32,
) -> usize {
    if label_map.is_empty() {
        return match map.entry(label.to_string()) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(score);
                1
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if score > *entry.get() {
                    entry.insert(score);
                }
                0
            }
        };
    }

    if let Some(tags) = label_map.get(label) {
        let mut added = 0usize;
        for tag in tags {
            match map.entry(tag.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(score);
                    added += 1;
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if score > *entry.get() {
                        entry.insert(score);
                    }
                }
            }
        }
        return added;
    }
    0
}

fn apply_detection_label(
    map: &mut HashMap<String, f32>,
    label_map: &HashMap<String, Vec<String>>,
    label: &str,
    score: f32,
) {
    if let Some(tags) = label_map.get(label) {
        for tag in tags {
            let entry = map.entry(tag.clone()).or_insert(0.0);
            if score > *entry {
                *entry = score;
            }
        }
    }
}

fn default_detection_tag(label: &str) -> Option<&'static str> {
    match label {
        "person" => Some("person"),
        "cat" => Some("cat"),
        "dog" => Some("dog"),
        "bird" => Some("bird"),
        "horse" | "sheep" | "cow" | "elephant" | "bear" | "zebra" | "giraffe" => Some("animal"),
        "bicycle"
        | "car"
        | "motorcycle"
        | "airplane"
        | "bus"
        | "train"
        | "truck"
        | "boat" => Some("vehicle"),
        _ => None,
    }
}

fn detection_class_scores(outputs: &[OrtOwnedTensor<f32, IxDyn>]) -> HashMap<usize, f32> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    if outputs.len() == 2 {
        if let Some(paired) = detection_scores_from_pair(outputs) {
            return paired;
        }
    }
    for output in outputs {
        let shape = output.shape();
        let Some(slice) = output.as_slice() else {
            continue;
        };
        if shape.len() >= 4 && shape[shape.len() - 1] >= 5 {
            let stride = shape[shape.len() - 1] as usize;
            for row in slice.chunks_exact(stride) {
                let obj = sigmoid(row[4]);
                if !obj.is_finite() {
                    continue;
                }
                for (idx, prob) in row[5..].iter().enumerate() {
                    if !prob.is_finite() {
                        continue;
                    }
                    let score = (obj * sigmoid(*prob)).max(0.0).min(1.0);
                    if score < DETECTION_MIN_SCORE {
                        continue;
                    }
                    let entry = scores.entry(idx).or_insert(0.0);
                    if score > *entry {
                        *entry = score;
                    }
                }
            }
        } else if shape.len() == 3 {
            let dim1 = shape[1] as usize;
            let dim2 = shape[2] as usize;
            if dim1 >= 5 && dim2 > dim1 {
                for col in 0..dim2 {
                    let base = col * dim1;
                    let obj = sigmoid(slice[base + 4]);
                    if !obj.is_finite() {
                        continue;
                    }
                    for cls in 0..(dim1 - 5) {
                        let prob = slice[base + 5 + cls];
                        if !prob.is_finite() {
                            continue;
                        }
                        let score = (obj * sigmoid(prob)).max(0.0).min(1.0);
                        if score < DETECTION_MIN_SCORE {
                            continue;
                        }
                        let entry = scores.entry(cls).or_insert(0.0);
                        if score > *entry {
                            *entry = score;
                        }
                    }
                }
            } else if dim2 >= 5 && dim1 > dim2 {
                let stride = dim2;
                for row in slice.chunks_exact(stride) {
                    let obj = sigmoid(row[4]);
                    if !obj.is_finite() {
                        continue;
                    }
                    for (idx, prob) in row[5..].iter().enumerate() {
                        if !prob.is_finite() {
                            continue;
                        }
                        let score = (obj * sigmoid(*prob)).max(0.0).min(1.0);
                        if score < DETECTION_MIN_SCORE {
                            continue;
                        }
                        let entry = scores.entry(idx).or_insert(0.0);
                        if score > *entry {
                            *entry = score;
                        }
                    }
                }
            }
        } else if shape.len() == 2 {
            let classes = shape[shape.len() - 1] as usize;
            if classes == 0 {
                continue;
            }
            let logits = &slice[..slice.len().min(classes)];
            let probs = softmax(logits);
            for (idx, prob) in probs.into_iter().enumerate() {
                if prob < DETECTION_MIN_SCORE {
                    continue;
                }
                let entry = scores.entry(idx).or_insert(0.0);
                if prob > *entry {
                    *entry = prob;
                }
            }
        }
    }
    scores
}

fn detection_scores_from_pair(
    outputs: &[OrtOwnedTensor<f32, IxDyn>],
) -> Option<HashMap<usize, f32>> {
    let mut scores_out: Option<&OrtOwnedTensor<f32, IxDyn>> = None;
    let mut boxes_out: Option<&OrtOwnedTensor<f32, IxDyn>> = None;
    for output in outputs {
        let shape = output.shape();
        if shape.len() == 3 && shape[2] == 2 {
            scores_out = Some(output);
        } else if shape.len() == 3 && shape[2] == 4 {
            boxes_out = Some(output);
        }
    }
    let scores_out = scores_out?;
    let _boxes_out = boxes_out?;
    let shape = scores_out.shape();
    let Some(slice) = scores_out.as_slice() else {
        return None;
    };
    let cols = shape[2] as usize;
    if cols != 2 {
        return None;
    }
    let mut scores: HashMap<usize, f32> = HashMap::new();
    for row in slice.chunks_exact(cols) {
        let raw = row.get(DETECTION_PAIR_FOREGROUND_INDEX).copied().unwrap_or(0.0);
        if !raw.is_finite() {
            continue;
        }
        let score = sigmoid(raw);
        if score < DETECTION_MIN_SCORE {
            continue;
        }
        let entry = scores.entry(DETECTION_PAIR_FOREGROUND_INDEX).or_insert(0.0);
        if score > *entry {
            *entry = score;
        }
    }
    Some(scores)
}

fn detection_outputs_pair(outputs: &[OrtOwnedTensor<f32, IxDyn>]) -> bool {
    if outputs.len() != 2 {
        return false;
    }
    let mut has_scores = false;
    let mut has_boxes = false;
    for output in outputs {
        let shape = output.shape();
        if shape.len() == 3 && shape[2] == 2 {
            has_scores = true;
        } else if shape.len() == 3 && shape[2] == 4 {
            has_boxes = true;
        }
    }
    has_scores && has_boxes
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
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
        } else if shape.len() >= 2 && shape[shape.len() - 1] >= 5 {
            // YOLO-style output: [..., 5 + classes] => [x, y, w, h, obj, ...class_probs]
            if let Some(slice) = output.as_slice() {
                let stride = shape[shape.len() - 1] as usize;
                for row in slice.chunks_exact(stride) {
                    let obj = row[4];
                    if !obj.is_finite() {
                        continue;
                    }
                    let class_max = if row.len() > 5 {
                        row[5..]
                            .iter()
                            .cloned()
                            .filter(|v| v.is_finite())
                            .fold(0.0, f32::max)
                    } else {
                        1.0
                    };
                    let score = (obj * class_max).max(0.0).min(1.0);
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

fn rgb32f_to_nhwc_normalized(img: &image::Rgb32FImage) -> Vec<f32> {
    let mut input: Vec<f32> = Vec::with_capacity((img.width() * img.height() * 3) as usize);
    for pixel in img.pixels() {
        input.extend_from_slice(&[
            (pixel[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
            (pixel[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
            (pixel[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
        ]);
    }
    input
}

fn rgb32f_to_nchw_normalized(img: &image::Rgb32FImage, w: u32, h: u32) -> Vec<f32> {
    let plane = (w * h) as usize;
    let mut input = vec![0.0; plane * 3];
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        input[idx] = (pixel[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        input[idx + plane] = (pixel[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        input[idx + plane * 2] = (pixel[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
    input
}

fn rgb32f_to_nhwc_tf(img: &image::Rgb32FImage) -> Vec<f32> {
    let mut input: Vec<f32> = Vec::with_capacity((img.width() * img.height() * 3) as usize);
    for pixel in img.pixels() {
        input.extend_from_slice(&[
            (pixel[0] * 2.0) - 1.0,
            (pixel[1] * 2.0) - 1.0,
            (pixel[2] * 2.0) - 1.0,
        ]);
    }
    input
}

fn rgb32f_to_nchw_tf(img: &image::Rgb32FImage, w: u32, h: u32) -> Vec<f32> {
    let plane = (w * h) as usize;
    let mut input = vec![0.0; plane * 3];
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        input[idx] = (pixel[0] * 2.0) - 1.0;
        input[idx + plane] = (pixel[1] * 2.0) - 1.0;
        input[idx + plane * 2] = (pixel[2] * 2.0) - 1.0;
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

fn run_scene_logits(
    session: &mut Session<'_>,
    resized: &image::Rgb32FImage,
    nchw: bool,
    w: u32,
    h: u32,
    mode: ScenePreprocess,
) -> Result<Vec<f32>> {
    let input = match (mode, nchw) {
        (ScenePreprocess::Imagenet, true) => rgb32f_to_nchw_normalized(resized, w, h),
        (ScenePreprocess::Imagenet, false) => rgb32f_to_nhwc_normalized(resized),
        (ScenePreprocess::Raw01, true) => rgb32f_to_nchw(resized, w, h),
        (ScenePreprocess::Raw01, false) => rgb32f_to_nhwc(resized),
        (ScenePreprocess::TfMinus1, true) => rgb32f_to_nchw_tf(resized, w, h),
        (ScenePreprocess::TfMinus1, false) => rgb32f_to_nhwc_tf(resized),
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
        log::warn!("Scene model returned no outputs");
        return Ok(Vec::new());
    }
    Ok(outputs
        .get(0)
        .and_then(|t| t.as_slice())
        .unwrap_or(&[])
        .to_vec())
}

fn top1_prob(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in logits {
        sum += (v - max_val).exp();
    }
    if sum <= 0.0 {
        return 0.0;
    }
    1.0 / sum
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
        let res = engine.heuristic_tags(&dummy, &ExifMetadata::default());
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
