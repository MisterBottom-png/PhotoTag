use crate::config::{InferenceDevicePreference, TaggingConfig};
use crate::error::{Error, Result};
use crate::models::{ExifMetadata, InferenceModelStatus, InferenceStatus, TaggingResult};
use crate::onnx::{self, InferenceProvider, OrtRuntimeConfig, ProviderChoice};
use image::imageops::FilterType;
use lazy_static::lazy_static;
use ndarray::Array;
use ort::session::Session;
use ort::value::TensorRef;
use std::collections::{HashMap, HashSet};
use std::env;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

lazy_static! {
    static ref SESSION_CACHE: Mutex<HashMap<SessionCacheKey, Arc<SessionHandle>>> =
        Mutex::new(HashMap::new());
    static ref INFERENCE_WARNING: Mutex<Option<String>> = Mutex::new(None);
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SessionCacheKey {
    model_path: String,
    provider: ProviderChoice,
    device_id: Option<u32>,
}

struct SessionHandle {
    session: Mutex<Session>,
    provider: InferenceProvider,
    label: &'static str,
    model_path: &'static Path,
}

#[derive(Default, Clone)]
struct TimingStats {
    samples: usize,
    decode_preprocess_total: Duration,
    inference_total: Duration,
}

impl TimingStats {
    fn record(&mut self, decode_preprocess: Duration, inference: Duration) {
        self.samples = self.samples.saturating_add(1);
        self.decode_preprocess_total += decode_preprocess;
        self.inference_total += inference;
    }

    fn summary(&self) -> String {
        if self.samples == 0 {
            return "no samples".to_string();
        }
        let samples = self.samples as u32;
        let dp_ms = self.decode_preprocess_total.as_millis() as f64 / samples as f64;
        let inf_ms = self.inference_total.as_millis() as f64 / samples as f64;
        format!("avg decode+preprocess={dp_ms:.1}ms, avg inference={inf_ms:.1}ms")
    }
}

fn ort_runtime_version() -> Option<String> {
    onnx::ort_runtime_version()
}

fn log_runtime_diagnostics_once() {
    static LOGGED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if let Some(version) = ort_runtime_version() {
        log::info!("ONNX Runtime version: {}", version);
    } else {
        log::warn!("Unable to determine ONNX Runtime version");
    }
}

pub struct TaggingEngine {
    scene_session: Option<Arc<SessionHandle>>,
    detection_session: Option<Arc<SessionHandle>>,
    face_session: Option<Arc<SessionHandle>>,
    config: TaggingConfig,
    onnx_enabled: bool,
    scene_labels: Vec<String>,
    scene_label_map: HashMap<String, Vec<String>>,
    detection_labels: Vec<String>,
    detection_label_map: HashMap<String, Vec<String>>,
    scene_input: Vec<f32>,
    detection_input: Vec<f32>,
    face_input: Vec<f32>,
    timings: HashMap<&'static str, TimingStats>,
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
                scene_input: Vec::new(),
                detection_input: Vec::new(),
                face_input: Vec::new(),
                timings: HashMap::new(),
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

        let scene_labels = load_labels_from_model(&scene_path);
        let scene_label_map = load_label_map(&scene_path);
        let detection_labels = load_labels_from_model(&detect_path);
        let detection_label_map = load_label_map(&detect_path);
        if detection_labels.len() == 80 {
            if detection_labels
                .get(0)
                .map(|s| s.as_str())
                .unwrap_or_default()
                != "person"
            {
                log::warn!(
                    "Detection labels look non-COCO or misaligned: expected labels[0] == \"person\" for COCO-80."
                );
            }
        }

        let ort_cfg = ort_config_from_tagging(&config);
        let scene_session =
            get_or_create_session(scene_path.as_path(), "scene", ort_cfg, 224, 224);

        let detection_session =
            get_or_create_session(detect_path.as_path(), "detection", ort_cfg, 640, 640);
        let face_session =
            get_or_create_session(face_path.as_path(), "face", ort_cfg, 224, 224);
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
            scene_input: Vec::new(),
            detection_input: Vec::new(),
            face_input: Vec::new(),
            timings: HashMap::new(),
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

    fn record_timing(
        &mut self,
        label: &'static str,
        decode_preprocess: Duration,
        inference: Duration,
        provider: InferenceProvider,
        preview_path: &Path,
    ) {
        let entry = self.timings.entry(label).or_default();
        entry.record(decode_preprocess, inference);
        log::info!(
            "{label} timing for {}: decode+preprocess={}ms, inference={}ms, provider={} ({})",
            preview_path.display(),
            decode_preprocess.as_millis(),
            inference.as_millis(),
            provider.label(),
            entry.summary()
        );
    }

    fn run_scene(&mut self, preview_path: &Path) -> Result<HashMap<String, f32>> {
        if self.scene_session.is_none() {
            return Ok(HashMap::new());
        }

        let session_handle = self.scene_session.as_ref().unwrap().clone();
        let (w, h, nchw) = {
            let session = session_handle.session.lock().unwrap();
            let (w, h) = model_input_hw(&session, 224, 224);
            (w, h, model_expects_nchw(&session))
        };
        let decode_start = Instant::now();
        let img = image::open(preview_path)?;
        let resized = img.resize_exact(w, h, FilterType::Triangle).to_rgb32f();
        let mut decode_preprocess = decode_start.elapsed();
        let mut best_mode = ScenePreprocess::Imagenet;
        let (mut logits, prep_time, mut inference_total) = run_scene_logits(
            &session_handle,
            &resized,
            nchw,
            w,
            h,
            best_mode,
            &mut self.scene_input,
        )?;
        decode_preprocess += prep_time;
        let mut best_top1 = top1_prob(&logits);
        for mode in [ScenePreprocess::Raw01, ScenePreprocess::TfMinus1] {
            let (candidate, prep_time, infer_time) = run_scene_logits(
                &session_handle,
                &resized,
                nchw,
                w,
                h,
                mode,
                &mut self.scene_input,
            )?;
            decode_preprocess += prep_time;
            inference_total += infer_time;
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
        if cfg!(debug_assertions) {
            self.record_timing(
                "scene",
                decode_preprocess,
                inference_total,
                session_handle.provider,
                preview_path,
            );
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
        let session_handle = self.detection_session.as_ref().unwrap().clone();
        let (w, h, nchw) = {
            let session = session_handle.session.lock().unwrap();
            let (w, h) = model_input_hw(&session, 640, 640);
            (w, h, model_expects_nchw(&session))
        };
        let decode_start = Instant::now();
        let img = image::open(preview_path)?;
        let rgb = img.to_rgb8();
        let orig_w = rgb.width();
        let orig_h = rgb.height();
        // YOLOv5 expects letterboxed input; keep scale/padding to recover boxes.
        let (letterboxed, ratio, dw, dh) = letterbox_rgb(&rgb, w, h, 114);
        let input = if nchw {
            rgb8_to_nchw_into(&letterboxed, w, h, &mut self.detection_input);
            self.detection_input.clone()
        } else {
            rgb8_to_nhwc_into(&letterboxed, &mut self.detection_input);
            self.detection_input.clone()
        };
        let input_tensor = if nchw {
            Array::from_shape_vec((1, 3, h as usize, w as usize), input)
        } else {
            Array::from_shape_vec((1, h as usize, w as usize, 3), input)
        }
        .map_err(|e| Error::Init(format!("Invalid detection tensor shape: {e}")))?;
        let decode_preprocess = decode_start.elapsed();
        let infer_start = Instant::now();
        let mut session = session_handle.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![TensorRef::from_array_view(&input_tensor).map_err(
                |e| Error::Init(format!("Invalid detection tensor: {e}"))
            )?])
            .map_err(|e| Error::Init(format!("Failed to run detection model: {e}")))?;
        let inference_time = infer_start.elapsed();
        let output_tensors = collect_output_tensors(&outputs);
        if !output_tensors.is_empty() {
            let shapes = output_tensors
                .iter()
                .map(|o| format!("{:?}", o.shape))
                .collect::<Vec<_>>()
                .join(", ");
            log::info!("Detection outputs: {shapes}");
        }
        if detection_outputs_pair(&output_tensors) {
            let scores = detection_scores_from_pair(&output_tensors).unwrap_or_default();
            let score = scores
                .get(&DETECTION_PAIR_FOREGROUND_INDEX)
                .copied()
                .unwrap_or(0.0);
            if score <= 0.0 {
                return Ok(HashMap::new());
            }
            if self.detection_labels.len() != 2 {
                log::warn!(
                    "Detection outputs look like a 2-class detector; overriding labels and tagging as person."
                );
            }
            let label = if self.detection_labels.len() == 2 {
                self.detection_labels
                    .get(DETECTION_PAIR_FOREGROUND_INDEX)
                    .map(|s| s.as_str())
                    .unwrap_or("person")
            } else {
                "person"
            };
            let mut tags = HashMap::new();
            if !self.detection_label_map.is_empty() {
                apply_detection_label(&mut tags, &self.detection_label_map, label, score);
            } else if let Some(tag) = default_detection_tag(label) {
                let entry = tags.entry(tag.to_string()).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            } else {
                let entry = tags.entry(label.to_string()).or_insert(0.0);
                if score > *entry {
                    *entry = score;
                }
            }
            if cfg!(debug_assertions) {
                self.record_timing(
                    "detection",
                    decode_preprocess,
                    inference_time,
                    session_handle.provider,
                    preview_path,
                );
            }
            return Ok(tags);
        }
        if let Some(detections) = yolov5_detections_from_outputs(
            &output_tensors,
            ratio,
            dw,
            dh,
            orig_w,
            orig_h,
            self.config.detection_confidence_threshold,
            self.config.detection_iou_threshold,
        ) {
            if let Some(top) = detections.first() {
                let label = self
                    .detection_labels
                    .get(top.class_id)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                log::info!(
                    "Top detection for {}: cls_id={} label={} score={:.2} box=[{:.1},{:.1},{:.1},{:.1}]",
                    preview_path.display(),
                    top.class_id,
                    label,
                    top.score,
                    top.bbox[0],
                    top.bbox[1],
                    top.bbox[2],
                    top.bbox[3]
                );
            }
            let mut tags = HashMap::new();
            for det in detections {
                let label = match self.detection_labels.get(det.class_id) {
                    Some(name) => name.as_str(),
                    None => continue,
                };
                if !self.detection_label_map.is_empty() {
                    apply_detection_label(&mut tags, &self.detection_label_map, label, det.score);
                } else if let Some(tag) = default_detection_tag(label) {
                    let entry = tags.entry(tag.to_string()).or_insert(0.0);
                    if det.score > *entry {
                        *entry = det.score;
                    }
                } else {
                    let entry = tags.entry(label.to_string()).or_insert(0.0);
                    if det.score > *entry {
                        *entry = det.score;
                    }
                }
            }
            if cfg!(debug_assertions) {
                self.record_timing(
                    "detection",
                    decode_preprocess,
                    inference_time,
                    session_handle.provider,
                    preview_path,
                );
            }
            return Ok(tags);
        }
        let class_scores = detection_class_scores(&output_tensors);
        if class_scores.is_empty() {
            log::info!(
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
        if cfg!(debug_assertions) {
            self.record_timing(
                "detection",
                decode_preprocess,
                inference_time,
                session_handle.provider,
                preview_path,
            );
        }
        Ok(tags)
    }

    fn run_face(&mut self, preview_path: &Path) -> Result<f32> {
        if self.face_session.is_none() {
            return Ok(0.0);
        }
        let session_handle = self.face_session.as_ref().unwrap().clone();
        let (w, h, nchw) = {
            let session = session_handle.session.lock().unwrap();
            let (w, h) = model_input_hw(&session, 224, 224);
            (w, h, model_expects_nchw(&session))
        };
        let decode_start = Instant::now();
        let img = image::open(preview_path)?;
        let resized = img.resize_exact(w, h, FilterType::Triangle).to_rgb8();
        let input = if nchw {
            rgb8_to_nchw_into(&resized, w, h, &mut self.face_input);
            self.face_input.clone()
        } else {
            rgb8_to_nhwc_into(&resized, &mut self.face_input);
            self.face_input.clone()
        };
        let input_tensor = if nchw {
            Array::from_shape_vec((1, 3, h as usize, w as usize), input)
        } else {
            Array::from_shape_vec((1, h as usize, w as usize, 3), input)
        }
        .map_err(|e| Error::Init(format!("Invalid detector tensor shape: {e}")))?;
        let decode_preprocess = decode_start.elapsed();
        let infer_start = Instant::now();
        let mut session = session_handle.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![TensorRef::from_array_view(&input_tensor).map_err(
                |e| Error::Init(format!("Invalid face tensor: {e}"))
            )?])
            .map_err(|e| Error::Init(format!("Failed to run face detector: {e}")))?;
        let inference_time = infer_start.elapsed();
        if outputs.len() == 0 {
            log::warn!(
                "Face model returned no outputs for {}",
                preview_path.display()
            );
        }
        let output_tensors = collect_output_tensors(&outputs);
        let max_score = max_face_score(&output_tensors).unwrap_or(0.0).max(0.0).min(1.0);
        log::info!(
            "Face score for {}: {:.4} (threshold {:.4})",
            preview_path.display(),
            max_score,
            self.config.face_min_score
        );
        if max_score >= self.config.face_min_score {
            if cfg!(debug_assertions) {
                self.record_timing(
                    "face",
                    decode_preprocess,
                    inference_time,
                    session_handle.provider,
                    preview_path,
                );
            }
            Ok(max_score)
        } else {
            if cfg!(debug_assertions) {
                self.record_timing(
                    "face",
                    decode_preprocess,
                    inference_time,
                    session_handle.provider,
                    preview_path,
                );
            }
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

fn input_dims(session: &Session) -> Vec<Option<u32>> {
    let shape = session
        .inputs
        .get(0)
        .and_then(|i| i.input_type.tensor_shape());
    let Some(shape) = shape else {
        return Vec::new();
    };
    shape
        .iter()
        .map(|d| if *d > 0 { Some(*d as u32) } else { None })
        .collect()
}

fn model_expects_nchw(session: &Session) -> bool {
    let dims = input_dims(session);
    if dims.len() == 4 {
        match (dims[1], dims[2], dims[3]) {
            (Some(3), Some(_), Some(_)) => return true,
            (Some(_), Some(_), Some(3)) => return false,
            _ => {}
        }
    }
    true
}

fn model_input_hw(session: &Session, default_w: u32, default_h: u32) -> (u32, u32) {
    let dims = input_dims(session);
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
const SCENE_GROUP_MIN_LABELS: usize = 3;
const SCENE_UNRELATED_PENALTY: f32 = 0.5;
const DETECTION_MIN_SCORE: f32 = 0.38;
const DETECTION_PAIR_FOREGROUND_INDEX: usize = 1;
const DETECTION_TAG_BOOST: f32 = 0.20;
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

#[derive(Clone, Debug)]
struct Detection {
    class_id: usize,
    score: f32,
    bbox: [f32; 4],
}

struct OutputTensor {
    shape: Vec<i64>,
    data: Vec<f32>,
}

fn collect_output_tensors(outputs: &ort::session::SessionOutputs<'_>) -> Vec<OutputTensor> {
    let mut tensors = Vec::new();
    for (_, value) in outputs.iter() {
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            tensors.push(OutputTensor {
                shape: shape.iter().copied().collect(),
                data: data.to_vec(),
            });
        }
    }
    tensors
}

fn dim_to_usize(dim: i64) -> Option<usize> {
    if dim > 0 {
        Some(dim as usize)
    } else {
        None
    }
}

fn yolov5_detections_from_outputs(
    outputs: &[OutputTensor],
    ratio: f32,
    dw: f32,
    dh: f32,
    orig_w: u32,
    orig_h: u32,
    conf_thres: f32,
    iou_thres: f32,
) -> Option<Vec<Detection>> {
    let mut raw = Vec::new();
    let mut rows_opt = None;
    for output in outputs {
        if let Some(rows) = YoloRows::from_output(output) {
            rows_opt = Some(rows);
            break;
        }
    }
    let rows = rows_opt?;
    let class_count = rows.stride().saturating_sub(5);
    if class_count == 0 || ratio <= 0.0 {
        return None;
    }
    for row_idx in 0..rows.rows() {
        let x = rows.value(row_idx, 0);
        let y = rows.value(row_idx, 1);
        let w = rows.value(row_idx, 2);
        let h = rows.value(row_idx, 3);
        let obj = sigmoid(rows.value(row_idx, 4));
        if !x.is_finite()
            || !y.is_finite()
            || !w.is_finite()
            || !h.is_finite()
            || !obj.is_finite()
        {
            continue;
        }
        let mut best_id = 0usize;
        let mut best_prob = 0.0f32;
        for cls in 0..class_count {
            let prob = sigmoid(rows.value(row_idx, 5 + cls));
            if prob.is_finite() && prob > best_prob {
                best_prob = prob;
                best_id = cls;
            }
        }
        let score = obj * best_prob;
        if !score.is_finite() || score < conf_thres {
            continue;
        }
        let half_w = w / 2.0;
        let half_h = h / 2.0;
        let mut x1 = (x - half_w - dw) / ratio;
        let mut y1 = (y - half_h - dh) / ratio;
        let mut x2 = (x + half_w - dw) / ratio;
        let mut y2 = (y + half_h - dh) / ratio;
        x1 = x1.max(0.0).min(orig_w as f32);
        y1 = y1.max(0.0).min(orig_h as f32);
        x2 = x2.max(0.0).min(orig_w as f32);
        y2 = y2.max(0.0).min(orig_h as f32);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        // Store boxes in original-image coords for NMS and tagging.
        raw.push(Detection {
            class_id: best_id,
            score,
            bbox: [x1, y1, x2, y2],
        });
    }
    if raw.is_empty() {
        return Some(Vec::new());
    }
    let kept = nms_class_aware(raw, iou_thres);
    Some(kept)
}

fn nms_class_aware(mut dets: Vec<Detection>, iou_thres: f32) -> Vec<Detection> {
    // Standard class-aware NMS over descending scores.
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] || dets[i].class_id != dets[j].class_id {
                continue;
            }
            if iou(&dets[i].bbox, &dets[j].bbox) > iou_thres {
                suppressed[j] = true;
            }
        }
    }
    keep
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter = inter_w * inter_h;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    inter / (area_a + area_b - inter)
}

enum YoloRows<'a> {
    RowMajor {
        data: &'a [f32],
        rows: usize,
        stride: usize,
    },
    ChannelMajor {
        data: &'a [f32],
        rows: usize,
        stride: usize,
    },
}

impl<'a> YoloRows<'a> {
    fn from_output(output: &'a OutputTensor) -> Option<Self> {
        let shape = &output.shape;
        let data = output.data.as_slice();
        if shape.len() == 3 {
            let dim1 = dim_to_usize(shape[1])?;
            let dim2 = dim_to_usize(shape[2])?;
            if dim1 >= 6 && dim2 > dim1 && data.len() >= dim1 * dim2 {
                return Some(Self::ChannelMajor {
                    data,
                    rows: dim2,
                    stride: dim1,
                });
            }
        }
        if shape.len() >= 2 {
            let stride = dim_to_usize(*shape.last()?)?;
            if stride >= 6 {
                let rows = shape[..shape.len() - 1]
                    .iter()
                    .fold(1usize, |acc, d| {
                        acc.saturating_mul(dim_to_usize(*d).unwrap_or(0))
                    });
                if data.len() >= rows * stride {
                    return Some(Self::RowMajor { data, rows, stride });
                }
            }
        }
        None
    }

    fn rows(&self) -> usize {
        match self {
            Self::RowMajor { rows, .. } => *rows,
            Self::ChannelMajor { rows, .. } => *rows,
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::RowMajor { stride, .. } => *stride,
            Self::ChannelMajor { stride, .. } => *stride,
        }
    }

    fn value(&self, row: usize, idx: usize) -> f32 {
        match self {
            Self::RowMajor { data, stride, .. } => data[row * *stride + idx],
            Self::ChannelMajor { data, rows, .. } => data[idx * *rows + row],
        }
    }
}

fn detection_class_scores(outputs: &[OutputTensor]) -> HashMap<usize, f32> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    if outputs.len() == 2 {
        if let Some(paired) = detection_scores_from_pair(outputs) {
            return paired;
        }
    }
    for output in outputs {
        let shape = &output.shape;
        let slice = output.data.as_slice();
        if shape.len() >= 4 && shape[shape.len() - 1] >= 5 {
            let stride = match dim_to_usize(shape[shape.len() - 1]) {
                Some(stride) if stride > 0 => stride,
                _ => continue,
            };
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
            let dim1 = dim_to_usize(shape[1]).unwrap_or(0);
            let dim2 = dim_to_usize(shape[2]).unwrap_or(0);
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
            let classes = dim_to_usize(shape[shape.len() - 1]).unwrap_or(0);
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

fn detection_scores_from_pair(outputs: &[OutputTensor]) -> Option<HashMap<usize, f32>> {
    let mut scores_out: Option<&OutputTensor> = None;
    let mut boxes_out: Option<&OutputTensor> = None;
    for output in outputs {
        let shape = &output.shape;
        if shape.len() == 3 && shape[2] == 2 {
            scores_out = Some(output);
        } else if shape.len() == 3 && shape[2] == 4 {
            boxes_out = Some(output);
        }
    }
    let scores_out = scores_out?;
    let _boxes_out = boxes_out?;
    let shape = &scores_out.shape;
    let slice = scores_out.data.as_slice();
    let cols = dim_to_usize(shape[2]).unwrap_or(0);
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

fn detection_outputs_pair(outputs: &[OutputTensor]) -> bool {
    if outputs.len() != 2 {
        return false;
    }
    let mut has_scores = false;
    let mut has_boxes = false;
    for output in outputs {
        let shape = &output.shape;
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

fn max_face_score(outputs: &[OutputTensor]) -> Option<f32> {
    let mut best = None;
    for output in outputs {
        let shape = &output.shape;
        if shape.len() >= 2 && shape[shape.len() - 1] == 2 {
            for pair in output.data.as_slice().chunks_exact(2) {
                let score = pair[1];
                if !score.is_finite() {
                    continue;
                }
                best = Some(best.map_or(score, |b: f32| b.max(score)));
            }
        } else if shape.len() >= 2 && shape[shape.len() - 1] >= 5 {
            // YOLO-style output: [..., 5 + classes] => [x, y, w, h, obj, ...class_probs]
            let stride = match dim_to_usize(shape[shape.len() - 1]) {
                Some(stride) if stride > 0 => stride,
                _ => continue,
            };
            for row in output.data.as_slice().chunks_exact(stride) {
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
    best
}

fn rgb32f_to_nhwc_into(img: &image::Rgb32FImage, output: &mut Vec<f32>) {
    let len = (img.width() * img.height() * 3) as usize;
    output.clear();
    output.resize(len, 0.0);
    let mut idx = 0usize;
    for pixel in img.pixels() {
        output[idx] = pixel[0];
        output[idx + 1] = pixel[1];
        output[idx + 2] = pixel[2];
        idx += 3;
    }
}

fn rgb32f_to_nchw_into(img: &image::Rgb32FImage, w: u32, h: u32, output: &mut Vec<f32>) {
    let plane = (w * h) as usize;
    output.clear();
    output.resize(plane * 3, 0.0);
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        output[idx] = pixel[0];
        output[idx + plane] = pixel[1];
        output[idx + plane * 2] = pixel[2];
    }
}

fn rgb32f_to_nhwc_normalized_into(img: &image::Rgb32FImage, output: &mut Vec<f32>) {
    let len = (img.width() * img.height() * 3) as usize;
    output.clear();
    output.resize(len, 0.0);
    let mut idx = 0usize;
    for pixel in img.pixels() {
        output[idx] = (pixel[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        output[idx + 1] = (pixel[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        output[idx + 2] = (pixel[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        idx += 3;
    }
}

fn rgb32f_to_nchw_normalized_into(
    img: &image::Rgb32FImage,
    w: u32,
    h: u32,
    output: &mut Vec<f32>,
) {
    let plane = (w * h) as usize;
    output.clear();
    output.resize(plane * 3, 0.0);
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        output[idx] = (pixel[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        output[idx + plane] = (pixel[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        output[idx + plane * 2] = (pixel[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
}

fn rgb32f_to_nhwc_tf_into(img: &image::Rgb32FImage, output: &mut Vec<f32>) {
    let len = (img.width() * img.height() * 3) as usize;
    output.clear();
    output.resize(len, 0.0);
    let mut idx = 0usize;
    for pixel in img.pixels() {
        output[idx] = (pixel[0] * 2.0) - 1.0;
        output[idx + 1] = (pixel[1] * 2.0) - 1.0;
        output[idx + 2] = (pixel[2] * 2.0) - 1.0;
        idx += 3;
    }
}

fn rgb32f_to_nchw_tf_into(
    img: &image::Rgb32FImage,
    w: u32,
    h: u32,
    output: &mut Vec<f32>,
) {
    let plane = (w * h) as usize;
    output.clear();
    output.resize(plane * 3, 0.0);
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        output[idx] = (pixel[0] * 2.0) - 1.0;
        output[idx + plane] = (pixel[1] * 2.0) - 1.0;
        output[idx + plane * 2] = (pixel[2] * 2.0) - 1.0;
    }
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

fn rgb8_to_nhwc_into(img: &image::RgbImage, output: &mut Vec<f32>) {
    let len = (img.width() * img.height() * 3) as usize;
    output.clear();
    output.resize(len, 0.0);
    let mut idx = 0usize;
    for pixel in img.pixels() {
        output[idx] = pixel[0] as f32 / 255.0;
        output[idx + 1] = pixel[1] as f32 / 255.0;
        output[idx + 2] = pixel[2] as f32 / 255.0;
        idx += 3;
    }
}

fn rgb8_to_nchw_into(img: &image::RgbImage, w: u32, h: u32, output: &mut Vec<f32>) {
    let plane = (w * h) as usize;
    output.clear();
    output.resize(plane * 3, 0.0);
    for (x, y, pixel) in img.enumerate_pixels() {
        let idx = (y * w + x) as usize;
        output[idx] = pixel[0] as f32 / 255.0;
        output[idx + plane] = pixel[1] as f32 / 255.0;
        output[idx + plane * 2] = pixel[2] as f32 / 255.0;
    }
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

fn letterbox_rgb(
    img: &image::RgbImage,
    net_w: u32,
    net_h: u32,
    pad_val: u8,
) -> (image::RgbImage, f32, f32, f32) {
    let src_w = img.width().max(1);
    let src_h = img.height().max(1);
    let ratio = (net_w as f32 / src_w as f32).min(net_h as f32 / src_h as f32);
    let new_w = ((src_w as f32) * ratio).round().max(1.0) as u32;
    let new_h = ((src_h as f32) * ratio).round().max(1.0) as u32;
    let resized = image::imageops::resize(img, new_w, new_h, FilterType::Triangle);
    let mut padded = image::RgbImage::from_pixel(net_w, net_h, image::Rgb([pad_val; 3]));
    let dw = (net_w as f32 - new_w as f32) / 2.0;
    let dh = (net_h as f32 - new_h as f32) / 2.0;
    let pad_left = dw.floor().max(0.0) as u32;
    let pad_top = dh.floor().max(0.0) as u32;
    // Pad to the requested net size with the YOLOv5 default background value.
    image::imageops::replace(&mut padded, &resized, pad_left as i64, pad_top as i64);
    (padded, ratio, pad_left as f32, pad_top as f32)
}

fn run_scene_logits(
    session_handle: &SessionHandle,
    resized: &image::Rgb32FImage,
    nchw: bool,
    w: u32,
    h: u32,
    mode: ScenePreprocess,
    input_buf: &mut Vec<f32>,
) -> Result<(Vec<f32>, Duration, Duration)> {
    let prep_start = Instant::now();
    match (mode, nchw) {
        (ScenePreprocess::Imagenet, true) => {
            rgb32f_to_nchw_normalized_into(resized, w, h, input_buf)
        }
        (ScenePreprocess::Imagenet, false) => {
            rgb32f_to_nhwc_normalized_into(resized, input_buf)
        }
        (ScenePreprocess::Raw01, true) => rgb32f_to_nchw_into(resized, w, h, input_buf),
        (ScenePreprocess::Raw01, false) => rgb32f_to_nhwc_into(resized, input_buf),
        (ScenePreprocess::TfMinus1, true) => rgb32f_to_nchw_tf_into(resized, w, h, input_buf),
        (ScenePreprocess::TfMinus1, false) => rgb32f_to_nhwc_tf_into(resized, input_buf),
    };
    let input = input_buf.clone();
    let input_tensor = if nchw {
        Array::from_shape_vec((1, 3, h as usize, w as usize), input)
    } else {
        Array::from_shape_vec((1, h as usize, w as usize, 3), input)
    }
    .map_err(|e| Error::Init(format!("Invalid scene tensor shape: {e}")))?;
    let preprocess_time = prep_start.elapsed();
    let infer_start = Instant::now();
    let mut session = session_handle.session.lock().unwrap();
    let outputs = session
        .run(ort::inputs![TensorRef::from_array_view(&input_tensor).map_err(
            |e| Error::Init(format!("Invalid scene tensor: {e}"))
        )?])
        .map_err(|e| Error::Init(format!("Failed to run scene model: {e}")))?;
    let inference_time = infer_start.elapsed();
    if outputs.len() == 0 {
        log::warn!("Scene model returned no outputs");
        return Ok((Vec::new(), preprocess_time, inference_time));
    }
    let (_, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Init(format!("Failed to extract scene outputs: {e}")))?;
    Ok((data.to_vec(), preprocess_time, inference_time))
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

fn ort_config_from_tagging(config: &TaggingConfig) -> OrtRuntimeConfig {
    let provider = match config.inference_device {
        InferenceDevicePreference::Auto => ProviderChoice::Auto,
        InferenceDevicePreference::Gpu => ProviderChoice::DirectMLOnly,
        InferenceDevicePreference::Cpu => ProviderChoice::CpuOnly,
    };
    OrtRuntimeConfig {
        provider,
        device_id: config.inference_device_id,
    }
}

fn session_cache_key(model_path: &Path, cfg: OrtRuntimeConfig) -> SessionCacheKey {
    SessionCacheKey {
        model_path: model_path.to_string_lossy().to_string(),
        provider: cfg.provider,
        device_id: cfg.device_id,
    }
}

fn get_or_create_session(
    model_path: &Path,
    label: &'static str,
    cfg: OrtRuntimeConfig,
    default_w: u32,
    default_h: u32,
) -> Option<Arc<SessionHandle>> {
    if !model_path.exists() {
        return None;
    }
    let key = session_cache_key(model_path, cfg);
    {
        let cache = SESSION_CACHE.lock().unwrap();
        if let Some(handle) = cache.get(&key) {
            return Some(handle.clone());
        }
    }

    let created = match create_session_with_preference(model_path, label, cfg, default_w, default_h) {
        Ok(handle) => Arc::new(handle),
        Err(err) => {
            log::warn!("Failed to create {label} session: {err}");
            return None;
        }
    };

    let mut cache = SESSION_CACHE.lock().unwrap();
    let handle = cache.entry(key).or_insert_with(|| created.clone());
    Some(handle.clone())
}

fn create_session_with_preference(
    model_path: &Path,
    label: &'static str,
    cfg: OrtRuntimeConfig,
    default_w: u32,
    default_h: u32,
) -> std::result::Result<SessionHandle, String> {
    if !model_path.exists() {
        return Err(format!("Model not found: {}", model_path.display()));
    }
    log_runtime_diagnostics_once();

    let model_path_static: &'static Path =
        Box::leak(model_path.to_path_buf().into_boxed_path());
    let mut warning: Option<String> = None;
    let (mut session, provider) = match onnx::build_session(model_path_static, cfg) {
        Ok((session, provider)) => (session, provider),
        Err(err) => return Err(format!("{err}")),
    };
    if matches!(cfg.provider, ProviderChoice::DirectMLOnly)
        && matches!(provider, InferenceProvider::Cpu)
    {
        let msg = format!("DirectML provider unavailable for {label}; using CPU");
        warning = Some(msg.clone());
        log::warn!("{msg}");
    }

    if let Some(message) = warning {
        *INFERENCE_WARNING.lock().unwrap() = Some(message);
    }

    warmup_session(&mut session, label, default_w, default_h);

    log::info!(
        "ONNX session ready for {label}: provider={}",
        provider.label()
    );

    Ok(SessionHandle {
        session: Mutex::new(session),
        provider,
        label,
        model_path: model_path_static,
    })
}

fn warmup_session(session: &mut Session, label: &str, default_w: u32, default_h: u32) {
    let (w, h) = model_input_hw(session, default_w, default_h);
    let nchw = model_expects_nchw(session);
    let (shape, len) = if nchw {
        ((1, 3, h as usize, w as usize), (w * h * 3) as usize)
    } else {
        ((1, h as usize, w as usize, 3), (w * h * 3) as usize)
    };
    let input = vec![0.0f32; len];
    let input_tensor = Array::from_shape_vec(shape, input);
    let input_tensor = match input_tensor {
        Ok(t) => t,
        Err(err) => {
            log::warn!("Warmup tensor build failed for {label}: {err}");
            return;
        }
    };
    let input_value = match TensorRef::from_array_view(&input_tensor) {
        Ok(value) => value,
        Err(err) => {
            log::warn!("Warmup tensor build failed for {label}: {err}");
            return;
        }
    };
    if let Err(err) = session.run(ort::inputs![input_value]) {
        log::warn!("Warmup run failed for {label}: {err}");
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

pub fn clear_session_cache() {
    SESSION_CACHE.lock().unwrap().clear();
    *INFERENCE_WARNING.lock().unwrap() = None;
}

pub fn inference_status(config: &TaggingConfig) -> InferenceStatus {
    let preference = config.inference_device;
    let ort_cfg = ort_config_from_tagging(config);
    let mut models = Vec::new();
    let mut provider_label = "Unavailable".to_string();
    let mut had_provider = false;

    let try_model = |label: &'static str,
                     model_path: &Path,
                     default_w: u32,
                     default_h: u32|
     -> Option<String> {
        if !model_path.exists() {
            return None;
        }
        let handle = get_or_create_session(model_path, label, ort_cfg, default_w, default_h)?;
        Some(handle.provider.label().to_string())
    };

    if let Some(provider) =
        try_model("scene", &config.scene_model_path, 224, 224)
    {
        provider_label = provider.clone();
        had_provider = true;
        models.push(InferenceModelStatus {
            label: "scene".to_string(),
            provider,
        });
    }
    if let Some(provider) =
        try_model("detection", &config.detection_model_path, 640, 640)
    {
        if !had_provider {
            provider_label = provider.clone();
            had_provider = true;
        }
        models.push(InferenceModelStatus {
            label: "detection".to_string(),
            provider,
        });
    }
    if let Some(provider) =
        try_model("face", &config.face_model_path, 224, 224)
    {
        if !had_provider {
            provider_label = provider.clone();
        }
        models.push(InferenceModelStatus {
            label: "face".to_string(),
            provider,
        });
    }

    let warning = INFERENCE_WARNING.lock().unwrap().clone();
    let runtime_version = ort_runtime_version();

    InferenceStatus {
        preference: preference.label().to_string(),
        provider: provider_label,
        warning,
        runtime_version,
        models,
    }
}

pub fn inference_backend_info(config: &TaggingConfig) -> crate::models::InferenceBackendInfo {
    let ort_cfg = ort_config_from_tagging(config);
    let mut provider = InferenceProvider::Cpu;
    let try_model = |label: &'static str,
                     model_path: &Path,
                     default_w: u32,
                     default_h: u32|
     -> Option<InferenceProvider> {
        if !model_path.exists() {
            return None;
        }
        let handle = get_or_create_session(model_path, label, ort_cfg, default_w, default_h)?;
        Some(handle.provider)
    };

    if let Some(p) = try_model("scene", &config.scene_model_path, 224, 224) {
        provider = p;
    } else if let Some(p) = try_model("detection", &config.detection_model_path, 640, 640) {
        provider = p;
    } else if let Some(p) = try_model("face", &config.face_model_path, 224, 224) {
        provider = p;
    }

    crate::models::InferenceBackendInfo {
        provider: match provider {
            InferenceProvider::Cpu => "cpu".to_string(),
            InferenceProvider::DirectML { .. } => "directml".to_string(),
        },
        device_id: provider.device_id(),
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
