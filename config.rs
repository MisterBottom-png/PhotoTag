use serde::Deserialize;

#[derive(Deserialize)]
pub struct TaggingConfig {
    pub scene_model_path: String,
    pub detection_model_path: String,
    pub confidence_threshold: f32,
    pub suggestion_threshold: f32,
}

impl Default for TaggingConfig {
    fn default() -> Self {
        Self {
            scene_model_path: "models/scene_classifier.onnx".to_string(),
            detection_model_path: "models/face_detector.onnx".to_string(),
            confidence_threshold: 0.65,
            suggestion_threshold: 0.45,
        }
    }
}