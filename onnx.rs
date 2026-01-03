use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

#[cfg(target_os = "windows")]
use ort::execution_providers::{DirectMLExecutionProvider, ExecutionProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceProvider {
    Cpu,
    DirectML { device_id: u32 },
}

impl InferenceProvider {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::DirectML { .. } => "GPU (DirectML)",
        }
    }

    pub fn device_id(self) -> Option<u32> {
        match self {
            Self::DirectML { device_id } => Some(device_id),
            Self::Cpu => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderChoice {
    Auto,
    CpuOnly,
    DirectMLOnly,
}

#[derive(Debug, Clone, Copy)]
pub struct OrtRuntimeConfig {
    pub provider: ProviderChoice,
    pub device_id: Option<u32>,
}

impl OrtRuntimeConfig {
    fn resolved_device_id(self) -> u32 {
        self.device_id.unwrap_or(0)
    }
}

pub fn init_ort_dylib_path(app_handle: &tauri::AppHandle) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        let resource_dir = app_handle.path_resolver().resource_dir();
        if let Some((dll_path, dll_dir)) = pick_ort_dll(ort_candidate_paths(resource_dir.as_deref()))
        {
            set_ort_dylib_path(&dll_path, &dll_dir);
        } else {
            return Err(Error::Path(
                "onnxruntime.dll not found in bundle".to_string(),
            ));
        }
    }
    Ok(())
}

fn ort_candidate_paths(resource_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(resource_dir) = resource_dir {
        candidates.push(resource_dir.join("onnxruntime").join("onnxruntime.dll"));
    }
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        candidates.push(
            Path::new(&manifest_dir)
                .join("vendor")
                .join("onnxruntime")
                .join("win-x64-directml")
                .join("onnxruntime.dll"),
        );
    }
    candidates
}

fn pick_ort_dll(candidates: Vec<PathBuf>) -> Option<(PathBuf, PathBuf)> {
    for candidate in candidates {
        if candidate.exists() {
            let dir = candidate.parent()?.to_path_buf();
            return Some((candidate, dir));
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn prepend_path_dir(dir: &Path) {
    let paths = std::env::var_os("PATH").unwrap_or_default();
    let mut new_paths = std::ffi::OsString::new();
    new_paths.push(dir);
    new_paths.push(";");
    new_paths.push(&paths);
    std::env::set_var("PATH", new_paths);
}

#[cfg(not(target_os = "windows"))]
fn prepend_path_dir(_dir: &Path) {}

#[cfg(target_os = "windows")]
fn resolve_ort_dylib_path() -> Option<(PathBuf, PathBuf)> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            let dir = path.parent()?.to_path_buf();
            return Some((path, dir));
        }
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("onnxruntime").join("onnxruntime.dll"));
        }
    }
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        candidates.push(
            Path::new(&manifest_dir)
                .join("vendor")
                .join("onnxruntime")
                .join("win-x64-directml")
                .join("onnxruntime.dll"),
        );
    }
    pick_ort_dll(candidates)
}

#[cfg(not(target_os = "windows"))]
fn resolve_ort_dylib_path() -> Option<(PathBuf, PathBuf)> {
    None
}

#[cfg(target_os = "windows")]
fn set_ort_dylib_path(dll_path: &Path, dll_dir: &Path) {
    std::env::set_var("ORT_DYLIB_PATH", dll_path);
    prepend_path_dir(dll_dir);
}

fn ensure_environment() -> Result<()> {
    let committed = ort::init()
        .with_name("photo-tagging")
        .commit()
        .map_err(|e| Error::Init(format!("Failed to init ORT environment: {e}")))?;
    if committed {
        if let Ok(env) = ort::environment::get_environment() {
            env.set_log_level(ort::logging::LogLevel::Warning);
        }
    }
    Ok(())
}

pub fn build_session(
    model_path: &Path,
    cfg: OrtRuntimeConfig,
) -> Result<(Session, InferenceProvider)> {
    if !model_path.exists() {
        return Err(Error::Init(format!(
            "Model not found: {}",
            model_path.display()
        )));
    }
    #[cfg(target_os = "windows")]
    {
        if let Some((dll_path, dll_dir)) = resolve_ort_dylib_path() {
            set_ort_dylib_path(&dll_path, &dll_dir);
        } else {
            return Err(Error::Init(
                "onnxruntime.dll not found; run scripts/fetch_onnxruntime_directml.ps1".into(),
            ));
        }
    }
    ensure_environment()?;
    let device_id = cfg.resolved_device_id();

    let try_build = |use_dml: bool| -> Result<Session> {
        let build = || -> Result<Session> {
            let mut builder = Session::builder()
                .map_err(|e| Error::Init(format!("{e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .map_err(|e| Error::Init(format!("{e}")))?
                .with_parallel_execution(false)
                .map_err(|e| Error::Init(format!("{e}")))?;
            if use_dml {
                #[cfg(target_os = "windows")]
                {
                    builder = builder
                        .with_memory_pattern(false)
                        .map_err(|e| Error::Init(format!("{e}")))?;
                    let ep = DirectMLExecutionProvider::default()
                        .with_device_id(device_id as i32)
                        .build();
                    builder = builder
                        .with_execution_providers([ep])
                        .map_err(|e| Error::Init(format!("{e}")))?;
                }
            }
            builder
                .commit_from_file(model_path)
                .map_err(|e| Error::Init(format!("{e}")))
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(build)) {
            Ok(res) => res,
            Err(_) => Err(Error::Init(
                "ONNX Runtime panicked while building session".into(),
            )),
        }
    };

    let wants_dml = matches!(cfg.provider, ProviderChoice::Auto | ProviderChoice::DirectMLOnly);
    #[cfg(target_os = "windows")]
    {
        if wants_dml {
            if let Ok(available) = DirectMLExecutionProvider::default().is_available() {
                if available {
                    if let Ok(session) = try_build(true) {
                        return Ok((
                            session,
                            InferenceProvider::DirectML { device_id },
                        ));
                    }
                }
            }
        }
    }

    if wants_dml {
        log::warn!(
            "DirectML execution provider unavailable; falling back to CPU for {}",
            model_path.display()
        );
    }

    let session = try_build(false)?;
    Ok((session, InferenceProvider::Cpu))
}

pub fn ort_runtime_version() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        if resolve_ort_dylib_path().is_none() {
            return None;
        }
    }
    let info = ort::info();
    if let Some(start) = info.find("git-branch=rel-") {
        let tail = &info[start + "git-branch=rel-".len()..];
        if let Some(end) = tail.find(',') {
            return Some(tail[..end].to_string());
        }
        return Some(tail.to_string());
    }
    Some(format!("1.{}.x", ort::MINOR_VERSION))
}
