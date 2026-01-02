use crate::config::{AppPaths, TaggingConfig};
use crate::db::{self, DbPool};
use crate::error::{Error, Result};
use crate::embedding;
use crate::exiftool;
use crate::models::{ExifMetadata, ImportProgressEvent, PhotoRecord, StageProgress, TaggingResult};
use crate::tagging::TaggingEngine;
use crate::thumbnails;
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::collections::HashSet;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;
use uuid::Uuid;
use walkdir::WalkDir;
use xxhash_rust::xxh3::xxh3_128;

const SUPPORTED_EXT: &[&str] = &[
    "jpg", "jpeg", "png", "tiff", "tif", "cr2", "nef", "arw", "dng", "raf",
];

const STAGES: [&str; 5] = ["exif", "thumbnail", "hash", "tagging", "embedding"];

#[derive(Clone, Default)]
pub struct JobManager {
    inner: Arc<JobManagerInner>,
}

#[derive(Default)]
struct JobManagerInner {
    current: Mutex<Option<JobHandle>>,
}

#[derive(Clone)]
struct JobHandle {
    id: String,
    cancel: Arc<AtomicBool>,
    cancel_files: Arc<Mutex<HashSet<String>>>,
}

impl JobManager {
    pub fn start_import(
        &self,
        app: tauri::AppHandle,
        root: PathBuf,
        pool: DbPool,
        paths: AppPaths,
        tagging: TaggingConfig,
    ) -> Result<String> {
        let mut current = self.inner.current.lock().unwrap();
        if current.is_some() {
            return Err(Error::Init(
                "Import already running; cancel before starting a new one.".into(),
            ));
        }

        let job_id = Uuid::new_v4().to_string();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_files = Arc::new(Mutex::new(HashSet::new()));
        let tracker = ProgressTracker::new(app.clone());

        *current = Some(JobHandle {
            id: job_id.clone(),
            cancel: cancel.clone(),
            cancel_files: cancel_files.clone(),
        });

        let handles = spawn_pipeline(
            app,
            root,
            pool,
            paths,
            tagging,
            cancel,
            cancel_files,
            tracker.clone(),
        )?;

        let manager = self.clone();
        let job_id_for_thread = job_id.clone();
        thread::spawn(move || {
            for handle in handles {
                let _ = handle.join();
            }
            manager.finish_job(&job_id_for_thread, &tracker);
        });

        Ok(job_id)
    }

    pub fn cancel_current(&self) -> Result<()> {
        let current = self.inner.current.lock().unwrap();
        if let Some(handle) = current.as_ref() {
            handle.cancel.store(true, Ordering::Relaxed);
            return Ok(());
        }
        Err(Error::Init("No import running".into()))
    }

    pub fn cancel_file(&self, path: String) -> Result<()> {
        let current = self.inner.current.lock().unwrap();
        if let Some(handle) = current.as_ref() {
            let mut canceled = handle.cancel_files.lock().unwrap();
            canceled.insert(path);
            return Ok(());
        }
        Err(Error::Init("No import running".into()))
    }

    pub fn is_importing(&self) -> bool {
        self.inner.current.lock().unwrap().is_some()
    }

    fn finish_job(&self, job_id: &str, tracker: &ProgressTracker) {
        let mut current = self.inner.current.lock().unwrap();
        if let Some(handle) = current.as_ref() {
            if handle.id == job_id {
                tracker.emit_progress(true);
                *current = None;
            }
        }
    }
}

#[derive(Clone)]
struct ProgressTracker {
    app: tauri::AppHandle,
    state: Arc<ProgressState>,
}

struct ProgressState {
    discovered: AtomicUsize,
    processed: AtomicUsize,
    errors: AtomicUsize,
    canceled: AtomicBool,
    current_file: Mutex<Option<String>>,
    current_stage: Mutex<Option<String>>,
    last_emit: Mutex<Instant>,
    started_at: Instant,
    stages: Vec<StageCounters>,
}

#[derive(Clone)]
struct StageCounters {
    name: &'static str,
    pending: Arc<AtomicUsize>,
    in_progress: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    errors: Arc<AtomicUsize>,
    started_at: Instant,
}

impl ProgressTracker {
    fn new(app: tauri::AppHandle) -> Self {
        let stages = STAGES
            .iter()
            .map(|name| StageCounters {
                name,
                pending: Arc::new(AtomicUsize::new(0)),
                in_progress: Arc::new(AtomicUsize::new(0)),
                completed: Arc::new(AtomicUsize::new(0)),
                errors: Arc::new(AtomicUsize::new(0)),
                started_at: Instant::now(),
            })
            .collect();
        Self {
            app,
            state: Arc::new(ProgressState {
                discovered: AtomicUsize::new(0),
                processed: AtomicUsize::new(0),
                errors: AtomicUsize::new(0),
                canceled: AtomicBool::new(false),
                current_file: Mutex::new(None),
                current_stage: Mutex::new(None),
                last_emit: Mutex::new(Instant::now()),
                started_at: Instant::now(),
                stages,
            }),
        }
    }

    fn mark_canceled(&self) {
        self.state.canceled.store(true, Ordering::Relaxed);
    }

    fn on_discovered(&self) {
        self.state.discovered.fetch_add(1, Ordering::Relaxed);
    }

    fn on_processed(&self) {
        self.state.processed.fetch_add(1, Ordering::Relaxed);
    }

    fn on_error(&self) {
        self.state.errors.fetch_add(1, Ordering::Relaxed);
    }

    fn stage_pending_inc(&self, stage: usize) {
        if let Some(stage) = self.state.stages.get(stage) {
            stage.pending.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn stage_pending_dec(&self, stage: usize) {
        if let Some(stage) = self.state.stages.get(stage) {
            stage.pending.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn stage_start(&self, stage: usize, path: &Path) {
        if let Some(stage) = self.state.stages.get(stage) {
            stage.in_progress.fetch_add(1, Ordering::Relaxed);
        }
        *self.state.current_file.lock().unwrap() = path.to_str().map(|s| s.to_string());
        *self.state.current_stage.lock().unwrap() = Some(STAGES[stage].to_string());
    }

    fn stage_complete(&self, stage: usize) {
        if let Some(stage) = self.state.stages.get(stage) {
            stage.in_progress.fetch_sub(1, Ordering::Relaxed);
            stage.completed.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn stage_error(&self, stage: usize) {
        if let Some(stage) = self.state.stages.get(stage) {
            stage.in_progress.fetch_sub(1, Ordering::Relaxed);
            stage.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn emit_progress(&self, force: bool) {
        let now = Instant::now();
        {
            let mut last = self.state.last_emit.lock().unwrap();
            if !force && now.duration_since(*last) < Duration::from_millis(200) {
                return;
            }
            *last = now;
        }

        let discovered = self.state.discovered.load(Ordering::Relaxed);
        let processed = self.state.processed.load(Ordering::Relaxed);
        let errors = self.state.errors.load(Ordering::Relaxed);
        let current_file = self.state.current_file.lock().unwrap().clone();
        let current_stage = self.state.current_stage.lock().unwrap().clone();
        let elapsed = self.state.started_at.elapsed().as_secs_f32();
        let throughput = if elapsed > 0.0 {
            Some(processed as f32 / elapsed)
        } else {
            None
        };
        let stages = self
            .state
            .stages
            .iter()
            .map(|stage| {
                let elapsed = stage.started_at.elapsed().as_secs_f32();
                let completed = stage.completed.load(Ordering::Relaxed);
                let items_per_sec = if elapsed > 0.0 {
                    Some(completed as f32 / elapsed)
                } else {
                    None
                };
                StageProgress {
                    stage: stage.name.to_string(),
                    pending: stage.pending.load(Ordering::Relaxed),
                    in_progress: stage.in_progress.load(Ordering::Relaxed),
                    completed,
                    errors: stage.errors.load(Ordering::Relaxed),
                    items_per_sec,
                }
            })
            .collect();
        let canceled = self.state.canceled.load(Ordering::Relaxed);
        let _ = self.app.emit_all(
            "import-progress",
            ImportProgressEvent {
                discovered,
                processed,
                errors,
                current_file,
                current_stage,
                throughput,
                stages,
                canceled,
            },
        );
    }
}

#[derive(Debug)]
struct FileWork {
    path: PathBuf,
    mtime: i64,
    size: i64,
    exif: ExifMetadata,
    preview_path: Option<PathBuf>,
    thumb_path: Option<PathBuf>,
    hash: Option<String>,
    import_batch_id: String,
    dhash: Option<i64>,
    photo_id: Option<i64>,
}

fn spawn_pipeline(
    app: tauri::AppHandle,
    root: PathBuf,
    pool: DbPool,
    paths: AppPaths,
    tagging: TaggingConfig,
    cancel: Arc<AtomicBool>,
    cancel_files: Arc<Mutex<HashSet<String>>>,
    tracker: ProgressTracker,
) -> Result<Vec<thread::JoinHandle<()>>> {
    let (exif_tx, exif_rx) = bounded::<PathBuf>(256);
    let (thumb_tx, thumb_rx) = bounded::<FileWork>(128);
    let (hash_tx, hash_rx) = bounded::<FileWork>(128);
    let (tag_tx, tag_rx) = bounded::<FileWork>(64);
    let (embed_tx, embed_rx) = bounded::<FileWork>(64);
    let import_batch_id = Uuid::new_v4().to_string();

    let mut handles = Vec::new();

    handles.push(spawn_discovery(
        app.clone(),
        root,
        pool.clone(),
        exif_tx,
        cancel.clone(),
        tracker.clone(),
    ));

    for _ in 0..2 {
        let rx = exif_rx.clone();
        let tx = thumb_tx.clone();
        let pool = pool.clone();
        let paths = paths.clone();
        let cancel = cancel.clone();
        let cancel_files = cancel_files.clone();
        let tracker = tracker.clone();
        let import_batch_id = import_batch_id.clone();
        handles.push(thread::spawn(move || {
            run_exif_stage(
                rx,
                tx,
                pool,
                paths,
                import_batch_id,
                cancel,
                cancel_files,
                tracker,
            );
        }));
    }

    for _ in 0..2 {
        let rx = thumb_rx.clone();
        let tx = hash_tx.clone();
        let paths = paths.clone();
        let cancel = cancel.clone();
        let cancel_files = cancel_files.clone();
        let tracker = tracker.clone();
        handles.push(thread::spawn(move || {
            run_thumbnail_stage(rx, tx, paths, cancel, cancel_files, tracker);
        }));
    }

    for _ in 0..2 {
        let rx = hash_rx.clone();
        let tx = tag_tx.clone();
        let cancel = cancel.clone();
        let cancel_files = cancel_files.clone();
        let tracker = tracker.clone();
        handles.push(thread::spawn(move || {
            run_hash_stage(rx, tx, cancel, cancel_files, tracker);
        }));
    }

    for _ in 0..1 {
        let rx = tag_rx.clone();
        let tx = embed_tx.clone();
        let pool = pool.clone();
        let paths = paths.clone();
        let tagging = tagging.clone();
        let cancel = cancel.clone();
        let cancel_files = cancel_files.clone();
        let tracker = tracker.clone();
        handles.push(thread::spawn(move || {
            run_tagging_stage(rx, tx, pool, paths, tagging, cancel, cancel_files, tracker);
        }));
    }

    for _ in 0..1 {
        let rx = embed_rx.clone();
        let pool = pool.clone();
        let cancel = cancel.clone();
        let cancel_files = cancel_files.clone();
        let tracker = tracker.clone();
        handles.push(thread::spawn(move || {
            run_embedding_stage(rx, pool, cancel, cancel_files, tracker);
        }));
    }

    Ok(handles)
}

fn spawn_discovery(
    app: tauri::AppHandle,
    root: PathBuf,
    pool: DbPool,
    exif_tx: Sender<PathBuf>,
    cancel: Arc<AtomicBool>,
    tracker: ProgressTracker,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let root_str = root.to_string_lossy().to_string();
        let existing_paths = {
            let conn = pool.get();
            if let Ok(conn) = conn {
                db::list_paths_with_prefix(&conn, &root_str).unwrap_or_default()
            } else {
                HashSet::new()
            }
        };

        for entry in WalkDir::new(&root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if cancel.load(Ordering::Relaxed) {
                tracker.mark_canceled();
                break;
            }
            let path = entry.into_path();
            if !is_supported(&path) {
                continue;
            }
            let path_str = path.to_string_lossy().to_string();
            if existing_paths.contains(&path_str) {
                continue;
            }
            tracker.on_discovered();
            tracker.stage_pending_inc(0);
            if exif_tx.send(path).is_err() {
                break;
            }
            tracker.emit_progress(false);
        }

        let _ = app.emit_all(
            "import-progress",
            ImportProgressEvent {
                discovered: tracker.state.discovered.load(Ordering::Relaxed),
                processed: tracker.state.processed.load(Ordering::Relaxed),
                errors: tracker.state.errors.load(Ordering::Relaxed),
                current_file: Some(root_str),
                current_stage: Some("scan".to_string()),
                throughput: None,
                stages: tracker
                    .state
                    .stages
                    .iter()
                    .map(|stage| StageProgress {
                        stage: stage.name.to_string(),
                        pending: stage.pending.load(Ordering::Relaxed),
                        in_progress: stage.in_progress.load(Ordering::Relaxed),
                        completed: stage.completed.load(Ordering::Relaxed),
                        errors: stage.errors.load(Ordering::Relaxed),
                        items_per_sec: None,
                    })
                    .collect(),
                canceled: tracker.state.canceled.load(Ordering::Relaxed),
            },
        );
    })
}

fn run_exif_stage(
    rx: Receiver<PathBuf>,
    tx: Sender<FileWork>,
    pool: DbPool,
    paths: AppPaths,
    import_batch_id: String,
    cancel: Arc<AtomicBool>,
    cancel_files: Arc<Mutex<HashSet<String>>>,
    tracker: ProgressTracker,
) {
    loop {
        if cancel.load(Ordering::Relaxed) && rx.is_empty() {
            tracker.mark_canceled();
            break;
        }
        let path = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(path) => path,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        tracker.stage_pending_dec(0);
        if is_canceled(&path, &cancel, &cancel_files) {
            tracker.mark_canceled();
            continue;
        }
        tracker.stage_start(0, &path);

        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(err) => {
                tracker.on_error();
                tracker.stage_error(0);
                log::warn!("Metadata read failed for {}: {}", path.display(), err);
                tracker.emit_progress(false);
                continue;
            }
        };
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let size = metadata.len() as i64;

        if let Ok(conn) = pool.get() {
            if let Ok(Some((existing_mtime, existing_size))) =
                db::get_photo_status(&conn, path.to_string_lossy().as_ref())
            {
                if existing_mtime == mtime && existing_size == size {
                    tracker.stage_complete(0);
                    tracker.emit_progress(false);
                    continue;
                }
            }
        }

        let exif = exiftool::read_metadata(&paths, &path).unwrap_or_default();
        let work = FileWork {
            path,
            mtime,
            size,
            exif,
            preview_path: None,
            thumb_path: None,
            hash: None,
            import_batch_id: import_batch_id.clone(),
            dhash: None,
            photo_id: None,
        };
        tracker.stage_complete(0);
        if tx.send(work).is_err() {
            break;
        }
        tracker.stage_pending_inc(1);
        tracker.emit_progress(false);
    }
}

fn run_thumbnail_stage(
    rx: Receiver<FileWork>,
    tx: Sender<FileWork>,
    paths: AppPaths,
    cancel: Arc<AtomicBool>,
    cancel_files: Arc<Mutex<HashSet<String>>>,
    tracker: ProgressTracker,
) {
    loop {
        if cancel.load(Ordering::Relaxed) && rx.is_empty() {
            tracker.mark_canceled();
            break;
        }
        let mut work = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(work) => work,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        tracker.stage_pending_dec(1);
        if is_canceled(&work.path, &cancel, &cancel_files) {
            tracker.mark_canceled();
            continue;
        }
        tracker.stage_start(1, &work.path);

        let hash_hint = name_hint(&work.path);
        let preview_output = paths.previews_dir.join(format!("{hash_hint}_preview.jpg"));
        let has_preview =
            exiftool::extract_preview(&paths, &work.path, &preview_output).unwrap_or(false);
        let preview_path = if has_preview && preview_output.exists() {
            Some(preview_output)
        } else {
            if thumbnails::is_supported_image(&work.path) {
                match thumbnails::build_preview(&work.path, &paths.previews_dir) {
                    Ok(path) if path.exists() => Some(path),
                    Ok(path) => {
                        log::warn!("Preview output missing for {}", path.display());
                        None
                    }
                    Err(err) => {
                        log::warn!("Preview generation failed for {}: {}", work.path.display(), err);
                        None
                    }
                }
            } else {
                log::warn!(
                    "No embedded preview found for {}; skipping preview generation",
                    work.path.display()
                );
                None
            }
        };
        let thumb_path = preview_path.as_ref().and_then(|preview| {
            thumbnails::build_thumbnail(preview, &paths.thumbs_dir)
                .map_err(|err| {
                    log::warn!("Thumbnail generation failed for {}: {}", preview.display(), err);
                    err
                })
                .ok()
        });

        work.preview_path = preview_path;
        work.thumb_path = thumb_path;

        tracker.stage_complete(1);
        if tx.send(work).is_err() {
            break;
        }
        tracker.stage_pending_inc(2);
        tracker.emit_progress(false);
    }
}

fn run_hash_stage(
    rx: Receiver<FileWork>,
    tx: Sender<FileWork>,
    cancel: Arc<AtomicBool>,
    cancel_files: Arc<Mutex<HashSet<String>>>,
    tracker: ProgressTracker,
) {
    loop {
        if cancel.load(Ordering::Relaxed) && rx.is_empty() {
            tracker.mark_canceled();
            break;
        }
        let mut work = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(work) => work,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        tracker.stage_pending_dec(2);
        if is_canceled(&work.path, &cancel, &cancel_files) {
            tracker.mark_canceled();
            continue;
        }
        tracker.stage_start(2, &work.path);

        match compute_hash(&work.path) {
            Ok(hash) => {
                work.hash = Some(hash);
                if let Some(preview_path) = work.preview_path.as_ref() {
                    match compute_dhash(preview_path) {
                        Ok(dhash) => work.dhash = Some(dhash as i64),
                        Err(err) => {
                            tracker.on_error();
                            log::warn!(
                                "dHash failed for {}: {}",
                                preview_path.display(),
                                err
                            );
                        }
                    }
                }
                tracker.stage_complete(2);
                if tx.send(work).is_err() {
                    break;
                }
                tracker.stage_pending_inc(3);
            }
            Err(err) => {
                tracker.on_error();
                tracker.stage_error(2);
                log::warn!("Hash failed for {}: {}", work.path.display(), err);
            }
        }
        tracker.emit_progress(false);
    }
}

fn run_tagging_stage(
    rx: Receiver<FileWork>,
    tx: Sender<FileWork>,
    pool: DbPool,
    paths: AppPaths,
    tagging: TaggingConfig,
    cancel: Arc<AtomicBool>,
    cancel_files: Arc<Mutex<HashSet<String>>>,
    tracker: ProgressTracker,
) {
    let mut engine = TaggingEngine::new(tagging).unwrap_or_else(|err| {
        log::warn!("Tagging engine init failed: {err}");
        TaggingEngine::new(TaggingConfig::default())
            .expect("Failed to initialize fallback tagging engine")
    });
    loop {
        if cancel.load(Ordering::Relaxed) && rx.is_empty() {
            tracker.mark_canceled();
            break;
        }
        let mut work = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(work) => work,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        tracker.stage_pending_dec(3);
        if is_canceled(&work.path, &cancel, &cancel_files) {
            tracker.mark_canceled();
            continue;
        }
        tracker.stage_start(3, &work.path);

        let hash = work.hash.clone().unwrap_or_else(|| "unknown".into());
        let file_name = work
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let ext = work
            .path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();

        let mut photo = PhotoRecord {
            id: None,
            path: work.path.to_string_lossy().to_string(),
            hash: hash.clone(),
            file_name,
            ext,
            size: work.size,
            mtime: work.mtime,
            width: work.exif.width,
            height: work.exif.height,
            make: work.exif.make.clone(),
            model: work.exif.model.clone(),
            lens: work.exif.lens.clone(),
            date_taken: work.exif.datetime_original,
            iso: work.exif.iso,
            fnumber: work.exif.fnumber,
            focal_length: work.exif.focal_length,
            exposure_time: work.exif.exposure_time,
            exposure_comp: work.exif.exposure_comp,
            gps_lat: work.exif.gps_lat,
            gps_lng: work.exif.gps_lng,
            thumb_path: work
                .thumb_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            preview_path: work
                .preview_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            dhash: work.dhash,
            rating: None,
            picked: false,
            rejected: false,
            last_modified: None,
            import_batch_id: Some(work.import_batch_id.clone()),
            created_at: None,
            updated_at: None,
        };

        let tagging = match work.preview_path.as_ref() {
            Some(preview_path) => match catch_unwind(AssertUnwindSafe(|| {
                engine.classify(preview_path, &work.exif)
            })) {
                Ok(Ok(tagging)) => tagging,
                Ok(Err(err)) => {
                    log::warn!(
                        "Auto tagging failed for {}: {}",
                        preview_path.display(),
                        err
                    );
                    TaggingResult::default()
                }
                Err(_) => {
                    log::warn!(
                        "ONNX runtime panicked while tagging {}; skipping auto tags",
                        preview_path.display()
                    );
                    TaggingResult::default()
                }
            },
            None => TaggingResult::default(),
        };

        match pool.get() {
            Ok(conn) => {
                match db::upsert_photo(&conn, &photo) {
                    Ok(photo_id) => {
                        photo.id = Some(photo_id);
                        work.photo_id = Some(photo_id);
                        if let Err(err) = db::replace_auto_tags(&conn, photo_id, tagging, &work.exif)
                        {
                            tracker.on_error();
                            tracker.stage_error(3);
                            log::warn!("Tag persistence failed for {}: {}", photo.path, err);
                        } else {
                            tracker.stage_complete(3);
                            if tx.send(work).is_err() {
                                break;
                            }
                            tracker.stage_pending_inc(4);
                        }
                    }
                    Err(err) => {
                        tracker.on_error();
                        tracker.stage_error(3);
                        log::warn!("Photo upsert failed for {}: {}", photo.path, err);
                    }
                }
            }
            Err(err) => {
                tracker.on_error();
                tracker.stage_error(3);
                log::warn!("DB connection failed for {}: {}", photo.path, err);
            }
        }
        tracker.emit_progress(false);
    }
}

fn run_embedding_stage(
    rx: Receiver<FileWork>,
    pool: DbPool,
    cancel: Arc<AtomicBool>,
    cancel_files: Arc<Mutex<HashSet<String>>>,
    tracker: ProgressTracker,
) {
    loop {
        if cancel.load(Ordering::Relaxed) && rx.is_empty() {
            tracker.mark_canceled();
            break;
        }
        let work = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(work) => work,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        tracker.stage_pending_dec(4);
        if is_canceled(&work.path, &cancel, &cancel_files) {
            tracker.mark_canceled();
            continue;
        }
        tracker.stage_start(4, &work.path);

        let mut success = true;
        if let Some(photo_id) = work.photo_id {
            if let Some(preview_path) = work.preview_path.as_ref() {
                match embedding::compute_embedding(preview_path)
                    .map(|vec| embedding::normalize_embedding(&vec))
                {
                    Ok((embedding_vec, _norm)) => {
                        if let Ok(conn) = pool.get() {
                            if let Err(err) =
                                db::upsert_embedding(&conn, photo_id, &embedding_vec, 1.0)
                            {
                                tracker.on_error();
                                tracker.stage_error(4);
                                log::warn!(
                                    "Embedding persistence failed for {}: {}",
                                    preview_path.display(),
                                    err
                                );
                                success = false;
                            }
                        }
                    }
                    Err(err) => {
                        tracker.on_error();
                        tracker.stage_error(4);
                        log::warn!("Embedding failed for {}: {}", preview_path.display(), err);
                        success = false;
                    }
                }
            }
        }

        if success {
            tracker.stage_complete(4);
            tracker.on_processed();
        }
        tracker.emit_progress(false);
    }
}

fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| SUPPORTED_EXT.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn is_canceled(path: &Path, cancel: &AtomicBool, cancel_files: &Mutex<HashSet<String>>) -> bool {
    if cancel.load(Ordering::Relaxed) {
        return true;
    }
    let path_str = path.to_string_lossy().to_string();
    let canceled = cancel_files.lock().unwrap();
    canceled.contains(&path_str)
}

fn compute_hash(path: &Path) -> Result<String> {
    let data = fs::read(path)?;
    let digest = xxh3_128(&data);
    Ok(format!("{:x}", digest))
}

fn name_hint(path: &Path) -> String {
    let key = path.to_string_lossy();
    let digest = xxh3_128(key.as_bytes());
    format!("{:x}", digest)
}

fn compute_dhash(path: &Path) -> Result<u64> {
    let img = image::open(path)?.to_luma8();
    let resized = image::imageops::resize(&img, 9, 8, image::imageops::FilterType::Triangle);
    let mut hash: u64 = 0;
    for y in 0..8 {
        for x in 0..8 {
            let left = resized.get_pixel(x, y)[0] as i16;
            let right = resized.get_pixel(x + 1, y)[0] as i16;
            let bit = left > right;
            hash = (hash << 1) | (bit as u64);
        }
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::TrySendError;
    use image::GrayImage;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;

    #[test]
    fn bounded_channel_applies_backpressure() {
        let (tx, _rx) = bounded::<i32>(1);
        tx.send(1).unwrap();
        match tx.try_send(2) {
            Err(TrySendError::Full(_)) => {}
            other => panic!("Expected backpressure, got {other:?}"),
        }
    }

    #[test]
    fn cancel_token_blocks_paths() {
        let cancel = AtomicBool::new(false);
        let cancel_files = Mutex::new(HashSet::new());
        cancel_files
            .lock()
            .unwrap()
            .insert("C:\\test\\photo.jpg".into());
        let canceled_path = PathBuf::from("C:\\test\\photo.jpg");
        assert!(is_canceled(&canceled_path, &cancel, &cancel_files));

        cancel.store(true, Ordering::Relaxed);
        let other_path = PathBuf::from("C:\\test\\other.jpg");
        assert!(is_canceled(&other_path, &cancel, &cancel_files));
    }

    #[test]
    fn dhash_changes_for_different_images() {
        let dir = std::env::temp_dir();
        let path_a = dir.join("pt_dhash_a.png");
        let path_b = dir.join("pt_dhash_b.png");
        let mut img_a = GrayImage::new(9, 8);
        for (x, y, pixel) in img_a.enumerate_pixels_mut() {
            *pixel = image::Luma([(x + y) as u8]);
        }
        img_a.save(&path_a).unwrap();

        let mut img_b = GrayImage::new(9, 8);
        for (x, y, pixel) in img_b.enumerate_pixels_mut() {
            *pixel = image::Luma([(x.wrapping_mul(2) + y) as u8]);
        }
        img_b.save(&path_b).unwrap();

        let hash_a = compute_dhash(&path_a).unwrap();
        let hash_b = compute_dhash(&path_b).unwrap();
        assert_ne!(hash_a, hash_b);
    }
}
