import { useEffect, useMemo, useRef, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/tauri";
import { open } from "@tauri-apps/api/dialog";
import { listen } from "@tauri-apps/api/event";
import { appWindow } from "@tauri-apps/api/window";

const SORT_OPTIONS = [
  { value: "date_taken", label: "Date (newest)" },
  { value: "last_modified", label: "Last modified" },
  { value: "rating", label: "Rating" },
  { value: "picked", label: "Picked" },
  { value: "rejected", label: "Rejected" },
  { value: "file_name", label: "Filename" },
];

const INFERENCE_DEVICE_OPTIONS = [
  { value: "auto", label: "Auto (GPU if available)" },
  { value: "gpu", label: "GPU (DirectML)" },
  { value: "cpu", label: "CPU" },
];

const DEFAULT_FILTERS = { search: "", tags: [], sort_by: "date_taken", sort_dir: "DESC" };

function usePersistentState(key, defaultValue) {
  const [value, setValue] = useState(() => {
    const stored = window.localStorage.getItem(key);
    return stored ? JSON.parse(stored) : defaultValue;
  });
  useEffect(() => {
    window.localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);
  return [value, setValue];
}

function resolvePath(path) {
  return path ? convertFileSrc(path) : null;
}

function formatExposureTime(value) {
  if (!value) return "n/a";
  if (value >= 1) return `${value.toFixed(1)}`;
  const denom = Math.round(1 / value);
  if (!denom || !Number.isFinite(denom)) return `${value.toFixed(4)}`;
  return `1/${denom}`;
}

function formatFNumber(value) {
  if (!value) return "n/a";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num % 1 === 0 ? num.toFixed(0) : num.toFixed(1);
}

function formatFocalLength(value) {
  if (!value) return "n/a";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num % 1 === 0 ? num.toFixed(0) : num.toFixed(1);
}

function useResizeObserver(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const observer = new ResizeObserver((entries) => {
      if (!entries.length) return;
      const entry = entries[0];
      setSize({
        width: Math.round(entry.contentRect.width),
        height: Math.round(entry.contentRect.height),
      });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

function StarIcon({ filled }) {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill={filled ? "url(#starGradient)" : "none"}
      stroke={filled ? "none" : "var(--muted)"}
    >
      <defs>
        <linearGradient id="starGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="var(--accent)" />
          <stop offset="100%" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
      <path
        d="M12 3l2.9 5.9 6.5.9-4.7 4.6 1.1 6.4L12 17.8 6.2 20.8l1.1-6.4L2.6 9.8l6.5-.9z"
        strokeWidth="1.3"
      />
    </svg>
  );
}

function RatingStars({ value, onChange, compact, showClear = false }) {
  return (
    <div className={`rating-stars ${compact ? "compact" : ""}`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} className={value >= n ? "filled" : ""} onClick={() => onChange(n)} aria-label={`${n} star`}>
          <StarIcon filled={value >= n} />
        </button>
      ))}
      {showClear && value > 0 && (
        <button className="clear" onClick={() => onChange(null)}>
          Clear
        </button>
      )}
    </div>
  );
}

function ArrowIcon({ direction = "left" }) {
  const isLeft = direction === "left";
  return (
    <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d={isLeft ? "M15 4L7 12L15 20" : "M9 4L17 12L9 20"}
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TagList({ tags, onRemove, limit, showAll, onToggle }) {
  const sorted = tags.slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  const visible = showAll || !limit ? sorted : sorted.slice(0, limit);
  const hiddenCount = sorted.length - visible.length;
  return (
    <div className="tag-list">
      {visible.map((tag) => (
        <span key={`${tag.tag}-${tag.source}`} className="tag-pill">
          <span>{tag.tag}</span>
          {tag.confidence != null && <span className="confidence">{Math.round(tag.confidence * 100)}%</span>}
          <span className="source">{tag.source}</span>
          {onRemove && tag.source === "manual" && (
            <button onClick={() => onRemove(tag.tag)} aria-label={`Remove ${tag.tag}`}>
              x
            </button>
          )}
        </span>
      ))}
      {!sorted.length && <div className="muted">No tags yet</div>}
      {hiddenCount > 0 && onToggle && (
        <button className="tag-toggle" onClick={onToggle}>
          {showAll ? "Show top 3" : `Show all (${hiddenCount + visible.length})`}
        </button>
      )}
    </div>
  );
}

function ThumbCard({ photo, selected, onSelect, onDoubleClick, thumbSize, variant, style }) {
  const sizingStyle = thumbSize ? { "--thumb-size": `${thumbSize}px` } : {};
  const mergedStyle = { ...sizingStyle, ...style };
  const isFilmstrip = variant === "filmstrip";
  const shutter = formatExposureTime(photo.exposure_time);
  const fNumber = formatFNumber(photo.fnumber);
  const iso = photo.iso ?? "n/a";
  const hoverMeta = `${shutter} | ${fNumber} | ${iso}`;
  return (
    <div
      className={`thumb ${isFilmstrip ? "filmstrip" : ""} ${selected ? "selected" : ""} ${
        photo.picked ? "picked" : photo.rejected ? "rejected" : ""
      }`}
      onClick={onSelect}
      onDoubleClick={onDoubleClick}
      style={mergedStyle}
    >
      {photo.thumb_path ? (
        <img src={resolvePath(photo.thumb_path)} alt={photo.file_name} />
      ) : (
        <div className="thumb-placeholder">No preview</div>
      )}
      {(photo.picked || photo.rejected) && (
        <div className={`thumb-badge ${photo.rejected ? "reject" : "pick"}`}>
          {photo.rejected ? "Reject" : "Pick"}
        </div>
      )}
      {isFilmstrip ? (
        <div className="thumb-overlay" title={`${photo.file_name} | ${hoverMeta}`}>
          <span className="thumb-title">{photo.file_name}</span>
          <span className="thumb-meta">{hoverMeta}</span>
        </div>
      ) : (
        <div className="thumb-caption">
          <div className="filename" title={photo.file_name}>
            {photo.file_name}
          </div>
          <div className="meta-row">
            {photo.rating ? <span className="pill rating-pill">R{photo.rating}</span> : null}
            {photo.picked ? <span className="pill pick">Pick</span> : null}
            {photo.rejected ? <span className="pill reject">Reject</span> : null}
          </div>
        </div>
      )}
    </div>
  );
}

function VirtualizedGallery({ photos, thumbSize, selection, onSelect, onDoubleClick, activeIndex }) {
  const scrollRef = useRef(null);
  const { width, height } = useResizeObserver(scrollRef);
  const [scrollTop, setScrollTop] = useState(0);
  const gap = 12;
  const padding = 8;
  const rowHeight = thumbSize + 70;
  const columns = Math.max(1, Math.floor((width - padding * 2 + gap) / (thumbSize + gap)));
  const totalRows = Math.ceil(photos.length / columns);
  const totalHeight = totalRows * rowHeight + padding * 2;
  const overscan = 2;
  const startRow = Math.max(0, Math.floor((scrollTop - overscan * rowHeight) / rowHeight));
  const endRow = Math.min(
    totalRows - 1,
    Math.ceil((scrollTop + height + overscan * rowHeight) / rowHeight)
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => setScrollTop(el.scrollTop);
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!scrollRef.current || activeIndex == null || activeIndex < 0) return;
    const row = Math.floor(activeIndex / columns);
    const rowTop = row * rowHeight;
    const rowBottom = rowTop + rowHeight;
    const viewTop = scrollRef.current.scrollTop;
    const viewBottom = viewTop + height;
    if (rowTop < viewTop) {
      scrollRef.current.scrollTo({ top: Math.max(rowTop - padding, 0), behavior: "smooth" });
    } else if (rowBottom > viewBottom) {
      scrollRef.current.scrollTo({ top: rowBottom - height + padding, behavior: "smooth" });
    }
  }, [activeIndex, columns, height, rowHeight, padding]);

  const visibleItems = useMemo(() => {
    const items = [];
    for (let row = startRow; row <= endRow; row += 1) {
      for (let col = 0; col < columns; col += 1) {
        const index = row * columns + col;
        if (index >= photos.length) break;
        const photo = photos[index];
        items.push(
          <ThumbCard
            key={photo.photo.id}
            photo={photo.photo}
            selected={selection.includes(photo.photo.id)}
            onSelect={(e) => onSelect(photo.photo.id, index, e)}
            onDoubleClick={() => onDoubleClick(photo.photo.id)}
            thumbSize={thumbSize}
            style={{
              position: "absolute",
              top: padding + row * rowHeight,
              left: padding + col * (thumbSize + gap),
              width: thumbSize,
            }}
          />
        );
      }
    }
    return items;
  }, [photos, selection, thumbSize, columns, startRow, endRow, padding, gap, rowHeight, onSelect, onDoubleClick]);

  return (
    <div className="gallery virtualized" ref={scrollRef}>
      <div className="gallery-inner" style={{ height: totalHeight }}>
        {visibleItems}
      </div>
    </div>
  );
}

function SelectionBar({ count, onRate, onPick, onReject, onClear, onTag }) {
  const [tagText, setTagText] = useState("");
  if (count <= 1) return null;
  return (
    <div className="selection-bar">
      <span>{count} selected</span>
      <div className="selection-actions">
        <RatingStars value={0} onChange={onRate} compact showClear />
        <button className="ghost" onClick={onPick}>
          Pick
        </button>
        <button className="ghost" onClick={onReject}>
          Reject
        </button>
        <button className="ghost" onClick={onClear}>
          Clear rating
        </button>
        <div className="tag-batch">
          <input
            type="text"
            placeholder="Add tag"
            value={tagText}
            onChange={(e) => setTagText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && tagText.trim()) {
                onTag(tagText.trim());
                setTagText("");
              }
            }}
          />
          <button onClick={() => tagText.trim() && (onTag(tagText.trim()), setTagText(""))}>Add</button>
        </div>
      </div>
    </div>
  );
}

function Toast({ toast, onUndo, onClose }) {
  if (!toast) return null;
  return (
    <div className="toast">
      <span>{toast.message}</span>
      {toast.canUndo && (
        <button className="ghost" onClick={onUndo}>
          Undo
        </button>
      )}
      <button className="ghost" onClick={onClose}>
        x
      </button>
    </div>
  );
}

function EmptyState({ onImport, onClearFilters }) {
  return (
    <div className="empty-state">
      <div className="empty-title">No photos match this view.</div>
      <div className="empty-actions">
        <button onClick={onClearFilters}>Clear filters</button>
        <button className="ghost" onClick={onImport}>
          Import folder
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = usePersistentState("pt-mode", "CULL");
  const [theme, setTheme] = usePersistentState("pt-theme", "dark");
  const [inferenceDevice, setInferenceDevice] = usePersistentState("pt-inference-device", "auto");
  const [smartView, setSmartView] = useState("ALL");
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [thumbSize, setThumbSize] = usePersistentState("pt-thumb-size", 190);
  const [autoAdvance, setAutoAdvance] = usePersistentState("pt-auto-advance", true);
  const [lastImportPath, setLastImportPath] = usePersistentState("pt-last-import", "");
  const [photos, setPhotos] = useState([]);
  const [selection, setSelection] = useState([]);
  const [cursorIndex, setCursorIndex] = useState(0);
  const [progress, setProgress] = useState({
    discovered: 0,
    processed: 0,
    errors: 0,
    current_file: "",
    current_stage: "",
    throughput: null,
    stages: [],
    canceled: false,
  });
  const [importJobId, setImportJobId] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [rerunLoading, setRerunLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [toast, setToast] = useState(null);
  const [inferenceStatus, setInferenceStatus] = useState(null);
  const [inferenceBusy, setInferenceBusy] = useState(false);
  const [zoomMode, setZoomMode] = usePersistentState("pt-zoom-mode", "FIT");
  const [fullscreen, setFullscreen] = useState(false);
  const [imageReady, setImageReady] = useState(false);
  const [showAllTags, setShowAllTags] = useState(false);
  const [duplicatesOpen, setDuplicatesOpen] = useState(false);
  const [duplicateGroups, setDuplicateGroups] = useState([]);
  const [duplicateThreshold, setDuplicateThreshold] = usePersistentState("pt-dup-threshold", 8);
  const [duplicatesLoading, setDuplicatesLoading] = useState(false);
  const [reviewedGroups, setReviewedGroups] = useState(() => new Set());
  const [similarOpen, setSimilarOpen] = useState(false);
  const [similarResults, setSimilarResults] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const lastActionRef = useRef(null);
  const searchRef = useRef(null);
  const anchorRef = useRef(null);
  const filmstripRef = useRef(null);
  const toastTimerRef = useRef(null);

  const activePhoto = useMemo(() => {
    if (!selection.length) return photos[0] || null;
    return photos.find((p) => p.photo.id === selection[0]) || photos[0] || null;
  }, [photos, selection]);

  useEffect(() => {
    const unlisten = listen("import-progress", (event) => {
      setProgress((prev) => ({ ...prev, ...event.payload }));
    });
    refreshPhotos({ resetCursor: true });
    invoke("is_importing")
      .then((value) => setImporting(Boolean(value)))
      .catch(() => null);
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    invoke("get_inference_status")
      .then((status) => setInferenceStatus(status))
      .catch(() => null);
  }, []);

  useEffect(() => {
    let canceled = false;
    setInferenceBusy(true);
    invoke("set_inference_device", { device: inferenceDevice })
      .then((status) => {
        if (canceled) return;
        setInferenceStatus(status);
        if (status?.warning) {
          setToast({ message: status.warning, canUndo: false });
        }
      })
      .catch((err) => {
        if (canceled) return;
        setErrorMessage(`Inference device update failed: ${err}`);
      })
      .finally(() => {
        if (!canceled) setInferenceBusy(false);
      });
    return () => {
      canceled = true;
    };
  }, [inferenceDevice]);

  useEffect(() => {
    setImageReady(false);
  }, [activePhoto?.photo?.preview_path]);

  useEffect(() => {
    if (mode === "CULL") {
      setShowAllTags(false);
    }
  }, [activePhoto?.photo?.id, mode]);

  useEffect(() => {
    setSimilarOpen(false);
    setSimilarResults([]);
  }, [activePhoto?.photo?.id]);

  useEffect(() => {
    if (!duplicatesOpen || mode !== "BROWSE") return;
    const handle = setTimeout(() => {
      loadDuplicates();
    }, 250);
    return () => clearTimeout(handle);
  }, [duplicatesOpen, duplicateThreshold, mode]);

  useEffect(() => {
    setSmartView(mode === "CULL" ? "UNSORTED" : "ALL");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  useEffect(() => {
    if (!toast) return;
    if (toastTimerRef.current) {
      clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = setTimeout(() => {
      setToast(null);
      toastTimerRef.current = null;
    }, 3500);
    return () => {
      if (toastTimerRef.current) {
        clearTimeout(toastTimerRef.current);
        toastTimerRef.current = null;
      }
    };
  }, [toast]);

  useEffect(() => {
    let unlisten;
    appWindow
      .onFileDropEvent(async (event) => {
        if (event.payload?.type !== "drop") return;
        const paths = event.payload?.paths || [];
        if (!paths.length) return;
        handleDroppedPaths(paths);
      })
      .then((stop) => {
        unlisten = stop;
      })
      .catch(() => null);
    return () => {
      if (unlisten) unlisten();
    };
  }, [importing, lastImportPath, mode]);

  useEffect(() => {
    if (!importing) return;
    const hasStages = Array.isArray(progress.stages) && progress.stages.length > 0;
    const pending = hasStages ? progress.stages.reduce((sum, s) => sum + (s.pending || 0) + (s.in_progress || 0), 0) : 0;
    const done = progress.discovered > 0 && pending === 0;
    if (done) {
      setImporting(false);
      setImportJobId(null);
      refreshPhotos({ resetCursor: true });
    }
  }, [importing, progress]);

  useEffect(() => {
    refreshPhotos({ resetCursor: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [smartView, filters.sort_by, filters.search, mode]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    const handler = (e) => {
      const target = e.target;
      const isInput = target instanceof HTMLElement && ["INPUT", "TEXTAREA"].includes(target.tagName);
      if (isInput && e.key !== "Escape") return;
      if (!photos.length) return;
      switch (e.key) {
        case " ":
          e.preventDefault();
          setZoomMode((mode) => (mode === "FIT" ? "ONE_TO_ONE" : "FIT"));
          break;
        case "ArrowRight":
          e.preventDefault();
          moveCursor(1);
          break;
        case "ArrowLeft":
          e.preventDefault();
          moveCursor(-1);
          break;
        case "0":
          e.preventDefault();
          applyCullChange({ rating: null, label: "Rating cleared" });
          break;
        case "1":
        case "2":
        case "3":
        case "4":
        case "5":
          e.preventDefault();
          applyCullChange({ rating: Number(e.key), label: `Rated ${e.key}` });
          break;
        case "p":
        case "P":
          e.preventDefault();
          togglePick();
          break;
        case "x":
        case "X":
          e.preventDefault();
          toggleReject();
          break;
        case "/":
          e.preventDefault();
          searchRef.current?.focus();
          break;
        case "Escape":
          setSelection([]);
          break;
        case "f":
        case "F":
          e.preventDefault();
          handleToggleFullscreen();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [photos.length, cursorIndex, selection, smartView]);

  useEffect(() => {
    const onFullscreenChange = () => {
      setFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  const updateSelectionAfterRefresh = (list, options) => {
    if (!list.length) {
      setSelection([]);
      setCursorIndex(0);
      return;
    }
    const availableIds = new Set(list.map((p) => p.photo.id));
    const stillSelected = selection.filter((id) => availableIds.has(id));
    if (stillSelected.length) {
      setSelection(stillSelected);
      const nextIdx = list.findIndex((p) => p.photo.id === stillSelected[0]);
      setCursorIndex(nextIdx === -1 ? 0 : nextIdx);
      anchorRef.current = nextIdx === -1 ? 0 : nextIdx;
      return;
    }
    const preferredIndex = Math.min(
      options?.preferredIndex ?? (options?.resetCursor ? 0 : cursorIndex),
      list.length - 1
    );
    setSelection([list[preferredIndex].photo.id]);
    setCursorIndex(preferredIndex);
    anchorRef.current = preferredIndex;
  };

  const refreshPhotos = async (options = {}) => {
    try {
      const limit = mode === "CULL" ? 500 : 800;
      const effectiveSmartView = mode === "CULL" ? "UNSORTED" : smartView;
      const result = await invoke("query_photos", {
        filters: {
          ...filters,
          mode,
          smart_view: effectiveSmartView === "ALL" ? null : effectiveSmartView,
          sort_by: filters.sort_by || (mode === "CULL" ? "last_modified" : "date_taken"),
          limit,
        },
      });
      setPhotos(result);
      updateSelectionAfterRefresh(result, options);
    } catch (err) {
      setErrorMessage(`Failed to load photos: ${err}`);
    }
  };

  const moveCursor = (delta) => {
    if (!photos.length) return;
    const next = Math.min(Math.max((cursorIndex ?? 0) + delta, 0), photos.length - 1);
    setCursorIndex(next);
    setSelection([photos[next].photo.id]);
    anchorRef.current = next;
  };

  const scrollFilmstrip = (direction) => {
    const el = filmstripRef.current;
    if (!el) return;
    const amount = Math.max(el.clientWidth * 0.8, 220) * direction;
    el.scrollBy({ left: amount, behavior: "smooth" });
  };

  const applyCullChange = async ({ rating, picked, rejected, label }) => {
    const ids = selection.length
      ? selection
      : activePhoto
      ? [activePhoto.photo.id]
      : [];
    if (!ids.length) return;
    const before = photos
      .filter((p) => ids.includes(p.photo.id))
      .map((p) => ({
        id: p.photo.id,
        rating: p.photo.rating,
        picked: p.photo.picked,
        rejected: p.photo.rejected,
      }));

    try {
      const payload = { photoIds: ids };
      if (rating !== undefined) payload.rating = rating;
      if (picked !== undefined) payload.picked = picked;
      if (rejected !== undefined) payload.rejected = rejected;

      // When only clearing rating, some bindings ignore null in batch; call dedicated setter.
      if (rating === null && picked === undefined && rejected === undefined) {
        await Promise.all(ids.map((id) => invoke("set_rating", { photoId: id, rating: null })));
      } else {
        await invoke("batch_update_cull", payload);
      }
      lastActionRef.current = { before };
      setToast({ message: label || "Updated", canUndo: true });
      const preferredIndex = autoAdvance ? cursorIndex + 1 : cursorIndex;
      await refreshPhotos({ preferredIndex });
    } catch (err) {
      setErrorMessage(`Update failed: ${err}`);
    }
  };

  const togglePick = () => {
    const current = activePhoto?.photo;
    if (!current) return;
    const next = !current.picked;
    applyCullChange({ picked: next, rejected: next ? false : undefined, label: next ? "Picked" : "Unpicked" });
  };

  const toggleReject = () => {
    const current = activePhoto?.photo;
    if (!current) return;
    const next = !current.rejected;
    applyCullChange({ rejected: next, picked: next ? false : undefined, label: next ? "Rejected" : "Restored" });
  };

  const handleImport = async () => {
    let pickedDir = null;
    try {
      pickedDir = await open({
        directory: true,
        recursive: true,
        defaultPath: lastImportPath || undefined,
      });
      if (pickedDir) {
        setImporting(true);
        const startedJobId = await invoke("import_folder", { path: pickedDir });
        setImportJobId(startedJobId);
        setLastImportPath(pickedDir);
        setSmartView(mode === "CULL" ? "UNSORTED" : "ALL");
      }
    } catch (err) {
      setErrorMessage(`Import failed: ${err}`);
      setImporting(false);
    } finally {
      if (!pickedDir) {
        setImporting(false);
      }
    }
  };

  const handleTestInference = async () => {
    try {
      await invoke("test_inference", { count: 24 });
      setToast({ message: "Inference test started (see dev logs)", canUndo: false });
    } catch (err) {
      setErrorMessage(`Test inference failed: ${err}`);
    }
  };

  const handleCancelImport = async () => {
    try {
      await invoke("cancel_import");
      setImporting(false);
      setImportJobId(null);
    } catch (err) {
      setErrorMessage(`Cancel failed: ${err}`);
    }
  };

  const handleAddTag = async (tagText) => {
    if (!tagText || !selection.length) return;
    try {
      await Promise.all(selection.map((id) => invoke("add_manual_tag", { photoId: id, tag: tagText })));
      setToast({ message: `Added tag "${tagText}"`, canUndo: false });
      await refreshPhotos();
    } catch (err) {
      setErrorMessage(`Add tag failed: ${err}`);
    }
  };

  const handleRemoveTag = async (tag) => {
    if (!activePhoto) return;
    try {
      await invoke("remove_manual_tag", { photoId: activePhoto.photo.id, tag });
      await refreshPhotos();
    } catch (err) {
      setErrorMessage(`Remove tag failed: ${err}`);
    }
  };

  const handleRerun = async () => {
    if (!activePhoto) return;
    try {
      setRerunLoading(true);
      await invoke("rerun_auto", { photoId: activePhoto.photo.id });
      await refreshPhotos();
    } catch (err) {
      setErrorMessage(`Auto detection failed: ${err}`);
    } finally {
      setRerunLoading(false);
    }
  };

  const handleShowInFolder = async () => {
    if (!activePhoto?.photo?.path) return;
    try {
      await invoke("show_in_folder", { path: activePhoto.photo.path });
    } catch (err) {
      setErrorMessage(`Show in folder failed: ${err}`);
    }
  };

  const loadDuplicates = async () => {
    try {
      setDuplicatesLoading(true);
      const groups = await invoke("find_duplicates", { threshold: duplicateThreshold });
      setDuplicateGroups(groups || []);
    } catch (err) {
      setErrorMessage(`Find duplicates failed: ${err}`);
    } finally {
      setDuplicatesLoading(false);
    }
  };

  const markGroupReviewed = (groupId) => {
    setReviewedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  };

  const resolveBestPhotoId = (group) => {
    const activeId = activePhoto?.photo?.id;
    const activeInGroup = group.photos.find((p) => p.id === activeId);
    if (activeInGroup) return activeInGroup.id;
    return group.photos.reduce((best, current) => {
      const bestArea = (best.width || 0) * (best.height || 0);
      const currentArea = (current.width || 0) * (current.height || 0);
      if (currentArea !== bestArea) return currentArea > bestArea ? current : best;
      return current.size > best.size ? current : best;
    }, group.photos[0]).id;
  };

  const handlePickBest = async (group) => {
    const bestId = resolveBestPhotoId(group);
    const otherIds = group.photos.filter((p) => p.id !== bestId).map((p) => p.id);
    try {
      await invoke("batch_update_cull", { photoIds: [bestId], picked: true, rejected: false });
      if (otherIds.length) {
        await invoke("batch_update_cull", { photoIds: otherIds, rejected: true, picked: false });
      }
      setToast({ message: "Picked best and rejected others.", canUndo: true });
      refreshPhotos();
    } catch (err) {
      setErrorMessage(`Pick best failed: ${err}`);
    }
  };

  const handleRejectOthers = async (group) => {
    const keepId = resolveBestPhotoId(group);
    const otherIds = group.photos.filter((p) => p.id !== keepId).map((p) => p.id);
    if (!otherIds.length) return;
    try {
      await invoke("batch_update_cull", { photoIds: otherIds, rejected: true, picked: false });
      setToast({ message: "Rejected duplicates.", canUndo: true });
      refreshPhotos();
    } catch (err) {
      setErrorMessage(`Reject others failed: ${err}`);
    }
  };

  const handleFindSimilar = async () => {
    if (!activePhoto) return;
    try {
      setSimilarLoading(true);
      setSimilarOpen(true);
      const results = await invoke("find_similar", { photoId: activePhoto.photo.id, limit: 12 });
      setSimilarResults(results || []);
    } catch (err) {
      setErrorMessage(`Find similar failed: ${err}`);
    } finally {
      setSimilarLoading(false);
    }
  };

  const handleDroppedPaths = async (paths) => {
    if (!paths.length) return;
    if (importing && !window.confirm("Import already running. Cancel and start a new one?")) {
      return;
    }
    if (importing) {
      await handleCancelImport();
    }
    for (const path of paths) {
      try {
        const isDir = await invoke("is_directory", { path });
        if (isDir) {
          setImporting(true);
          const startedJobId = await invoke("import_folder", { path });
          setImportJobId(startedJobId);
          setLastImportPath(path);
          setSmartView(mode === "CULL" ? "UNSORTED" : "ALL");
          return;
        }
      } catch (err) {
        setErrorMessage(`Drop import failed: ${err}`);
        return;
      }
    }
    setToast({ message: "Dropped files ignored (folders only).", canUndo: false });
  };

  const handleToggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch (err) {
      setErrorMessage(`Fullscreen failed: ${err}`);
    }
  };

  const handleUndo = async () => {
    if (!lastActionRef.current) return;
    const { before } = lastActionRef.current;
    try {
      for (const entry of before) {
        const changes = {
          rating: entry.rating ?? null,
          picked: entry.picked,
          rejected: entry.rejected,
        };
        if (changes.rating === null && changes.picked === undefined && changes.rejected === undefined) {
          await invoke("set_rating", { photoId: entry.id, rating: null });
        } else {
          await invoke("batch_update_cull", {
            photoIds: [entry.id],
            ...changes,
          });
        }
      }
      await refreshPhotos();
    } catch (err) {
      setErrorMessage(`Undo failed: ${err}`);
    } finally {
      setToast(null);
      lastActionRef.current = null;
    }
  };

  const onSelectPhoto = (photoId, idx, evt) => {
    setSelection((prev) => {
      let next = [];
      if (evt?.shiftKey && prev.length && anchorRef.current != null) {
        const start = Math.min(anchorRef.current, idx);
        const end = Math.max(anchorRef.current, idx);
        next = photos.slice(start, end + 1).map((p) => p.photo.id);
      } else if (evt?.metaKey || evt?.ctrlKey) {
        const set = new Set(prev);
        if (set.has(photoId)) set.delete(photoId);
        else set.add(photoId);
        next = Array.from(set);
        anchorRef.current = idx;
      } else {
        next = [photoId];
        anchorRef.current = idx;
      }
      setCursorIndex(idx);
      return next;
    });
  };

  const importProgressText = useMemo(() => {
    if (!progress.discovered) return importing ? "Importing..." : "Idle";
    const throughput = progress.throughput ? ` • ${progress.throughput.toFixed(1)}/sec` : "";
    return `Processing ${progress.processed}/${progress.discovered}${throughput}`;
  }, [progress, importing]);

  const stageSummary = useMemo(() => {
    if (!Array.isArray(progress.stages) || !progress.stages.length) return null;
    return progress.stages.map((stage) => {
      const pending = (stage.pending || 0) + (stage.in_progress || 0);
      const rate = stage.items_per_sec ? `${stage.items_per_sec.toFixed(1)}/sec` : null;
      return { ...stage, pending, rate };
    });
  }, [progress.stages]);

  return (
    <div className={`app-shell ${fullscreen ? "fullscreen" : ""} mode-${mode.toLowerCase()}`}>
      <header>
        <div className="brand">
          <h1>PhotoTag</h1>
          <div className="mode-toggle">
            {["CULL", "BROWSE"].map((m) => (
              <button key={m} className={mode === m ? "active" : ""} onClick={() => setMode(m)}>
                {m}
              </button>
            ))}
          </div>
        </div>
        <div className="actions">
          <div className="theme-toggle" role="group" aria-label="Theme">
            {["dark", "light"].map((value) => (
              <button
                key={value}
                className={theme === value ? "active" : ""}
                onClick={() => setTheme(value)}
                aria-pressed={theme === value}
              >
                {value === "dark" ? "Dark" : "Light"}
              </button>
            ))}
          </div>
          <div className="search">
            <input
              ref={searchRef}
              type="search"
              placeholder="Search by filename or camera (/ to focus)"
              value={filters.search}
              onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))}
            />
          </div>
          <button onClick={handleImport} disabled={importing}>
            {importing ? "Importing..." : "Import Folder"}
          </button>
          {mode === "BROWSE" && (
            <button
              className={duplicatesOpen ? "ghost active" : "ghost"}
              onClick={() => {
                const next = !duplicatesOpen;
                setDuplicatesOpen(next);
                if (next) loadDuplicates();
              }}
            >
              Duplicates
            </button>
          )}
          {importing && (
            <button className="ghost" onClick={handleCancelImport}>
              Cancel Import
            </button>
          )}
          <span className="progress">{importProgressText}</span>
          {importing && stageSummary && (
            <div className="progress-detail">
              <div className="progress-stage">
                <strong>Stage:</strong> {progress.current_stage || "scan"}
              </div>
              <div className="progress-grid">
                {stageSummary.map((stage) => (
                  <div key={stage.stage} className="progress-row">
                    <span className="label">{stage.stage}</span>
                    <span className="value">{stage.pending || 0} pending</span>
                    <span className="value">{stage.completed || 0} done</span>
                    {stage.rate && <span className="muted">{stage.rate}</span>}
                  </div>
                ))}
              </div>
              {progress.errors > 0 && <div className="progress-errors">{progress.errors} errors</div>}
            </div>
          )}
        </div>
      </header>

      {rerunLoading && <div className="banner info-banner">Re-running auto detection...</div>}
      {errorMessage && <div className="banner error-banner">{errorMessage}</div>}

      <div className="content">
        <main className={mode === "CULL" ? "cull" : "browse"}>
          {mode !== "CULL" && (
            <div className="view-toolbar">
            <div className="toolbar-group">
              <label className="toolbar-item">
                <span>Sort by</span>
                <select
                  value={filters.sort_by}
                  onChange={(e) => setFilters((f) => ({ ...f, sort_by: e.target.value }))}
                >
                  {SORT_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </label>
              {mode !== "CULL" && (
                <label className="toolbar-item slider">
                  <span>Thumbnail size</span>
                  <input
                    type="range"
                    min="120"
                    max="320"
                    value={thumbSize}
                    onChange={(e) => setThumbSize(Number(e.target.value))}
                  />
                </label>
              )}
            </div>
            <label className="toolbar-toggle">
              <input
                type="checkbox"
                checked={autoAdvance}
                onChange={(e) => setAutoAdvance(e.target.checked)}
              />
              Auto-advance after action
            </label>
            </div>
          )}
          {photos.length === 0 ? (
            <EmptyState
              onImport={handleImport}
              onClearFilters={() => {
                setFilters(DEFAULT_FILTERS);
                if (mode === "CULL") {
                  setSmartView("UNSORTED");
                } else {
                  setSmartView("ALL");
                }
              }}
            />
          ) : mode === "CULL" ? (
            <div className="cull-view">
              <div className="cull-preview">
                <div className="preview-bar">
                  <div className="preview-meta">
                    <div className="file-name" title={activePhoto?.photo.file_name}>
                      {activePhoto?.photo.file_name}
                    </div>
                    <div className="muted truncate" title={activePhoto?.photo.path}>
                      {activePhoto?.photo.path}
                    </div>
                  </div>
                  <div className="preview-actions">
                    <RatingStars
                      value={activePhoto?.photo.rating || 0}
                      onChange={(v) => applyCullChange({ rating: v, label: `Rated ${v ?? "clear"}` })}
                      compact
                      showClear={(activePhoto?.photo.rating || 0) > 0}
                    />
                    <div className="preview-chips">
                      {activePhoto?.photo.picked && <span className="preview-chip">Picked</span>}
                      {activePhoto?.photo.rejected && <span className="preview-chip reject">Rejected</span>}
                    </div>
                    <div className="preview-actions-right">
                      <button className={activePhoto?.photo.picked ? "ghost active" : "ghost"} onClick={togglePick}>
                        Pick
                      </button>
                      <button
                        className={activePhoto?.photo.rejected ? "ghost active reject" : "ghost"}
                        onClick={toggleReject}
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                </div>
                <div className="preview-stage">
                  <div className="preview-count">
                    {cursorIndex + 1} / {photos.length}
                  </div>
                  <div className="preview-frame">
                    <div className="preview-gutter left">
                      <button className="nav-arrow prev" onClick={() => moveCursor(-1)} aria-label="Previous photo">
                        <ArrowIcon direction="left" />
                      </button>
                    </div>
                    <div
                      className={`preview-media ${zoomMode === "ONE_TO_ONE" ? "zoomed" : ""} ${
                        activePhoto?.photo.picked ? "picked" : activePhoto?.photo.rejected ? "rejected" : ""
                      }`}
                    >
                      {activePhoto?.photo.preview_path ? (
                        <>
                          <img
                            src={resolvePath(activePhoto.photo.preview_path)}
                            alt=""
                            className={`preview-blur ${imageReady ? "ready" : ""}`}
                            aria-hidden="true"
                          />
                          <img
                            src={resolvePath(activePhoto.photo.preview_path)}
                            alt={activePhoto.photo.file_name}
                            className={`preview-large ${zoomMode === "ONE_TO_ONE" ? "zoomed" : ""} ${
                              imageReady ? "ready" : ""
                            }`}
                            onLoad={() => setImageReady(true)}
                          />
                        </>
                      ) : (
                        <div className="thumb-placeholder large">No preview</div>
                      )}
                      {(activePhoto?.photo.picked || activePhoto?.photo.rejected) && (
                        <div className={`preview-status ${activePhoto?.photo.rejected ? "reject" : "pick"}`}>
                          {activePhoto?.photo.rejected ? "Rejected" : "Picked"}
                        </div>
                      )}
                    </div>
                    <div className="zoom-indicator">{zoomMode === "ONE_TO_ONE" ? "1:1" : "Fit"}</div>
                    <div className="preview-gutter right">
                      <button className="nav-arrow next" onClick={() => moveCursor(1)} aria-label="Next photo">
                        <ArrowIcon direction="right" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              <div className="filmstrip-wrap">
                <button
                  className="strip-arrow prev"
                  onClick={() => scrollFilmstrip(-1)}
                  aria-label="Scroll filmstrip left"
                >
                  <ArrowIcon direction="left" />
                </button>
                <div
                  className="filmstrip"
                  style={{ ["--thumb-size"]: "var(--filmstrip-thumb-size)" }}
                  ref={filmstripRef}
                >
                  {photos.map((p, idx) => (
                    <ThumbCard
                      key={p.photo.id}
                      photo={p.photo}
                      selected={selection.includes(p.photo.id)}
                      onSelect={(e) => onSelectPhoto(p.photo.id, idx, e)}
                      onDoubleClick={() => setSelection([p.photo.id])}
                      variant="filmstrip"
                    />
                  ))}
                </div>
                <button
                  className="strip-arrow next"
                  onClick={() => scrollFilmstrip(1)}
                  aria-label="Scroll filmstrip right"
                >
                  <ArrowIcon direction="right" />
                </button>
              </div>
            </div>
          ) : (
            <VirtualizedGallery
              photos={photos}
              thumbSize={thumbSize}
              selection={selection}
              activeIndex={cursorIndex}
              onSelect={onSelectPhoto}
              onDoubleClick={(id) => setSelection([id])}
            />
          )}
        </main>

        <aside className="panel details">
          {activePhoto ? (
            <>
              {mode === "BROWSE" && duplicatesOpen && (
                <div className="duplicates-panel">
                  <div className="section-head">
                    <h4>Near-duplicates</h4>
                    <button className="ghost" onClick={loadDuplicates} disabled={duplicatesLoading}>
                      {duplicatesLoading ? "Scanning..." : "Refresh"}
                    </button>
                  </div>
                  <div className="dup-controls">
                    <label>
                      Threshold
                      <input
                        type="range"
                        min="4"
                        max="16"
                        value={duplicateThreshold}
                        onChange={(e) => setDuplicateThreshold(Number(e.target.value))}
                      />
                    </label>
                    <span className="muted">{duplicateThreshold}</span>
                  </div>
                  {duplicateGroups.length === 0 && !duplicatesLoading && (
                    <div className="muted">No near-duplicates found.</div>
                  )}
                  <div className="dup-groups">
                    {duplicateGroups.map((group) => (
                      <div
                        key={group.representative}
                        className={`dup-group ${reviewedGroups.has(group.representative) ? "reviewed" : ""}`}
                      >
                        <div className="dup-row">
                          <span className="muted">{group.photos.length} photos</span>
                          <div className="dup-actions">
                            <button className="small ghost" onClick={() => handlePickBest(group)}>
                              Pick best
                            </button>
                            <button className="small ghost" onClick={() => handleRejectOthers(group)}>
                              Reject others
                            </button>
                            <button className="small ghost" onClick={() => markGroupReviewed(group.representative)}>
                              {reviewedGroups.has(group.representative) ? "Unreview" : "Reviewed"}
                            </button>
                          </div>
                        </div>
                        <div className="dup-thumbs">
                          {group.photos.map((photo) => (
                            <button
                              key={photo.id}
                              className="dup-thumb"
                              onClick={() => setSelection([photo.id])}
                            >
                              {photo.thumb_path ? (
                                <img src={resolvePath(photo.thumb_path)} alt={photo.file_name} />
                              ) : (
                                <div className="thumb-placeholder">No preview</div>
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="detail-head">
                <div className="detail-title">
                  <div className="file-title">{activePhoto.photo.file_name}</div>
                  <div className="muted truncate">{activePhoto.photo.path}</div>
                </div>
                <div className="detail-actions">
                  <button className="ghost show-folder" onClick={handleShowInFolder}>
                    Show in folder
                  </button>
                  <button className="ghost" onClick={handleFindSimilar}>
                    Find similar
                  </button>
                </div>
              </div>
              {mode === "BROWSE" && (
                <div className="action-block">
                  <div className="rating-stack">
                    <RatingStars
                      value={activePhoto.photo.rating || 0}
                      onChange={(v) => applyCullChange({ rating: v, label: `Rated ${v ?? "clear"}` })}
                      showClear={(activePhoto?.photo.rating || 0) > 0}
                    />
                  </div>
                </div>
              )}
              <div className="meta">
                <div className="meta-grid">
                  <span>
                    <strong>Camera:</strong> {activePhoto.photo.make || "Unknown"} {activePhoto.photo.model || ""}
                  </span>
                  <span>
                    <strong>Lens:</strong> {activePhoto.photo.lens || "Unknown"}
                  </span>
                  <span>
                    <strong>ISO:</strong> {activePhoto.photo.iso ?? "n/a"}
                  </span>
                  <span>
                    <strong>F-number:</strong> {activePhoto.photo.fnumber ?? "n/a"}
                  </span>
                  <span>
                    <strong>Focal:</strong>{" "}
                    {activePhoto.photo.focal_length ? `${activePhoto.photo.focal_length}mm` : "n/a"}
                  </span>
                  <span>
                    <strong>Shutter:</strong> {formatExposureTime(activePhoto.photo.exposure_time)}
                  </span>
                </div>
              </div>
              <div className="tags-block">
                <div className="section-head">
                  <h4>Tags</h4>
                  <div className="tag-input">
                    <input
                      type="text"
                      placeholder="Add tag (Enter)"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && e.target.value.trim()) {
                          handleAddTag(e.target.value.trim());
                          e.target.value = "";
                        }
                      }}
                    />
                  </div>
                </div>
                <TagList
                  tags={activePhoto.tags}
                  onRemove={handleRemoveTag}
                  limit={mode === "CULL" ? 3 : null}
                  showAll={mode === "CULL" ? showAllTags : true}
                  onToggle={mode === "CULL" ? () => setShowAllTags((v) => !v) : null}
                />
              </div>
              <div className="settings-panel">
                <div className="section-head">
                  <h4>Inference</h4>
                  {import.meta.env.DEV && (
                    <button className="ghost small" onClick={handleTestInference} disabled={inferenceBusy}>
                      {inferenceBusy ? "Updating..." : "Test Inference"}
                    </button>
                  )}
                </div>
                <label className="settings-row">
                  <span>Device</span>
                  <select
                    value={inferenceDevice}
                    onChange={(e) => setInferenceDevice(e.target.value)}
                    disabled={inferenceBusy}
                  >
                    {INFERENCE_DEVICE_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="settings-meta">
                  <span>
                    <strong>Active:</strong> {inferenceStatus?.provider || "Unknown"}
                  </span>
                  {inferenceStatus?.runtime_version && (
                    <span className="muted">ORT {inferenceStatus.runtime_version}</span>
                  )}
                </div>
                {inferenceStatus?.models?.length > 0 && (
                  <div className="settings-models">
                    {inferenceStatus.models.map((model) => (
                      <div key={model.label} className="settings-model">
                        <span className="muted">{model.label}</span>
                        <span>{model.provider}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {similarOpen && (
                <div className="similar-panel">
                  <div className="section-head">
                    <h4>Similar photos</h4>
                    <button className="ghost small" onClick={() => setSimilarOpen(false)}>
                      Hide
                    </button>
                  </div>
                  {similarLoading && <div className="muted">Searching...</div>}
                  {!similarLoading && similarResults.length === 0 && (
                    <div className="muted">No embeddings yet for similar search.</div>
                  )}
                  <div className="similar-grid">
                    {similarResults.map((item) => (
                      <button
                        key={item.id}
                        className="similar-card"
                        onClick={() => setSelection([item.id])}
                      >
                        {item.thumb_path ? (
                          <img src={resolvePath(item.thumb_path)} alt={item.file_name} />
                        ) : (
                          <div className="thumb-placeholder">No preview</div>
                        )}
                        <div className="similar-meta">
                          <span>{item.file_name}</span>
                          <span className="muted">{item.score.toFixed(2)}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className="detail-actions dual">
                <button className="secondary" onClick={handleRerun} disabled={rerunLoading}>
                  {rerunLoading ? "Processing..." : "Re-run auto tagging"}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="muted">Select a photo to see details</div>
              <div className="settings-panel">
                <div className="section-head">
                  <h4>Inference</h4>
                  {import.meta.env.DEV && (
                    <button className="ghost small" onClick={handleTestInference} disabled={inferenceBusy}>
                      {inferenceBusy ? "Updating..." : "Test Inference"}
                    </button>
                  )}
                </div>
                <label className="settings-row">
                  <span>Device</span>
                  <select
                    value={inferenceDevice}
                    onChange={(e) => setInferenceDevice(e.target.value)}
                    disabled={inferenceBusy}
                  >
                    {INFERENCE_DEVICE_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="settings-meta">
                  <span>
                    <strong>Active:</strong> {inferenceStatus?.provider || "Unknown"}
                  </span>
                  {inferenceStatus?.runtime_version && (
                    <span className="muted">ORT {inferenceStatus.runtime_version}</span>
                  )}
                </div>
                {inferenceStatus?.models?.length > 0 && (
                  <div className="settings-models">
                    {inferenceStatus.models.map((model) => (
                      <div key={model.label} className="settings-model">
                        <span className="muted">{model.label}</span>
                        <span>{model.provider}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </aside>
      </div>

      <SelectionBar
        count={selection.length}
        onRate={(value) => applyCullChange({ rating: value, label: `Rated ${value ?? "clear"}` })}
        onPick={togglePick}
        onReject={toggleReject}
        onClear={() => applyCullChange({ rating: null, label: "Rating cleared" })}
        onTag={handleAddTag}
      />
      <Toast toast={toast} onUndo={handleUndo} onClose={() => setToast(null)} />
    </div>
  );
}
