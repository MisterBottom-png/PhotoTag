import { useEffect, useMemo, useRef, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/tauri";
import { open } from "@tauri-apps/api/dialog";
import { listen } from "@tauri-apps/api/event";
import { open as openExternal } from "@tauri-apps/api/shell";
import { dirname } from "@tauri-apps/api/path";

const SMART_VIEWS = [
  { key: "UNSORTED", label: "Unsorted", helper: "Unrated, unpicked, unrejected" },
  { key: "PICKS", label: "Picks", helper: "Picked and not rejected" },
  { key: "REJECTS", label: "Rejects", helper: "Marked as rejected" },
  { key: "LAST_IMPORT", label: "Last import", helper: "Most recent batch" },
  { key: "ALL", label: "All photos", helper: "Everything in the library" },
];

const SORT_OPTIONS = [
  { value: "date_taken", label: "Date (newest)" },
  { value: "last_modified", label: "Last modified" },
  { value: "rating", label: "Rating" },
  { value: "picked", label: "Picked" },
  { value: "rejected", label: "Rejected" },
  { value: "file_name", label: "Filename" },
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
  if (value >= 1) return `${value.toFixed(1)}s`;
  const denom = Math.round(1 / value);
  if (!denom || !Number.isFinite(denom)) return `${value.toFixed(4)}s`;
  return `1/${denom}s`;
}

function StarIcon({ filled }) {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill={filled ? "url(#starGradient)" : "none"}
      stroke={filled ? "none" : "#6b7a94"}
    >
      <defs>
        <linearGradient id="starGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="100%" stopColor="#22d3ee" />
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
      {showClear && (
        <button className="clear" onClick={() => onChange(null)}>
          Clear
        </button>
      )}
    </div>
  );
}

function TagList({ tags, onRemove }) {
  const sorted = tags.slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  return (
    <div className="tag-list">
      {sorted.map((tag) => (
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
    </div>
  );
}

function ThumbCard({ photo, selected, onSelect, onDoubleClick, thumbSize }) {
  return (
    <div
      className={`thumb ${selected ? "selected" : ""}`}
      onClick={onSelect}
      onDoubleClick={onDoubleClick}
      style={{ minHeight: thumbSize * 0.75 + 36 }}
    >
      {photo.thumb_path ? (
        <img src={resolvePath(photo.thumb_path)} alt={photo.file_name} style={{ height: thumbSize * 0.75 }} />
      ) : (
        <div className="thumb-placeholder">No preview</div>
      )}
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
  const [smartView, setSmartView] = useState("UNSORTED");
  const [browseSmartView, setBrowseSmartView] = usePersistentState("pt-browse-smart-view", "ALL");
  const [filters, setFilters] = useState(DEFAULT_FILTERS);
  const [thumbSize, setThumbSize] = usePersistentState("pt-thumb-size", 190);
  const [autoAdvance, setAutoAdvance] = usePersistentState("pt-auto-advance", true);
  const [photos, setPhotos] = useState([]);
  const [selection, setSelection] = useState([]);
  const [cursorIndex, setCursorIndex] = useState(0);
  const [counts, setCounts] = useState({ unsorted: 0, picks: 0, rejects: 0, last_import: 0, all: 0 });
  const [progress, setProgress] = useState({ discovered: 0, processed: 0, current_file: "" });
  const [errorMessage, setErrorMessage] = useState("");
  const [rerunLoading, setRerunLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [toast, setToast] = useState(null);
  const lastActionRef = useRef(null);
  const searchRef = useRef(null);
  const anchorRef = useRef(null);
  const toastTimerRef = useRef(null);

  const activePhoto = useMemo(() => {
    if (!selection.length) return photos[0] || null;
    return photos.find((p) => p.photo.id === selection[0]) || photos[0] || null;
  }, [photos, selection]);

  useEffect(() => {
    const unlisten = listen("import-progress", (event) => {
      setProgress(event.payload);
    });
    refreshPhotos({ resetCursor: true });
    refreshCounts();
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    if (mode === "CULL") {
      setSmartView("UNSORTED");
    } else {
      setSmartView(browseSmartView || "ALL");
    }
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
    refreshPhotos({ resetCursor: true });
    refreshCounts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [smartView, filters.sort_by, filters.search, mode]);

  useEffect(() => {
    const handler = (e) => {
      const target = e.target;
      const isInput = target instanceof HTMLElement && ["INPUT", "TEXTAREA"].includes(target.tagName);
      if (isInput && e.key !== "Escape") return;
      if (!photos.length) return;
      switch (e.key) {
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
        default:
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [photos.length, cursorIndex, selection, smartView]);

  const refreshCounts = async () => {
    try {
      const data = await invoke("get_smart_views_counts");
      setCounts(data);
    } catch (err) {
      setErrorMessage(`Failed to load counts: ${err}`);
    }
  };

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
      await refreshCounts();
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
    try {
      const dir = await open({ directory: true, recursive: true });
      if (dir) {
        setImporting(true);
        await invoke("import_folder", { path: dir });
        setSmartView("UNSORTED");
        await refreshPhotos({ resetCursor: true });
        await refreshCounts();
      }
    } catch (err) {
      setErrorMessage(`Import failed: ${err}`);
    } finally {
      setImporting(false);
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
      const folder = await dirname(activePhoto.photo.path);
      const normalized = folder.replace(/\//g, "\\");
      await openExternal(normalized);
    } catch (err) {
      setErrorMessage(`Show in folder failed: ${err}`);
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
      await refreshCounts();
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
    return `Processing ${progress.processed}/${progress.discovered}`;
  }, [progress, importing]);

  return (
    <div className="app-shell">
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
          <span className="progress">{importProgressText}</span>
        </div>
      </header>

      {rerunLoading && <div className="banner info-banner">Re-running auto detection...</div>}
      {errorMessage && <div className="banner error-banner">{errorMessage}</div>}

      <div className="content">
        <aside className="panel sidebar">
          <div className="section">
            <div className="section-head">
              <h3>Smart views</h3>
              <button className="ghost small" onClick={refreshCounts}>
                Refresh
              </button>
            </div>
            <div className="smart-list">
              {SMART_VIEWS.filter((v) => mode === "CULL" ? v.key === "UNSORTED" : true).map((view) => {
                const countKey = view.key.toLowerCase();
                const countValue = counts[countKey] ?? (view.key === "ALL" ? counts.all : 0);
                const disabled = mode === "CULL";
                return (
                  <button
                    key={view.key}
                    className={`smart-item ${smartView === view.key ? "active" : ""} ${disabled ? "disabled" : ""}`}
                    onClick={() => {
                      if (disabled) return;
                      setSmartView(view.key);
                      setBrowseSmartView(view.key);
                    }}
                    disabled={disabled}
                  >
                    <div>
                      <div className="label">{view.label}</div>
                      <div className="helper">
                        {disabled ? "Other smart views are available in Browse." : view.helper}
                      </div>
                    </div>
                    <span className="count">{countValue ?? 0}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="section">
            <h3>Filters</h3>
            <label className="stacked">
              Sort by
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
            <label className="stacked slider">
              Thumbnail size
              <input
                type="range"
                min="120"
                max="320"
                value={thumbSize}
                onChange={(e) => setThumbSize(Number(e.target.value))}
              />
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={autoAdvance}
                onChange={(e) => setAutoAdvance(e.target.checked)}
              />
              Auto-advance after action
            </label>
          </div>
        </aside>

        <main className={mode === "CULL" ? "cull" : "browse"}>
          {photos.length === 0 ? (
            <EmptyState
              onImport={handleImport}
              onClearFilters={() => {
                setFilters(DEFAULT_FILTERS);
                if (mode === "CULL") {
                  setSmartView("UNSORTED");
                } else {
                  setSmartView("ALL");
                  setBrowseSmartView("ALL");
                }
              }}
            />
          ) : mode === "CULL" ? (
            <div className="cull-view">
              <div className="cull-preview">
                {activePhoto?.photo.preview_path ? (
                  <img
                    src={resolvePath(activePhoto.photo.preview_path)}
                    alt={activePhoto.photo.file_name}
                    className="preview-large"
                  />
                ) : (
                  <div className="thumb-placeholder large">No preview</div>
                )}
                <div className="cull-controls">
                  <RatingStars
                    value={activePhoto?.photo.rating || 0}
                    onChange={(v) => applyCullChange({ rating: v, label: `Rated ${v ?? "clear"}` })}
                  />
                  <div className="cull-buttons">
                    <button className={activePhoto?.photo.picked ? "active" : ""} onClick={togglePick}>
                      Pick (P)
                    </button>
                    <button className={activePhoto?.photo.rejected ? "active reject" : ""} onClick={toggleReject}>
                      Reject (X)
                    </button>
                  </div>
                  <div className="cull-nav">
                    <button className="ghost" onClick={() => moveCursor(-1)}>
                      Prev
                    </button>
                    <button className="ghost" onClick={() => moveCursor(1)}>
                      Next
                    </button>
                  </div>
                </div>
              </div>
              <div className="filmstrip" style={{ ["--thumb-size"]: `${thumbSize}px` }}>
                {photos.map((p, idx) => (
                  <ThumbCard
                    key={p.photo.id}
                    photo={p.photo}
                    selected={selection.includes(p.photo.id)}
                    onSelect={(e) => onSelectPhoto(p.photo.id, idx, e)}
                    onDoubleClick={() => setSelection([p.photo.id])}
                    thumbSize={thumbSize}
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="gallery" style={{ ["--thumb-size"]: `${thumbSize}px` }}>
              {photos.map((p, idx) => (
                <ThumbCard
                  key={p.photo.id}
                  photo={p.photo}
                  selected={selection.includes(p.photo.id)}
                  onSelect={(e) => onSelectPhoto(p.photo.id, idx, e)}
                  onDoubleClick={() => setSelection([p.photo.id])}
                  thumbSize={thumbSize}
                />
              ))}
            </div>
          )}
        </main>

        <aside className="panel details">
          {activePhoto ? (
            <>
              <div className="detail-head">
                <div>
                  <h3 className="file-title">{activePhoto.photo.file_name}</h3>
                  <div className="muted truncate">{activePhoto.photo.path}</div>
                </div>
                <div className="detail-actions">
                  <button className="ghost" onClick={handleShowInFolder}>
                    Show in folder
                  </button>
                </div>
              </div>
              <div className="action-block">
                <RatingStars
                  value={activePhoto.photo.rating || 0}
                  onChange={(v) => applyCullChange({ rating: v, label: `Rated ${v ?? "clear"}` })}
                />
                <div className="pill-row">
                  <button className={activePhoto.photo.picked ? "active" : ""} onClick={togglePick}>
                    Pick
                  </button>
                  <button className={activePhoto.photo.rejected ? "active reject" : ""} onClick={toggleReject}>
                    Reject
                  </button>
                </div>
              </div>
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
                <TagList tags={activePhoto.tags} onRemove={handleRemoveTag} />
              </div>
              <div className="detail-actions dual">
                <button className="secondary" onClick={handleRerun} disabled={rerunLoading}>
                  {rerunLoading ? "Processing..." : "Re-run auto tagging"}
                </button>
                <button className="ghost" onClick={() => applyCullChange({ rating: null, label: "Rating cleared" })}>
                  Clear rating
                </button>
              </div>
            </>
          ) : (
            <div className="muted">Select a photo to see details</div>
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
