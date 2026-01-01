import { useEffect, useMemo, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/tauri";
import { open } from "@tauri-apps/api/dialog";
import { listen } from "@tauri-apps/api/event";
import { open as openExternal } from "@tauri-apps/api/shell";
import { dirname } from "@tauri-apps/api/path";

const TAGS = ["street", "landscape", "portrait", "nature"];

function FiltersPanel({ filters, onChange, onApply }) {
  return (
    <div className="panel filters">
      <h3>Filters</h3>
      <label>
        Search
        <input
          type="text"
          value={filters.search || ""}
          onChange={(e) => onChange({ ...filters, search: e.target.value })}
        />
      </label>
      <label>
        Tags
        <div className="tags">
          {TAGS.map((tag) => (
            <label key={tag} className="tag-checkbox">
              <input
                type="checkbox"
                checked={filters.tags?.includes(tag)}
                onChange={(e) => {
                  const next = new Set(filters.tags || []);
                  if (e.target.checked) next.add(tag);
                  else next.delete(tag);
                  onChange({ ...filters, tags: Array.from(next) });
                }}
              />
              {tag}
            </label>
          ))}
        </div>
      </label>
      <label>
        Camera Make
        <input
          type="text"
          value={filters.camera_make || ""}
          onChange={(e) => onChange({ ...filters, camera_make: e.target.value })}
        />
      </label>
      <label>
        Camera Model
        <input
          type="text"
          value={filters.camera_model || ""}
          onChange={(e) => onChange({ ...filters, camera_model: e.target.value })}
        />
      </label>
      <label>
        Lens
        <input
          type="text"
          value={filters.lens || ""}
          onChange={(e) => onChange({ ...filters, lens: e.target.value })}
        />
      </label>
      <div className="range-row">
        <label>
          ISO Min
          <input
            type="number"
            value={filters.iso_min || ""}
            onChange={(e) => onChange({ ...filters, iso_min: e.target.value ? Number(e.target.value) : null })}
          />
        </label>
        <label>
          ISO Max
          <input
            type="number"
            value={filters.iso_max || ""}
            onChange={(e) => onChange({ ...filters, iso_max: e.target.value ? Number(e.target.value) : null })}
          />
        </label>
      </div>
      <div className="range-row">
        <label>
          Aperture Min
          <input
            type="number"
            value={filters.aperture_min || ""}
            onChange={(e) => onChange({ ...filters, aperture_min: e.target.value ? Number(e.target.value) : null })}
          />
        </label>
        <label>
          Aperture Max
          <input
            type="number"
            value={filters.aperture_max || ""}
            onChange={(e) => onChange({ ...filters, aperture_max: e.target.value ? Number(e.target.value) : null })}
          />
        </label>
      </div>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={filters.has_gps || false}
          onChange={(e) => onChange({ ...filters, has_gps: e.target.checked })}
        />
        Has GPS
      </label>
      <label>
        Sort By
        <select
          value={filters.sort_by || "date_taken"}
          onChange={(e) => onChange({ ...filters, sort_by: e.target.value })}
        >
          <option value="date_taken">Date</option>
          <option value="iso">ISO</option>
          <option value="focal_length">Focal Length</option>
          <option value="fnumber">Aperture</option>
          <option value="exposure_time">Shutter</option>
          <option value="file_name">Filename</option>
        </select>
      </label>
      <button onClick={onApply}>Apply</button>
    </div>
  );
}

function resolvePath(path) {
  return path ? convertFileSrc(path) : null;
}

function formatExposureTime(value) {
  if (!value) return "–";
  if (value >= 1) return `${value.toFixed(1)}s`;
  const denom = Math.round(1 / value);
  if (!denom || !Number.isFinite(denom)) return `${value.toFixed(4)}s`;
  return `1/${denom}s`;
}

function GalleryGrid({ photos, onSelect, selectedId }) {
  return (
    <div className="gallery">
      {photos.map((p) => (
        <div
          key={p.photo.id}
          className={`thumb ${selectedId === p.photo.id ? "selected" : ""}`}
          onClick={() => onSelect(p)}
        >
          {p.photo.thumb_path ? (
            <img src={resolvePath(p.photo.thumb_path)} alt={p.photo.file_name} />
          ) : (
            <div className="thumb-placeholder">No preview</div>
          )}
          <div className="thumb-caption">{p.photo.file_name}</div>
        </div>
      ))}
    </div>
  );
}

function TagList({ tags, onRemove }) {
  return (
    <div className="tag-list">
      {tags.map((tag) => (
        <span key={tag.tag} className="tag-pill">
          {tag.tag}
          {tag.confidence != null && (
            <span className="confidence">{(tag.confidence * 100).toFixed(0)}%</span>
          )}
          <span className="source">{tag.source}</span>
          {onRemove && tag.source === "manual" && (
            <button onClick={() => onRemove(tag.tag)} aria-label="remove tag">
              ×
            </button>
          )}
        </span>
      ))}
    </div>
  );
}

function DetailsPanel({ selected, onAddTag, onRemoveTag, onRerun, onShowInFolder }) {
  const [newTag, setNewTag] = useState("");

  if (!selected) {
    return (
      <div className="panel details">
        <h3>Details</h3>
        <p>Select a photo to see details.</p>
      </div>
    );
  }

  const { photo, tags } = selected;
  return (
    <div className="panel details">
      <h3>{photo.file_name}</h3>
      {photo.preview_path && (
        <img className="preview" src={resolvePath(photo.preview_path)} alt={photo.file_name} />
      )}
      <div className="meta">
        <div>
          <strong>Camera:</strong> {photo.make || "Unknown"} {photo.model || ""}
        </div>
        <div>
          <strong>Lens:</strong> {photo.lens || "Unknown"}
        </div>
        <div className="meta-grid">
          <span>ISO: {photo.iso || "–"}</span>
          <span>F/{photo.fnumber || "–"}</span>
          <span>{photo.focal_length ? `${photo.focal_length}mm` : "–"}</span>
          <span>{formatExposureTime(photo.exposure_time)}</span>
        </div>
      </div>
      <h4>Tags</h4>
      <TagList tags={tags} onRemove={onRemoveTag} />
      <div className="tag-input">
        <input
          type="text"
          placeholder="Add tag"
          value={newTag}
          onChange={(e) => setNewTag(e.target.value)}
        />
        <button
          onClick={() => {
            if (newTag.trim()) {
              onAddTag(newTag.trim());
              setNewTag("");
            }
          }}
        >
          Add
        </button>
      </div>
      <div className="detail-actions">
        <button className="secondary" onClick={onRerun}>
          Re-run auto detection
        </button>
        <button className="ghost" onClick={onShowInFolder}>
          Show in folder
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [photos, setPhotos] = useState([]);
  const [selected, setSelected] = useState(null);
  const [filters, setFilters] = useState({ tags: [], sort_by: "date_taken", sort_dir: "DESC" });
  const [progress, setProgress] = useState({ discovered: 0, processed: 0, current_file: "" });

  useEffect(() => {
    const unlisten = listen("import-progress", (event) => {
      setProgress(event.payload);
    });
    refresh();
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const refresh = async () => {
    const result = await invoke("query_photos", { filters: { ...filters, limit: 200 } });
    setPhotos(result);
    if (selected) {
      const updated = result.find((p) => p.photo.id === selected.photo.id);
      setSelected(updated || null);
    }
  };

  const handleImport = async () => {
    const dir = await open({ directory: true, recursive: true });
    if (dir) {
      await invoke("import_folder", { path: dir });
      await refresh();
    }
  };

  const handleAddTag = async (tag) => {
    if (!selected) return;
    await invoke("add_manual_tag", { photo_id: selected.photo.id, tag });
    await refresh();
  };

  const handleRemoveTag = async (tag) => {
    if (!selected) return;
    await invoke("remove_manual_tag", { photo_id: selected.photo.id, tag });
    await refresh();
  };

  const handleRerun = async () => {
    if (!selected) return;
    await invoke("rerun_auto", { photo_id: selected.photo.id });
    await refresh();
  };

  const handleShowInFolder = async () => {
    if (!selected?.photo?.path) return;
    const folder = await dirname(selected.photo.path);
    await openExternal(folder);
  };

  const importProgressText = useMemo(() => {
    if (!progress.discovered) return "Idle";
    return `Processing ${progress.processed}/${progress.discovered}`;
  }, [progress]);

  return (
    <div className="app-shell">
      <header>
        <h1>PhotoTag</h1>
        <div className="actions">
          <button onClick={handleImport}>Import Folder</button>
          <span className="progress">{importProgressText}</span>
        </div>
      </header>
      <div className="content">
        <FiltersPanel filters={filters} onChange={setFilters} onApply={refresh} />
        <GalleryGrid photos={photos} selectedId={selected?.photo.id} onSelect={setSelected} />
        <DetailsPanel
          selected={selected}
          onAddTag={handleAddTag}
          onRemoveTag={handleRemoveTag}
          onRerun={handleRerun}
          onShowInFolder={handleShowInFolder}
        />
      </div>
    </div>
  );
}
