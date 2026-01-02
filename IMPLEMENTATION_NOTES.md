PhotoTag Implementation Notes

Overview
- Added a multi-stage, bounded import pipeline (exif, thumbnail, hash, tagging, embedding) with cancel support and real-time progress events.
- Implemented dHash-based near-duplicate detection with adjustable thresholds and a UI panel for reviewing groups.
- Added a lightweight embedding pipeline (color histogram) and a "Find similar" UI action with cosine scoring.
- Improved cull UX: keyboard shortcuts, zoom indicator, stronger pick/reject state, compact header, and smoother image transitions.
- Virtualized the browse grid for large libraries and added drag-and-drop folder import + Explorer reveal/select.

How to use
- Import: Use "Import Folder" or drag a folder onto the app. Cancel from the header if needed.
- Culling shortcuts: Space toggles Fit/1:1, F toggles fullscreen.
- Duplicates: In Browse mode, click "Duplicates" in the header, adjust threshold, and use Pick/Reject actions per group.
- Similarity: Select a photo and click "Find similar" in the right panel to see top matches with scores.

Tradeoffs and notes
- Embeddings use a lightweight color-histogram baseline for offline similarity; an ONNX model can replace this later.
- Duplicate grouping uses a bucketed Hamming search for speed; it is approximate but practical for large sets.
- Existing photos only get embeddings/dHash when imported with this version; re-import or rescan to backfill.
- Explorer context menu integration is not written to the registry yet; a future installer task can add it cleanly.
