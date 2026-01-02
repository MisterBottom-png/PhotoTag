# PhotoTag - Offline Photo Catalog

PhotoTag is a fully offline, high-performance desktop application for Windows designed to catalog, browse, and auto-tag large photo collections. It uses a Rust + Tauri backend for performance-critical tasks (EXIF extraction, thumbnailing, ML inference, SQLite) and a React frontend for a modern, responsive user interface.

## Features

* **Fully Offline**: No cloud services needed. Your photos and data stay on your machine.
* **Recursive Folder Import**: Add entire folders of images (JPEG, PNG, TIFF, CR2, NEF, ARW, DNG).
* **Advanced Metadata Extraction**: Pulls detailed EXIF and camera information using a bundled ExifTool.
* **Offline AI Auto-Tagging**: Classifies photos into categories like `street`, `landscape`, `portrait`, and `nature` using local ONNX models. Portrait tagging uses face/person detection heuristics to require a dominant subject.
* **Powerful Filtering**: Search and filter your library by tags, camera/lens model, date, and other EXIF fields.
* **Fast Browsing**: Generates thumbnails and previews for a smooth gallery experience.
* **Incremental Imports**: Efficiently scans for new or modified files, skipping those already processed.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Rust**: https://www.rust-lang.org/tools/install
2.  **Node.js**: https://nodejs.org/
3.  **Tauri Prerequisites**: Follow the official Tauri guide for your OS, especially the "Build" section for WebView2. [Tauri Prerequisites Guide](https://tauri.app/v1/guides/getting-started/prerequisites)

## Acquiring Models and Binaries

This application relies on external binaries and machine learning models that must be acquired and placed in the correct directory.

### 1. ExifTool

* **What it is**: A command-line utility for reading, writing, and editing meta information in a wide variety of files.
* **How to get it**:
  1. Download the **Windows Executable** from the ExifTool Website.
  2. Rename `exiftool(-k).exe` to `exiftool.exe`.
  3. Create a `bin` directory at the project root and place `exiftool.exe` inside it. The final path should be `bin/exiftool.exe`.

### 2. ONNX Models

* **What they are**: Pre-trained machine learning models for scene classification and object detection.
* **How to get them**:
  * **Scene Classifier**: Download a CPU-friendly model like [EfficientNet-Lite4](https://github.com/onnx/models/blob/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx) from the ONNX Model Zoo for scene classification.
  * **Face/Person Detector**: Download a lightweight model like YOLO or a specialized face detector.
* **Setup**:
  1. Create a `models` directory at the project root.
  2. Place your downloaded `.onnx` files into this `models/` directory.
  3. Add a labels sidecar for the scene model (e.g. `scene_classifier.labels.txt`) with one label per line.
  4. Update the `config.rs` file with the correct model filenames or adjust `TaggingConfig` accordingly.

The `build.rs` script is configured to automatically copy the `bin/` and `models/` directories into your final application bundle, ensuring they are available at runtime.

## Development Setup

1.  **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd PhotoTag
    ```

2.  **Install frontend dependencies**:
    ```sh
    npm install
    ```

3. **Run the development server**:
   This command will launch the Tauri application in development mode with hot-reloading for both the frontend and backend.
   ```sh
   npm run tauri dev
   ```

## Production Build

To create a standalone executable for distribution:

1.  **Build the application**:
    ```sh
    npm run tauri build
    ```

2.  **Find the executable**:
    The installer (`.msi`) and executable will be located in `target/release/bundle/`.

## Working Without PR or File-Creation Permissions

If you are working in a restricted environment (for example, you cannot open a pull request or cannot create/delete files directly), you can still share changes safely:

1. **Commit Locally**: Create one or more local commits that contain your changes. Avoid uncommitted work when generating artifacts for review.
2. **Generate a Patch File**: Use `git format-patch -1` (or with the desired commit count) to produce `.patch` files. These capture the exact changes without requiring you to open a PR or modify repository permissions.
3. **Share the Patch**: Send the generated patch files to a collaborator who has repository write access. They can apply them with `git am < patchfile` and open the PR on your behalf.
4. **Archive for Handoff**: If patch files are not feasible, produce a zip/tar archive of the repository including the `.git` directory so commit history is preserved.

These steps keep the workflow auditable while respecting permission constraints.

## How to Add New Tags

The auto-tagging system uses the scene model's labels as tags. To add or rename tags (e.g., `animal`):

1. **Update Labels**: Edit the scene labels sidecar (e.g. `scene_classifier.labels.txt`) so it includes the desired label names.

2. **Adjust Heuristics**: If the tag relies on object detection (like `portrait`), adjust the heuristics in `tagging.rs` to interpret detection results (center bias, dominant face size, focal-length boosts, etc.).

3. **Re-run Tagging**: Use the "Re-run Auto Detection" button in the UI to apply the new logic to your existing photos.

Manual tags can always be added directly through the UI without any code changes.

Manual tags can always be added directly through the UI without any code changes.
