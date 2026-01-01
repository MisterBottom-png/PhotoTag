# PhotoTag - Offline Photo Catalog

PhotoTag is a fully offline, high-performance desktop application for Windows designed to catalog, browse, and auto-tag large photo collections. It uses a Rust backend for performance-critical tasks and a React frontend for a modern, responsive user interface.

## Features

*   **Fully Offline**: No cloud services needed. Your photos and data stay on your machine.
*   **Recursive Folder Import**: Add entire folders of images (JPEG, PNG, TIFF, CR2, NEF, ARW, DNG).
*   **Advanced Metadata Extraction**: Pulls detailed EXIF and camera information using a bundled ExifTool.
*   **Offline AI Auto-Tagging**: Classifies photos into categories like `street`, `landscape`, `portrait`, and `nature` using local ONNX models.
*   **Powerful Filtering**: Search and filter your library by tags, camera/lens model, date, and other EXIF fields.
*   **Fast Browsing**: Generates thumbnails and previews for a smooth gallery experience.
*   **Incremental Imports**: Efficiently scans for new or modified files, skipping those already processed.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Rust**: https://www.rust-lang.org/tools/install
2.  **Node.js**: https://nodejs.org/
3.  **Tauri Prerequisites**: Follow the official Tauri guide for your OS, especially the "Build" section for WebView2. Tauri Guide

## Acquiring Models and Binaries

This application relies on external binaries and machine learning models that must be acquired and placed in the correct directory.

### 1. ExifTool

*   **What it is**: A command-line utility for reading, writing, and editing meta information in a wide variety of files.
*   **How to get it**:
    1.  Download the **Windows Executable** from the ExifTool Website.
    2.  Rename `exiftool(-k).exe` to `exiftool.exe`.
    3.  Create a `bin` directory in `src-tauri` and place `exiftool.exe` inside it. The final path should be `src-tauri/bin/exiftool.exe`.

### 2. ONNX Models

*   **What they are**: Pre-trained machine learning models for scene classification and object detection.
*   **How to get them**:
    *   **Scene Classifier**: Download a CPU-friendly model like MobileNet or EfficientNet from the ONNX Model Zoo.
    *   **Face/Person Detector**: Download a lightweight model like YOLO or a specialized face detector.
*   **Setup**:
    1.  Create a `models` directory inside `src-tauri`.
    2.  Place your downloaded `.onnx` files into this directory.
    3.  Update the `src-tauri/src/config.rs` file with the correct model filenames.

The `build.rs` script will automatically bundle these files into your final application executable.

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

3.  **Run the development server**:
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
    The installer (`.msi`) and executable will be located in `src-tauri/target/release/bundle/`.

## How to Add New Tags

The auto-tagging system is based on mapping ML model outputs to a fixed set of tags. To add a new tag (e.g., `animal`):

1.  **Update Model Mappings**: In `src-tauri/src/tagging.rs`, modify the function that maps scene classification labels to your application's tags. You might need a more advanced classification model if the existing one doesn't recognize the concept.

2.  **Adjust Heuristics**: If the tag relies on object detection (like `portrait`), you may need to add new heuristics in `src-tauri/src/tagging.rs` to interpret the detection results.

3.  **Re-run Tagging**: Use the "Re-run Auto Detection" feature in the UI to apply the new logic to your existing photos.

Manual tags can always be added directly through the UI without any code changes.
