# GPU Inference (DirectML) - Windows Dev Guide

This app uses ONNX Runtime and can run inference on Windows GPU via the DirectML Execution Provider.
GPU is optional and will fall back to CPU when DirectML is unavailable.

## Requirements
- Windows 10 1903+ or Windows 11
- Updated GPU drivers (DirectX 12 / DirectML capable)
- A DirectML-capable ONNX Runtime build available at runtime

## Device Selection
In the UI, open the **Inference** section and choose:
- **Auto (GPU if available)**: tries DirectML, silently falls back to CPU
- **GPU (DirectML)**: tries DirectML, falls back to CPU and shows a warning
- **CPU**: forces CPU execution

The active provider is displayed in the UI and logged at startup.

## Ensuring DirectML Loads in Dev
The default `onnxruntime-sys` build downloads the standard ONNX Runtime package.
If DirectML is not available, use a DirectML-enabled package and point the build to it:

1) Download or install the DirectML build of ONNX Runtime
   - Example: `onnxruntime-directml` NuGet package
2) Set environment variables before building:

```powershell
$env:ORT_STRATEGY="system"
$env:ORT_LIB_LOCATION="C:\\path\\to\\onnxruntime-directml"
```

The folder should contain `onnxruntime.dll` and related libraries.
If the DirectML entry point is missing, the app logs:
`DirectML provider unavailable ...` and falls back to CPU.

## Troubleshooting
- **No GPU provider in UI**: Check logs for `DirectML provider unavailable`.
- **Missing DLL**: Ensure `onnxruntime.dll` is on PATH or in `ORT_LIB_LOCATION`.
- **DirectML symbol not found**: Use the DirectML build of ONNX Runtime.
- **Fallback to CPU**: This is expected if DirectML cannot load or is unsupported.

## Verifying GPU is Active
- In dev mode, use the **Test Inference** button and check logs.
- Logs include provider selection and per-image timings.

## Notes
- Sessions are cached per model and reused.
- Inference runs off the UI thread.
- DirectML session options use sequential execution and disable memory patterns.

## Other GPU Work
- Embedding histogram computation can run on the GPU when available (Windows only).
- Thumbnail/preview resizing uses the GPU when available (Windows only), with CPU fallback.
