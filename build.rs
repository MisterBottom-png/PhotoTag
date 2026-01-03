use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/onnxruntime/win-x64-directml");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        if let Err(err) = copy_directml_dlls() {
            println!("cargo:warning=Failed to copy DirectML DLLs: {err}");
        }
    }

    tauri_build::build()
}

fn copy_directml_dlls() -> std::io::Result<()> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let target_dir = out_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "missing target dir"))?;
    let src_dir = manifest_dir
        .join("vendor")
        .join("onnxruntime")
        .join("win-x64-directml");
    if !src_dir.exists() {
        return Ok(());
    }
    let dest_dir = target_dir.join("onnxruntime");
    fs::create_dir_all(&dest_dir)?;
    for entry in fs::read_dir(&src_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()).unwrap_or("") != "dll" {
            continue;
        }
        let file_name = entry.file_name();
        fs::copy(&path, dest_dir.join(file_name))?;
    }
    Ok(())
}
