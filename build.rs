fn main() {
    // This tells Cargo to re-run the build script if `build.rs` changes.
    println!("cargo:rerun-if-changed=build.rs");
    tauri_build::build()
}
