use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

use crate::{Profile, RustBackendConfig};

const CARGO_TOML: &str = "\
[package]
name = \"anvyx-gen\"
version = \"0.0.0\"
edition = \"2021\"

[workspace]
";

// TODO: we need to find a way to bump this manually and not depend on me to do it
// Bump this after changing the cache format, artifact paths, build command,
// or toolchain assumptions so old cached binaries are rebuilt
const CACHE_SCHEMA_VERSION: u8 = 1;

pub fn compile_and_run(source: &str, config: &RustBackendConfig) -> Result<String, String> {
    let cache_root = cache_root()?;
    let key = content_key(source, config.profile);
    let artifact = artifact_binary(&cache_root, &key);

    // the artifact already exists, run it immediately without any locking
    if artifact.exists() {
        return run_binary(&artifact);
    }

    // if not exists compiles the artifcat allowing parallel invocations
    let lock_path = cache_root.join("workspace.lock");
    fs::create_dir_all(&cache_root).map_err(|e| format!("Failed to create cache root: {e}"))?;
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| format!("Failed to open workspace lock: {e}"))?;

    let mut lock = fd_lock::RwLock::new(lock_file);
    let _guard = lock
        .write()
        .map_err(|e| format!("Failed to acquire workspace lock: {e}"))?;

    // another process may have built this artifact while we waited
    if artifact.exists() {
        return run_binary(&artifact);
    }

    build_artifact(source, config, &cache_root, &key, &artifact)?;
    run_binary(&artifact)
}

fn build_artifact(
    source: &str,
    config: &RustBackendConfig,
    cache_root: &Path,
    key: &str,
    artifact: &Path,
) -> Result<(), String> {
    let workspace = cache_root.join("workspace");
    let src_dir = workspace.join("src");
    let target_dir = workspace.join("target");
    fs::create_dir_all(&src_dir).map_err(|e| format!("Failed to create workspace: {e}"))?;

    fs::write(workspace.join("Cargo.toml"), CARGO_TOML)
        .map_err(|e| format!("Failed to write Cargo.toml: {e}"))?;
    fs::write(src_dir.join("main.rs"), source)
        .map_err(|e| format!("Failed to write main.rs: {e}"))?;

    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .current_dir(&workspace)
        .env_remove("CARGO_TARGET_DIR")
        .arg("--target-dir")
        .arg(&target_dir);

    if config.profile == Profile::Release {
        cmd.arg("--release");
    }

    let build = cmd
        .output()
        .map_err(|e| format!("Failed to run cargo build: {e}"))?;

    if !build.status.success() {
        let stderr = String::from_utf8_lossy(&build.stderr);
        return Err(format!("Rust backend codegen error:\n{stderr}"));
    }

    // copy the built binary into an immutable artifact entry via temp + rename
    // so readers never observe a partial write
    let built = workspace_binary(&target_dir, config.profile);
    let artifact_dir = cache_root.join("artifacts").join(key);
    fs::create_dir_all(&artifact_dir)
        .map_err(|e| format!("Failed to create artifact directory: {e}"))?;

    let tmp = artifact_dir.join("anvyx-gen.tmp");
    fs::copy(&built, &tmp).map_err(|e| format!("Failed to copy binary: {e}"))?;
    fs::rename(&tmp, artifact).map_err(|e| format!("Failed to publish artifact: {e}"))?;

    Ok(())
}

fn run_binary(binary: &Path) -> Result<String, String> {
    let run = Command::new(binary)
        .output()
        .map_err(|e| format!("Failed to execute generated binary: {e}"))?;

    let stderr = String::from_utf8_lossy(&run.stderr);
    if !run.status.success() {
        return Err(format!("Runtime error:\n{stderr}"));
    }
    eprint!("{stderr}");

    Ok(String::from_utf8_lossy(&run.stdout).into_owned())
}

fn content_key(source: &str, profile: Profile) -> String {
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    CARGO_TOML.hash(&mut h);
    profile.hash(&mut h);
    CACHE_SCHEMA_VERSION.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn artifact_binary(cache_root: &Path, key: &str) -> PathBuf {
    let name = binary_name();
    cache_root.join("artifacts").join(key).join(name)
}

fn workspace_binary(target_dir: &Path, profile: Profile) -> PathBuf {
    let profile_dir = match profile {
        Profile::Debug => "debug",
        Profile::Release => "release",
    };
    target_dir.join(profile_dir).join(binary_name())
}

fn binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "anvyx-gen.exe"
    } else {
        "anvyx-gen"
    }
}

fn cache_root() -> Result<PathBuf, String> {
    let cwd = env::current_dir().map_err(|e| format!("Failed to get current directory: {e}"))?;
    Ok(cwd.join(".anvyx").join("cache").join("rust"))
}
