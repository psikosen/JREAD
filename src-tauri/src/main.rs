use std::{env, path::PathBuf, process::Command};
use tauri::{WebviewUrl, WebviewWindowBuilder};

fn app_dir() -> PathBuf {
    if let Some(path) = env::var_os("JREAD_APP_DIR") {
        return path.into();
    }
    let exe = env::current_exe().expect("locate executable");
    if let Some(root) = exe.parent().and_then(|p| p.parent()) {
        let installed = root.join("app");
        if installed.join("scripts/launch.py").is_file() {
            return installed;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn main() {
    let root = app_dir();
    let output = Command::new("python3")
        .arg(root.join("scripts/launch.py"))
        .arg("--server-only")
        .output()
        .expect("start JREAD backend");
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
    let url = String::from_utf8(output.stdout)
        .expect("launcher output is UTF-8")
        .trim()
        .parse::<url::Url>()
        .expect("launcher returned a URL");
    assert_eq!(url.origin().ascii_serialization(), "http://127.0.0.1:8765");

    tauri::Builder::default()
        .setup(move |app| {
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.clone()))
                .title("JREAD")
                .on_navigation(|url| url.origin().ascii_serialization() == "http://127.0.0.1:8765")
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("run JREAD desktop");
}
