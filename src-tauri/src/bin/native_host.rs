use serde_json::{json, Value};
use std::{
    env, fs,
    io::{self, Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
};

fn parse_request(bytes: &[u8]) -> Result<Value, String> {
    if bytes.len() < 4 {
        return Err("incomplete native message length".into());
    }
    let len = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    if len == 0 || len > 4096 || bytes.len() != len + 4 {
        return Err("invalid native message size".into());
    }
    serde_json::from_slice(&bytes[4..]).map_err(|e| e.to_string())
}

fn app_dir() -> PathBuf {
    let exe = env::current_exe().expect("locate native host");
    let root = exe
        .parent()
        .and_then(|p| p.parent())
        .expect("native host must be in <root>/bin");
    let installed = root.join("app");
    if installed.join("scripts/launch.py").is_file() {
        installed
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf()
    }
}

fn respond(value: &Value) -> io::Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let mut out = io::stdout().lock();
    out.write_all(&(bytes.len() as u32).to_le_bytes())?;
    out.write_all(&bytes)?;
    out.flush()
}

fn run() -> Result<(), String> {
    let origin = env::args().nth(1).ok_or("missing browser origin")?;
    let id = fs::read_to_string(app_dir().join("scripts/extension-id.txt"))
        .map_err(|e| format!("extension allowlist unavailable: {e}"))?;
    let id = id.trim();
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        || origin != format!("chrome-extension://{id}/")
    {
        return Err("browser origin is not allowed".into());
    }

    let mut stdin = io::stdin().lock();
    let mut len = [0; 4];
    stdin.read_exact(&mut len).map_err(|e| e.to_string())?;
    let len = u32::from_le_bytes(len) as usize;
    if len == 0 || len > 4096 {
        return Err("invalid native message size".into());
    }
    let mut bytes = vec![0; len + 4];
    bytes[..4].copy_from_slice(&(len as u32).to_le_bytes());
    stdin
        .read_exact(&mut bytes[4..])
        .map_err(|e| e.to_string())?;
    let request = parse_request(&bytes)?;
    if request != json!({"action":"start"}) {
        return Err("unsupported request".into());
    }

    let root = app_dir();
    let output = Command::new("python3")
        .arg(root.join("scripts/launch.py"))
        .arg("--desktop")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("could not open JREAD: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    respond(&json!({"base_url":"http://127.0.0.1:8765","version":1})).map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        let _ = respond(&json!({"error":error}));
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_frame_size_is_little_endian() {
        let frame = serde_json::to_vec(&serde_json::json!({"action":"start"})).unwrap();
        let mut message = (frame.len() as u32).to_le_bytes().to_vec();
        message.extend(frame);
        assert_eq!(
            super::parse_request(&message).unwrap(),
            serde_json::json!({"action":"start"})
        );
        assert!(super::parse_request(&[1, 0, 0, 0, b'!']).is_err());
    }
}
