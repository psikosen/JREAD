#!/usr/bin/env python3
"""Start JREAD's local backend and optionally open its desktop window."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("JREAD_DATA_DIR", Path.home() / ".local/share/JREAD")).expanduser()
URL = "http://127.0.0.1:8765"


def ready():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_server():
    if not ready():
        python = ROOT / ".venv/bin/python"
        if not python.is_file():
            raise RuntimeError(f"Backend environment missing: {python}; install backend requirements first")
        DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_path = DATA / "server.log"
        log = log_path.open("ab")
        os.chmod(log_path, 0o600)
        subprocess.Popen(
            [str(python), "-m", "jread"], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and not ready():
            time.sleep(0.25)
    token_path = DATA / "token"
    if not ready() or not token_path.is_file():
        raise RuntimeError(f"JREAD backend did not become ready; inspect {DATA / 'server.log'}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("JREAD backend token file is empty")
    return f"{URL}/#token={token}"


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server-only", action="store_true")
    mode.add_argument("--desktop", action="store_true")
    args = parser.parse_args()
    try:
        ui_url = start_server()
        if args.server_only:
            print(ui_url)
            return
        install_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "JREAD"
        installed_bin = install_home / "bin/jread-desktop"
        development_bins = [ROOT / "src-tauri/target/release/jread-desktop", ROOT / "src-tauri/target/debug/jread-desktop"]
        desktop = installed_bin if installed_bin.is_file() else next((p for p in development_bins if p.is_file()), None)
        if not desktop:
            raise RuntimeError("JREAD desktop binary is missing; build src-tauri first")
        subprocess.Popen([str(desktop)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except (OSError, RuntimeError) as error:
        print(f"JREAD: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)
