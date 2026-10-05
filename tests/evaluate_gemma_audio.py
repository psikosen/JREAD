"""Isolated direct Japanese-audio Gemma 4 smoke test; inference is opt-in."""
import argparse
import base64
import hashlib
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / ".local/share/JREAD"
RUNTIME = DATA / "runtime/llama-cpp"
MODEL = DATA / "models/gemma4-e4b/gemma-4-E4B_q4_0-it.gguf"
PROJECTOR = Path.home() / ".cache/jread-experiments/gemma4-audio/gemma-4-E4B-it-mmproj.gguf"
MANIFEST = ROOT / "tests/fixtures/accuracy-corpus.json"
RUNTIME_MANIFEST = ROOT / "docs/runtime.json"
REPORT = ROOT / "test-results/gemma4-audio-smoke.json"
PORT = 18768
HOST = "127.0.0.1"
PROMPT = ("Transcribe the Japanese speech, then translate it into English. "
          "Return only two lines: Japanese: <transcript> and English: <translation>.")
MODEL_SHA256 = "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee"
PROJECTOR_SHA256 = "7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578"
opener = build_opener(ProxyHandler({}))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture():
    corpus = json.loads(MANIFEST.read_text())
    case = next(c for c in corpus["cases"] if c["id"] == 1514)
    audio = ROOT / case["audio_path"]
    audio_dir = (ROOT / "test-results/accuracy-audio").resolve()
    if (case["partition"] != "development" or not audio.is_file()
            or audio.resolve().parent != audio_dir):
        raise SystemExit("Expected development case 1514 audio is unavailable.")
    with wave.open(str(audio), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    if duration > 30 or case["duration_seconds"] > 30:
        raise SystemExit("Audio fixture exceeds the 30-second Gemma 4 limit.")
    return case, audio


def preflight():
    case, audio = fixture()
    if sha256(audio) != case["audio_sha256"]:
        raise SystemExit("Case 1514 audio checksum does not match the corpus manifest.")
    assert case["partition"] == "development" and case["id"] == 1514
    assert "Japanese speech" in PROMPT and "English:" in PROMPT
    print(json.dumps({"self_check": "ok", "case": case["id"], "audio_sha256": sha256(audio),
                      "prompt": PROMPT}, ensure_ascii=False))


def request(path, key, payload=None, timeout=2):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = Request(f"http://{HOST}:{PORT}{path}", data=body,
                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.loads(response.read(8 * 1024 * 1024))


def port_available():
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, PORT))
            return True
        except OSError:
            return False


def runtime_info():
    manifest = json.loads(RUNTIME_MANIFEST.read_text())
    runtime = RUNTIME / "llama-server"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "HF_"))}
    env.pop("LD_PRELOAD", None)
    env["LD_LIBRARY_PATH"] = str(RUNTIME)
    result = subprocess.run([str(runtime), "--version"], capture_output=True, text=True,
                            timeout=10, env=env, check=True)
    actual = result.stdout + result.stderr
    if (manifest.get("name") != "llama.cpp" or not manifest.get("build_tag")
            or f"build {manifest['build_tag'].lstrip('b')}" not in actual
            or manifest["commit"][:9] not in actual):
        raise RuntimeError("Pinned llama-server version does not match docs/runtime.json.")
    return {"build_tag": manifest["build_tag"], "commit": manifest["commit"],
            "reported_version": actual.strip(), "encoder": "Gemma 4 audio encoder via llama.cpp mmproj",
            "device": "CPU", "threads": 8}


def serve(audio, case_name, allow_empty=False, on_lifecycle=None):
    if not MODEL.is_file() or sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("Pinned Gemma E4B model is missing or has the wrong checksum.")
    if not PROJECTOR.is_file() or sha256(PROJECTOR) != PROJECTOR_SHA256:
        raise RuntimeError("Pinned Gemma E4B audio projector is missing or has the wrong checksum.")
    runtime = RUNTIME / "llama-server"
    if not runtime.is_file():
        raise RuntimeError("Pinned llama-server runtime is missing.")
    if not port_available():
        raise RuntimeError(f"Experiment port {PORT} is occupied; leaving its owner untouched.")
    runtime_data = runtime_info()

    key = secrets.token_urlsafe(32)
    fd, key_path = tempfile.mkstemp(prefix="jread-gemma-audio-", suffix=".key")
    process = None
    lifecycle = {"owned_pid": None, "process_exit_code": None, "terminated_by_runner": False,
                 "port_released": None, "key_removed": False}
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key + "\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "HF_"))}
        env.pop("LD_PRELOAD", None)
        env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
                   LD_LIBRARY_PATH=str(RUNTIME))
        args = [str(runtime), "--model", str(MODEL), "--mmproj", str(PROJECTOR),
                "--no-mmproj-offload", "--host", HOST, "--port", str(PORT),
                "--ctx-size", "4096", "--threads", "8", "--n-gpu-layers", "0",
                "--parallel", "1", "--jinja", "--reasoning-budget", "0",
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--api-key-file", key_path, "--no-webui", "--no-slots", "--offline", "--log-disable"]
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        lifecycle["owned_pid"] = process.pid
        if on_lifecycle:
            on_lifecycle(lifecycle.copy())
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"isolated llama-server exited with code {process.returncode}")
            try:
                health = request("/health", key)
                if health.get("status") == "ok":
                    break
            except (URLError, TimeoutError, HTTPError):
                pass
            time.sleep(.25)
        else:
            raise TimeoutError("isolated llama-server did not become ready in 120 seconds")

        audio_bytes = audio.read_bytes()
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"), "format": "wav"}},
        ]}], "temperature": 0, "max_tokens": 512}
        started = time.monotonic()
        result = request("/v1/chat/completions", key, payload, timeout=600)
        wall = time.monotonic() - started
        choice = result["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        fields = {line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip()
                  for line in text.splitlines() if ":" in line}
        protocol_errors = []
        if choice.get("finish_reason") != "stop":
            protocol_errors.append(f"unexpected finish_reason: {choice.get('finish_reason')}")
        if "japanese" not in fields:
            protocol_errors.append("missing Japanese transcript field")
        if "english" not in fields:
            protocol_errors.append("missing English translation field")
        if not allow_empty and not fields.get("japanese", "").strip():
            protocol_errors.append("empty Japanese transcript")
        if not allow_empty and not fields.get("english", "").strip():
            protocol_errors.append("empty English translation")
        reasoning = message.get("reasoning_content")
        row = {"case": case_name, "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
               "prompt": PROMPT, "response": text,
               "japanese_transcript": fields.get("japanese"), "english_translation": fields.get("english"),
               "finish_reason": choice.get("finish_reason"), "reasoning_content": reasoning,
               "protocol_errors": protocol_errors, "status": "failed" if protocol_errors else "complete",
               "wall_seconds": round(wall, 3),
               "timings": result.get("timings"),
               "weights": {"model_sha256": MODEL_SHA256, "mmproj_sha256": PROJECTOR_SHA256},
               "runtime": runtime_data,
               "scope": "One development fixture; references were not sent to the model."}
        return row
    finally:
        if process is not None and process.poll() is None:
            lifecycle["terminated_by_runner"] = True
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if process is not None:
            lifecycle["process_exit_code"] = process.poll()
        lifecycle["port_released"] = port_available()
        Path(key_path).unlink(missing_ok=True)
        lifecycle["key_removed"] = not Path(key_path).exists()
        if on_lifecycle:
            on_lifecycle(lifecycle.copy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="check development fixture and prompt without inference")
    parser.add_argument("--include-silence", action="store_true", help="also submit a generated silence WAV control")
    args = parser.parse_args()
    preflight()
    if not args.self_check:
        case, audio = fixture()
        try:
            fd = os.open(REPORT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise SystemExit(f"Refusing to overwrite existing report: {REPORT}") from None
        report = {"status": "running", "cases": []}

        def save():
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
            os.fsync(fd)

        def run_case(case_name, case_audio, allow_empty=False):
            row = {"case": case_name, "status": "running", "lifecycle": {
                "owned_pid": None, "process_exit_code": None, "terminated_by_runner": False,
                "port_released": None, "key_removed": None}}
            report["cases"].append(row)
            save()

            def lifecycle_update(data):
                row["lifecycle"].update(data)
                save()

            try:
                row.update(serve(case_audio, case_name, allow_empty, lifecycle_update))
                if row["protocol_errors"]:
                    report["status"] = "failed"
            except BaseException as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                report["status"] = "failed"
                save()
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
            save()

        try:
            run_case(f"manifest-{case['id']}", audio)
            if args.include_silence:
                with tempfile.NamedTemporaryFile(suffix=".wav") as silence:
                    with wave.open(silence.name, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(b"\0\0" * 16000)
                    run_case("generated-silence-control", Path(silence.name), allow_empty=True)
            if report["status"] == "running":
                report["status"] = "complete"
            save()
        finally:
            os.close(fd)
        print(json.dumps(report, ensure_ascii=False))
