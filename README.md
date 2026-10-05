# JREAD

JREAD is a local Linux development build for Japanese-to-English subtitles. A Tauri desktop window hosts the companion UI; a local Python service manages media, jobs, captions, and models. The Chrome/Brave extension pairs with the companion to apply saved captions to accessible video players and to capture tab audio for delayed live captions.

This is an implementation in progress, not a packaged release. It targets Linux only. It does not bypass DRM or promise support for every site or player. See [development status](docs/development-status.md) for current evidence and limits.

## Run on Linux

For Debian/Ubuntu, install [Tauri 2's Linux prerequisites](https://v2.tauri.app/start/prerequisites/) and the media tools:

```sh
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
  libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev ffmpeg
```

Install Rust stable with Cargo, Python 3.12 or newer with `venv`, and Node.js/npm. This machine was tested with Python 3.13. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
npm ci
npm run check:extension
npm run build:extension
cargo run --manifest-path src-tauri/Cargo.toml --bin jread-desktop
```

The first launch starts the backend on `127.0.0.1:8765` and opens the companion. Choose **small** or **medium** for Japanese speech recognition, then choose an English translation engine: **Whisper translation** is the fresh-install default and comparison baseline, while **Gemma 4 E4B** is a separately selected local translator. Models are not fetched automatically; install the selected model in Setup. Gemma’s pinned 5.15 GB Q4 model and the pinned llama.cpp CPU runtime download on request and run locally. The Gemma path is Linux x86-64 only. A small initial comparison is available; see its results and limits in [development status](docs/development-status.md). **CPU** is the default device; **CUDA** is available for Whisper on a compatible NVIDIA setup, and **Automatic** currently resolves to CPU.

For a user-level Linux install instead of a development run, first install the Python and Node dependencies above, then run:

```sh
bash scripts/install-linux.sh
```

This builds the desktop and native-messaging binaries, installs a launcher and registers the host for Chrome, Brave, and Chromium under your user account. It is not a signed package or a one-click dependency installer. Start JREAD from the application menu after installing.

## Pair the browser extension

For extension pairing, first run the Linux installer above so the browser's native-messaging host is registered. In Brave or Chrome, open the extensions page, enable **Developer mode**, choose **Load unpacked**, and select this repository’s `extension/` directory. `npm run build:extension` creates the extension scripts before loading. Restart the browser if it was open during installation. Then:

1. Open JREAD and use the pairing code shown above Local engine. Choose **Refresh code** if it has expired.
2. Open the JREAD extension popup and enter that code. It expires after five minutes.
3. Use the extension’s explicit controls to select an accessible player, apply saved subtitles, or start live captions.

Pairing lasts up to one day; pair again after a browser restart clears its session storage. The service listens only on loopback; media and inference stay local. Model downloads require a network connection and use pinned faster-whisper revisions or pinned Gemma 4 GGUF files. No cloud inference is used.

For live captions, select **Spoken language** before starting capture. Japanese, Spanish, English, French, German, Italian, Portuguese, Korean, Chinese, Russian, Arabic, and Hindi are selectable; output stays English. The extension remembers your choice and locks it while capturing. Automatic detection is off. Whisper receives the selected language, and Gemma receives its matching translation prompt; Japanese show hints apply only to Japanese. English speech is transcribed directly. These language-routing checks do not establish translation accuracy for each language. After updating an unpacked extension, reload JREAD on the browser's extensions page to load the new controls.

## Use the companion

Import a local video or WAV, MP3, FLAC, OGG, M4A, or AAC file to create a Japanese-to-English job. The companion shows its processing stage and progress, lets you cancel or resume supported jobs, and stores captions in the searchable library. You can also import SRT/WebVTT, edit cue text and timing, undo an edit, play local media with native captions, and export SRT or WebVTT.

The default library and model directory is `~/.local/share/JREAD/` (`library.sqlite3`, `media/`, `models/`, and a local token file). Set `JREAD_DATA_DIR` before launch to use another data directory. Gemma weights are stored under `models/gemma4-e4b/` or `models/gemma4-12b/`; the private llama.cpp runtime is stored under `runtime/llama-cpp/`. **Clear subtitle library** removes episodes and caption files; downloaded models remain until removed separately in Setup.

**Gemma 4 12B** is also selectable under English translation. Its pinned Q4 model requires about 7.0 GB of disk space; [model provenance](docs/gemma-12b-model.json) records its exact revision and checksum. Each job records its translator, so changing the selection does not relabel E4B jobs. The runtime loads one Gemma preset at a time. Both Gemma sizes use CPU; the larger model's accuracy and speed must be measured on the intended dialogue.

Gemma caption options include an explicit **Aikatsu × PriPara names** glossary, **Remember recent live dialogue**, and **Combine overlapping live captions**. The glossary supplies preferred names to Japanese recognition and translation; leave it off for other shows. Live memory carries a short Japanese source history and resets on seeking or a new capture. Overlap correction conservatively updates a repeated unfinished fragment under its original cue ID and marks the correction for review; it adds no model call or buffering wait. These options are individually selectable and are saved with each new job. They are disabled for Whisper's direct-translation mode. Optional hints can also introduce errors; evaluate the intended material.

**Large v3 Turbo** is an optional speech recognizer, separate from the Gemma
translator. Its pinned 1.62 GB model requires Gemma for English translation;
the UI and API prevent pairing it with direct Whisper translation. See
[model provenance](docs/whisper-turbo-model.json). On this machine CUDA runs the
recognizer while Gemma remains on CPU. Lower transcription error alone does not
establish better caption meaning; current evaluation criteria and limits are in
[the accuracy target](docs/accuracy-target.md).

## Checks

```sh
.venv/bin/python -m pytest -q tests/test_core.py tests/test_gemma.py tests/test_caption_hints.py tests/test_live_context.py
npm run check:extension
node extension/tests/run.mjs
cargo test --manifest-path src-tauri/Cargo.toml
```

Additional integration checks use a running companion, `/usr/bin/brave-browser`, and the installed model. Browser checks launch disposable profiles. Run the inference fixture first to create the demo video used by the player checks:

```sh
npm run test:inference
npm run test:ui
npm run test:browser
npm run test:live
JREAD_TEST_LIVE=1 npm run test:browser
```

The real inference test downloads no model and leaves its completed demo in the library. Browser/API tests remove their temporary jobs. Reports and screenshots are saved under `test-results/`. These checks do not establish subtitle quality or release readiness.

The fixture attribution and provenance are in [tests/fixtures/README.md](tests/fixtures/README.md).

The opt-in `PYTHONPATH=. .venv/bin/python tests/compare_caption_combos.py` replays 24 saved/synthetic transcript cases through four E4B glossary/context conditions, then applies both overlap policies to derive eight combinations without duplicating inference. It uses isolated port 18767, records input hashes, source text, translations and processing times, and downloads no models. It also replays the short FLEURS audio with all four recognition-hint conditions; use `--skip-asr` to retain the existing audio report. `--reconcile-only` recomputes caption overlap results without model inference. Reports distinguish transcript replay from actual live delay; the sample is not a human-certified accuracy benchmark.
