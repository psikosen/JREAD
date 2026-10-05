#!/usr/bin/env python3
"""Pinned Kotoba Whisper v2.2 config for the shared causal ASR/replay runner."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = Path.home() / '.cache/jread-experiments/kotoba-whisper-v2.2-faster'
MODEL = Path.home() / '.cache/jread-experiments/kotoba-whisper-v2.2-faster-align-fallback'
REVISION = '77dacdfbfa2f022e53974e8f463cf24051d0a6df'
MODEL_BIN_SIZE = 3_025_831_547
MODEL_BIN_SHA256 = '16b050f845c0465a62ed1d2c72d7b517cc71186cd11edb05f6a46e108ddc6c97'
ORIGINAL_CONFIG_SHA256 = 'f277d9390e65adc1630b349151eca38aaf7420c41c814271928854edcfd6ffbe'
DERIVED_CONFIG_SHA256 = '90c55f775cc4e0bb17293d0bf12f96557a486f20dea886fabd8e6075a3588b21'
DERIVED_CONFIG_SIZE = 2690
SIDE_FILES = {
    'config.json': DERIVED_CONFIG_SIZE,
    'tokenizer.json': 3_931_381,
    'vocabulary.json': 1_119_981,
    'preprocessor_config.json': 340,
}
ASR_OUTPUT = ROOT / 'test-results/accuracy-streaming-kotoba-speech-development.json'
REPLAY_OUTPUT = ROOT / 'test-results/accuracy-streaming-kotoba-boundary-replay.json'
HELPER_PATH = ROOT / 'tests/evaluate_streaming_turbo.py'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helper():
    spec = importlib.util.spec_from_file_location('jread_streaming_runner_config', HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load shared streaming runner: {HELPER_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_config() -> dict:
    return {
        'model_name': 'RoachLin/kotoba-whisper-v2.2-faster (derived decoder-alignment-head fallback config)',
        'model_directory': str(MODEL),
        'revision': REVISION,
        'model_bin_sha256': MODEL_BIN_SHA256,
        'model_config_sha256': DERIVED_CONFIG_SHA256,
        'model_config_provenance': model_config_provenance(),
        'asr_output': str(ASR_OUTPUT),
        'replay_output': str(REPLAY_OUTPUT),
        'wrapper_path': str(Path(__file__).resolve()),
        'wrapper_sha256': sha256(Path(__file__).resolve()),
        'required_files': SIDE_FILES,
}


def model_config_provenance() -> dict:
    provenance_path = MODEL / 'derived_provenance.json'
    record = json.loads(provenance_path.read_text())
    return {
        'path': str(provenance_path),
        'sha256': sha256(provenance_path),
        'original_config_sha256': record['original_config_sha256'],
        'derived_config_sha256': record['derived_config_sha256'],
        'source_provenance_sha256': record['source_provenance_sha256'],
        'conversion': record['conversion'],
        'upstream': record['upstream'],
        'repair_rationale': record['repair_rationale'],
        'quality_note': record['quality_note'],
    }


def validate_model_files() -> None:
    marker = MODEL / '.verified'
    if not marker.is_file() or marker.read_text().strip() != REVISION:
        raise ValueError('Kotoba model marker does not match the pinned Hub revision.')
    original_config = SOURCE_MODEL / 'config.json'
    if sha256(original_config) != ORIGINAL_CONFIG_SHA256:
        raise ValueError('Pinned original Kotoba config differs from the verified source hash.')
    derived_config = MODEL / 'config.json'
    if derived_config.stat().st_size != DERIVED_CONFIG_SIZE or sha256(derived_config) != DERIVED_CONFIG_SHA256:
        raise ValueError('Derived Kotoba config differs from its frozen size or SHA256.')
    config = json.loads(derived_config.read_text())
    if config.get('alignment_heads') != [[1, h] for h in range(20)]:
        raise ValueError('Derived alignment heads differ from the architecture-based fallback.')
    model_bin = MODEL / 'model.bin'
    if model_bin.stat().st_size != MODEL_BIN_SIZE or sha256(model_bin) != MODEL_BIN_SHA256:
        raise ValueError('Kotoba model.bin size or SHA256 differs from pinned provenance.')
    for name, size in SIDE_FILES.items():
        if (MODEL / name).stat().st_size != size:
            raise ValueError(f'Kotoba model side file has unexpected size: {name}')


def self_test(helper) -> None:
    config = model_config()
    assert config['revision'] == REVISION
    assert 'derived decoder-alignment-head fallback config' in config['model_name']
    assert config['asr_output'].endswith('accuracy-streaming-kotoba-speech-development.json')
    assert config['replay_output'].endswith('accuracy-streaming-kotoba-boundary-replay.json')
    assert config['required_files'] == SIDE_FILES
    helper.configure(config)
    helper.self_test(check_model_files=False)
    manifest = json.loads((ROOT / 'tests/fixtures/accuracy-corpus.json').read_text())
    development = [row for row in manifest['cases'] if row.get('partition') == 'development']
    assert len(development) == 20 and len({row['id'] for row in development}) == 20
    assert helper.POLICY == 'sentence_terminal_eof'
    if ASR_OUTPUT.exists() or REPLAY_OUTPUT.exists():
        raise RuntimeError('Refusing to overwrite existing Kotoba artifacts.')


def main() -> int:
    helper = load_helper()
    self_test(helper)
    if '--self-test' in sys.argv:
        print('Kotoba config, frozen corpus shape, shared capture helpers, and corrected replay self-tests passed.')
        return 0
    validate_model_files()
    result = helper.run_configured(model_config())
    print(json.dumps({
        'status': result.get('status'),
        'completed_cases': result.get('completed_cases'),
        'source_diagnostic_cer': result.get('aggregate', {}).get('streaming_diagnostic_cer'),
        'report': str(ASR_OUTPUT.relative_to(ROOT)),
        'replay_report': str(REPLAY_OUTPUT.relative_to(ROOT)) if REPLAY_OUTPUT.exists() else None,
        'owned_gpu_process_released': result.get('owned_gpu_process_released'),
    }, ensure_ascii=False))
    return 0 if result.get('status') == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
