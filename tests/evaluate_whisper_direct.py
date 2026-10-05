"""Bounded direct Whisper Japanese-to-English development evaluation."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'tests/fixtures/accuracy-corpus.json'
OUTPUT = ROOT / 'test-results/accuracy-whisper-direct-development.json'
WARMUP = ROOT / 'tests/fixtures/japanese-speech.wav'
REVISION = '08e178d48790749d25932bbc082711ddcfdfbc4f'
FLAGS = dict(language='ja', task='translate', vad_filter=True, beam_size=5,
             condition_on_previous_text=False)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(report):
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUTPUT)


def dev_cases(manifest):
    cases = [case for case in manifest['cases'] if case.get('partition') == 'development']
    if len(cases) != 20 or len({case['id'] for case in cases}) != 20:
        raise ValueError('Expected exactly 20 distinct development rows.')
    return cases


def gpu_snapshot():
    from asr_cuda_pilot import gpu_snapshot as snapshot
    return snapshot()


def transcribe(model, samples):
    started = time.perf_counter()
    segments, info = model.transcribe(samples, **FLAGS)
    duration = len(samples) / 16000
    rows = []
    for segment in segments:
        start, end = max(0.0, float(segment.start)), min(duration, float(segment.end))
        text = segment.text.strip()
        if start < end and text:
            rows.append({'start': start, 'end': end, 'text': text})
    return {'segments': rows, 'transcription_seconds': time.perf_counter() - started,
            'detected_language': getattr(info, 'language', None)}


def worker():
    import numpy as np
    from faster_whisper import WhisperModel
    from jread import engine

    report = json.loads(OUTPUT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if digest(MANIFEST) != report['corpus']['manifest_sha256']:
        raise ValueError('Corpus manifest changed after preflight.')
    cases = dev_cases(manifest)
    model = None
    try:
        marker = engine.store.MODELS / 'medium' / '.verified'
        if not marker.exists() or marker.read_text().strip() != REVISION:
            raise ValueError('Existing Whisper medium revision marker is not the pinned revision.')
        load_started = time.perf_counter()
        model = WhisperModel(str(marker.parent), device='cuda', compute_type='int8_float16',
                             cpu_threads=4, num_workers=1, local_files_only=True)
        report['model']['load_seconds'] = time.perf_counter() - load_started
        write(report)

        warm_samples = engine.read_audio(WARMUP)
        warm = transcribe(model, warm_samples)
        report['warmup'] = {
            'excluded': True, 'path': str(WARMUP.relative_to(ROOT)),
            'audio_sha256': digest(WARMUP), 'duration_seconds': len(warm_samples) / 16000,
            'japanese_source': 'こんにちは。', **warm,
        }
        report['status'] = 'running'
        write(report)

        for case in cases:
            audio_path = (ROOT / case['audio_path']).resolve()
            if not audio_path.is_relative_to(ROOT / 'test-results/accuracy-audio'):
                raise ValueError(f"Audio path escaped development audio directory for {case['id']}.")
            if digest(audio_path) != case['audio_sha256']:
                raise ValueError(f"Audio hash mismatch for development case {case['id']}.")
            samples = engine.read_audio(audio_path)
            duration = len(samples) / 16000
            if abs(duration - case['duration_seconds']) > 1 / 16000:
                raise ValueError(f"Audio duration mismatch for development case {case['id']}.")
            result = transcribe(model, samples)
            report['cases'].append({
                'id': case['id'], 'partition': 'development',
                'source_provenance': {
                    'dataset': manifest['dataset'], 'revision': manifest['revision'],
                    'license': manifest['license'], 'attribution_url': manifest['attribution_url'],
                    'row_index': case['row_index'], 'source_filename': case['source_filename'],
                    'audio_path': case['audio_path'], 'audio_sha256': case['audio_sha256'],
                    'download_sha256': case['download_sha256'],
                    'duration_seconds': case['duration_seconds'],
                    'japanese_reference': case['japanese_reference'],
                    'english_reference': case['english_reference'],
                },
                'input': {'kind': 'complete normalized PCM recording', 'start': 0.0,
                          'end': duration, 'duration_seconds': duration},
                **result,
            })
            report['completed_cases'] = len(report['cases'])
            write(report)
            print(json.dumps({'completed_cases': report['completed_cases'],
                              'development_id': case['id']}), flush=True)

        silence = transcribe(model, np.zeros(8 * 16000, dtype=np.float32))
        report['zero_pcm_control'] = {
            'duration_seconds': 8, 'expected_empty': True,
            'passed': not silence['segments'], **silence,
        }
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if model is not None:
            model.model.unload_model()
            report['owned_model_unloaded'] = True
        model = None
        engine._model = engine._model_key = None
        gc.collect()
        write(report)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    if OUTPUT.exists():
        raise SystemExit(f'Preserving existing artifact: {OUTPUT}')
    manifest = json.loads(MANIFEST.read_text())
    cases = dev_cases(manifest)
    from asr_cuda_pilot import gpu_snapshot
    report = {
        'schema_version': 1, 'status': 'starting',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'review_scope': '20 development audio clips only; heldout rows are not evaluated.',
        'corpus': {key: value for key, value in manifest.items() if key != 'cases'} |
                  {'manifest': str(MANIFEST.relative_to(ROOT)),
                   'manifest_sha256': digest(MANIFEST)},
        'partition': 'development', 'development_ids': [case['id'] for case in cases],
        'completed_cases': 0, 'cases': [],
        'model': {'name': 'faster-whisper medium', 'revision': REVISION,
                  'device': 'cuda', 'compute_type': 'int8_float16',
                  'cpu_threads': 4, 'num_workers': 1, 'local_files_only': True},
        'flags': FLAGS,
        'timing': {'clock': 'per complete recording',
                   'warmup_excluded': 'tests/fixtures/japanese-speech.wav',
                   'not_live_latency': True},
        'versions': {name: importlib.metadata.version(name)
                     for name in ('faster-whisper', 'ctranslate2', 'numpy')},
        'limitations': ['Development-set tuning/evaluation only; not independent heldout evidence.',
                        'Japanese reference remains authoritative for semantic review.',
                        'Direct Whisper translation is compared without Gemma; outputs are not a live-latency measure.'],
    }
    child = None
    try:
        report['gpu_before'] = gpu_snapshot()
        if not any(row['index'] == 0 and row['free_mib'] > 4096
                   for row in report['gpu_before']['gpus']):
            report['status'] = 'skipped_insufficient_gpu_memory'
            write(report)
            return
        env = os.environ | {'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1',
                            'TRANSFORMERS_OFFLINE': '1'}
        write(report)
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                 cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        report['owned_worker_pid'] = child.pid
        try:
            stdout, stderr = child.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            report['status'] = 'timed_out'
        current = json.loads(OUTPUT.read_text())
        current.update(owned_worker_pid=child.pid, worker_exit_code=child.returncode,
                       worker_stdout=stdout[-4000:], worker_stderr=stderr[-4000:])
        if child.returncode:
            current['status'] = 'failed'
        report = current
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        report['gpu_after'] = gpu_snapshot()
        report['owned_gpu_process_released'] = (
            child is None or all(row['pid'] != child.pid for row in report['gpu_after']['processes']))
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(report)
        print(json.dumps({'status': report['status'],
                          'completed_cases': report['completed_cases'],
                          'report': str(OUTPUT.relative_to(ROOT)),
                          'owned_gpu_process_released': report['owned_gpu_process_released']}),
              flush=True)


if __name__ == '__main__':
    main()
