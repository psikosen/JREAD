#!/usr/bin/env python3
"""One frozen exploratory Whisper pass over the authorized anime audio window."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import prepare_adapter_asr as audio_helpers

AUDIO = Path('/media/psikosen/newvolume/jread-translategemma/anime-staging/aikatsu-pripara-episode1-11m17-12m17.wav')
AUDIO_SHA256 = '2362e8953c462f80fef7484cb29f6b963b6d87cabed19a67f7100757e40ff883'
AUDIO_SECONDS = 59.971125
OFFSET_SECONDS = 677.0
OUT = AUDIO.parent / 'whisper-turbo-v1'
LOCK = Path('/media/psikosen/newvolume/jread-finetune/runtime/audio-readiness.lock')
FLAGS = dict(language='ja', task='transcribe', vad_filter=True, beam_size=5,
             condition_on_previous_text=False)
TIMEOUT_SECONDS = 900
MAX_SOURCE_ROWS = 128


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def release_metadata(worker_pid: int | None, worker_exit_code: int | None, gpu_after: dict) -> dict:
    return dict(worker_pid=worker_pid, worker_exit_code=worker_exit_code,
                worker_released=worker_pid is None or worker_exit_code is not None,
                gpu_after=gpu_after,
                owned_gpu_process_released=(worker_pid is None or all(
                    item['pid'] != worker_pid for item in gpu_after['processes'])))


def worker_succeeded(child_exit_code: int | None, worker_report: dict, gpu_process_released: bool) -> bool:
    return (child_exit_code == 0 and worker_report.get('status') == 'asr_complete'
            and worker_report.get('owned_model_unloaded') is True and gpu_process_released)


def wav_info(path: Path = AUDIO) -> dict:
    if sha(path) != AUDIO_SHA256:
        raise ValueError('Captured anime audio SHA256 differs from the frozen input.')
    with wave.open(str(path), 'rb') as w:
        details = dict(channels=w.getnchannels(), sample_width_bytes=w.getsampwidth(),
                       sample_rate=w.getframerate(), frames=w.getnframes(),
                       duration_seconds=w.getnframes() / w.getframerate(),
                       compression=w.getcomptype())
    if details != dict(channels=1, sample_width_bytes=2, sample_rate=16000,
                       frames=959538, duration_seconds=AUDIO_SECONDS, compression='NONE'):
        raise ValueError(f'Captured WAV format/duration changed: {details}')
    return details


def source_rows(segments: list[dict], audio_sha: str = AUDIO_SHA256) -> list[dict]:
    rows = []
    previous_start = -1.0
    for i, segment in enumerate(segments, 1):
        text = str(segment.get('text', ''))
        start, end = float(segment['start']), float(segment['end'])
        if start < previous_start or not (0 <= start <= end <= AUDIO_SECONDS + 0.1):
            raise ValueError('Raw ASR segments are not chronological/in-bounds.')
        previous_start = start
        if not text.strip():
            continue
        rows.append(dict(
            id=f'aikatsu-pripara-e1-11m17-12m17-seg-{i:04d}',
            source=text.strip(),
            start=OFFSET_SECONDS + start,
            end=OFFSET_SECONDS + end,
            clip_start=start,
            clip_end=end,
            offset_seconds=OFFSET_SECONDS,
            audio_sha256=audio_sha,
            segment_index=i,
        ))
    if len(rows) > MAX_SOURCE_ROWS:
        raise ValueError(f'Found {len(rows)} nonempty segments; refusing to truncate above {MAX_SOURCE_ROWS}.')
    return rows


def preflight() -> dict:
    audio = wav_info()
    model = audio_helpers.validate_model()
    return dict(runner_sha256=sha(Path(__file__)), helper_sha256=sha(Path(audio_helpers.__file__)),
                audio=dict(path=str(AUDIO), sha256=AUDIO_SHA256, **audio),
                model=dict(name='faster-whisper large-v3-turbo', revision=model['revision'],
                           model_bin_sha256=model['files']['model.bin']['sha256'],
                           device='cuda:0', compute_type='int8_float16', cpu_threads=4,
                           num_workers=1, local_files_only=True),
                flags=FLAGS, offset_seconds=OFFSET_SECONDS,
                limitations=['Exploratory anime source only; ASR and all translations are uncertified.',
                              'One full-clip decode; no chunking, warmup, references, hints or previous context.'])


def self_test() -> None:
    raw = [dict(start=0.2, end=1.0, text='  日本語です。  ', tokens=[1], words=None),
           dict(start=1.2, end=1.3, text='  ', tokens=[], words=None),
           dict(start=3.0, end=4.0, text='次です。', tokens=[2], words=None)]
    rows = source_rows(raw)
    assert len(rows) == 2 and rows[0]['source'] == '日本語です。'
    assert rows[0]['id'].endswith('0001') and rows[1]['id'].endswith('0003')
    assert rows[0]['start'] == 677.2 and rows[1]['end'] == 681.0
    assert all('target' not in row and 'reference' not in row for row in rows)
    assert release_metadata(42, 0, {'processes': []})['owned_gpu_process_released'] is True
    assert release_metadata(42, 0, {'processes': [{'pid': 42}]})['owned_gpu_process_released'] is False
    child_report = {'status': 'asr_complete', 'owned_model_unloaded': True}
    assert worker_succeeded(0, child_report, gpu_process_released=True)
    assert not worker_succeeded(0, child_report, gpu_process_released=False)
    try:
        source_rows([dict(start=4, end=3, text='壊れた')])
    except ValueError:
        pass
    else:
        raise AssertionError('Non-monotonic segments were accepted.')
    print('Anime replay self-check passed: raw segment order, blank filtering, stable IDs and +677s offset.')


def worker(output: Path) -> None:
    import numpy as np
    from faster_whisper import WhisperModel
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text())
    model = None
    failure = None
    report.update(status='running', raw_segments=[], worker_pid=os.getpid())
    try:
        frozen = preflight()
        if frozen != report['preflight']:
            raise ValueError('Frozen audio/model preflight changed after parent validation.')
        with wave.open(str(AUDIO), 'rb') as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
        if len(samples) != 959538 or not np.isfinite(samples).all():
            raise ValueError('Decoded PCM does not match the frozen 16 kHz clip.')
        model = WhisperModel(str(audio_helpers.MODEL_DIR), device='cuda', device_index=0,
                             compute_type='int8_float16', cpu_threads=4, num_workers=1,
                             local_files_only=True)
        started = time.perf_counter()
        segments, info = model.transcribe(samples, **FLAGS)
        rows = []
        previous_start = -1.0
        for segment in segments:
            raw = dataclasses.asdict(segment)
            if raw['start'] < previous_start:
                raise ValueError('Whisper yielded out-of-order segment times.')
            previous_start = raw['start']
            rows.append(raw)
            report['raw_segments'] = rows
            write_json(report_path, report)
        report.update(status='asr_complete',
                      asr_language=info.language,
                      language_was_forced=True,
                      asr_language_probability=float(info.language_probability),
                      asr_language_note='Japanese was forced with language=ja; probability is not independent language verification.',
                      model_duration_seconds=float(info.duration),
                      inference_seconds=time.perf_counter() - started)
    except BaseException as error:
        failure = error
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        unloaded = False
        if model is not None:
            try:
                model.model.unload_model()
                unloaded = True
            except Exception as error:
                report['unload_error'] = f'{type(error).__name__}: {error}'
            model = None
        gc.collect()
        report['owned_model_unloaded'] = unloaded
        if failure is None and not unloaded:
            report['status'] = 'failed'
            report['error'] = 'ASR worker did not verify owned model unload.'
        write_json(report_path, report)
    if failure:
        raise failure


def run(output: Path) -> None:
    if output.exists():
        raise SystemExit(f'Refusing to overwrite existing run directory: {output}')
    frozen = preflight()
    before = audio_helpers.gpu_snapshot()
    if not any(g['index'] == 0 and g['free_mib'] >= 4096 for g in before['gpus']):
        raise SystemExit('Need at least4096MiB free on GPU0; no other process is touched.')
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'report.json'
    report = dict(schema_version=1, status='starting', started_at=time.time(), preflight=frozen,
                  flags=FLAGS, gpu_before=before, raw_segments=[], source_rows_path=None,
                  source_rows_sha256=None, source_rows_count=None, asr_source_certified=False,
                  translation_accuracy_claimed=False, worker_pid=None, worker_exit_code=None)
    write_json(report_path, report)
    child = None
    env = os.environ | dict(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    cuda_lib = '/usr/local/cuda-12.8/lib64'
    env['LD_LIBRARY_PATH'] = cuda_lib + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    lock_path = LOCK.resolve()
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (output / 'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                    '--output', str(output)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    child.wait(timeout=TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    report['timeout_seconds'] = TIMEOUT_SECONDS
    except (BlockingIOError, FileExistsError) as error:
        report['status'] = 'blocked_gpu_lock'
        report['error'] = str(error)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        current = None
        if report.get('status') not in ('blocked_gpu_lock',):
            current = json.loads(report_path.read_text())
            report.update(current)
        gpu_after = audio_helpers.gpu_snapshot()
        report.update(release_metadata(child.pid if child else None,
                                      child.returncode if child else None, gpu_after),
                      finished_at=time.time())
        if report.get('status') not in ('blocked_gpu_lock',):
            assert current is not None
            if (child is None or not worker_succeeded(child.returncode, current,
                                                       report['owned_gpu_process_released'])):
                report['status'] = 'failed'
                if child and child.returncode:
                    report.setdefault('error', f'ASR child exited {child.returncode}; inspect worker.log.')
            else:
                try:
                    rows = source_rows(current['raw_segments'])
                    source_path = output / 'sources.jsonl'
                    with source_path.open('x', encoding='utf-8') as f:
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                        f.flush(); os.fsync(f.fileno())
                    report.update(status='complete', source_rows_path=str(source_path),
                                  source_rows_sha256=sha(source_path), source_rows_count=len(rows),
                                  raw_segment_count=len(current['raw_segments']),
                                  empty_segment_count=len(current['raw_segments']) - len(rows))
                except Exception as error:
                    report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(report_path, report)
    print(json.dumps(dict(status=report['status'], report=str(report_path),
                          source_rows=report.get('source_rows_count'), worker_released=report.get('worker_released'))))
    if report['status'] != 'complete':
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='validate only; does not load models or initialize CUDA')
    parser.add_argument('--run-asr', action='store_true', help='one full-clip GPU pass; root will launch after review')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.worker:
        worker(args.output.resolve())
    elif args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.run_asr:
        run(args.output.resolve())
    else:
        parser.error('Choose --self-test, --preflight, or --run-asr.')


if __name__ == '__main__':
    main()
