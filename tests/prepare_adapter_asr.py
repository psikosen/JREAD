#!/usr/bin/env python3
"""Create reviewed-data Turbo ASR candidates; this helper never translates."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/media/psikosen/newvolume/jread-finetune/data/covost2-ja-en-source-check').resolve()
AUDIO_ROOT = (DATA_ROOT / 'audio').resolve()
MODEL_DIR = Path.home() / '.local/share/JREAD/models/large-v3-turbo'
MODEL_META = ROOT / 'docs/whisper-turbo-model.json'
CORPUS = ROOT / 'tests/fixtures/accuracy-corpus.json'
CORPUS_SHA256 = '66825d56bfa271a58037a07ef68a59c1998c07f65b7740833b9e30d2d44cd578'
MODEL_REVISION = '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'
MODEL_BIN_SHA256 = 'e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da'
SEED = '20261004'
FLAGS = dict(language='ja', task='transcribe', vad_filter=True, beam_size=5,
              condition_on_previous_text=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(raw: bytes) -> list[dict]:
    rows = []
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Input row {number} is not a JSON object.')
            rows.append(row)
    return rows


def select_rows(rows: list[dict], train_count: int | None = 32,
                group_ids: list[str] | None = None, seed: str = SEED) -> list[dict]:
    by_group: dict[str, dict] = {}
    for row in rows:
        group_id = row.get('group_id')
        if not isinstance(group_id, str) or not group_id:
            raise ValueError('Every clean input row needs a group_id.')
        if group_id in by_group:
            raise ValueError(f'Duplicate clean group: {group_id}')
        by_group[group_id] = row
    if group_ids is not None:
        if train_count is not None:
            raise ValueError('Choose either explicit group IDs or train-count selection.')
        if not group_ids:
            raise ValueError('Explicit group ID selection cannot be empty.')
        if len(group_ids) != len(set(group_ids)):
            raise ValueError('Explicit group IDs must be unique.')
        missing = [group_id for group_id in group_ids if group_id not in by_group]
        if missing:
            raise ValueError('Unknown group IDs: ' + ', '.join(missing))
        return [by_group[group_id] for group_id in group_ids]
    if not isinstance(train_count, int) or train_count < 1:
        raise ValueError('--train-count must be a positive integer.')
    validation = sorted((r for r in rows if r.get('partition') == 'validation'),
                        key=lambda r: r['group_id'])
    train = [r for r in rows if r.get('partition') == 'train']
    if len(validation) != 20 or len({r['group_id'] for r in validation}) != 20:
        raise ValueError('Count mode requires all 20 distinct validation groups.')
    train.sort(key=lambda r: (hashlib.sha256(f'{seed}|{r["group_id"]}'.encode()).hexdigest(), r['group_id']))
    if train_count > len(train):
        raise ValueError(f'Requested {train_count} train groups; only {len(train)} are available.')
    chosen_train = train[:train_count]
    return chosen_train + validation


def evaluation_audio_metadata() -> tuple[str, set[str], set[Path]]:
    raw = CORPUS.read_bytes()
    if sha256_bytes(raw) != CORPUS_SHA256:
        raise ValueError('Frozen evaluation corpus manifest changed.')
    corpus = json.loads(raw)
    if len(corpus.get('cases', [])) != 70:
        raise ValueError('Expected the frozen 20-development + 50-heldout manifest.')
    hashes, paths = set(), set()
    for case in corpus['cases']:
        for key in ('audio_sha256', 'download_sha256'):
            value = case.get(key)
            if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value):
                hashes.add(value)
        value = case.get('audio_path')
        if isinstance(value, str) and value:
            paths.add((ROOT / value).resolve())
    return sha256_bytes(raw), hashes, paths


def validate_inputs(rows: list[dict], evaluation_hashes: set[str],
                    evaluation_paths: set[Path], check_audio: bool = True) -> None:
    ids = set()
    for row in rows:
        for key in ('id', 'group_id', 'source', 'target', 'partition', 'variant', 'review_status', 'audio_path', 'audio_sha256'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Input row is missing nonempty {key}.")
        if row['id'] in ids:
            raise ValueError(f"Duplicate clean row ID: {row['id']}")
        ids.add(row['id'])
        if row['partition'] not in ('train', 'validation'):
            raise ValueError(f"Unsupported partition for {row['id']}.")
        if row['variant'] != 'clean' or row['review_status'] != 'accepted':
            raise ValueError(f"Only reviewed, accepted clean rows are allowed: {row['id']}.")
        if not re.fullmatch(r'[0-9a-f]{64}', row['audio_sha256']):
            raise ValueError(f"Invalid audio SHA256 for {row['id']}.")
        provenance = row.get('provenance')
        if (not isinstance(provenance, dict)
                or provenance.get('dataset') != 'ku-nlp/covost2NativeJa'
                or provenance.get('revision') != '52f01d398c24017d104021897e022676117bbe7b'
                or provenance.get('split') != row['partition']):
            raise ValueError(f"Unexpected native corpus provenance for {row['id']}.")
        if provenance.get('clean_source_transcript') != row['source']:
            raise ValueError(f"Clean source/provenance mismatch for {row['id']}.")
        path = Path(row['audio_path']).resolve()
        try:
            path.relative_to(AUDIO_ROOT)
        except ValueError as exc:
            raise ValueError(f"Audio path for {row['id']} is outside the local native corpus.") from exc
        if path in evaluation_paths or row['audio_sha256'] in evaluation_hashes:
            raise ValueError(f"Input {row['id']} overlaps frozen evaluation audio metadata.")
        if check_audio:
            if not path.is_file() or sha256_file(path) != row['audio_sha256']:
                raise ValueError(f"Native audio path/hash mismatch for {row['id']}.")


def validate_model() -> dict:
    meta = json.loads(MODEL_META.read_text())
    if meta.get('revision') != MODEL_REVISION:
        raise ValueError('Turbo model metadata revision differs from the frozen revision.')
    if not MODEL_DIR.is_dir() or (MODEL_DIR / '.verified').read_text().strip() != MODEL_REVISION:
        raise ValueError('The local Turbo model is absent or has the wrong verified revision.')
    files = meta.get('files', {})
    if files.get('model.bin', {}).get('sha256') != MODEL_BIN_SHA256:
        raise ValueError('Turbo model metadata does not contain the expected model.bin SHA256.')
    for name, info in files.items():
        path = MODEL_DIR / name
        expected_size = info.get('size_bytes', info.get('size'))
        if (not path.is_file() or expected_size is None or path.stat().st_size != expected_size
                or sha256_file(path) != info['sha256']):
            raise ValueError(f'Local Turbo file failed pinned size/SHA256 check: {name}.')
    return meta


def gpu_snapshot() -> dict:
    tests_dir = ROOT / 'tests'
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from asr_cuda_pilot import gpu_snapshot as snapshot
    return snapshot()


def noisy_rows(clean_rows: list[dict], cases: list[dict], report_path: Path,
               report_sha256: str, model: dict) -> list[dict]:
    by_group = {case['group_id']: case for case in cases}
    output = []
    for clean in clean_rows:
        case = by_group[clean['group_id']]
        row = dict(clean)
        row.update(id=clean['id'] + '-whisper', source=case.get('transcript', ''),
                   clean_source=clean['source'], variant='whisper', review_status='pending',
                   asr_report_sha256=report_sha256)
        provenance = dict(clean['provenance'])
        provenance.update({
            'asr_report': str(report_path.resolve()),
            'asr_case_status': case['status'],
            'asr_model': model['name'],
            'asr_model_revision': model['revision'],
            'asr_model_sha256': model['model_bin_sha256'],
            'asr_method': 'full-clip Japanese transcription; independent per-row decode, no hints, references or previous-case context',
        })
        row['provenance'] = provenance
        output.append(row)
    return output


def decode_audio(path: Path):
    """Use the installed FFmpeg path; avoid the incompatible PyAV file decoder."""
    import numpy as np
    decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2',
        '-protocol_whitelist', 'file,pipe', '-f', 'mp3', '-i', str(path),
        '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'],
        capture_output=True, timeout=30, check=True)
    samples = np.frombuffer(decoded.stdout, dtype='<f4').copy()
    if not 0 < len(samples) <= 30 * 16000 or not np.isfinite(samples).all():
        raise ValueError('Expected finite mono 16kHz audio lasting at most30 seconds.')
    return samples


def worker(input_path: Path, input_sha: str, group_ids: list[str], cases_path: Path) -> int:
    raw = input_path.read_bytes()
    if sha256_bytes(raw) != input_sha:
        raise ValueError('Reviewed input changed after preflight.')
    rows = read_jsonl(raw)
    _, eval_hashes, eval_paths = evaluation_audio_metadata()
    validate_inputs(rows, eval_hashes, eval_paths, check_audio=True)
    selected = select_rows(rows, train_count=None, group_ids=group_ids)
    if len(selected) != len(group_ids):
        raise ValueError('Selected rows changed after preflight.')
    import numpy  # noqa: F401 -- capture runtime version in provenance
    from faster_whisper import WhisperModel

    records = []
    model = None
    def save_case(record: dict, output_file) -> None:
        records.append(record)
        output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        output_file.flush()
        os.fsync(output_file.fileno())
        print(json.dumps({'event': 'case', 'case': record}, ensure_ascii=False), flush=True)
    try:
        model = WhisperModel(str(MODEL_DIR), device='cuda', device_index=0,
                             compute_type='int8_float16', cpu_threads=4,
                             num_workers=1, local_files_only=True)
        with cases_path.open('x', encoding='utf-8') as case_file:
            for clean in selected:
                path = Path(clean['audio_path']).resolve()
                record = {key: clean[key] for key in ('id', 'group_id', 'source', 'audio_path', 'audio_sha256', 'partition')}
                try:
                    if sha256_file(path) != clean['audio_sha256']:
                        raise ValueError('Audio hash changed immediately before inference.')
                    started = time.perf_counter()
                    samples = decode_audio(path)
                    record.update(decoded_samples=len(samples), decoded_pcm_sha256=sha256_bytes(samples.tobytes()))
                    segments, info = model.transcribe(samples, **FLAGS)
                    segment_rows = []
                    parts = []
                    for segment in segments:
                        text = str(segment.text)
                        parts.append(text)
                        segment_rows.append({
                            'start': float(segment.start), 'end': float(segment.end), 'text': text,
                            'avg_logprob': float(segment.avg_logprob),
                            'no_speech_prob': float(segment.no_speech_prob),
                        })
                    transcript = ''.join(parts).strip()
                    elapsed = time.perf_counter() - started
                    record.update(status='success' if transcript else 'empty', transcript=transcript,
                                  segments=segment_rows, duration_seconds=float(info.duration),
                                  detected_language=info.language,
                                  language_probability=float(info.language_probability),
                                  inference_seconds=elapsed, error=None)
                except Exception as error:
                    record.update(status='error', transcript='', segments=[], duration_seconds=None,
                                  detected_language=None, language_probability=None,
                                  inference_seconds=None, error=f'{type(error).__name__}: {error}')
                save_case(record, case_file)
    finally:
        unloaded = False
        if model is not None:
            try:
                model.model.unload_model()
                unloaded = True
            finally:
                model = None
                gc.collect()
        print(json.dumps({'event': 'owned_model_unloaded', 'value': unloaded}), flush=True)
    return 0


def self_test() -> None:
    rows = []
    for i in range(40):
        rows.append({'id': f't{i}', 'group_id': f'train-{i:02d}', 'partition': 'train'})
    for i in range(20):
        rows.append({'id': f'v{i}', 'group_id': f'validation-{i:02d}', 'partition': 'validation'})
    first = select_rows(rows, train_count=32, seed=SEED)
    second = select_rows(rows, train_count=32, seed=SEED)
    assert [row['group_id'] for row in first] == [row['group_id'] for row in second]
    assert len(first) == 52 and sum(r['partition'] == 'train' for r in first) == 32
    assert sum(r['partition'] == 'validation' for r in first) == 20
    explicit = select_rows(rows, train_count=None, group_ids=['train-03', 'validation-02'])
    assert [r['group_id'] for r in explicit] == ['train-03', 'validation-02']
    for bad in ([], ['train-03', 'train-03'], ['not-present']):
        try:
            select_rows(rows, train_count=None, group_ids=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid explicit selection was accepted.')
    good = [{'group_id': 'train-03', 'status': 'success'},
            {'group_id': 'validation-02', 'status': 'empty'}]
    result = completion_state(good, ['train-03', 'validation-02'], 0, True, True, True)
    assert result['status'] == 'complete' and result['traversal_complete']
    assert result['asr_preparation_success']  # Empty transcription is a valid observed result.
    failed = [dict(good[0], status='error'), good[1]]
    result = completion_state(failed, ['train-03', 'validation-02'], 0, True, True, True)
    assert result['status'] == 'failed' and result['traversal_complete']
    assert not result['asr_preparation_success']
    missing = completion_state(good[:1], ['train-03', 'validation-02'], 0, True, True, True)
    assert missing['status'] == 'failed' and not missing['traversal_complete']
    assert sha256_bytes(b'audio') == hashlib.sha256(b'audio').hexdigest()
    clean = {'id': 'x', 'group_id': 'x', 'source': '清潔原文', 'target': 'clean target',
             'partition': 'train', 'variant': 'clean', 'review_status': 'accepted',
             'audio_sha256': 'a' * 64, 'provenance': {'dataset': 'source'}}
    case = {'group_id': 'x', 'status': 'empty', 'transcript': '', 'source': '清潔原文'}
    pending = noisy_rows([clean], [case], Path('/tmp/asr-report.json'), 'b' * 64,
                         {'name': 'test-model', 'revision': 'r', 'model_bin_sha256': 'c' * 64})[0]
    assert pending['source'] == '' and pending['clean_source'] == '清潔原文'
    assert pending['review_status'] == 'pending' and pending['asr_report_sha256'] == 'b' * 64
    print('adapter ASR self-test passed: deterministic 32+20 selection, explicit selection and rejects')


def completion_state(cases: list[dict], selected_group_ids: list[str], child_exit_code: int | None,
                     model_unloaded: bool, process_released: bool,
                     input_unchanged: bool) -> dict:
    case_groups = [case.get('group_id') for case in cases]
    traversal_complete = (len(cases) == len(selected_group_ids)
                          and len(case_groups) == len(set(case_groups))
                          and set(case_groups) == set(selected_group_ids))
    statuses = [case.get('status') for case in cases]
    statuses_supported = all(status in ('success', 'empty', 'error') for status in statuses)
    asr_preparation_success = (traversal_complete and statuses_supported
                               and all(status in ('success', 'empty') for status in statuses))
    execution_success = (child_exit_code == 0 and model_unloaded and process_released
                         and input_unchanged)
    return {
        'traversal_complete': traversal_complete,
        'asr_preparation_success': asr_preparation_success,
        'execution_success': execution_success,
        'status': 'complete' if asr_preparation_success and execution_success else 'failed',
    }


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        f.flush(); os.fsync(f.fileno())


def rewrite_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    if args.timeout < 1:
        raise ValueError('--timeout must be positive.')
    input_path = args.input.resolve()
    raw = input_path.read_bytes()
    input_sha = sha256_bytes(raw)
    rows = read_jsonl(raw)
    corpus_sha, eval_hashes, eval_paths = evaluation_audio_metadata()
    validate_inputs(rows, eval_hashes, eval_paths, check_audio=True)
    selected = select_rows(rows, train_count=None if args.group_ids else args.train_count,
                           group_ids=args.group_ids, seed=args.seed)
    if not selected:
        raise ValueError('No rows selected.')
    meta = validate_model()
    versions = {name: importlib.metadata.version(name)
                for name in ('faster-whisper', 'ctranslate2', 'numpy')}
    gpu_before = gpu_snapshot()
    if not any(row['index'] == 0 and row['free_mib'] >= 4096 for row in gpu_before['gpus']):
        raise RuntimeError('At least4096MiB free GPU memory is required; no model was started.')
    report_path, noisy_path = args.report.resolve(), args.noisy_output.resolve()
    if report_path in (noisy_path, input_path) or noisy_path == input_path:
        raise ValueError('Input, report and noisy-output paths must be distinct.')
    if report_path.exists() or noisy_path.exists():
        raise FileExistsError('Refusing to overwrite an existing ASR report or candidate JSONL.')
    cases_path = report_path.with_suffix('.cases.partial.jsonl')
    groups_path = report_path.with_suffix('.groups.json')
    if len({report_path, noisy_path, cases_path, groups_path}) != 4:
        raise ValueError('Report, candidate and scratch artifact paths must be distinct.')
    if cases_path.exists() or groups_path.exists():
        raise FileExistsError('Refusing to overwrite existing partial ASR case/selection files.')
    selected_groups = [row['group_id'] for row in selected]
    groups_snapshot = (json.dumps(selected_groups, ensure_ascii=False) + '\n').encode('utf-8')
    report = {
        'schema_version': 1,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'input': {'path': str(input_path), 'sha256': input_sha,
                  'row_count': len(rows), 'reviewed_jsonl_snapshot_sha256': input_sha},
        'corpus_guard': {'manifest': str(CORPUS), 'manifest_sha256': corpus_sha,
                         'case_count': 70, 'evaluation_audio_opened': False,
                         'evaluation_references_used_for_inference': False},
        'selection': {'method': 'explicit group IDs' if args.group_ids else 'SHA256(seed|group_id), train subset plus all 20 validation groups',
                      'seed': None if args.group_ids else args.seed,
                      'train_count': sum(row['partition'] == 'train' for row in selected),
                      'validation_count': sum(row['partition'] == 'validation' for row in selected),
                      'group_ids': selected_groups,
                      'group_ids_snapshot_sha256': sha256_bytes(groups_snapshot)},
        'model': {'name': 'faster-whisper large-v3-turbo',
                  'repo': meta['repo'], 'revision': meta['revision'], 'license': meta['license'],
                  'model_directory': str(MODEL_DIR), 'model_bin_sha256': MODEL_BIN_SHA256,
                  'files': {name: {'size_bytes': info.get('size_bytes', info.get('size')), 'sha256': info['sha256']}
                            for name, info in meta['files'].items()},
                  'device': 'cuda:0', 'compute_type': 'int8_float16', 'cpu_threads': 4,
                  'num_workers': 1, 'local_files_only': True},
        'flags': dict(FLAGS),
        'audio_decode': {'method': 'FFmpeg MP3 to mono16000Hz float32 PCM; input array passed to Whisper',
                         'ffmpeg_version': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]},
        'method': 'Independent full-clip decode for each selected audio file. No Whisper prompt, prior case context, Japanese reference, English target, glossary, or streaming capture is passed to transcribe().',
        'versions': versions, 'gpu_before': gpu_before,
        'runner_sha256': sha256_file(Path(__file__).resolve()),
        'cases': [], 'owned_model_unloaded': False,
        'owned_gpu_process_released': False,
    }
    write_json_exclusive(report_path, report)
    child = None
    worker_events = []
    worker_stderr = ''
    child_exit_code = None
    try:
        env = os.environ | {'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1',
                            'TRANSFORMERS_OFFLINE': '1'}
        cuda_library = '/usr/local/cuda-12.8/lib64'
        old_library_path = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = cuda_library + (':' + old_library_path if old_library_path else '')
        report['worker_environment'] = {key: env.get(key, '') for key in
                                        ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'LD_LIBRARY_PATH')}
        rewrite_json(report_path, report)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--input', str(input_path), '--input-sha', input_sha,
                   '--groups-file', str(write_group_snapshot(report_path, selected_groups)),
                   '--cases-output', str(cases_path)]
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            stdout, worker_stderr = child.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            child.kill(); stdout, worker_stderr = child.communicate()
            report['worker_timed_out'] = True
        child_exit_code = child.returncode
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
                worker_events.append(event)
            except json.JSONDecodeError:
                continue
        if cases_path.exists():
            report['cases'] = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
        seen = {case['group_id'] for case in report['cases']}
        for clean in selected:
            if clean['group_id'] not in seen:
                report['cases'].append({
                    'id': clean['id'], 'group_id': clean['group_id'], 'source': clean['source'],
                    'audio_path': clean['audio_path'], 'audio_sha256': clean['audio_sha256'],
                    'partition': clean['partition'], 'status': 'error', 'transcript': '',
                    'segments': [], 'inference_seconds': None,
                    'error': 'Worker terminated before this selected case produced a record.',
                })
        report['worker_exit_code'] = child_exit_code
        report['worker_stderr'] = worker_stderr[-4000:]
        report['owned_model_unloaded'] = any(event.get('event') == 'owned_model_unloaded' and event.get('value') is True for event in worker_events)
        report['gpu_after'] = gpu_snapshot()
        report['owned_gpu_process_released'] = child is not None and all(p['pid'] != child.pid for p in report['gpu_after']['processes'])
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        input_unchanged = sha256_file(input_path) == input_sha
        report['input']['unchanged_after_run'] = input_unchanged
        report['case_status_counts'] = dict(Counter(c['status'] for c in report['cases']))
        completion = completion_state(
            report['cases'], selected_groups, child_exit_code,
            report['owned_model_unloaded'], report['owned_gpu_process_released'],
            input_unchanged,
        )
        report.update(completion)
        rewrite_json(report_path, report)
        if report['status'] != 'complete':
            raise RuntimeError(f"ASR run failed (traversal_complete={report['traversal_complete']}, "
                               f"case_status_counts={report['case_status_counts']}); inspect {report_path}.")
        report_sha = sha256_file(report_path)
        noisy = noisy_rows(selected, report['cases'], report_path, report_sha, report['model'])
        with noisy_path.open('x', encoding='utf-8') as f:
            for row in noisy:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            f.flush(); os.fsync(f.fileno())
        print(json.dumps({'status': 'complete', 'cases': len(report['cases']),
                          'case_status_counts': report['case_status_counts'],
                          'report': str(report_path), 'report_sha256': report_sha,
                          'noisy_candidates': str(noisy_path), 'noisy_candidate_count': len(noisy)}, ensure_ascii=False))
    finally:
        if child is not None and child.poll() is None:
            child.kill(); child.wait(timeout=10)


def write_group_snapshot(report_path: Path, group_ids: list[str]) -> Path:
    path = report_path.with_suffix('.groups.json')
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(group_ids, ensure_ascii=False) + '\n')
        f.flush(); os.fsync(f.fileno())
    return path


def worker_main(args: argparse.Namespace) -> None:
    try:
        groups = json.loads(args.groups_file.read_text())
        exit_code = worker(args.input, args.input_sha, groups, args.cases_output)
        raise SystemExit(exit_code)
    except SystemExit:
        raise
    except Exception as error:
        print(json.dumps({'event': 'worker_error', 'error': f'{type(error).__name__}: {error}'}), flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--noisy-output', type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--train-count', type=int, default=32)
    parser.add_argument('--seed', default=SEED)
    selection.add_argument('--group-ids', nargs='+')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--input-sha')
    parser.add_argument('--groups-file', type=Path)
    parser.add_argument('--cases-output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.worker:
        if not args.input or not args.input_sha or not args.groups_file or not args.cases_output:
            parser.error('Worker requires --input, --input-sha, --groups-file and --cases-output.')
        worker_main(args); return
    if not args.input or not args.report or not args.noisy_output:
        parser.error('--input, --report and --noisy-output are required for --run')
    run(args)


if __name__ == '__main__':
    main()
