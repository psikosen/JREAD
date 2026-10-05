#!/usr/bin/env python3
"""Pinned ReazonSpeech CPU recognition followed by unchanged E4B translation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/media/psikosen/newvolume/jread-reazon')
PROTOCOL = ROOT / 'tests/fixtures/accuracy-reazon.json'
SCRIPT = Path(__file__).resolve()
MODEL_FILES = {'tokens': 'tokens.txt', 'encoder': 'encoder-epoch-99-avg-1.onnx',
               'decoder': 'decoder-epoch-99-avg-1.onnx', 'joiner': 'joiner-epoch-99-avg-1.onnx'}
ASR_SETTINGS = dict(provider='cpu', num_threads=4, sample_rate=16000, feature_dim=80,
                    decoding_method='greedy_search', padding_seconds_each_side=0.9)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frozen(binding):
    path = Path(binding['path'])
    if sha(path) != binding['sha256']:
        raise ValueError(f'Frozen file changed: {path}')
    return json.loads(path.read_text())


def write(path, data):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def pcm(path):
    with wave.open(str(path), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('Expected mono PCM16 16kHz WAV.')
        frames = wav.getnframes()
        raw = wav.readframes(frames)
        if not 0 < frames <= 30 * 16000 or len(raw) != frames * 2:
            raise ValueError('Require complete audio within30 seconds, without truncation.')
        return raw, frames / 16000


def validate(protocol_path):
    protocol = json.loads(protocol_path.read_text())
    if protocol['status'] != 'frozen' or protocol['runner_sha256'] != sha(SCRIPT):
        raise ValueError('Require frozen protocol and unchanged runner.')
    if protocol['asr_settings'] != ASR_SETTINGS or protocol['case_count'] != 20:
        raise ValueError('ASR settings or denominator changed.')
    data = frozen(protocol['inputs'])
    if len(data) != 20 or [r['id'] for r in data] != protocol['case_ids']:
        raise ValueError('Require the20 frozen development IDs in order.')
    if len(set(protocol['case_ids'])) != 20:
        raise ValueError('Duplicate development ID.')
    if any(row.get('partition') != 'development' for row in data):
        raise ValueError('Only development audio is authorized.')
    for row in data + [protocol['speech_control']]:
        if sha(row['audio_path']) != row['audio_sha256']:
            raise ValueError('Audio hash changed.')
        _, duration = pcm(row['audio_path'])
        if abs(duration - row['duration_seconds']) > 1 / 16000:
            raise ValueError('Audio duration changed.')
    if protocol['silence_seconds'] != 8 or protocol['speech_control']['id'] != 1828:
        raise ValueError('Controls changed.')
    provenance = frozen(protocol['model_provenance'])
    if provenance['overall_status'] != 'verified' or provenance['revision'] != protocol['model_revision']:
        raise ValueError('Require verified pinned model.')
    marker = dict(line.split('=', 1) for line in (BASE / 'model/.verified').read_text().splitlines())
    if marker.get('revision') != protocol['model_revision'] or marker.get('repository') != provenance['repository']:
        raise ValueError('Model verification marker mismatch.')
    for name in MODEL_FILES.values():
        expected = provenance['files'][name]
        path = BASE / 'model' / name
        if path.stat().st_size != expected['size_bytes'] or sha(path) != expected['actual_sha256']:
            raise ValueError('Pinned model bytes changed.')
    runtime = frozen(protocol['runtime_manifest'])
    for binding in runtime['imported_file_hashes']:
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Pinned runtime file changed.')
    for binding in protocol['helpers'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Frozen helper changed.')
    return protocol, data


def free_ports():
    for port in (18767, 18768):
        with socket.socket() as sock:
            sock.settimeout(.25)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise ValueError(f'Private model port{port} occupied; no overlapping job.')


def recognize(model, raw, duration):
    import numpy as np
    audio = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    # Match official ReazonSpeech pad_audio:0.9 seconds on BOTH sides.
    audio = np.pad(audio, int(.9 * 16000))
    started = time.perf_counter()
    stream = model.create_stream()
    stream.accept_waveform(16000, audio)
    model.decode_stream(stream)
    result = stream.result
    seconds = time.perf_counter() - started
    return dict(status='ok', text=result.text, tokens=list(result.tokens),
                padded_audio_token_timestamps=list(result.timestamps), inference_seconds=seconds,
                original_duration_seconds=duration, padded_duration_seconds=len(audio) / 16000,
                timestamps_note='Token points refer to padded audio; no subtitle alignment is asserted.')


def worker(report_path, protocol_path):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    protocol, data = validate(protocol_path)
    if Path(sys.prefix).resolve() != (BASE / 'runtime/venv').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Require isolated CPU worker runtime.')
    import sherpa_onnx
    for name, expected in protocol['runtime_versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Runtime package version changed: {name}')
    report = json.loads(report_path.read_text())
    try:
        started = time.perf_counter()
        model = sherpa_onnx.OfflineRecognizer.from_transducer(
            **{key: str(BASE / 'model' / name) for key, name in MODEL_FILES.items()},
            **{k: v for k, v in ASR_SETTINGS.items() if k != 'padding_seconds_each_side'})
        report['model_load_seconds'] = time.perf_counter() - started
        if report['stage'] == 'smoke':
            control = protocol['speech_control']
            raw, duration = pcm(control['audio_path'])
            report['speech_control'] = recognize(model, raw, duration)
            report['silence_control'] = recognize(model, bytes(8 * 16000 * 2), 8)
        else:
            # Independent control warms the runtime; do not re-decode a development clip.
            raw, duration = pcm(protocol['speech_control']['audio_path'])
            report['warmup'] = recognize(model, raw, duration) | {'excluded': True}
            for row in report['cases']:
                try:
                    raw, duration = pcm(row['audio_path'])
                    row['asr'] = recognize(model, raw, duration)
                    if not row['asr']['text'].strip():
                        row['asr'].update(status='error', error='Empty speech transcript; retained in denominator.')
                except Exception as error:
                    row['asr'] = dict(status='error', error=f'{type(error).__name__}: {error}', inference_seconds=0)
                write(report_path, report)
        report['status'] = 'worker_complete'
        report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(report_path, report)


def asr_run(stage, protocol_path):
    protocol, data = validate(protocol_path)
    free_ports()
    path = Path(protocol['outputs'][stage])
    if stage == 'asr':
        gate = frozen(protocol['readiness_review'])
        smoke_path = Path(protocol['outputs']['smoke'])
        diagnostic = protocol.get('failed_readiness_diagnostic') is True
        if (gate['smoke_sha256'] != sha(smoke_path) or
                (gate['decision'] != 'pass' and not (diagnostic and gate['decision'] == 'fail'))):
            raise ValueError('Require reviewed readiness, or an explicitly frozen failed-readiness offline diagnostic.')
    report = dict(schema_version=1, status='running', stage=stage, partition='development' if stage == 'asr' else 'control',
        heldout_rows_used=0, created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=dict(path=str(protocol_path), sha256=sha(protocol_path), contents=protocol),
        failed_readiness_diagnostic=protocol.get('failed_readiness_diagnostic', False),
        production_eligible=False,
        cases=[row | {'asr': {'status': 'pending'}} for row in data] if stage == 'asr' else [],
        limits=['Whole clips decoded after EOF; no live-caption or streaming-delay claim.',
                'FP32 Reazon CPU4 versus historical Whisper CUDA; timing is not a matched hardware comparison.',
                'No lexical correction, VAD, glossary, prompt, tuning or retries.'])
    with path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    env = os.environ | dict(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                            OMP_NUM_THREADS='4', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    process = None
    stdout, stderr = b'', b''
    try:
        process = subprocess.Popen([str(BASE / 'runtime/venv/bin/python'), str(SCRIPT), '--worker', str(path),
                                    '--protocol', str(protocol_path)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=600)
        report = json.loads(path.read_text())
        report.update(worker_stdout=stdout.decode(errors='replace')[-4000:], worker_stderr=stderr.decode(errors='replace')[-4000:])
        if process.returncode != 0 or report['status'] != 'worker_complete':
            raise RuntimeError('ASR worker failed; inspect saved stderr and partial report.')
        report['status'] = 'complete_with_errors' if any(r['asr']['status'] != 'ok' for r in report['cases']) else 'complete'
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        # Preserve the worker's latest completed rows on timeout/interruption.
        report = json.loads(path.read_text())
        report.update(status='failed', worker_error=report.get('error'), error=f'{type(error).__name__}: {error}',
                      worker_stdout=stdout.decode(errors='replace')[-4000:], worker_stderr=stderr.decode(errors='replace')[-4000:])
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        report['cleanup'] = dict(owned_worker_pid=process.pid if process else None,
            worker_exit_code=process.returncode if process else None,
            owned_worker_released=process is None or process.poll() is not None)
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(path, report)
    return report


def translate_run(protocol_path):
    protocol, data = validate(protocol_path)
    import evaluate_e4b_punctuation as previous
    old_protocol, cases, baseline, cache = previous.validated_inputs(previous.PROTOCOL)
    gemma, ensemble = previous.gemma, previous.ensemble
    for case in cases:
        existing = cache.get(case['original_prompt_sha256'])
        if existing is not None and existing['text'] != case['baseline']['text']:
            raise ValueError('Conflicting historical outputs for an identical prompt; no silent selection.')
        cache.setdefault(case['original_prompt_sha256'], dict(text=case['baseline']['text'],
            translation_seconds=case['baseline_measured_seconds'], provenance={'kind': 'frozen_turbo_baseline', 'case_id': case['id']}))
    asr_path = Path(protocol['outputs']['asr'])
    asr = json.loads(asr_path.read_text())
    if (asr['status'] not in ('complete', 'complete_with_errors') or asr['cleanup']['worker_exit_code'] != 0
            or not asr['cleanup']['owned_worker_released'] or [r['id'] for r in asr['cases']] != protocol['case_ids']):
        raise ValueError('Require completed ASR and released worker.')
    if asr['protocol']['sha256'] != sha(protocol_path):
        raise ValueError('ASR protocol binding mismatch.')
    free_ports()
    path = Path(protocol['outputs']['translate'])
    report = dict(schema_version=1, status='running', partition='development', heldout_rows_used=0,
        failed_readiness_diagnostic=protocol.get('failed_readiness_diagnostic', False), production_eligible=False,
        protocol=dict(path=str(protocol_path), sha256=sha(protocol_path)),
        source_asr=dict(path=str(asr_path), sha256=sha(asr_path)), model=baseline['model'],
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        cases=[dict(id=r['id'], status='pending', text=None) for r in asr['cases']])
    with path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    old_port, old_base = gemma.PORT, gemma.BASE
    gemma.PORT, gemma.BASE = 18767, f'http://{gemma.HOST}:18767'
    try:
        if gemma._process is not None or not gemma.ready('gemma4') or not gemma.runtime_ready():
            raise ValueError('Require idle verified E4B runtime.')
        report['warmup'] = ensemble.complete('4b', gemma._messages([{'start': 0, 'end': 1, 'text': 'こんにちは。'}], '', None), 0, None, 1024)
        if report['warmup']['status'] != 'ok':
            raise ValueError('Translation warmup failed; no retry.')
        for source, baseline_case, row in zip(asr['cases'], cases, report['cases'], strict=True):
            row.update(asr=source['asr'], baseline=baseline_case['baseline'], status='error', text=None,
                       translation_seconds=0, new_translation_seconds=0, reused=False)
            if source['asr']['status'] == 'ok':
                unit = dict(start=0, end=source['duration_seconds'], text=source['asr']['text'])
                messages = gemma._messages([unit], '', None)
                prompt_hash = ensemble.digest(messages)
                row.update(source=unit, prompt_sha256=prompt_hash)
                if prompt_hash in cache:
                    saved = cache[prompt_hash]
                    row.update(status='ok', text=saved['text'], reused=True, cache_provenance=saved['provenance'],
                               translation_seconds=saved['translation_seconds'])
                else:
                    result = ensemble.complete('4b', messages, 0, None, 1024)
                    row.update(status=result['status'], text=result.get('text'), result=result,
                               translation_seconds=result['inference_seconds'], new_translation_seconds=result['inference_seconds'])
            else:
                row['error'] = 'ASR error retained without translation.'
            write(path, report)
        report['status'] = 'complete_with_errors' if any(r['status'] != 'ok' for r in report['cases']) else 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        process = gemma._process
        owner = process is None or gemma._owner_pid == os.getpid()
        gemma.shutdown()
        report['cleanup'] = dict(owned_process_pid=process.pid if process else None,
            owner_matches_this_evaluator=owner,
            owned_process_released=process is None or process.poll() is not None,
            port_free=not previous.translation.port_accepting_connections(18767))
        if not owner or not report['cleanup']['owned_process_released'] or not report['cleanup']['port_free']:
            report['status'] = 'failed'
        gemma.PORT, gemma.BASE = old_port, old_base
        for row in report['cases']:
            if row['status'] == 'pending':
                row.update(status='error', error='Run stopped before this case; retained in denominator.')
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(path, report)
    return report


def self_test():
    import tempfile
    from unittest.mock import patch, Mock
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'clip.wav'
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(bytes(32000))
        raw, duration = pcm(path)
        assert len(raw) == 32000 and duration == 1
        report_path = Path(directory) / 'report.json'
        write(report_path, {'case_count': 20})
        assert json.loads(report_path.read_text())['case_count'] == 20
        with patch.object(socket.socket, 'connect_ex', return_value=0):
            try:
                free_ports()
            except ValueError:
                pass
            else:
                raise AssertionError('Occupied model port accepted.')
        stream = Mock()
        stream.result = Mock(text='例', tokens=['例'], timestamps=[.9])
        model = Mock()
        model.create_stream.return_value = stream
        result = recognize(model, raw, duration)
        rate, waveform = stream.accept_waveform.call_args.args
        assert rate == 16000 and waveform.dtype == np.float32 and len(waveform) == 44800
        assert result['padded_duration_seconds'] == 2.8 and result['text'] == '例'
        # Exercise actual parent failure/cleanup using a fake worker; never construct ONNX.
        smoke = Path(directory) / 'smoke.json'
        write(smoke, {'status': 'complete'})
        gate = Path(directory) / 'gate.json'
        write(gate, {'decision': 'pass', 'smoke_sha256': sha(smoke)})
        output = Path(directory) / 'asr.json'
        protocol = {'outputs': {'smoke': str(smoke), 'asr': str(output)},
                    'readiness_review': {'path': str(gate), 'sha256': sha(gate)}}
        protocol_path = Path(directory) / 'protocol.json'
        write(protocol_path, protocol)
        process = Mock(pid=42, returncode=1)
        process.poll.return_value = 1
        def failed_worker(**kwargs):
            saved = json.loads(output.read_text())
            saved['cases'][0]['asr'] = {'status': 'ok', 'text': 'saved before failure'}
            saved['status'] = 'failed'
            write(output, saved)
            return b'', b'controlled worker error'
        process.communicate.side_effect = failed_worker
        with (patch(__name__ + '.validate', return_value=(protocol, [{'id': i} for i in range(20)])),
              patch(__name__ + '.free_ports'), patch.object(subprocess, 'Popen', return_value=process) as spawn):
            try:
                asr_run('asr', protocol_path)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Failed worker accepted.')
            saved = json.loads(output.read_text())
            assert len(saved['cases']) == 20 and saved['cases'][0]['asr']['text'] == 'saved before failure'
            assert saved['status'] == 'failed' and saved['cleanup']['owned_worker_released']
            before = output.read_bytes()
            try:
                asr_run('asr', protocol_path)
            except FileExistsError:
                pass
            else:
                raise AssertionError('Existing ASR report overwritten.')
            assert before == output.read_bytes() and spawn.call_count == 1
    print('CPU self-test passed: PCM/padding, occupied-port guard, owned worker failure/cleanup, saved partial20 and overwrite refusal; no model construction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--validate-only', action='store_true')
    group.add_argument('--stage', choices=['smoke', 'asr', 'translate'])
    group.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.worker:
        worker(args.worker, args.protocol); return
    if args.validate_only:
        _, data = validate(args.protocol)
        print(json.dumps({'validated_cases': len(data), 'inference_started': False})); return
    with (BASE / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = translate_run(args.protocol) if args.stage == 'translate' else asr_run(args.stage, args.protocol)
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']), 'cleanup': report['cleanup']}))
    if report['status'] not in ('complete', 'complete_with_errors'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
