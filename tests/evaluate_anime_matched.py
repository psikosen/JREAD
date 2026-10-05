#!/usr/bin/env python3
"""Caption-reference diagnostic: compare fixed Whisper/Gemma source conditions."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import evaluate_caption_verifiers as processes  # noqa: E402
import evaluate_index_translate as gemma_runtime  # noqa: E402
import evaluate_translation_ensemble as ensemble  # noqa: E402

BASE = Path('/media/psikosen/newvolume/jread-anime-matched')
PROTOCOL = ROOT / 'tests/fixtures/accuracy-anime-matched.json'
TARGET_PROTOCOL = ROOT / 'tests/fixtures/accuracy-anime-matched-targeted.json'
RUNNER = Path(__file__).resolve()
LOCK = Path('/media/psikosen/newvolume/jread-verifier/evaluation.lock')
HOST, PORT = '127.0.0.1', 18767
TIMEOUT = 1800
ASR_MODEL = Path.home() / '.local/share/JREAD/models/large-v3-turbo'
ASR_SETTINGS = {
    'model': 'large-v3-turbo', 'device': 'cuda', 'device_index': 0,
    'compute_type': 'int8_float16', 'cpu_threads': 4, 'num_workers': 1,
    'local_files_only': True, 'language': 'ja', 'task': 'transcribe',
    'vad_filter': True, 'beam_size': 5, 'condition_on_previous_text': False,
    'temperature': 0,
}
GEMMA_SETTINGS = {
    'preset': 'gemma4', 'device': 'CPU', 'threads': 8,
    'context_tokens': 4096, 'port': PORT, 'temperature': 0,
    'max_tokens': 1024, 'thinking': False, 'context': '', 'glossary': None,
}
PHASES = ('validate', 'asr-calibration', 'calibration-baseline', 'calibration-targeted',
          'asr-validation', 'validation')
PHASE_OUTPUTS = {
    'asr-calibration': 'asr_calibration', 'calibration-baseline': 'calibration_baseline',
    'calibration-targeted': 'calibration_targeted', 'asr-validation': 'asr_validation',
    'validation': 'validation',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def payload_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def bound(item):
    path = Path(item['path'])
    if not path.is_file() or sha(path) != item['sha256']:
        raise ValueError(f'Frozen binding changed: {path}')
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path), 'sha256': sha(path)}


def source_row(text, start, end):
    return {'start': start, 'end': end, 'text': text}


def translated_request(source, system_instruction=None):
    messages = ensemble.gemma._messages([source], '', None)
    if system_instruction:
        messages[0]['content'] += '\n' + system_instruction
    return ensemble.payload('4b', messages, 0, None, 1024)


def segment_text(raw_segments):
    """Join recognized segments in order; never use caption text to repair ASR."""
    if not isinstance(raw_segments, list):
        raise ValueError('ASR segments must be a list.')
    pieces = []
    for row in raw_segments:
        if not isinstance(row, dict) or not isinstance(row.get('text'), str):
            raise ValueError('Malformed ASR segment.')
        text = row['text'].strip()
        if text:
            pieces.append(text)
    return ''.join(pieces)


def expected_text_slots(phase, ids):
    prompts = ('baseline',) if phase == 'calibration-baseline' else (
        ('targeted',) if phase == 'calibration-targeted' else ('baseline', 'targeted'))
    partitions = ids
    return [(case_id, condition, prompt) for case_id in partitions
            for prompt in prompts for condition in ('clean-caption', 'recognized-asr')]


def phase_kind(phase):
    return 'asr' if phase in ('asr-calibration', 'asr-validation') else 'gemma'


def asr_phase_ids(protocol, phase):
    if phase == 'asr-calibration':
        return protocol['case_ids']['calibration']
    if phase == 'asr-validation':
        return protocol['case_ids']['validation']
    raise ValueError(f'Not an ASR phase: {phase}')


def released_with_free_ports(cleanup):
    return cleanup.get('released') is True and all(
        cleanup.get('private_ports_free', {}).get(str(port)) is True for port in (18767, 18768))


def protocol_data(protocol_path=PROTOCOL):
    if Path(protocol_path).resolve() != PROTOCOL.resolve():
        raise ValueError('Only the single frozen base protocol is accepted.')
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get('status') != 'frozen' or protocol.get('runner_sha256') != sha(RUNNER):
        raise ValueError('Require frozen protocol bound to this runner.')
    expected_runtime = protocol.get('runtime', {})
    if Path(sys.executable).resolve() != Path(expected_runtime.get('python', '')).resolve():
        raise ValueError('Run with the pinned ASR virtual-environment Python.')
    ct2 = tuple(int(piece) for piece in expected_runtime.get('versions', {}).get('ctranslate2', '0').split('.')[:3])
    if ct2 < (4, 8, 2):
        raise ValueError('Pinned CTranslate2 predates verified Whisper decoding behavior.')
    for package, version in expected_runtime.get('versions', {}).items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Pinned package version differs: {package}')
    settings = protocol.get('settings', {})
    if settings.get('asr') != ASR_SETTINGS or settings.get('gemma') != GEMMA_SETTINGS:
        raise ValueError('ASR or Gemma settings differ from this fixed experiment.')
    required = {
        str(RUNNER), str(ROOT / 'tests/evaluate_index_translate.py'),
        str(ROOT / 'tests/evaluate_translation_ensemble.py'),
        str(ROOT / 'tests/evaluate_asr_recovery_translation.py'),
        str(ROOT / 'tests/evaluate_caption_verifiers.py'), str(ROOT / 'tests/asr_cuda_pilot.py'),
        str(ROOT / 'tests/evaluate_reazon.py'),
        str(ROOT / 'tests/evaluate_streaming_combinations.py'),
        str(ROOT / 'tests/evaluate_streaming_translation.py'),
        str(ROOT / 'jread/gemma.py'), str(ROOT / 'jread/store.py'), str(ROOT / 'docs/runtime.json'),
        str(ROOT / 'docs/whisper-turbo-model.json'),
        str(ensemble.gemma.SERVER_PATH), str(ensemble.gemma.model_path('gemma4')),
        str(ASR_MODEL / 'model.bin'), str(ASR_MODEL / '.verified'),
        str(ASR_MODEL / 'config.json'), str(ASR_MODEL / 'preprocessor_config.json'),
        str(ASR_MODEL / 'tokenizer.json'), str(ASR_MODEL / 'vocabulary.json'),
        str(protocol['inputs']['audio_manifest']['path']), str(protocol['inputs']['clean_text']['path']),
    }
    actual = set()
    for row in protocol.get('bindings', []):
        if not isinstance(row, dict) or set(row) != {'path', 'sha256'}:
            raise ValueError('Malformed frozen binding.')
        path = str(Path(row['path']))
        if sha(path) != row['sha256']:
            raise ValueError(f'Frozen helper/model/input changed: {path}')
        actual.add(path)
    runtime_bindings = expected_runtime.get('bindings', [])
    if not isinstance(runtime_bindings, list) or not runtime_bindings:
        raise ValueError('Pinned ASR runtime bindings must be explicit and nonempty.')
    runtime_actual = set()
    for runtime_binding in runtime_bindings:
        if not isinstance(runtime_binding, dict) or set(runtime_binding) != {'path', 'sha256'}:
            raise ValueError('Malformed pinned ASR runtime binding.')
        path = str(Path(runtime_binding['path']))
        if sha(path) != runtime_binding['sha256']:
            raise ValueError(f'Pinned ASR runtime file changed: {path}')
        runtime_actual.add(path)
    if not required.issubset(actual) or not runtime_actual.issubset(actual):
        raise ValueError(f'Frozen bindings omit required helpers, models, runtime or inputs: {sorted((required | runtime_actual)-actual)}')
    model = protocol.get('asr_model', {})
    if (Path(model.get('directory', '')) != ASR_MODEL or
            model.get('revision') != '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf' or
            model.get('model_bin_sha256') != 'e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da' or
            (ASR_MODEL / '.verified').read_text().strip() != model.get('revision') or
            sha(ASR_MODEL / 'model.bin') != model.get('model_bin_sha256')):
        raise ValueError('Whisper Turbo model/revision bytes differ from the pinned local model.')
    if protocol.get('gemma_model') != gemma_runtime.current_gemma():
        raise ValueError('Pinned Gemma E4B model/runtime provenance differs.')
    if protocol.get('lock') != str(LOCK) or protocol.get('private_ports') != [18767, 18768]:
        raise ValueError('Global lock or private-port policy differs.')
    expected_folds = protocol.get('case_ids', {})
    if len(expected_folds.get('calibration', [])) != 5 or len(expected_folds.get('validation', [])) != 4:
        raise ValueError('Require the fixed five calibration and four validation clips.')
    if set(expected_folds['calibration']) & set(expected_folds['validation']):
        raise ValueError('Calibration and validation IDs must be disjoint.')
    audio = bound(protocol['inputs']['audio_manifest'])
    text = bound(protocol['inputs']['clean_text'])
    if audio.get('status') != 'frozen_caption_window_audio_unverified' or text.get('status') != 'frozen_caption_text_only_audio_correspondence_unverified':
        raise ValueError('Inputs must retain the explicit unverified-audio diagnostic status.')
    rows = audio.get('rows', [])
    cases = text.get('cases', [])
    if [r.get('id') for r in rows] != expected_folds['calibration'] + expected_folds['validation']:
        raise ValueError('Audio IDs/order must be the frozen Nakano calibration then Saga validation set.')
    if [r.get('id') for r in cases] != expected_folds['calibration'] + expected_folds['validation']:
        raise ValueError('Clean-caption IDs/order differ from the audio manifest.')
    if any(set(r) != {'id', 'film_group', 'audio'} for r in rows):
        raise ValueError('ASR input must contain only audio metadata, never captions or reference text.')
    if any(set(r) != {'id', 'partition', 'text', 'start', 'end'} for r in cases):
        raise ValueError('Clean-text input must contain only Japanese caption text and timing, no rubric or English.')
    for index, (audio_row, clean) in enumerate(zip(rows, cases, strict=True)):
        partition = 'calibration' if index < 5 else 'validation'
        expected_group = 'nakano_calibration' if partition == 'calibration' else 'saga_karatsu_validation'
        a = audio_row['audio']
        if (audio_row['id'] != clean['id'] or audio_row.get('film_group') != expected_group or
                clean.get('partition') != partition or not isinstance(clean.get('text'), str) or not clean['text'].strip() or
                clean.get('start') != 0 or clean.get('end') != a.get('duration_seconds') or
                a.get('sample_rate') != 16000 or a.get('sample_count') <= 0 or
                not math.isclose(a.get('duration_seconds', -1), a['sample_count'] / 16000, abs_tol=1/16000) or
                not Path(a.get('path', '')).is_absolute() or
                not math.isclose(a.get('source_end_seconds', -1)-a.get('source_start_seconds', -1),
                                 a['duration_seconds'], abs_tol=1/16000) or
                sha(a['path']) != a['sha256']):
            raise ValueError(f'Invalid/misbound source-only clip {audio_row.get("id")}.')
        with wave.open(a['path'], 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype(), wav.getnframes()) != (1, 2, 16000, 'NONE', a['sample_count']):
                raise ValueError(f'Clip format or sample count changed: {audio_row["id"]}.')
    output_dirs = protocol.get('output_dirs', {})
    if set(output_dirs) != set(PHASE_OUTPUTS.values()) or any(
            Path(output_dirs[name]) != BASE / name for name in output_dirs):
        raise ValueError('Output directories must use the non-overwriting external layout.')
    return protocol, audio, text


def validate_target_spec(path, protocol):
    spec = json.loads(Path(path).read_text())
    base_binding = {'path': str(PROTOCOL), 'sha256': sha(PROTOCOL)}
    if (spec.get('status') != 'frozen' or spec.get('runner_sha256') != sha(RUNNER) or
            spec.get('base_protocol') != base_binding or
            not isinstance(spec.get('system_instruction'), str) or not spec['system_instruction'].strip()):
        raise ValueError('Target prompt must be separately frozen and bind the calibration protocol.')
    report_ref = spec.get('calibration_report')
    grade_refs = spec.get('calibration_grades')
    if not isinstance(report_ref, dict):
        raise ValueError('Target prompt lacks a frozen calibration report binding.')
    if Path(report_ref['path']) != Path(protocol['output_dirs']['calibration_baseline']) / 'report.json':
        raise ValueError('Targeted prompt must bind the completed calibration baseline report.')
    bound(report_ref)
    validate_dependency(report_ref['path'], protocol, 'calibration-baseline')
    grade_path = BASE / 'review/calibration_baseline-grades.json'
    if (not isinstance(grade_refs, list) or len(grade_refs) != 1 or
            Path(grade_refs[0]['path']) != grade_path):
        raise ValueError('Target prompt must bind the reconciled calibration-only grades.')
    grades = bound(grade_refs[0])
    grade_slots = [(row.get('case_id'), row.get('condition'), row.get('prompt'))
                   for row in grades.get('slots', [])]
    if (grades.get('status') != 'frozen_reconciled_grades' or
            grades.get('phase') != 'calibration_baseline' or
            grade_slots != expected_text_slots('calibration-baseline', protocol['case_ids']['calibration']) or
            any(row.get('status') not in ('pass', 'fail', 'unclear') for row in grades['slots'])):
        raise ValueError('Target selection grades must cover only the five calibration cases.')
    return spec


def validate_dependency(report_path, protocol, phase):
    report = json.loads(Path(report_path).read_text())
    cleanup = report.get('cleanup', {})
    is_asr = phase.startswith('asr-')
    released = cleanup.get('owned_released') is True if is_asr else cleanup.get('released') is True
    expected_ids = (protocol['case_ids']['calibration'] if phase in ('asr-calibration', 'calibration-baseline', 'calibration-targeted')
                    else protocol['case_ids']['validation'])
    expected_slots = expected_text_slots(phase, expected_ids) if not is_asr else None
    if (report.get('status') not in ('complete', 'complete_with_errors') or report.get('protocol') != binding(PROTOCOL) or
            report.get('phase') != phase or not released):
        raise ValueError(f'Required completed/released {phase} report is not bound to this protocol.')
    if report.get('expected_ids') != expected_ids:
        raise ValueError(f'{phase} report does not retain the full ordered case denominator.')
    if is_asr:
        rows = report.get('rows', [])
        if ([row.get('id') for row in rows] != expected_ids or report.get('settings') != ASR_SETTINGS or
                report.get('inference_performed') is not True or report.get('model_unloaded') is not True or
                report.get('inputs', {}).get('audio_manifest') != protocol['inputs']['audio_manifest']):
            raise ValueError('ASR dependency lacks exact coverage/settings/inference/unload evidence.')
    else:
        clean_packet = bound(protocol['inputs']['clean_text'])
        clean_by_id = {row['id']: row for row in clean_packet['cases']}
        asr_phase = 'asr-validation' if phase == 'validation' else 'asr-calibration'
        asr_path = Path(protocol['output_dirs'][PHASE_OUTPUTS[asr_phase]]) / 'report.json'
        asr_report = json.loads(asr_path.read_text())
        asr_by_id = {row['id']: row for row in asr_report.get('rows', [])}
        target_system = None
        if phase != 'calibration-baseline':
            ref = report.get('target_spec')
            if not isinstance(ref, dict):
                raise ValueError(f'{phase} lacks its frozen target prompt binding.')
            target_spec = bound(ref)
            target_system = target_spec['system_instruction']
        actual_slots = [(case.get('id'), slot.get('condition'), slot.get('prompt'))
                        for case in report.get('cases', []) for slot in case.get('slots', [])]
        if (actual_slots != expected_slots or report.get('expected_slots') != len(expected_slots) or
                report.get('settings') != GEMMA_SETTINGS or report.get('model_provenance') != protocol['gemma_model'] or
                report.get('inference_performed') is not True or report.get('model_unloaded') is not True or
                report.get('inputs', {}).get('clean_text') != protocol['inputs']['clean_text']):
            raise ValueError(f'{phase} dependency lacks exact slots/settings/model/cleanup evidence.')
        if not released_with_free_ports(cleanup):
            raise ValueError(f'{phase} private ports were not verified free after cleanup.')
        for case, expected_id in zip(report.get('cases', []), expected_ids, strict=True):
            if case.get('id') != expected_id:
                raise ValueError(f'{phase} case order changed.')
            clean = clean_by_id[expected_id]
            asr_row = asr_by_id.get(expected_id, {})
            for slot in case.get('slots', []):
                condition, prompt = slot['condition'], slot['prompt']
                text = clean['text'] if condition == 'clean-caption' else asr_row.get('text', '')
                if condition == 'recognized-asr' and asr_row.get('status') != 'ok':
                    if slot.get('status') != 'error' or slot.get('attempted') is not False:
                        raise ValueError(f'{phase} must retain a failed ASR slot without rendering another source.')
                    continue
                expected_request = translated_request(source_row(text, clean['start'], clean['end']),
                    target_system if prompt == 'targeted' else None)
                raw_request = payload_bytes(expected_request)
                if (slot.get('request') != expected_request or
                        slot.get('request_sha256') != hashlib.sha256(raw_request).hexdigest() or
                        slot.get('request_bytes') != len(raw_request)):
                    raise ValueError(f'{phase} request/source/prompt bytes differ for {expected_id}/{condition}/{prompt}.')
                if slot.get('status') == 'ok':
                    raw_response = slot.get('raw_response_json_utf8')
                    if (not isinstance(raw_response, str) or
                            hashlib.sha256(raw_response.encode('utf-8')).hexdigest() != slot.get('response_sha256') or
                            len(raw_response.encode('utf-8')) != slot.get('response_bytes') or
                            not isinstance(slot.get('text'), str) or not slot['text'].strip()):
                        raise ValueError(f'{phase} response bytes/text are incomplete for {expected_id}/{condition}/{prompt}.')
    if is_asr and not all(cleanup.get('private_ports_free', {}).get(str(port)) is True for port in (18767, 18768)):
        raise ValueError(f'{phase} private ports were not verified free after cleanup.')
    if phase == 'asr-validation' and not isinstance(report.get('dependencies', {}).get('target_spec'), dict):
        raise ValueError('Validation ASR must bind the already-frozen calibration target spec.')
    return report, binding(report_path)


def validate_phase(protocol, phase, target_path=None):
    p, audio, text = protocol_data(protocol)
    deps = {}
    target = None
    if phase != 'validate':
        out = Path(p['output_dirs'][PHASE_OUTPUTS[phase]])
        if out.exists():
            raise ValueError(f'Refusing to overwrite prior output: {out}')
    if phase == 'asr-validation':
        target_path = target_path or TARGET_PROTOCOL
        target = validate_target_spec(target_path, p)
        deps['target_spec'] = binding(target_path)
        asr_report = None
    if phase in ('calibration-baseline', 'calibration-targeted', 'validation'):
        if phase == 'validation':
            target_path = target_path or TARGET_PROTOCOL
            target = validate_target_spec(target_path, p)
            deps['target_spec'] = binding(target_path)
        asr_phase = 'asr-validation' if phase == 'validation' else 'asr-calibration'
        asr_report, asr_binding = validate_dependency(
            Path(p['output_dirs'][PHASE_OUTPUTS[asr_phase]]) / 'report.json', p, asr_phase)
        deps[asr_phase.replace('-', '_')] = asr_binding
        if phase == 'validation' and asr_report.get('dependencies', {}).get('target_spec') != deps['target_spec']:
            raise ValueError('Validation ASR report does not bind the frozen targeted prompt spec.')
    else:
        asr_report = None
    if phase in ('calibration-targeted', 'validation') and target is None:
        if target_path is None:
            target_path = TARGET_PROTOCOL
        target = validate_target_spec(target_path, p)
        deps['target_spec'] = binding(target_path)
    if phase == 'validation':
        for prior in ('calibration-baseline', 'calibration-targeted'):
            report_path = Path(p['output_dirs'][PHASE_OUTPUTS[prior]]) / 'report.json'
            _, deps[prior.replace('-', '_')] = validate_dependency(report_path, p, prior)
    return p, audio, text, asr_report, target, deps


def asr_worker(protocol_path, output_path):
    import numpy as np
    from faster_whisper import WhisperModel
    protocol, audio_manifest, _ = protocol_data(protocol_path)
    report = json.loads(Path(output_path).read_text())
    expected_ids = report.get('expected_ids')
    rows_by_id = {row['id']: row for row in audio_manifest['rows']}
    if expected_ids not in (protocol['case_ids']['calibration'], protocol['case_ids']['validation']):
        raise ValueError('Worker received invalid phase-specific audio IDs.')
    audio_manifest = {'rows': [rows_by_id[case_id] for case_id in expected_ids]}
    report.update(status='running', model_loaded=False, inference_performed=False, rows=[])
    write(output_path, report)
    model = None
    try:
        started = time.perf_counter()
        model = WhisperModel(str(ASR_MODEL), device='cuda', device_index=0,
                             compute_type='int8_float16', cpu_threads=4, num_workers=1,
                             local_files_only=True)
        report.update(model_loaded=True, model_load_seconds=time.perf_counter()-started,
                      model_device=str(model.model.device), model_device_index=str(model.model.device_index),
                      model_compute_type=str(model.model.compute_type), inference_performed=True)
        if (model.model.device != 'cuda' or model.model.compute_type != 'int8_float16' or
                list(model.model.device_index) != [0]):
            raise ValueError('Actual ASR device/compute type differs from the frozen request.')
        write(output_path, report)
        failed = False
        for item in audio_manifest['rows']:
            a = item['audio']; row = {'id': item['id'], 'status': 'error', 'audio_sha256': a['sha256']}
            try:
                with wave.open(a['path'], 'rb') as wav:
                    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
                started = time.perf_counter()
                segments, info = model.transcribe(samples, language='ja', task='transcribe',
                    vad_filter=True, beam_size=5, condition_on_previous_text=False, temperature=0)
                rows = []
                for seg in segments:
                    rows.append({'start': float(seg.start), 'end': float(seg.end), 'text': str(seg.text),
                                 'avg_logprob': float(seg.avg_logprob), 'no_speech_prob': float(seg.no_speech_prob)})
                transcript = segment_text(rows)
                if not transcript:
                    raise ValueError('Whisper returned no nonempty transcript text.')
                row.update(status='ok', segments=rows, text=transcript,
                           language=str(info.language), language_probability=float(info.language_probability),
                           inference_seconds=time.perf_counter()-started,
                           score_note='Whisper segment scores are descriptive model outputs, not calibrated correctness confidence.')
            except Exception as error:
                failed = True; row['error'] = f'{type(error).__name__}: {error}'
            report['rows'].append(row); report['status'] = 'running'; write(output_path, report)
        report['status'] = 'complete_with_errors' if failed else 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        unload_error = None
        if model is not None:
            try:
                model.model.unload_model()
                report['model_unloaded'] = True
            except Exception as error:
                unload_error = f'{type(error).__name__}: {error}'
                report['model_unloaded'] = False
            del model
        import gc
        gc.collect()
        if unload_error:
            report['unload_error'] = unload_error
            report['status'] = 'failed'
        write(output_path, report)


def run_asr(protocol, audio_manifest, output_dir, dependencies):
    out = output_dir / 'report.json'
    worker_out = output_dir / 'worker.json'
    phase = 'asr-calibration' if output_dir.name == 'asr_calibration' else 'asr-validation'
    allowed_ids = asr_phase_ids(protocol, phase)
    audio_manifest = {'rows': [r for r in audio_manifest['rows'] if r['id'] in allowed_ids]}
    report = {'status': 'running', 'phase': phase, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': binding(PROTOCOL), 'dependencies': dependencies,
              'expected_ids': allowed_ids, 'model_loaded': False,
              'inference_performed': False, 'heldout_rows_used': 0, 'production_changed': False,
              'settings': ASR_SETTINGS, 'runtime_versions': protocol['runtime']['versions'],
              'inputs': {'audio_manifest': binding(protocol['inputs']['audio_manifest']['path'])},
              'python': sys.executable, 'worker': None, 'rows': [], 'cleanup': {}}
    write(out, report)
    write(worker_out, {'expected_ids': allowed_ids})
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HF_', 'TRANSFORMERS_', 'LLAMA_'))
           and k not in ('HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               LD_LIBRARY_PATH='/usr/local/cuda-12.8/lib64')
    child = subprocess.Popen([sys.executable, str(RUNNER), '--worker-asr', str(PROTOCOL), str(worker_out)],
                             cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, start_new_session=True)
    report['worker_pid'] = child.pid; write(out, report)
    members_before_cleanup = []
    stderr = b''
    try:
        _, stderr = child.communicate(timeout=TIMEOUT)
    except BaseException:
        members_before_cleanup = processes.group_members(child.pid)
        processes.signal_group(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            processes.signal_group(child.pid, signal.SIGKILL); child.wait(timeout=5)
        raise
    finally:
        remaining = processes.group_members(child.pid)
        if remaining:
            members_before_cleanup = members_before_cleanup or remaining
            processes.signal_group(child.pid, signal.SIGTERM)
            time.sleep(.2)
            remaining = processes.group_members(child.pid)
        if remaining:
            processes.signal_group(child.pid, signal.SIGKILL)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
            remaining = processes.group_members(child.pid)
        worker = json.loads(worker_out.read_text()) if worker_out.exists() else {}
        try:
            from asr_cuda_pilot import gpu_snapshot
            gpu_after = gpu_snapshot()
            worker_gpu_pids = [row['pid'] for row in gpu_after['processes'] if row['pid'] == child.pid]
        except Exception as error:
            gpu_after = {'error': f'{type(error).__name__}: {error}'}
            worker_gpu_pids = [child.pid]
        report.update(worker=worker, rows=worker.get('rows', []), model_loaded=worker.get('model_loaded', False),
                      inference_performed=worker.get('inference_performed', False), worker_exit_code=child.returncode,
                      worker_stderr=(stderr or b'').decode('utf-8', 'replace')[-4000:],
                      model_unloaded=worker.get('model_unloaded') is True,
                      cleanup={'owned_released': child.poll() is not None and not remaining and not worker_gpu_pids,
                               'group_before_cleanup': members_before_cleanup, 'group_remaining': remaining,
                               'worker_gpu_pids_after_exit': worker_gpu_pids, 'gpu_after_exit': gpu_after,
                               'private_ports_free': private_ports_free()})
        if [r.get('id') for r in report['rows']] != report['expected_ids']:
            report['status'] = 'failed'
            report['error'] = 'Worker did not preserve every ordered clip slot for this phase.'
        elif (report['cleanup']['owned_released'] and child.returncode == 0 and report['model_unloaded']
              and all(report['cleanup']['private_ports_free'].values())):
            report['status'] = worker.get('status') if worker.get('status') in ('complete', 'complete_with_errors') else 'failed'
        else:
            report['status'] = 'failed'
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(out, report)
    return report


def private_ports_free():
    checks = {}
    for port in (18767, 18768):
        with socket.socket() as probe:
            checks[str(port)] = probe.connect_ex((HOST, port)) != 0
    return checks


def source_for(row, text):
    return source_row(text, row['start'], row['end'])


def cache_from_report(report, report_path):
    cache = {}
    for case in report.get('cases', []):
        for slot in case.get('slots', []):
            if slot.get('status') == 'ok':
                raw = payload_bytes(slot['request'])
                if hashlib.sha256(raw).hexdigest() == slot.get('request_sha256'):
                    cache[raw] = slot | {'source_report': binding(report_path)}
    return cache


def run_gemma(protocol, phase, text_packet, asr_report, target, dependencies, output_dir):
    report_path = output_dir / 'report.json'
    rows_by_id = {x['id']: x for x in text_packet['cases']}
    asr_by_id = {x['id']: x for x in asr_report.get('rows', [])} if asr_report else {}
    ids = protocol['case_ids']['calibration'] if phase.startswith('calibration-') else protocol['case_ids']['validation']
    if phase != 'validation' and phase == 'calibration-targeted':
        prior_path = Path(protocol['output_dirs']['calibration_baseline'])/'report.json'
        prior = json.loads(prior_path.read_text())
        cache = cache_from_report(prior, prior_path)
    elif phase == 'validation':
        prior_base_path = Path(protocol['output_dirs']['calibration_baseline'])/'report.json'
        prior_target_path = Path(protocol['output_dirs']['calibration_targeted'])/'report.json'
        prior_base = json.loads(prior_base_path.read_text())
        prior_target = json.loads(prior_target_path.read_text())
        cache = cache_from_report(prior_base, prior_base_path) | cache_from_report(prior_target, prior_target_path)
    else:
        cache = {}
    prompts = ('baseline',) if phase == 'calibration-baseline' else (
        ('targeted',) if phase == 'calibration-targeted' else ('baseline', 'targeted'))
    report = {'status':'running','phase':phase,'started_at_utc':datetime.now(timezone.utc).isoformat(),
              'protocol':binding(PROTOCOL),'dependencies':dependencies,
              'expected_slots':len(ids)*2*len(prompts),'expected_ids':ids,'cases':[],
              'model_loaded':False,'inference_performed':False,'model_provenance':protocol['gemma_model'],
              'settings':GEMMA_SETTINGS,'heldout_rows_used':0,'production_changed':False,
              'inputs': {'audio_manifest': binding(protocol['inputs']['audio_manifest']['path']),
                         'clean_text': binding(protocol['inputs']['clean_text']['path'])},
              'target_spec':dependencies.get('target_spec'), 'cleanup':{}}
    write(report_path,report)
    old_port = (ensemble.gemma.PORT, ensemble.gemma.BASE)
    server = None; error = None
    slots=[]
    try:
        server, _ = gemma_runtime.start_gemma()
        report.update(model_loaded=True,model_load_seconds=server['load_seconds'],server=server)
        cache_req = cache.copy()
        for case_id in ids:
            clean = rows_by_id[case_id]
            asr = asr_by_id.get(case_id, {})
            case={'id':case_id,'caption_reference_diagnostic':True,'slots':[]}
            for prompt in prompts:
                for condition, text in (('clean-caption',clean['text']),('recognized-asr',asr.get('text',''))):
                    slot={'condition':condition,'prompt':prompt,'status':'error','text':None,'attempted':False}
                    try:
                        if condition=='recognized-asr' and (asr.get('status')!='ok' or not text.strip()):
                            raise ValueError('ASR text unavailable; preserve this slot as error.')
                        request=translated_request(source_for(clean,text),target['system_instruction'] if prompt=='targeted' else None)
                        raw=payload_bytes(request); digest=hashlib.sha256(raw).hexdigest()
                        slot.update(request=request,request_sha256=digest,request_bytes=len(raw),attempted=True)
                        cached=cache_req.get(raw)
                        if cached is not None:
                            slot.update(status='ok',text=cached['text'],seconds=0.0,
                                        historical_seconds=cached.get('historical_seconds',cached.get('seconds',0.0)),
                                        reused=True,source_report=cached.get('source_report'))
                            for key in ('raw_response','raw_response_json_utf8','response_sha256',
                                        'response_bytes','usage','stop_reason'):
                                if key in cached: slot[key]=cached[key]
                        else:
                            result=gemma_runtime.call(ensemble.gemma.BASE,request,auth=True,strict_json=True)
                            slot.update(result,reused=False)
                            if result.get('status')=='ok':
                                cache_req[raw]=slot
                        case['slots'].append(slot); slots.append(slot)
                    except Exception as exc:
                        slot['error']=f'{type(exc).__name__}: {exc}'; case['slots'].append(slot); slots.append(slot)
            report['cases'].append(case); report['inference_performed']=report['inference_performed'] or any(s.get('attempted') and not s.get('reused') for s in case['slots'])
            write(report_path,report)
        report['status']='complete_with_errors' if any(s['status']!='ok' for s in slots) else 'complete'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        report.update(status='failed',error=error)
        raise
    finally:
        if server is not None or getattr(ensemble.gemma, '_process', None) is not None:
            report['cleanup']=gemma_runtime.stop_gemma(old_port)
        else:
            report['cleanup']={'released':True,'no_server_started':True}
        report['cleanup']['private_ports_free'] = private_ports_free()
        report['model_unloaded']=report['cleanup'].get('released',False)
        report['sequential_inference_seconds']=sum(float(s.get('seconds',0)) for s in slots if not s.get('reused'))
        report['reused_historical_seconds']=sum(float(s.get('historical_seconds',0)) for s in slots if s.get('reused'))
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        if report.get('status') in ('running','complete','complete_with_errors') and not released_with_free_ports(report['cleanup']):
            report['status']='failed'; report['error']='Gemma server cleanup failed.'
        write(report_path,report)
    return report


def run_phase(phase, protocol_path=PROTOCOL, target_path=None):
    protocol,audio,text,asr,target,deps=validate_phase(protocol_path,phase,target_path)
    if phase=='validate':
        return {'status':'valid','model_loaded':False,'inference_performed':False,
                'case_counts':{'calibration':5,'validation':4},'audio_unverified':True}
    output_dir=Path(protocol['output_dirs'][PHASE_OUTPUTS[phase]])
    lock=None; created=False
    old_term=signal.getsignal(signal.SIGTERM)
    def on_term(_signum, _frame):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,on_term)
    old_deadline=gemma_runtime.DEADLINE
    try:
        gemma_runtime.DEADLINE=time.monotonic()+TIMEOUT
        LOCK.parent.mkdir(parents=True,exist_ok=True)
        lock=LOCK.open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for port in (18767,18768):
            with socket.socket() as probe:
                if probe.connect_ex((HOST,port))==0:
                    raise ValueError(f'Private port {port} is occupied; refusing overlap.')
        if phase in ('asr-calibration', 'asr-validation'):
            from asr_cuda_pilot import gpu_snapshot
            gpu=gpu_snapshot()
            if not any(row['index']==0 and row['free_mib']>=4096 for row in gpu['gpus']):
                raise RuntimeError('Require at least 4 GiB free on GPU 0; no resident process will be stopped.')
        output_dir.mkdir(parents=True,exist_ok=False); created=True
        if phase_kind(phase) == 'asr':
            return run_asr(protocol,audio,output_dir,deps)
        else:
            return run_gemma(protocol,phase,text,asr,target,deps,output_dir)
    except BaseException:
        if created and output_dir.exists():
            report_path=output_dir/'report.json'
            try:
                report=json.loads(report_path.read_text()) if report_path.exists() else {'phase':phase}
                report.update(status='interrupted_or_failed',failure='The phase did not complete; preserve partial rows.',finished_at_utc=datetime.now(timezone.utc).isoformat())
                write(report_path,report)
            except Exception:
                pass
        raise
    finally:
        gemma_runtime.DEADLINE=old_deadline
        if lock is not None:
            fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
        signal.signal(signal.SIGTERM,old_term)


def self_test():
    segments=[{'start':0.1,'end':0.6,'text':'  ありがとう。 '},{'start':0.6,'end':1.1,'text':'はい。'}]
    assert segment_text(segments)=='ありがとう。はい。'
    assert segment_text([])==''
    clean=source_row('話します。',0.0,3.0)
    req=translated_request(clean)
    expected=ensemble.payload('4b',ensemble.gemma._messages([clean],'',None),0,None,1024)
    assert req==expected and req['temperature']==0 and req['max_tokens']==1024
    targeted=translated_request(clean,'Preserve explicit participant roles.')
    assert targeted['messages'][0]['content'].endswith('\nPreserve explicit participant roles.')
    assert targeted!=req and targeted['messages'][1]==req['messages'][1]
    assert expected_text_slots('calibration-baseline',['n1','n2'])==[
        ('n1','clean-caption','baseline'),('n1','recognized-asr','baseline'),
        ('n2','clean-caption','baseline'),('n2','recognized-asr','baseline')]
    assert expected_text_slots('validation',['s1'])==[
        ('s1','clean-caption','baseline'),('s1','recognized-asr','baseline'),
        ('s1','clean-caption','targeted'),('s1','recognized-asr','targeted')]
    assert phase_kind('asr-calibration')=='asr' and phase_kind('asr-validation')=='asr'
    assert phase_kind('validation')=='gemma'
    mock_protocol={'case_ids':{'calibration':['n1','n2','n3','n4','n5'],'validation':['s1','s2','s3','s4']}}
    assert asr_phase_ids(mock_protocol,'asr-calibration')==mock_protocol['case_ids']['calibration']
    assert asr_phase_ids(mock_protocol,'asr-validation')==mock_protocol['case_ids']['validation']
    try:
        asr_phase_ids(mock_protocol,'validation')
        raise AssertionError('A translation phase must not dispatch as ASR.')
    except ValueError:
        pass
    assert released_with_free_ports({'released':True,'private_ports_free':{'18767':True,'18768':True}})
    assert not released_with_free_ports({'released':True,'private_ports_free':{'18767':True,'18768':False}})
    assert not released_with_free_ports({'released':False,'private_ports_free':{'18767':True,'18768':True}})
    # Exercise actual dispatch/ownership with fake models and a real temporary lock.
    import tempfile
    from contextlib import ExitStack
    from unittest.mock import patch
    module = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
        temp = Path(temp)
        mock_protocol['output_dirs'] = {name: str(temp/name) for name in PHASE_OUTPUTS.values()}
        stack.enter_context(patch.object(module, 'LOCK', temp/'evaluation.lock'))
        stack.enter_context(patch.object(module, 'validate_phase', return_value=(mock_protocol, {}, {}, None, None, {})))
        probe = stack.enter_context(patch.object(socket, 'socket'))
        probe.return_value.__enter__.return_value.connect_ex.return_value = 1
        stack.enter_context(patch('asr_cuda_pilot.gpu_snapshot', return_value={'gpus':[{'index':0,'free_mib':8192}]}))
        fake_asr = stack.enter_context(patch.object(module, 'run_asr', return_value={'kind':'asr'}))
        fake_gemma = stack.enter_context(patch.object(module, 'run_gemma', return_value={'kind':'gemma'}))
        for phase in PHASE_OUTPUTS:
            assert run_phase(phase)['kind'] == phase_kind(phase)
        assert fake_asr.call_count == 2 and fake_gemma.call_count == 3
        sentinel = temp/'calibration_baseline'/'report.json'
        sentinel.write_text('existing output')
        try:
            run_phase('calibration-baseline')
            raise AssertionError('Existing output must not be replaced.')
        except FileExistsError:
            pass
        assert sentinel.read_text() == 'existing output'
        with (temp/'evaluation.lock').open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run_phase('calibration-baseline')
                raise AssertionError('An occupied global lock must stop dispatch.')
            except BlockingIOError:
                pass
        fake_gemma.side_effect = KeyboardInterrupt('test only')
        (temp/'calibration_targeted').rmdir()
        try:
            run_phase('calibration-targeted')
            raise AssertionError('Interrupt must propagate.')
        except KeyboardInterrupt:
            pass
        partial = json.loads((temp/'calibration_targeted'/'report.json').read_text())
        assert partial['status'] == 'interrupted_or_failed'
        with (temp/'evaluation.lock').open('a') as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('Anime matched runner self-test passed; no model loaded or request sent.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=(*PHASES,'self-test'))
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--target-spec',type=Path,default=None)
    parser.add_argument('--worker-asr',nargs=2,metavar=('PROTOCOL','REPORT'))
    args=parser.parse_args()
    if args.worker_asr:
        asr_worker(Path(args.worker_asr[0]),Path(args.worker_asr[1]));return
    if args.phase is None:
        parser.error('--phase is required')
    if args.phase=='self-test': self_test();return
    result=run_phase(args.phase,args.protocol,args.target_spec)
    print(json.dumps({'phase':args.phase,'status':result.get('status'),
                      'output':str(Path(args.protocol).resolve())},ensure_ascii=False))


if __name__=='__main__': main()
