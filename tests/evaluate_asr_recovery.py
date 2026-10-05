#!/usr/bin/env python3
"""Bounded native top-three Turbo decoding on audio-only development inputs."""
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
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone

import evaluate_caption_verifiers as processes
import evaluate_reazon as shared

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/media/psikosen/newvolume/jread-asr-recovery')
PROTOCOL = ROOT / 'tests/fixtures/accuracy-asr-recovery.json'
LOCK = Path('/media/psikosen/newvolume/jread-verifier/evaluation.lock')
MODEL_DEFAULT = Path.home() / '.local/share/JREAD/models/large-v3-turbo'
SETTINGS = {'device': 'cuda', 'device_index': 0, 'compute_type': 'int8_float16',
            'cpu_threads': 4, 'num_workers': 1, 'language': 'ja', 'task': 'transcribe',
            'vad_filter': False, 'without_timestamps': True, 'condition_on_previous_text': False,
            'beam_size': 5, 'num_hypotheses': 3, 'patience': 1.0, 'length_penalty': 1.0,
            'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0, 'max_length': 448,
            'sampling_enabled': False, 'temperature': 0.0, 'return_scores': True,
            'return_no_speech_prob': True}
CT2_MAX_GENERATED_TOKENS = 224  # CT2 4.8.2 uses min(max_length / 2, max_length - start_step).
PACKETS = ('development_audio', 'controls_audio', 'context_audio')
CONTEXT_METADATA = ('parent_audio_path', 'parent_audio_sha256', 'parent_start_frame', 'parent_end_frame',
                    'parent_sample_rate', 'target_waveform_start_seconds', 'target_waveform_end_seconds')


def binding(path):
    return {'path': str(path), 'sha256': shared.sha(path)}


def read_bound(item):
    if shared.sha(item['path']) != item['sha256']:
        raise ValueError(f'Frozen input changed: {item["path"]}')
    return json.loads(Path(item['path']).read_text())


def write(path, data):
    shared.write(path, data)


def gpu_snapshot():
    from asr_cuda_pilot import gpu_snapshot as snapshot
    return snapshot()


def model_path(protocol):
    return Path(protocol['model'].get('directory', MODEL_DEFAULT))


def frozen_protocol():
    p = json.loads(PROTOCOL.read_text())
    if p.get('status') != 'frozen' or p.get('runner_sha256') != shared.sha(__file__):
        raise ValueError('Require the frozen protocol and exact runner hash.')
    for item in p['bindings'] + p['model']['bindings'] + p['runtime']['bindings']:
        if shared.sha(item['path']) != item['sha256']:
            raise ValueError(f'Pinned file changed: {item["path"]}')
    if Path(sys.executable).resolve() != Path(p['runtime']['python']).resolve():
        raise ValueError('Run with the pinned JREAD virtual-environment Python.')
    for name, version in p['runtime']['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Pinned runtime package changed: {name}')
    if p['settings'] != SETTINGS:
        raise ValueError('Native decode settings differ from the frozen plan.')
    directory = model_path(p)
    if (directory / '.verified').read_text().strip() != p['model']['revision']:
        raise ValueError('Local Turbo verification marker changed.')
    if shared.sha(directory / 'model.bin') != p['model']['model_bin_sha256']:
        raise ValueError('Local Turbo model bytes changed.')
    return p


def audio_info(row):
    path = Path(row['audio_path'])
    if not path.is_absolute() or shared.sha(path) != row['audio_sha256']:
        raise ValueError(f'Audio path/hash mismatch for {row["id"]}.')
    with wave.open(str(path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError(f'Expected mono PCM16 16kHz WAV for {row["id"]}.')
        frames = audio.getnframes()
        raw = audio.readframes(frames)
    duration = frames / 16000
    if not 0 < frames <= 30 * 16000 or len(raw) != frames * 2 or abs(duration - row['duration_seconds']) > 1 / 16000:
        raise ValueError(f'Invalid or truncated audio for {row["id"]}.')
    return raw, duration


def packet_cases(protocol):
    result = {}
    for key in PACKETS:
        packet = read_bound(protocol[key])
        rows = packet.get('cases', [])
        expected = {'development_audio': 20, 'controls_audio': 3, 'context_audio': 2}[key]
        if len(rows) != expected or len({str(r['id']) for r in rows}) != expected:
            raise ValueError(f'{key} must contain {expected} unique audio-only rows.')
        for row in rows:
            if any(k in row for k in ('japanese_reference', 'english_reference', 'transcript', 'translation', 'text')):
                raise ValueError(f'{key} must not expose annotations or case text to the worker.')
            if not isinstance(row.get('duration_seconds'), (int, float)) or not 0 < row['duration_seconds'] <= 30:
                raise ValueError(f'Invalid duration in {key}.')
            if not Path(row['audio_path']).is_absolute() or len(row.get('audio_sha256', '')) != 64:
                raise ValueError(f'{key} must bind an absolute audio path and SHA-256.')
        if [str(r['id']) for r in rows] != [str(x) for x in protocol['case_ids'][key]]:
            raise ValueError(f'{key} IDs/order differ from frozen protocol.')
        result[key] = rows
    if len({str(r['id']) for key in PACKETS for r in result[key]}) != 25:
        raise ValueError('Development, controls, and context IDs must be disjoint.')
    return result


def validate_real_context(rows):
    by_id = {str(r['id']): r for r in rows}
    target, wide = by_id.get('anime-target'), by_id.get('anime-left-context')
    if not target or not wide:
        raise ValueError('Context packet must bind target and left-context windows.')
    for row in (target, wide):
        if any(k not in row for k in CONTEXT_METADATA):
            raise ValueError('Context rows lack exact parent-waveform provenance.')
    if (target['parent_audio_path'] != wide['parent_audio_path'] or
            target['parent_audio_sha256'] != wide['parent_audio_sha256'] or
            target['parent_sample_rate'] != 16000 or wide['parent_sample_rate'] != 16000 or
            target['parent_end_frame'] != wide['parent_end_frame'] or
            target['target_waveform_start_seconds'] != wide['target_waveform_start_seconds'] or
            target['target_waveform_end_seconds'] != wide['target_waveform_end_seconds']):
        raise ValueError('Context pair does not share a pinned parent recording and target interval.')
    start, end, left = (target['parent_start_frame'], target['parent_end_frame'], wide['parent_start_frame'])
    if (start >= end or left >= start or wide['parent_end_frame'] != end or
            start / 16000 != target['target_waveform_start_seconds'] or
            end / 16000 != target['target_waveform_end_seconds'] or
            abs((start - left) / 16000 - 14.0) > 1 / 16000 or
            abs((end - start) / 16000 - 16.0) > 1 / 16000 or
            abs((end - left) / 16000 - 30.0) > 1 / 16000):
        raise ValueError('Context pair must be the fixed 16s target and 30s window with 14s real preceding audio.')
    parent_path = Path(target['parent_audio_path'])
    if shared.sha(parent_path) != target['parent_audio_sha256']:
        raise ValueError('Pinned parent recording hash changed.')
    with wave.open(str(parent_path), 'rb') as parent:
        if (parent.getnchannels(), parent.getsampwidth(), parent.getframerate(), parent.getcomptype()) != (1, 2, 16000, 'NONE') or end > parent.getnframes():
            raise ValueError('Parent recording is not the pinned full PCM waveform.')
        parent.setpos(left)
        parent_window = parent.readframes(end - left)
    for row in (target, wide):
        raw, _ = audio_info(row)
        offset = row['parent_start_frame'] - left
        if raw != parent_window[offset * 2:offset * 2 + len(raw)]:
            raise ValueError(f'Context WAV {row["id"]} is not an exact parent-waveform crop.')


def validate_audio_features(rows, protocol):
    import numpy as np
    from faster_whisper.feature_extractor import FeatureExtractor
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer
    from tokenizers import Tokenizer as LocalTokenizer

    directory = model_path(protocol)
    preprocessor = json.loads((directory / 'preprocessor_config.json').read_text())
    model_config = json.loads((directory / 'config.json').read_text())
    if preprocessor.get('feature_size') != 128:
        raise ValueError('Turbo requires its pinned 128-mel preprocessing config.')
    extractor = FeatureExtractor(**{k: v for k, v in preprocessor.items()
                                    if k in ('feature_size', 'sampling_rate', 'hop_length', 'chunk_length', 'n_fft')})
    hf_tokenizer = LocalTokenizer.from_file(str(directory / 'tokenizer.json'))
    japanese_id = hf_tokenizer.token_to_id('<|ja|>')
    if japanese_id is None or japanese_id not in model_config.get('lang_ids', []):
        raise ValueError('Pinned model config and tokenizer do not include Japanese.')
    tokenizer = Tokenizer(hf_tokenizer, True, task='transcribe', language='ja')
    if not tokenizer.encode('確認'):
        raise ValueError('Pinned Japanese tokenizer did not encode local validation text.')
    audit = []
    for row in rows:
        raw, duration = audio_info(row)
        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
        features = extractor(samples)
        frames = features.shape[-1] - 1  # faster-whisper reserves the final mel frame.
        padded = pad_or_trim(features)
        if (samples.dtype != np.float32 or features.dtype != np.float32 or padded.dtype != np.float32 or
                features.shape[0] != 128 or not 0 < frames <= 3000 or padded.shape != (128, 3000) or
                not np.isfinite(features).all() or not np.isfinite(padded).all()):
            raise ValueError(f'Invalid Turbo mel shape, dtype or finite values for {row["id"]}.')
        audit.append({'id': row['id'], 'audio_sha256': row['audio_sha256'], 'duration_seconds': duration,
                      'sample_count': len(samples), 'sample_dtype': str(samples.dtype),
                      'mel_bins': features.shape[0], 'real_mel_frames': frames,
                      'model_mel_frames': padded.shape[-1], 'mel_dtype': str(padded.dtype),
                      'tokenizer_probe_tokens': len(tokenizer.encode('確認'))})
    return audit


def exact_preflight(protocol, all_rows):
    report = read_bound(binding(BASE / 'validate/report.json'))
    if (report.get('status') != 'complete' or report.get('protocol', {}).get('sha256') != shared.sha(PROTOCOL) or
            report.get('packet_bindings') != {key: protocol[key] for key in PACKETS}):
        raise ValueError('Require exact complete validation under this frozen protocol.')
    expected = [[key, str(r['id']), r['audio_sha256'], r['duration_seconds']] for key in PACKETS for r in all_rows[key]]
    audit = report.get('feature_preflight', [])
    if (report.get('model_loaded') is not False or report.get('inference_performed') is not False or
            report.get('model_load_seconds') != 0.0 or report.get('validated_audio') != expected or
            len(expected) != 25 or [r.get('id') for r in audit] != [item[1] for item in expected] or
            any(r.get('mel_bins') != 128 or r.get('model_mel_frames') != 3000 or r.get('mel_dtype') != 'float32' for r in audit)):
        raise ValueError('Validation did not cover these exact 25 audio inputs.')
    return report


def decode_case(model, row):
    import numpy as np
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import get_suppressed_tokens

    total_started = time.perf_counter()
    raw, duration = audio_info(row)
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    feature_started = time.perf_counter()
    mel = model.feature_extractor(samples)
    real_frames = mel.shape[-1] - 1
    encoded = model.encode(pad_or_trim(mel))
    feature_seconds = time.perf_counter() - feature_started
    tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task='transcribe', language='ja')
    prompt = model.get_prompt(tokenizer, [], without_timestamps=True)
    generation_started = time.perf_counter()
    generated = model.model.generate(
        encoded, [prompt], beam_size=5, num_hypotheses=3, patience=1.0,
        length_penalty=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
        max_length=448, return_scores=True, return_no_speech_prob=True,
        suppress_blank=True, suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
        sampling_topk=1, sampling_temperature=0.0)[0]
    generation_seconds = time.perf_counter() - generation_started
    ids = generated.sequences_ids
    scores = generated.scores
    no_speech = float(generated.no_speech_prob)
    hypotheses = []
    alignable = []
    for rank, token_ids in enumerate(ids):
        token_ids = [int(t) for t in token_ids]
        text = tokenizer.decode(token_ids).strip()
        score = float(scores[rank]) if rank < len(scores) else None
        clipped = len(token_ids) >= CT2_MAX_GENERATED_TOKENS
        hypotheses.append({'rank': rank, 'text': text, 'token_ids': token_ids,
                           'sequence_score': score if score is None or math.isfinite(score) else None,
                           'sequence_score_kind': 'native CT2 beam sequence score; not calibrated correctness confidence',
                           'sequence_length': len(token_ids), 'potentially_truncated_at_max_length': clipped,
                           'unexpected_timestamp_tokens': any(t >= tokenizer.timestamp_begin for t in token_ids),
                           'terminal_note': 'CTranslate2 omits EOS from returned sequences; EOS observation is unavailable.',
                           'token_evidence': {'status': 'pending' if text else 'skipped_empty',
                                              'kind': 'forced_audio_alignment',
                                              'meaning': 'Model alignment evidence only; not calibrated transcript correctness.'}})
        if text and token_ids:
            alignable.append((rank, token_ids))
    alignment_started = time.perf_counter()
    for rank, token_ids in alignable:
        try:
            evidence = model.model.align(encoded, tokenizer.sot_sequence, [token_ids], real_frames)[0]
            probs = [float(v) for v in evidence.text_token_probs]
            times = [[int(a), int(b)] for a, b in evidence.alignments]
            okay = (len(probs) == len(token_ids) and bool(times) and
                    all(math.isfinite(v) and 0 <= v <= 1 for v in probs) and
                    all(0 <= token <= len(token_ids) and 0 <= frame < real_frames // 2 for token, frame in times))
            hypotheses[rank]['token_evidence'] = {
                'status': 'ok' if okay else 'error', 'kind': 'forced_audio_alignment',
                'text_token_probs': probs if okay else [], 'alignments': times,
                'alignment_times_seconds': [frame / 50.0 for _, frame in times],
                'notes': 'Forced alignment on this same encoded full clip. These values do not establish lexical truth.'}
            if not okay:
                hypotheses[rank]['token_evidence']['error'] = 'Invalid token-probability length or token/frame alignment.'
        except Exception as error:
            hypotheses[rank]['token_evidence'].update(status='error', error=f'{type(error).__name__}: {error}')
    alignment_seconds = time.perf_counter() - alignment_started
    unique = unique_candidates(row['id'], hypotheses)
    good = (len(hypotheses) == 3 and len(scores) == 3 and math.isfinite(no_speech) and
            all(h['sequence_score'] is not None and
                not h['potentially_truncated_at_max_length'] and
                (not h['text'] or h['token_evidence']['status'] == 'ok') and
                not h.get('unexpected_timestamp_tokens') for h in hypotheses))
    safe_metadata = {k: row[k] for k in ('kind', 'condition', 'pair_id', 'source_audio_sha256',
                     'source_start_seconds', 'source_end_seconds', 'target_start_seconds',
                     'target_end_seconds', 'added_real_context_seconds', *CONTEXT_METADATA) if k in row}
    return {'id': str(row['id']), 'status': 'ok' if good else 'error', **safe_metadata,
            'duration_seconds': duration, 'audio_sha256': row['audio_sha256'],
            'input_sample_count': len(samples), 'real_mel_frames': real_frames,
            'model_mel_frames': 3000, 'model_required_mel_padding_frames': 3000 - real_frames,
            'model_required_mel_padding_is_audio_context': False,
            'raw_hypotheses': hypotheses, 'unique_candidates': unique,
            'unique_candidate_count': len(unique),
            'no_speech_probability': no_speech if math.isfinite(no_speech) else None,
            'actual_result_rank': 0 if hypotheses else None,
            'actual_result_text': hypotheses[0]['text'] if hypotheses else None,
            'timing_seconds': {'features_encode': feature_seconds, 'native_generate': generation_seconds,
                               'alignment': alignment_seconds, 'total': time.perf_counter() - total_started},
            'decode_completeness': 'below length cap; EOS is not exposed by CT2'}


def unique_candidates(case_id, hypotheses):
    unique, seen = [], {}
    for hyp in hypotheses:
        text = hyp['text']
        if text in seen:
            hyp['duplicate_of_rank'] = seen[text]
        else:
            seen[text] = hyp['rank']
            unique.append({'candidate_id': 'c' + hashlib.sha256(
                json.dumps([case_id, text], ensure_ascii=False).encode()).hexdigest()[:16],
                'rank': hyp['rank'], 'text': text})
    return unique


def run_worker(protocol, phase, rows, output):
    import gc
    from faster_whisper import WhisperModel

    result = {'status': 'running', 'phase': phase, 'rows': [], 'model_loaded': False,
              'inference_performed': False, 'cuda_initialized': False, 'model_load_seconds': 0.0}
    write(output, result)
    directory = model_path(protocol)
    model = None
    try:
        started = time.perf_counter()
        model = WhisperModel(str(directory), device='cuda', device_index=0,
                             compute_type='int8_float16', cpu_threads=4,
                             num_workers=1, local_files_only=True)
        result.update(model_loaded=True, model_load_seconds=time.perf_counter() - started,
                      cuda_initialized=True, model_device=str(model.model.device),
                      model_device_index=str(model.model.device_index),
                      model_compute_type=str(model.model.compute_type), torch_threads=None)
        if (model.model.device != 'cuda' or model.model.compute_type != 'int8_float16'
                or list(model.model.device_index) != [0]):
            raise ValueError('Loaded CTranslate2 device/compute type differs from the frozen request.')
        result.update(status='running', inference_performed=True)
        write(output, result)
        for row in rows:
            started = time.perf_counter()
            try:
                case = decode_case(model, row)
                case['decode_seconds'] = time.perf_counter() - started
            except Exception as error:
                case = {'id': str(row['id']), 'status': 'error', 'audio_sha256': row['audio_sha256'],
                        'error': f'{type(error).__name__}: {error}', 'raw_hypotheses': [], 'unique_candidates': []}
            result['rows'].append(case)
            write(output, result)
        result['status'] = 'complete' if all(r['status'] == 'ok' for r in result['rows']) else 'complete_with_errors'
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if model is not None:
            del model
            gc.collect()
        write(output, result)
    return result


def checked_worker(expected, raw, phase):
    if raw.get('status') not in ('complete', 'complete_with_errors') or raw.get('model_loaded') is not True or raw.get('inference_performed') is not True or raw.get('cuda_initialized') is not True:
        raise ValueError('Worker lacks complete native model/inference evidence.')
    if raw.get('model_device') != 'cuda' or raw.get('model_compute_type') != 'int8_float16':
        raise ValueError('Worker did not use actual CUDA int8_float16 execution.')
    rows = raw.get('rows', [])
    if [r.get('id') for r in rows] != [str(r['id']) for r in expected]:
        raise ValueError('Worker rows are missing, reordered, or duplicated.')
    for row, source in zip(rows, expected, strict=True):
        if row.get('audio_sha256') != source['audio_sha256']:
            raise ValueError('Worker audio hash differs from the exact input packet.')
        if row.get('status') != 'ok':
            continue
        hyps = row.get('raw_hypotheses', [])
        if len(hyps) != 3 or [h.get('rank') for h in hyps] != [0, 1, 2]:
            raise ValueError('Successful case must preserve exactly native ranks 0, 1, 2.')
        if (not isinstance(row.get('no_speech_probability'), (int, float)) or
                not math.isfinite(row['no_speech_probability']) or not 0 <= row['no_speech_probability'] <= 1 or
                any(not isinstance(h.get('sequence_score'), (int, float)) or not math.isfinite(h['sequence_score']) for h in hyps)):
            raise ValueError('Native hypotheses have missing/nonfinite scores or no-speech probability.')
        if any(h.get('potentially_truncated_at_max_length') for h in hyps):
            raise ValueError('Length-capped hypotheses remain preserved but invalidate completeness.')
        if any(not isinstance(h.get('text'), str) or not isinstance(h.get('token_ids'), list) or
               any(type(token) is not int for token in h['token_ids']) for h in hyps):
            raise ValueError('Malformed decoded text/token-ID evidence.')
        if any(h.get('sequence_length') != len(h['token_ids']) for h in hyps):
            raise ValueError('Native token length does not match preserved IDs.')
        for hyp in hyps:
            evidence = hyp.get('token_evidence', {})
            if hyp['text']:
                probs = evidence.get('text_token_probs', [])
                alignments = evidence.get('alignments', [])
                if (evidence.get('status') != 'ok' or len(probs) != len(hyp['token_ids']) or not alignments or
                        any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs) or
                        any(not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair) or
                            not 0 <= pair[0] <= len(hyp['token_ids']) or not 0 <= pair[1] < row['real_mel_frames'] // 2 for pair in alignments)):
                    raise ValueError('Nonempty native hypothesis lacks finite per-token alignment evidence.')
            elif evidence.get('status') != 'skipped_empty':
                raise ValueError('Empty native hypothesis has unexpected alignment status.')
            if hyp.get('unexpected_timestamp_tokens'):
                raise ValueError('No-timestamp decode emitted timestamp tokens.')
    if phase == 'smoke' and any(r['status'] != 'ok' for r in rows):
        raise ValueError('Smoke must complete every control with finite three-rank output.')
    return rows


def report_path(phase):
    return BASE / phase / 'report.json'


def run(phase):
    p = frozen_protocol()
    cases = packet_cases(p)
    validated = [(key, r['id'], r['audio_sha256'], r['duration_seconds'])
                 for key in PACKETS for r in cases[key]]
    audit = validate_audio_features([r for key in PACKETS for r in cases[key]], p) if phase == 'validate' else None
    if phase in ('validate', 'context'):
        validate_real_context(cases['context_audio'])
    if phase != 'validate':
        previous = exact_preflight(p, cases)
        if phase == 'asr':
            prior_smoke = read_bound(binding(report_path('smoke')))
            if prior_smoke['status'] != 'complete' or prior_smoke['protocol']['sha256'] != shared.sha(PROTOCOL) or not prior_smoke['cleanup']['owned_worker_released']:
                raise ValueError('Require successful same-protocol smoke and worker release.')
            checked_worker(cases['controls_audio'], read_bound(prior_smoke['worker']), 'smoke')
        if phase == 'context':
            prior_asr = read_bound(binding(report_path('asr')))
            if prior_asr['status'] not in ('complete', 'complete_with_errors') or prior_asr['protocol']['sha256'] != shared.sha(PROTOCOL) or not prior_asr['cleanup']['owned_worker_released']:
                raise ValueError('Require complete main development run and worker release before context diagnostics.')
            checked_worker(cases['development_audio'], read_bound(prior_asr['worker']), 'asr')
    rows = {'validate': [r for key in PACKETS for r in cases[key]],
            'smoke': cases['controls_audio'], 'asr': cases['development_audio'],
            'context': cases['context_audio']}[phase]
    out = BASE / phase
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs.json'
    request = {'phase': phase, 'cases': rows}
    write(inputs, request)
    report = {'status': 'running', 'phase': phase, 'protocol': binding(PROTOCOL),
              'packet_bindings': {key: p[key] for key in PACKETS},
              'inputs': binding(inputs), 'expected_cases': len(rows),
              'heldout_rows_used': 0, 'original_reports_changed': False,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'differences_from_historical_live_and_streaming_asr': [
                  'Whole-clip decode instead of two-second prefix updates.',
                  'No VAD, no timestamp tokens, native CT2 top-three beam candidates.',
                  'Sequence scores/alignment values are not calibrated transcript correctness probabilities.']}
    if phase == 'validate':
        report.update(status='complete', model_loaded=False, inference_performed=False,
                      model_load_seconds=0.0, validated_audio=validated,
                      feature_preflight=audit,
                      worker=None, cleanup={'owned_worker_released': True})
        write(out / 'report.json', report)
        print(json.dumps({'status': report['status'], 'report': str(out / 'report.json')}))
        return 0
    report.update(gpu_before=gpu_snapshot())
    if not any(g['index'] == 0 and g['free_mib'] >= 4096 for g in report['gpu_before']['gpus']):
        report.update(status='failed_preflight', error='GPU 0 must have at least 4096 MiB free; no model started.')
        report['cleanup'] = {'owned_worker_released': True, 'owned_gpu_process_absent': True}
        write(out / 'report.json', report)
        return 1
    worker_path = out / 'worker.json'
    child = lock = None
    env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', USE_TF='0',
               HF_HOME=str(BASE / 'hf-cache'), XDG_CACHE_HOME=str(BASE / 'cache'))
    worker_env = out / 'worker-env.json'
    write(worker_env, {'phase': phase, 'protocol_sha256': shared.sha(PROTOCOL), 'inputs': binding(inputs),
                       'environment': {k: env.get(k, '') for k in ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE',
                           'TRANSFORMERS_OFFLINE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'LD_LIBRARY_PATH')}})
    report['worker_environment'] = binding(worker_env)
    try:
        lock = LOCK.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shared.free_ports()
        with (out / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', '--input', str(inputs),
                                      '--output', str(worker_path)], cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            child.wait(timeout=1800)
        raw = json.loads(worker_path.read_text()) if worker_path.exists() else {}
        report.update(worker=binding(worker_path) if worker_path.exists() else None,
                      worker_exit_code=child.returncode)
        worker_rows = checked_worker(rows, raw, phase) if child.returncode == 0 else raw.get('rows', [])
        if child.returncode != 0:
            report.update(status='failed', error='Owned worker failed; preserve raw worker output and log.')
        else:
            report['cases'] = worker_rows
            if phase == 'smoke':
                report['control_quality_diagnostic'] = [{'id': r['id'], 'is_silence_input': r['id'] == 'zero-pcm-1s',
                    'top1_text': r['raw_hypotheses'][0]['text'],
                    'top1_no_speech_probability': r['no_speech_probability'],
                    'empty_top1': not bool(r['raw_hypotheses'][0]['text'].strip())} for r in worker_rows]
                report['technical_gate'] = True
                report['semantic_gate'] = None
            report['status'] = 'complete' if all(r['status'] == 'ok' for r in worker_rows) else 'complete_with_errors'
            if phase == 'context':
                report['interpretation'] = 'Separate qualitative context diagnostic; no verified Japanese ground truth or direct accuracy comparison.'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if worker_path.exists(): report['partial_worker'] = binding(worker_path)
    finally:
        if child is not None and child.poll() is None:
            processes.signal_group(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                processes.signal_group(child.pid, signal.SIGKILL); child.wait()
        survivors = processes.group_members(child.pid) if child else []
        if survivors:
            processes.signal_group(child.pid, signal.SIGKILL)
            report.update(status='failed', cleanup_error='Owned process descendants remained after worker exit.')
        remaining = processes.group_members(child.pid) if child else []
        try:
            after = gpu_snapshot()
        except Exception as error:
            after = {'gpus': [], 'processes': [], 'error': f'{type(error).__name__}: {error}'}
            report.update(status='failed', cleanup_error='Could not verify GPU cleanup.')
        report.update(gpu_after=after, cleanup={'owned_worker_pid': child.pid if child else None,
            'worker_exit_code': child.returncode if child else None,
            'owned_worker_released': (child is None or child.poll() is not None) and not remaining,
            'descendants_after_worker_exit': survivors, 'remaining_group_members': remaining,
            'owned_gpu_process_absent': 'error' not in after and (child is None or all(r['pid'] != child.pid for r in after['processes']))})
        if not report['cleanup']['owned_worker_released'] or not report['cleanup']['owned_gpu_process_absent']:
            report.update(status='failed', cleanup_error='Owned GPU worker or process group remains.')
        if phase == 'smoke' and report.get('status') != 'complete': report['technical_gate'] = False
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        try: write(out / 'report.json', report)
        finally:
            if lock is not None: lock.close()
    print(json.dumps({'status': report['status'], 'report': str(out / 'report.json')}))
    return 0 if report['status'] in ('complete', 'complete_with_errors') else 1


def self_test():
    import hashlib
    case = {'id': 'm', 'audio_sha256': '0' * 64, 'kind': 'mock'}
    raw = {'status': 'complete', 'model_loaded': True, 'inference_performed': True,
           'cuda_initialized': True, 'model_device': 'cuda', 'model_compute_type': 'int8_float16',
           'rows': [{'id': 'm', 'audio_sha256': '0' * 64, 'status': 'ok', 'no_speech_probability': .1, 'real_mel_frames': 100,
               'raw_hypotheses': [{'rank': i, 'sequence_score': -float(i), 'text': f'hyp {i}', 'token_ids': [i], 'sequence_length': 1,
                'potentially_truncated_at_max_length': False,
                'unexpected_timestamp_tokens': False,
               'token_evidence': {'status': 'ok', 'text_token_probs': [.25], 'alignments': [[0, 0]]}} for i in range(3)]}]}
    assert checked_worker([case], raw, 'asr')[0]['raw_hypotheses'][2]['rank'] == 2
    mock = [{'rank': 0, 'text': 'same'}, {'rank': 1, 'text': 'same'}, {'rank': 2, 'text': 'other'}]
    unique = unique_candidates('m', mock)
    assert [h['rank'] for h in mock] == [0, 1, 2] and len(unique) == 2 and mock[1]['duplicate_of_rank'] == 0
    case2 = {'id': 'n', 'audio_sha256': '1' * 64}
    row2 = raw['rows'][0] | {'id': 'n', 'audio_sha256': case2['audio_sha256']}
    raw2 = raw | {'rows': raw['rows'] + [row2]}
    try: checked_worker([case, case2], raw2 | {'rows': list(reversed(raw2['rows']))}, 'asr')
    except ValueError: pass
    else: raise AssertionError('Wrong row order accepted.')
    try: checked_worker([case], raw | {'rows': [raw['rows'][0] | {'raw_hypotheses': raw['rows'][0]['raw_hypotheses'][:2]}]}, 'asr')
    except ValueError: pass
    else: raise AssertionError('Missing native rank accepted.')
    short_evidence = raw['rows'][0]['raw_hypotheses'][0] | {'token_evidence': {'status': 'ok', 'text_token_probs': [], 'alignments': [[0, 0]]}}
    badrow = raw['rows'][0] | {'raw_hypotheses': [short_evidence, *raw['rows'][0]['raw_hypotheses'][1:]]}
    try: checked_worker([case], raw | {'rows': [badrow]}, 'asr')
    except ValueError: pass
    else: raise AssertionError('Missing per-token evidence accepted.')
    print('Pure ASR self-test passed: exact ranks, finite scores, per-token evidence and row order; no model load.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['validate', 'smoke', 'asr', 'context'])
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test: self_test()
    elif args.worker:
        p = frozen_protocol()
        payload = json.loads(args.input.read_text())
        worker_result = run_worker(p, payload['phase'], payload['cases'], args.output)
        raise SystemExit(0 if worker_result['status'] in ('complete', 'complete_with_errors') else 1)
    elif args.phase: raise SystemExit(run(args.phase))
    else: parser.error('Require --self-test or --phase.')
