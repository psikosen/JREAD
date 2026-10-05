#!/usr/bin/env python3
"""Compare live-like packet translation with causal stable-unit translation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jread import engine, gemma  # noqa: E402

EVALUATOR_PATH = ROOT / 'tests/evaluate_translation.py'
CORPUS_PATH = ROOT / 'tests/fixtures/accuracy-corpus.json'
OUTPUT_PATH = ROOT / 'test-results/accuracy-streaming-translation-development.json'
POLICY_A_OUTPUT_PATH = ROOT / 'test-results/accuracy-streaming-repaired-translation-development.json'
BOUNDARY_REPLAY_PATH = ROOT / 'test-results/accuracy-streaming-boundary-replay.json'
BASELINE_REPORT_PATH = OUTPUT_PATH
SOURCE_ASR_PATH = ROOT / 'test-results/accuracy-streaming-speech-development.json'
EXPECTED_CASES = 20
PORT = 18767
PRESET = 'gemma4'
QUEUE_CAPACITY = 3  # one worker plus asyncio.Queue(maxsize=2)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location('jread_translation_evaluator', EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load the frozen translation evaluator helpers.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def port_accepting_connections(port: int) -> bool:
    probe = socket.socket()
    probe.settimeout(0.2)
    try:
        probe.connect((gemma.HOST, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def load_development_cases() -> tuple[dict, list[dict]]:
    manifest = json.loads(CORPUS_PATH.read_text())
    cases = [row for row in manifest['cases'] if row.get('partition') == 'development']
    if len(cases) != EXPECTED_CASES or len({str(row['id']) for row in cases}) != EXPECTED_CASES:
        raise ValueError('Frozen corpus must contain exactly 20 unique development cases.')
    return manifest, cases


def validate_asr_report(path: Path, manifest: dict, cases: list[dict]) -> tuple[dict, dict[str, dict], dict[str, list[dict]]]:
    report = json.loads(path.read_text())
    if report.get('status') != 'complete':
        raise ValueError(f"Streaming ASR report must be complete; got {report.get('status')!r}.")
    if report.get('partition') != 'development':
        raise ValueError('Streaming ASR report must be development-only.')
    if report.get('owned_model_unloaded') is not True or report.get('owned_gpu_process_released') is not True:
        raise ValueError('Streaming ASR cleanup must confirm model unload and owned process release.')
    if report.get('worker_exit_code') != 0:
        raise ValueError('Streaming ASR worker must have exited successfully.')
    corpus_sha = sha256(CORPUS_PATH)
    recorded_sha = report.get('corpus', {}).get('manifest_sha256')
    if recorded_sha != corpus_sha:
        raise ValueError('Streaming ASR manifest SHA256 does not match the frozen accuracy corpus.')
    report_cases = report.get('cases')
    if not isinstance(report_cases, list) or len(report_cases) != EXPECTED_CASES:
        raise ValueError('Streaming ASR report must contain exactly 20 cases.')
    ids = [str(row.get('id')) for row in report_cases]
    expected_ids = {str(row['id']) for row in cases}
    if len(set(ids)) != EXPECTED_CASES or set(ids) != expected_ids:
        raise ValueError('Streaming ASR IDs must match the 20 development IDs exactly.')
    by_id = {str(row['id']): row for row in report_cases}
    manifest_by_id = {str(row['id']): row for row in cases}
    normalized_updates = {}
    for case_id, row in by_id.items():
        frozen = manifest_by_id[case_id]
        if row.get('partition') != 'development':
            raise ValueError(f'ASR case {case_id} is not marked development.')
        if row.get('audio_sha256') != frozen.get('audio_sha256'):
            raise ValueError(f'Audio SHA256 mismatch for ASR case {case_id}.')
        if row.get('japanese_reference') != frozen.get('japanese_reference') or row.get('english_reference') != frozen.get('english_reference'):
            raise ValueError(f'Frozen reference fields mismatch for ASR case {case_id}.')
        if row.get('status', 'complete') not in ('complete', 'ok'):
            raise ValueError(f"ASR case {case_id} is not complete: {row.get('status')!r}.")
        _validate_baseline(row.get('overlap_baseline'), float(frozen['duration_seconds']), case_id)
        normalized_updates[case_id] = _prepare_streaming_updates(row, float(frozen['duration_seconds']), case_id)
    return report, by_id, normalized_updates


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number.')
    return float(value)


def _validate_segment(row: dict, duration: float, label: str) -> None:
    start = _finite_number(row.get('start', row.get('start_seconds')), f'{label}.start')
    end = _finite_number(row.get('end', row.get('end_seconds')), f'{label}.end')
    if not 0 <= start < end <= duration + 1e-4 or not isinstance(row.get('text'), str) or not row['text'].strip():
        raise ValueError(f'{label} has invalid source text or clocks.')


def _validate_baseline(baseline: dict | None, duration: float, case_id: str) -> None:
    chunks = (baseline or {}).get('chunks')
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f'ASR case {case_id} lacks overlap_baseline.chunks.')
    previous_end = -1.0
    for index, chunk in enumerate(chunks):
        start = _finite_number(chunk.get('start_seconds', chunk.get('start')), f'{case_id}.baseline[{index}].start_seconds')
        end = _finite_number(chunk.get('end_seconds', chunk.get('end', chunk.get('audio_end_seconds'))), f'{case_id}.baseline[{index}].end_seconds')
        inference = _finite_number(chunk.get('inference_seconds'), f'{case_id}.baseline[{index}].inference_seconds')
        if inference < 0 or not 0 <= start < end <= duration + 1e-4 or end < previous_end:
            raise ValueError(f'ASR case {case_id} has invalid baseline order or duration.')
        if abs(start - 7.0 * index) > 1e-3 or abs(end - min(duration, start + 8.0)) > 1e-3:
            raise ValueError(f'ASR case {case_id} baseline chunk {index} is not an 8s/1s-overlap window.')
        previous_end = end
        if not isinstance(chunk.get('segments'), list):
            raise ValueError(f'ASR case {case_id} baseline chunk {index} lacks segment rows.')
        for j, segment in enumerate(chunk['segments']):
            _validate_segment(segment, duration, f'{case_id}.baseline[{index}].segments[{j}]')


def _unit_signature(unit: dict) -> tuple:
    return (
        unit.get('text'),
        round(float(unit.get('source_start_seconds', -1)), 6),
        round(float(unit.get('source_end_seconds', -1)), 6),
        round(float(unit.get('emitted_at_audio_seconds', -1)), 6),
        unit.get('stability'),
    )


def _prepare_streaming_updates(case: dict, duration: float, case_id: str) -> list[dict]:
    streaming = case.get('streaming') or {}
    alignment_status = streaming.get('transcript_alignment_status')
    if alignment_status not in ('aligned', 'unresolved_committed_prefix_conflict'):
        raise ValueError(f'ASR case {case_id} has an unknown transcript_alignment_status.')
    updates = copy.deepcopy(streaming.get('updates'))
    if not isinstance(updates, list) or not updates:
        raise ValueError(f'ASR case {case_id} lacks streaming.updates.')
    tails = streaming.get('final_tail_units', case.get('final_tail_units', []))
    if not isinstance(tails, list):
        raise ValueError(f'ASR case {case_id} final_tail_units must be a list.')
    if tails and not updates[-1].get('emitted_units'):
        updates[-1]['emitted_units'] = []
    emitted_before = [unit for update in updates for unit in update.get('emitted_units', [])]
    counts = {}
    for unit in emitted_before:
        signature = _unit_signature(unit)
        counts[signature] = counts.get(signature, 0) + 1
    for tail in tails:
        signature = _unit_signature(tail)
        existing = counts.get(signature, 0)
        if existing > 1:
            raise ValueError(f'ASR case {case_id} duplicates an EOF tail before adapter normalization.')
        if existing == 0:
            # Compatibility for a report that records the final tail separately.
            # It inherits the final update's cost; no synthetic ASR update is added.
            updates[-1]['emitted_units'].append(copy.deepcopy(tail))
            counts[signature] = 1
    flush_count = updates[-1].get('eof_flush_unit_count')
    if tails and flush_count is not None and flush_count != len(tails):
        raise ValueError(f'ASR case {case_id} EOF flush count does not match final_tail_units.')
    if tails and flush_count is None:
        updates[-1]['eof_flush_unit_count'] = len(tails)
    flush_count = updates[-1].get('eof_flush_unit_count', 0)
    if type(flush_count) is not int or flush_count < 0 or flush_count > len(updates[-1].get('emitted_units', [])):
        raise ValueError(f'ASR case {case_id} has an invalid eof_flush_unit_count.')
    if flush_count:
        actual_tail = updates[-1]['emitted_units'][-flush_count:]
        allowed_eof_stability = {'end_of_recording_unconfirmed', 'stable_sentence_terminal_eof'}
        if any(unit.get('stability') not in allowed_eof_stability for unit in actual_tail):
            raise ValueError(f'ASR case {case_id} EOF flush units lack an accepted terminal/ unconfirmed stability marker.')
        if tails and [_unit_signature(unit) for unit in actual_tail] != [_unit_signature(unit) for unit in tails]:
            raise ValueError(f'ASR case {case_id} EOF units are not present exactly once at the end of the final update.')
    audit_units = streaming.get('translation_units')
    if audit_units is not None:
        if not isinstance(audit_units, list):
            raise ValueError(f'ASR case {case_id} translation_units audit field must be a list.')
        actual_units = [unit for update in updates for unit in update.get('emitted_units', [])]
        if [_unit_signature(unit) for unit in audit_units] != [_unit_signature(unit) for unit in actual_units]:
            raise ValueError(f'ASR case {case_id} emitted units disagree with the translation_units audit list.')
    _validate_streaming({'updates': updates}, duration, case_id)
    return updates


def alignment_evidence(case: dict) -> dict:
    streaming = case.get('streaming') or {}
    updates = streaming.get('updates') or []
    per_update = [
        {'update_index': index, 'audio_end_seconds': update.get('audio_end_seconds'), 'boundary_conflict': update.get('boundary_conflict')}
        for index, update in enumerate(updates) if update.get('boundary_conflict') is not None
    ]
    return {
        'transcript_alignment_status': streaming.get('transcript_alignment_status'),
        'unresolved_alignment': streaming.get('transcript_alignment_status') == 'unresolved_committed_prefix_conflict',
        'boundary_conflicts': streaming.get('boundary_conflicts', []),
        'per_update_boundary_conflicts': per_update,
        'unresolved_final_hypothesis_present': bool(streaming.get('unresolved_final_hypothesis')),
    }


def _validate_streaming(streaming: dict | None, duration: float, case_id: str) -> None:
    updates = (streaming or {}).get('updates')
    if not isinstance(updates, list) or not updates:
        raise ValueError(f'ASR case {case_id} lacks streaming.updates.')
    previous_end = -1.0
    previous_emit = -1.0
    for index, update in enumerate(updates):
        end = _finite_number(update.get('audio_end_seconds'), f'{case_id}.stream[{index}].audio_end_seconds')
        inference = _finite_number(update.get('inference_seconds'), f'{case_id}.stream[{index}].inference_seconds')
        if inference < 0 or not 0 < end <= duration + 1e-4 or (previous_end >= 0 and end <= previous_end):
            raise ValueError(f'ASR case {case_id} has invalid streaming order or duration.')
        previous_end = end
        units = update.get('emitted_units')
        if not isinstance(units, list):
            raise ValueError(f'ASR case {case_id} update {index} lacks emitted_units.')
        for j, unit in enumerate(units):
            start = _finite_number(unit.get('source_start_seconds'), f'{case_id}.unit[{j}].source_start_seconds')
            finish = _finite_number(unit.get('source_end_seconds'), f'{case_id}.unit[{j}].source_end_seconds')
            emitted = _finite_number(unit.get('emitted_at_audio_seconds'), f'{case_id}.unit[{j}].emitted_at_audio_seconds')
            if not 0 <= start < finish <= duration + 1e-4 or not finish <= emitted <= end + 1e-4:
                raise ValueError(f'ASR case {case_id} unit {j} has invalid or noncausal source/emission clocks.')
            if emitted < previous_emit or not isinstance(unit.get('text'), str) or not unit['text'].strip():
                raise ValueError(f'ASR case {case_id} emitted-unit order or text is invalid.')
            previous_emit = emitted


def _source_segments_for_update(mode: str, update: dict, update_index: int) -> list[dict]:
    if mode == 'fixed_8s_packet_baseline':
        rows = update['segments']
        result = []
        for index, row in enumerate(rows):
            result.append({
                'start': float(row.get('start', row.get('start_seconds'))),
                'end': float(row.get('end', row.get('end_seconds'))), 'text': row['text'],
                '_translation_key': f'p{update_index}-s{index}',
                'source_metadata': row,
            })
        return result
    result = []
    for index, unit in enumerate(update['emitted_units']):
        result.append({
            'start': float(unit['source_start_seconds']),
            'end': float(unit['source_end_seconds']),
            'text': unit['text'],
            '_translation_key': f'u{update_index}-{index}',
            'source_metadata': unit,
        })
    return result


def _call_logger(original):
    measured, calls = EVALUATOR.call_log_wrapper(original)

    def tracked(segments, context='', preset='gemma4', glossary=None):
        original_request = gemma._request
        usage = {}

        def capture_usage(path, payload=None, timeout=gemma.REQUEST_TIMEOUT):
            response = original_request(path, payload, timeout)
            if path == '/v1/chat/completions' and isinstance(response, dict) and isinstance(response.get('usage'), dict):
                usage.update({key: response['usage'].get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
            return response

        gemma._request = capture_usage
        try:
            return measured(segments, context=context, preset=preset, glossary=glossary)
        finally:
            gemma._request = original_request
            if calls:
                calls[-1].update({
                    'translation_keys': [row.get('_translation_key') for row in segments],
                    'source_characters': sum(len(row['text']) for row in segments),
                    'token_usage': usage or None,
                })

    return tracked, calls


def _run_update(mode: str, update: dict, index: int, translation_fn) -> tuple[list[dict], list[dict]]:
    sources = _source_segments_for_update(mode, update, index)
    if not sources:
        return [], []
    measured, calls = _call_logger(translation_fn)
    gemma.translate_batch = measured
    try:
        outputs = list(engine.gemma_segments(sources, preset=PRESET, context='', glossary=None))
        translated = []
        for source, output in zip(sources, outputs, strict=True):
            metadata = source['source_metadata']
            emitted = (float(update['audio_end_seconds']) if mode == 'fixed_8s_packet_baseline'
                       else float(metadata['emitted_at_audio_seconds']))
            translated.append({
                'key': source['_translation_key'],
                'source_start_seconds': float(source['start']),
                'source_end_seconds': float(source['end']),
                'emitted_at_audio_seconds': emitted,
                'source_text': source['text'],
                'translation': output.text,
                'source_characters': len(source['text']),
                'translation_characters': len(output.text),
                'needs_review': bool(getattr(output, 'needs_review', False)),
                'stability': metadata.get('stability') if mode != 'fixed_8s_packet_baseline' else None,
                'boundary_kind': metadata.get('boundary_kind') if mode != 'fixed_8s_packet_baseline' else None,
                'forced_fragment': bool(metadata.get('forced_fragment', False)) if mode != 'fixed_8s_packet_baseline' else False,
                'contains_unstable_tail': bool(metadata.get('contains_unstable_tail', False)) if mode != 'fixed_8s_packet_baseline' else False,
            })
        if len(outputs) != len(sources):
            raise ValueError(f'Expected {len(sources)} outputs, received {len(outputs)}.')
        return translated, calls
    finally:
        gemma.translate_batch = translation_fn


def simulate_case(mode: str, updates: list[dict], translate_fn) -> dict:
    emitted = []
    event_rows = []
    previous_finish = 0.0
    task_finish_times = []
    overflow_updates = []
    pending_peak = 0
    for update_index, update in enumerate(updates):
        audio_arrival = float(update['audio_end_seconds'])
        pending_before = sum(finish > audio_arrival for finish in task_finish_times)
        pending_peak = max(pending_peak, pending_before + 1)
        if pending_before >= QUEUE_CAPACITY:
            overflow_updates.append(update_index)
        scheduled_start = max(audio_arrival, previous_finish)
        asr_finish = scheduled_start + float(update['inference_seconds'])
        translated, calls = _run_update(mode, update, update_index, translate_fn)
        completion_by_key = {}
        call_finish = asr_finish
        for call in calls:
            call_finish += float(call['elapsed_seconds'])
            for key in call['translation_keys']:
                if key:
                    completion_by_key[key] = call_finish
        for row in translated:
            finish = completion_by_key.get(row['key'], asr_finish)
            row['scheduled_translation_complete_audio_seconds'] = round(finish, 4)
            row['simulated_latency_after_asr_emission_seconds'] = round(max(0.0, finish - row['emitted_at_audio_seconds']), 4)
            row['simulated_latency_after_source_end_seconds'] = round(finish - row['source_end_seconds'], 4)
            emitted.append(row)
        previous_finish = max(call_finish, asr_finish)
        task_finish_times.append(previous_finish)
        event_rows.append({
            'update_index': update_index,
            'audio_arrival_seconds': audio_arrival,
            'asr_inference_seconds': float(update['inference_seconds']),
            'scheduled_asr_start_seconds': round(scheduled_start, 4),
            'scheduled_asr_complete_seconds': round(asr_finish, 4),
            'translation_calls': calls,
            'translation_seconds': round(sum(float(call['elapsed_seconds']) for call in calls), 4),
            'scheduled_update_complete_seconds': round(previous_finish, 4),
            'backlog_wait_before_asr_seconds': round(scheduled_start - audio_arrival, 4),
            'pending_updates_before_arrival': pending_before,
            'would_exceed_live_queue_capacity': pending_before >= QUEUE_CAPACITY,
            'emitted_unit_count': len(translated),
        })
    return {
        'updates': event_rows,
        'outputs': emitted,
        'measured_asr_seconds': round(sum(float(row['inference_seconds']) for row in updates), 4),
        'measured_translation_seconds': round(sum(call['elapsed_seconds'] for row in event_rows for call in row['translation_calls']), 4),
        'measured_serial_compute_seconds': round(sum(float(row['inference_seconds']) for row in updates) + sum(call['elapsed_seconds'] for row in event_rows for call in row['translation_calls']), 4),
        'simulated_completion_audio_clock_seconds': round(previous_finish, 4),
        'max_pending_updates_including_active': pending_peak,
        'queue_capacity': QUEUE_CAPACITY,
        'would_exceed_live_queue_capacity': bool(overflow_updates),
        'capacity_exceeded_update_indices': overflow_updates,
        'duplicate_overlapping_translations_retained': _overlapping_duplicate_count(emitted),
    }


def _overlapping_duplicate_count(rows: list[dict]) -> int:
    duplicates = 0
    seen = []
    for row in rows:
        if any(old['translation'] == row['translation'] and old['source_start_seconds'] < row['source_end_seconds'] and old['source_end_seconds'] > row['source_start_seconds'] for old in seen):
            duplicates += 1
        seen.append(row)
    return duplicates


def self_test() -> None:
    # Canonical units are grouped by commit horizon exactly once. Event rows
    # are cross-checks and must not be appended as a second source.
    grouped = group_policy_units(
        [{'update_index': 0}, {'update_index': 1}],
        [{'text': 'terminal。', 'emitted_update_index': 1, 'stability': 'la2_stable'},
         {'text': 'tail', 'emitted_update_index': 1, 'stability': 'end_of_recording_unconfirmed'}],
        'tail-only',
    )
    assert grouped == [[], [
        {'text': 'terminal。', 'emitted_update_index': 1, 'stability': 'la2_stable'},
        {'text': 'tail', 'emitted_update_index': 1, 'stability': 'end_of_recording_unconfirmed'},
    ]]

    baseline = [
        {'audio_end_seconds': 8, 'inference_seconds': 2, 'segments': []},
        {'audio_end_seconds': 15, 'inference_seconds': 2, 'segments': []},
    ]
    # An empty-ASR update still consumes its measured compute time and delays the next update.
    result = simulate_case('fixed_8s_packet_baseline', baseline, lambda *args, **kwargs: [])
    assert result['updates'][0]['scheduled_asr_complete_seconds'] == 10
    assert result['updates'][1]['scheduled_asr_start_seconds'] == 15
    stable = [
        {'audio_end_seconds': 1, 'inference_seconds': 4.1, 'emitted_units': []},
        {'audio_end_seconds': 2, 'inference_seconds': 4.1, 'emitted_units': []},
        {'audio_end_seconds': 3, 'inference_seconds': 4.1, 'emitted_units': []},
        {'audio_end_seconds': 4, 'inference_seconds': 4.1, 'emitted_units': []},
    ]
    queued = simulate_case('la2_stable_commit_batches', stable, lambda *args, **kwargs: [])
    assert queued['would_exceed_live_queue_capacity']
    assert queued['capacity_exceeded_update_indices'] == [3]
    assert _overlapping_duplicate_count([
        {'translation': 'repeat', 'source_start_seconds': 0, 'source_end_seconds': 2},
        {'translation': 'repeat', 'source_start_seconds': 1, 'source_end_seconds': 3},
    ]) == 1

    # A separately-recorded EOF tail is attached to the last update once, with no
    # extra ASR cost; the original report object stays unchanged.
    tail = {
        'text': '残った尾。', 'source_start_seconds': 1.2, 'source_end_seconds': 1.8,
        'emitted_at_audio_seconds': 2.0, 'stability': 'end_of_recording_unconfirmed',
        'word_timestamps': [],
    }
    case = {'streaming': {
        'transcript_alignment_status': 'aligned',
        'updates': [{'audio_end_seconds': 2.0, 'inference_seconds': 0.6, 'emitted_units': []}],
        'final_tail_units': [tail], 'translation_units': [tail],
    }}
    normalized = _prepare_streaming_updates(case, 2.0, 'tail-only')
    assert case['streaming']['updates'][0]['emitted_units'] == []
    assert normalized[0]['inference_seconds'] == 0.6
    assert normalized[0]['eof_flush_unit_count'] == 1
    translated = simulate_case('la2_stable_commit_batches', normalized, lambda segments, **kwargs: ['translated'] * len(segments))
    assert len(translated['outputs']) == 1
    assert translated['outputs'][0]['stability'] == 'end_of_recording_unconfirmed'
    assert sum(update['asr_inference_seconds'] for update in translated['updates']) == 0.6
    assert sum(len(update['translation_calls']) for update in translated['updates']) == 1

    # The current ASR convention may already append the single EOF tail to the
    # final update and keep a provenance copy. The adapter must leave it alone.
    already_flushed = {'streaming': {
        'transcript_alignment_status': 'aligned',
        'updates': [{'audio_end_seconds': 2.0, 'inference_seconds': 0.6,
                     'emitted_units': [tail], 'eof_flush_unit_count': 1}],
        'final_tail_units': [tail], 'translation_units': [tail],
    }}
    once = _prepare_streaming_updates(already_flushed, 2.0, 'pre-appended-tail')
    assert len(once[-1]['emitted_units']) == 1
    assert once[-1]['eof_flush_unit_count'] == 1
    no_double = simulate_case('la2_stable_commit_batches', once, lambda segments, **kwargs: ['translated'] * len(segments))
    assert len(no_double['outputs']) == 1
    assert sum(len(update['translation_calls']) for update in no_double['updates']) == 1


def _new_report(asr_path: Path, asr: dict, cases: list[dict], model: dict) -> dict:
    return {
        'schema_version': 1,
        'status': 'running',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Only the frozen 20 development audio cases. Compares fixed 8-second overlapping ASR packets translated in live per-packet grouping with context off against causal LA2 stable commit units. No held-out cases, transcript/reference text as model input, glossary, joining of future text, or live-browser timing claims.',
        'corpus': {
            'path': str(CORPUS_PATH.relative_to(ROOT)),
            'manifest_sha256': sha256(CORPUS_PATH),
            'partition': 'development',
            'case_ids': [row['id'] for row in cases],
            'heldout_rows_used': 0,
        },
        'asr_report': {
            'path': str(asr_path.resolve().relative_to(ROOT)),
            'sha256': sha256(asr_path),
            'asr_model': asr.get('model'),
            'causal_method': (asr.get('streaming', {}).get('causality_note')
                              or asr.get('method', {}).get('streaming_causality')),
        },
        'model': model,
        'translation_policy': {
            'preset': PRESET,
            'prompt_builder_source_sha256': model['prompt_builder_source_sha256'],
            'engine_gemma_segments_source_sha256': hashlib.sha256(inspect.getsource(engine.gemma_segments).encode()).hexdigest(),
            'temperature': 0,
            'context': 'off for both modes; reset at each ASR update and recording',
            'glossary': 'none',
            'strict_output_validation': 'Existing gemma.translate_batch strict ordered-ID JSON/protocol validation; every error retained, no retries.',
            'model_lock_simulation': 'Serial ASR plus translation per update. Every ASR update cost is charged, including updates with no emitted units; translate after the update ASR pass completes.',
            'translation_grouping': 'engine.gemma_segments is called once per ASR update, preserving its production max-4-segment / 900-byte request batching.',
        },
        'stages': {
            'fixed_8s_packet_baseline': {
                'status': 'pending',
                'description': 'Translate each 8-second/1-second-overlap ASR packet independently, using the exact live per-packet source grouping and empty context. Preserve repeated English cues; caption_boundaries is off in the production baseline.',
                'timing_basis': 'Measured ASR update durations plus measured offline translation-call durations, serially replayed against media-clock packet arrival. Reconstructed schedule, not browser latency.',
                'cases': [], 'errors': [],
            },
            'la2_stable_commit_batches': {
                'status': 'pending_policy_repair',
                'pending_reason': 'Await root decision on six unresolved final-prefix cases; these remain failures in the 20-case denominator and must not be replaced by incomplete committed text.',
                'description': 'Translate only stable units emitted by each causal ASR update, in the same update/commit order. End-of-recording unconfirmed residual units are labeled, not removed.',
                'timing_basis': 'Measured ASR update durations plus measured offline translation-call durations, serially replayed against media-clock update arrival. Reconstructed schedule, not browser latency.',
                'cases': [], 'errors': [],
            },
        },
        'interpretation_limits': [
            'This comparison isolates translation grouping on two ASR schedules; the ASR update schedules themselves are not treated as equivalent recognition systems.',
            'Model-assisted semantic review is required before accuracy claims. Repeated English caused by overlapping packets remains visible and is not reconciled away.',
            'Offline model-call runtimes are used only for a compute-aware simulated serial schedule; they are not live extension/browser latency measurements.',
            'Queue-capacity flags indicate reconstructed updates that would arrive while three worker/queued packets are still pending; cases are never dropped to improve the comparison.',
            'Unresolved committed-prefix alignment is retained as an explicit failure; partial committed units do not convert the case to a pass.',
        ],
    }


def run(asr_path: Path, stage_key: str) -> dict:
    allowed_stages = ('fixed_8s_packet_baseline', 'la2_stable_commit_batches')
    if stage_key not in allowed_stages:
        raise ValueError(f'Stage must be one of {allowed_stages}.')
    manifest, cases = load_development_cases()
    asr, asr_cases, normalized_stream_updates = validate_asr_report(asr_path, manifest, cases)
    model = EVALUATOR.model_info()
    if OUTPUT_PATH.exists():
        report = json.loads(OUTPUT_PATH.read_text())
        if report.get('schema_version') != 1 or report.get('corpus', {}).get('manifest_sha256') != sha256(CORPUS_PATH):
            raise RuntimeError('Existing report corpus/schema does not match; refusing to overwrite it.')
        if report.get('asr_report', {}).get('sha256') != sha256(asr_path):
            raise RuntimeError('Existing report was built from a different ASR artifact; refusing to merge.')
        frozen_model = report.get('model', {})
        if frozen_model.get('preset') != model.get('preset') or frozen_model.get('provenance') != model.get('provenance') or frozen_model.get('prompt_builder_source_sha256') != model.get('prompt_builder_source_sha256'):
            raise RuntimeError('Existing report Gemma provenance or prompt builder changed; refusing to merge.')
        if report.get('translation_policy', {}).get('engine_gemma_segments_source_sha256') != hashlib.sha256(inspect.getsource(engine.gemma_segments).encode()).hexdigest():
            raise RuntimeError('Existing report engine.gemma_segments changed; refusing to merge.')
    else:
        if stage_key != 'fixed_8s_packet_baseline':
            raise RuntimeError('The LA2 stage cannot start without the existing baseline report.')
        report = _new_report(asr_path, asr, cases, model)
        write_json(OUTPUT_PATH, report)
    stage = report['stages'].get(stage_key)
    if not isinstance(stage, dict) or stage.get('cases') or stage.get('status') not in ('pending', 'pending_policy_repair'):
        raise RuntimeError(f'Stage {stage_key} is absent, already started, or already complete; preserving existing report.')
    if stage_key == 'la2_stable_commit_batches' and report['stages']['fixed_8s_packet_baseline'].get('status') not in ('complete', 'complete_with_errors'):
        raise RuntimeError('LA2 cannot start until the fixed-packet baseline is complete.')
    stage['status'] = 'running'
    stage['warmup'] = {'excluded': True, 'description': 'One synthetic greeting before this stage.'}
    stage['source_report_sha256'] = sha256(asr_path)
    report['status'] = f'running_{stage_key}'
    write_json(OUTPUT_PATH, report)
    original = gemma.translate_batch
    gemma.PORT, gemma.BASE = PORT, f'http://{gemma.HOST}:{PORT}'
    try:
        warm_start = time.monotonic()
        original([{'start': 0.0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        stage['warmup']['elapsed_seconds_excluded'] = round(time.monotonic() - warm_start, 4)
        input_key, list_key = (('overlap_baseline', 'chunks') if stage_key == 'fixed_8s_packet_baseline'
                               else ('streaming', 'updates'))
        for frozen in cases:
            source_case = asr_cases[str(frozen['id'])]
            source_mode = source_case[input_key]
            raw_updates = source_mode[list_key]
            if input_key == 'overlap_baseline':
                updates = [{
                    'audio_end_seconds': row.get('end_seconds', row.get('end', row.get('audio_end_seconds'))),
                    'inference_seconds': row['inference_seconds'],
                    'segments': row['segments'],
                } for row in raw_updates]
            else:
                updates = normalized_stream_updates[str(frozen['id'])]
            streaming_source = source_case.get('streaming') or {}
            raw_stream_updates = streaming_source.get('updates') or []
            tail_units = streaming_source.get('final_tail_units', source_case.get('final_tail_units', [])) or []
            tail_audit = None
            if stage_key == 'la2_stable_commit_batches':
                raw_last_units = raw_stream_updates[-1].get('emitted_units', []) if raw_stream_updates else []
                raw_counts = {}
                for unit in raw_last_units:
                    signature = _unit_signature(unit)
                    raw_counts[signature] = raw_counts.get(signature, 0) + 1
                adapter_attached = sum(raw_counts.get(_unit_signature(unit), 0) == 0 for unit in tail_units)
                tail_audit = {
                    'final_tail_unit_count': len(tail_units),
                    'source_last_update_eof_flush_unit_count': raw_stream_updates[-1].get('eof_flush_unit_count', 0) if raw_stream_updates else 0,
                    'adapter_attached_separate_tail_count': adapter_attached,
                    'normalized_last_update_eof_flush_unit_count': updates[-1].get('eof_flush_unit_count', 0) if updates else 0,
                }
            case_started = time.monotonic()
            try:
                replay = simulate_case(stage_key, updates, original)
                evidence = alignment_evidence(source_case)
                status = ('forced_fail_unresolved_alignment'
                          if stage_key == 'la2_stable_commit_batches' and evidence['unresolved_alignment']
                          else 'forced_fail_missing_translation_input'
                          if not replay['outputs'] else 'ok')
                row = {
                    'id': frozen['id'], 'partition': 'development',
                    'audio_sha256': frozen.get('audio_sha256'),
                    'duration_seconds': frozen['duration_seconds'],
                    'english_reference': frozen['english_reference'],
                    'asr_updates': raw_updates,
                    'replay': replay,
                    'asr_alignment_evidence': evidence,
                    'eof_tail_audit': tail_audit,
                    'missing_translation_input': not replay['outputs'],
                    'offline_translation_wall_seconds': round(time.monotonic() - case_started, 4),
                    'status': status,
                    'failure_reason': ('Committed-prefix alignment remained unresolved after final ASR update.'
                                      if status == 'forced_fail_unresolved_alignment'
                                      else 'No translation source units or captions were emitted.'
                                      if status == 'forced_fail_missing_translation_input' else None),
                }
            except Exception as exc:
                row = {
                    'id': frozen['id'], 'partition': 'development',
                    'audio_sha256': frozen.get('audio_sha256'),
                    'duration_seconds': frozen['duration_seconds'],
                    'english_reference': frozen['english_reference'],
                    'asr_updates': raw_updates,
                    'replay': None,
                    'asr_alignment_evidence': alignment_evidence(source_case),
                    'eof_tail_audit': tail_audit,
                    'missing_translation_input': True,
                    'offline_translation_wall_seconds': round(time.monotonic() - case_started, 4),
                    'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc),
                }
                stage['errors'].append({'id': frozen['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            stage['cases'].append(row)
            report[f'completed_{stage_key}_cases'] = len(stage['cases'])
            write_json(OUTPUT_PATH, report)
        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['aggregate'] = {
            'case_count': len(stage['cases']),
            'error_count': len(stage['errors']),
            'forced_fail_count': sum(row['status'].startswith('forced_fail') for row in stage['cases']),
            'offline_translation_wall_seconds': round(sum(row['offline_translation_wall_seconds'] for row in stage['cases']), 4),
            'measured_asr_seconds': round(sum((row.get('replay') or {}).get('measured_asr_seconds', 0) for row in stage['cases']), 4),
            'measured_translation_seconds': round(sum((row.get('replay') or {}).get('measured_translation_seconds', 0) for row in stage['cases']), 4),
            'measured_serial_compute_seconds': round(sum((row.get('replay') or {}).get('measured_serial_compute_seconds', 0) for row in stage['cases']), 4),
            'capacity_exceeded_case_ids': [row['id'] for row in stage['cases'] if (row.get('replay') or {}).get('would_exceed_live_queue_capacity')],
        }
        statuses = [item.get('status') for item in report['stages'].values()]
        if all(value in ('complete', 'complete_with_errors') for value in statuses):
            report['status'] = 'complete_with_errors' if any(value == 'complete_with_errors' for value in statuses) else 'complete'
        else:
            report['status'] = 'baseline_complete_candidate_pending_policy_repair' if stage_key == 'fixed_8s_packet_baseline' else 'candidate_complete'
        report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT_PATH, report)
        return report
    finally:
        gemma.translate_batch = original
        owned_process = gemma._process
        owner_matches = gemma._owner_pid == os.getpid() if owned_process is not None else True
        gemma.shutdown()
        process_exit_code = owned_process.poll() if owned_process is not None else None
        stage['cleanup'] = {
            'shutdown_called': True,
            'owner_matches_this_evaluator': owner_matches,
            'owned_process_pid': owned_process.pid if owned_process is not None else None,
            'owned_model_unloaded': gemma._process is None and (owned_process is None or process_exit_code is not None),
            'owned_process_released': owned_process is None or process_exit_code is not None,
            'owned_process_exit_code': process_exit_code,
            'port_accepting_connections_after_shutdown': port_accepting_connections(PORT),
        }
        write_json(OUTPUT_PATH, report)


def _policy_unit_signature(unit: dict) -> tuple:
    return (
        unit.get('text'),
        round(float(unit.get('source_start_seconds', -1)), 6),
        round(float(unit.get('source_end_seconds', -1)), 6),
        round(float(unit.get('emitted_at_audio_seconds', -1)), 6),
        unit.get('emitted_update_index'),
        unit.get('boundary_kind'),
        unit.get('stability'),
        bool(unit.get('forced_fragment', False)),
    )


def group_policy_units(events: list[dict], units: list[dict], case_id: str) -> list[list[dict]]:
    """Group the canonical replay units; event emissions are validation-only copies."""
    grouped = [[] for _ in events]
    for index, unit in enumerate(units):
        emit_index = unit.get('emitted_update_index')
        if type(emit_index) is not int or not 0 <= emit_index < len(events):
            raise ValueError(f'Policy A case {case_id} has unit {index} with an invalid emitted update index.')
        grouped[emit_index].append(unit)
    return grouped


def validate_policy_a_replay(replay_path: Path, asr_path: Path, baseline_path: Path,
                             cases: list[dict]) -> tuple[dict, dict[str, dict], dict[str, list[dict]]]:
    replay = json.loads(replay_path.read_text())
    asr = json.loads(asr_path.read_text())
    if replay.get('status') != 'complete':
        raise ValueError('Boundary replay report must be complete.')
    provenance = replay.get('provenance') or {}
    manifest_sha = sha256(CORPUS_PATH)
    asr_sha = sha256(asr_path)
    if provenance.get('partition') != 'development' or provenance.get('manifest_sha256') != manifest_sha:
        raise ValueError('Boundary replay report does not reference the frozen development manifest.')
    if provenance.get('asr_report_sha256') != asr_sha:
        raise ValueError('Boundary replay report ASR hash does not match the original streaming ASR report.')
    if asr.get('status') != 'complete' or asr.get('owned_model_unloaded') is not True or asr.get('owned_gpu_process_released') is not True or asr.get('worker_exit_code') != 0:
        raise ValueError('Original ASR report must be complete with cleanup verified.')
    if asr.get('corpus', {}).get('manifest_sha256') != manifest_sha:
        raise ValueError('Original ASR report manifest does not match the frozen corpus.')
    asr_rows = asr.get('cases')
    if not isinstance(asr_rows, list) or len(asr_rows) != EXPECTED_CASES:
        raise ValueError('Original ASR report must contain exactly 20 development rows.')
    asr_ids = [str(row.get('id')) for row in asr_rows]
    case_ids = {str(row['id']) for row in cases}
    if len(set(asr_ids)) != EXPECTED_CASES or set(asr_ids) != case_ids:
        raise ValueError('Original ASR case IDs do not match the frozen development IDs.')
    asr_by_id = {str(row['id']): row for row in asr_rows}

    baseline = json.loads(baseline_path.read_text())
    if baseline.get('corpus', {}).get('manifest_sha256') != manifest_sha or baseline.get('asr_report', {}).get('sha256') != asr_sha:
        raise ValueError('Completed fixed-packet baseline does not use the same corpus and ASR report.')
    if baseline.get('stages', {}).get('fixed_8s_packet_baseline', {}).get('status') != 'complete':
        raise ValueError('Fixed-packet baseline must be complete before this candidate comparison.')
    if baseline.get('stages', {}).get('la2_stable_commit_batches', {}).get('cases'):
        raise ValueError('Original negative LA2 attempt exists; refusing to contaminate the selected policy artifact.')

    policy = replay.get('policies', {}).get('sentence_terminal_eof')
    if not isinstance(policy, dict):
        raise ValueError('Selected sentence_terminal_eof policy is missing.')
    policy_cases = policy.get('cases')
    summary = policy.get('summary') or {}
    if not isinstance(policy_cases, list) or len(policy_cases) != EXPECTED_CASES:
        raise ValueError('Selected policy must contain exactly 20 development cases.')
    replay_ids = [str(row.get('id')) for row in policy_cases]
    if len(set(replay_ids)) != EXPECTED_CASES or set(replay_ids) != case_ids:
        raise ValueError('Selected policy case IDs do not match the development corpus.')
    expected_summary = {
        'case_count': 20,
        'update_count': 141,
        'cer_coverage_count': 20,
        'cer_uncovered_count': 0,
        'failed_published_prefix_cases': [],
        'published_prefix_conflict_failures': [],
        'non_eof_flush_unit_count': 19,
        'strict_before_eof_emission_unit_count': 8,
        'strict_before_eof_emission_case_count': 8,
        'eof_flush_unit_count': 3,
        'emitted_unit_count': 22,
        'forced_fragment_count': 0,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f'Selected policy summary {key}={summary.get(key)!r}, expected {value!r}.')
    if summary.get('policy') != 'sentence_terminal_eof':
        raise ValueError('Selected replay policy name is inconsistent.')
    metrics = replay.get('metrics', {}).get('sentence_terminal_eof') or {}
    if metrics.get('policy') != 'sentence_terminal_eof':
        raise ValueError('Selected policy metrics are missing or inconsistent.')
    for key, value in expected_summary.items():
        if metrics.get(key) != value:
            raise ValueError(f'Selected policy metrics {key}={metrics.get(key)!r}, expected {value!r}.')
    if metrics.get('same_saved_asr_decode_count') != 141 or abs(float(metrics.get('same_saved_asr_inference_seconds', -1)) - 40.8444) > 1e-4:
        raise ValueError('Selected policy does not preserve all original ASR update costs.')
    if abs(float(metrics.get('unchanged_8s_overlap_baseline_inference_seconds', -1)) - 12.7716) > 1e-4:
        raise ValueError('Selected policy altered the recorded 8-second baseline ASR cost.')

    manifest_by_id = {str(row['id']): row for row in cases}
    updates_by_id = {}
    units_total = eof_total = pre_eof_total = pre_eof_cases = 0
    for case_id, policy_case in {str(row['id']): row for row in policy_cases}.items():
        frozen = manifest_by_id[case_id]
        source_case = asr_by_id[case_id]
        if source_case.get('partition') != 'development' or source_case.get('audio_sha256') != frozen.get('audio_sha256'):
            raise ValueError(f'Original ASR provenance mismatch for case {case_id}.')
        if policy_case.get('transcript_alignment_status') != 'aligned' or policy_case.get('protocol_status') != 'pass' or policy_case.get('boundary_conflicts'):
            raise ValueError(f'Policy A case {case_id} failed alignment/protocol/conflict review.')
        events = policy_case.get('events')
        original_updates = source_case.get('streaming', {}).get('updates')
        units = policy_case.get('units')
        if not isinstance(events, list) or not isinstance(original_updates, list) or len(events) != len(original_updates):
            raise ValueError(f'Policy A case {case_id} update count differs from original ASR cost rows.')
        if int(policy_case.get('update_count', -1)) != len(events):
            raise ValueError(f'Policy A case {case_id} declares an unexpected update count.')
        if not isinstance(units, list):
            raise ValueError(f'Policy A case {case_id} has no canonical units list.')
        grouped = group_policy_units(events, units, case_id)
        previous_emit = -1.0
        eof_count = 0
        strict_count = 0
        for unit_index, unit in enumerate(units):
            start = _finite_number(unit.get('source_start_seconds'), f'{case_id}.units[{unit_index}].source_start_seconds')
            end = _finite_number(unit.get('source_end_seconds'), f'{case_id}.units[{unit_index}].source_end_seconds')
            emitted = _finite_number(unit.get('emitted_at_audio_seconds'), f'{case_id}.units[{unit_index}].emitted_at_audio_seconds')
            if not isinstance(unit.get('text'), str) or not unit['text'].strip() or not 0 <= start < end <= emitted <= float(frozen['duration_seconds']) + 1e-4:
                raise ValueError(f'Policy A case {case_id} has invalid unit text or source/emission clocks.')
            for word_index, word in enumerate(unit.get('word_timestamps', [])):
                word_start = _finite_number(word.get('start_seconds'), f'{case_id}.units[{unit_index}].word[{word_index}].start_seconds')
                word_end = _finite_number(word.get('end_seconds'), f'{case_id}.units[{unit_index}].word[{word_index}].end_seconds')
                if not 0 <= word_start <= word_end <= float(frozen['duration_seconds']) + 1e-4:
                    raise ValueError(f'Policy A case {case_id} has an invalid word timestamp.')
            if emitted < previous_emit:
                raise ValueError(f'Policy A case {case_id} units are not in causal emission order.')
            previous_emit = emitted
            if unit.get('forced_fragment') is not False:
                raise ValueError(f'Policy A case {case_id} unexpectedly contains a forced fragment.')
            if unit.get('stability') == 'end_of_recording_unconfirmed':
                eof_count += 1
            elif unit.get('stability') != 'la2_stable':
                raise ValueError(f'Policy A case {case_id} has an unknown unit stability state.')
            if emitted < float(frozen['duration_seconds']) - 1e-6:
                strict_count += 1
        if eof_count != int(policy_case.get('eof_flush_count', -1)):
            raise ValueError(f'Policy A case {case_id} EOF units are missing or duplicated.')
        if eof_count > 1:
            raise ValueError(f'Policy A case {case_id} emitted more than one EOF tail.')
        units_total += len(units)
        eof_total += eof_count
        pre_eof_total += strict_count
        pre_eof_cases += strict_count > 0

        normalized = []
        for index, (event, original_update, canonical_units) in enumerate(zip(events, original_updates, grouped, strict=True)):
            audio_end = _finite_number(event.get('audio_end_seconds'), f'{case_id}.events[{index}].audio_end_seconds')
            inference = _finite_number(original_update.get('inference_seconds'), f'{case_id}.asr_updates[{index}].inference_seconds')
            source_end = _finite_number(original_update.get('audio_end_seconds'), f'{case_id}.asr_updates[{index}].audio_end_seconds')
            if event.get('update_index') != index or abs(audio_end - source_end) > 1e-6:
                raise ValueError(f'Policy A case {case_id} event clocks/order differ from original ASR.')
            if abs(_finite_number(event.get('inference_seconds'), f'{case_id}.events[{index}].inference_seconds') - inference) > 1e-9:
                raise ValueError(f'Policy A case {case_id} ASR cost differs from original decode update.')
            event_units = event.get('emitted_units')
            if not isinstance(event_units, list):
                raise ValueError(f'Policy A case {case_id} event {index} lacks emitted-unit list.')
            if [_policy_unit_signature(unit) for unit in event_units] != [_policy_unit_signature(unit) for unit in canonical_units]:
                raise ValueError(f'Policy A case {case_id} event emission differs from the canonical units list.')
            for unit in canonical_units:
                if abs(float(unit['emitted_at_audio_seconds']) - audio_end) > 1e-6:
                    raise ValueError(f'Policy A case {case_id} unit was not emitted at its ASR update horizon.')
            normalized.append({
                'audio_end_seconds': audio_end,
                'inference_seconds': inference,
                'emitted_units': canonical_units,
            })
        updates_by_id[case_id] = normalized
    if units_total != 22 or eof_total != 3 or pre_eof_total != 8 or pre_eof_cases != 8:
        raise ValueError('Independent policy-A unit/EOF/pre-EOF totals do not match the reviewed summary.')

    total_updates = sum(len(row['streaming']['updates']) for row in asr_rows)
    total_asr_seconds = sum(float(update['inference_seconds']) for row in asr_rows for update in row['streaming']['updates'])
    if total_updates != 141 or abs(total_asr_seconds - 40.8444) > 1e-4:
        raise ValueError('Original streaming ASR input must contain all 141 updates and the reviewed 40.8444 seconds of inference.')
    return replay, {str(row['id']): row for row in policy_cases}, updates_by_id


def run_policy_a(replay_path: Path = BOUNDARY_REPLAY_PATH,
                 asr_path: Path = SOURCE_ASR_PATH,
                 baseline_path: Path = BASELINE_REPORT_PATH) -> dict:
    if POLICY_A_OUTPUT_PATH.exists():
        raise RuntimeError(f'Preserve the existing policy-A report; refusing to overwrite {POLICY_A_OUTPUT_PATH.relative_to(ROOT)}.')
    manifest, cases = load_development_cases()
    replay, policy_cases, event_updates = validate_policy_a_replay(replay_path, asr_path, baseline_path, cases)
    model = EVALUATOR.model_info()
    baseline = json.loads(baseline_path.read_text())
    frozen_model = baseline.get('model', {})
    if frozen_model.get('preset') != model.get('preset') or frozen_model.get('provenance') != model.get('provenance') or frozen_model.get('prompt_builder_source_sha256') != model.get('prompt_builder_source_sha256'):
        raise RuntimeError('Policy-A Gemma preset, weights or base prompt differ from the completed packet baseline.')
    if baseline.get('translation_policy', {}).get('engine_gemma_segments_source_sha256') != hashlib.sha256(inspect.getsource(engine.gemma_segments).encode()).hexdigest():
        raise RuntimeError('engine.gemma_segments changed since the completed packet baseline.')
    if port_accepting_connections(PORT):
        raise RuntimeError(f'Isolated E4B port {PORT} is already occupied; refusing model contention.')

    asr_sha = sha256(asr_path)
    replay_sha = sha256(replay_path)
    report = {
        'schema_version': 1,
        'status': 'running',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Separate selected-policy candidate: only the frozen 20 development audio cases, with policy A sentence-terminal commits plus exactly-once EOF flush. Original fixed-packet baseline is preserved and not rerun or overwritten. English references are review-only and are never sent to Gemma.',
        'corpus': {
            'path': str(CORPUS_PATH.relative_to(ROOT)), 'manifest_sha256': sha256(CORPUS_PATH),
            'partition': 'development', 'case_ids': [row['id'] for row in cases], 'heldout_rows_used': 0,
        },
        'source_asr': {
            'path': str(asr_path.resolve().relative_to(ROOT)), 'sha256': asr_sha,
            'model': json.loads(asr_path.read_text()).get('model'),
            'original_decode_update_count': sum(len(row['streaming']['updates']) for row in json.loads(asr_path.read_text())['cases']),
            'original_decode_inference_seconds': round(sum(update['inference_seconds'] for row in json.loads(asr_path.read_text())['cases'] for update in row['streaming']['updates']), 4),
        },
        'boundary_replay': {
            'path': str(replay_path.resolve().relative_to(ROOT)), 'sha256': replay_sha,
            'selected_policy': 'sentence_terminal_eof',
            'runner_sha256': replay.get('provenance', {}).get('replay_runner_sha256'),
            'summary': replay['policies']['sentence_terminal_eof']['summary'],
        },
        'baseline_comparison': {
            'path': str(baseline_path.resolve().relative_to(ROOT)), 'sha256_before_candidate': sha256(baseline_path),
            'stage': 'fixed_8s_packet_baseline',
            'baseline_stage_status': baseline['stages']['fixed_8s_packet_baseline']['status'],
            'baseline_translation_seconds': baseline['stages']['fixed_8s_packet_baseline']['aggregate']['measured_translation_seconds'],
            'rewritten_or_rerun': False,
        },
        'model': model,
        'translation_policy': {
            'preset': PRESET,
            'prompt_builder_source_sha256': model['prompt_builder_source_sha256'],
            'engine_gemma_segments_source_sha256': hashlib.sha256(inspect.getsource(engine.gemma_segments).encode()).hexdigest(),
            'temperature': 0,
            'thinking': 'off', 'context': 'off/reset for every update and recording', 'glossary': 'none',
            'source_inputs': 'Only canonical policy-A units; never the pristine Japanese references, future hypotheses, or English references.',
            'strict_output_validation': 'Existing gemma.translate_batch strict ordered-ID JSON/protocol validation; every error retained, no retries.',
            'grouping': 'Canonical units are grouped by emitted_update_index and passed once per ASR update through engine.gemma_segments; no future unit is batched early.',
            'latency_simulation': 'Charge each original ASR update inference_seconds once, including updates with no emitted unit; serially schedule Gemma requests after their update ASR completes. These are reconstructed times, not browser latency.',
        },
        'warmup': {'excluded': True, 'description': 'One synthetic greeting before measured policy-A translations.'},
        'stage': {'status': 'pending', 'cases': [], 'errors': []},
        'interpretation_limits': [
            'Policy-A replay diagnostics and English translation quality are different measurements; source CER is not meaning accuracy.',
            'Six cases failed in the original streaming-prefix state; policy A was independently replayed as aligned with no published-prefix conflicts. This candidate uses only its canonical aligned units.',
            'Offline model-call durations are replayed with original measured ASR costs to estimate end-to-end scheduling and backlog; they are not browser or live-capture latency.',
            'Only 20 development cases were accessed; no held-out recordings or reference translations were provided to Gemma.',
        ],
    }
    stage = report['stage']
    stage['status'] = 'running'
    stage['warmup'] = {'excluded': True}
    write_json(POLICY_A_OUTPUT_PATH, report)
    original = gemma.translate_batch
    gemma.PORT, gemma.BASE = PORT, f'http://{gemma.HOST}:{PORT}'
    try:
        warm_start = time.monotonic()
        original([{'start': 0.0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        stage['warmup']['elapsed_seconds_excluded'] = round(time.monotonic() - warm_start, 4)
        for frozen in cases:
            case_id = str(frozen['id'])
            policy_case = policy_cases[case_id]
            updates = event_updates[case_id]
            case_started = time.monotonic()
            try:
                replayed = simulate_case('la2_stable_commit_batches', updates, original)
                status = 'ok' if replayed['outputs'] else 'forced_fail_missing_translation_input'
                row = {
                    'id': frozen['id'], 'partition': 'development',
                    'audio_sha256': frozen.get('audio_sha256'),
                    'duration_seconds': frozen['duration_seconds'],
                    'english_reference': frozen['english_reference'],
                    'policy_alignment_status': policy_case['transcript_alignment_status'],
                    'protocol_status': policy_case['protocol_status'],
                    'boundary_conflicts': policy_case['boundary_conflicts'],
                    'canonical_source_unit_count': len(policy_case['units']),
                    'eof_flush_unit_count': policy_case['eof_flush_count'],
                    'strict_pre_eof_unit_count': sum(unit['emitted_at_audio_seconds'] < frozen['duration_seconds'] - 1e-6 for unit in policy_case['units']),
                    'asr_cost_update_count': len(updates),
                    'asr_cost_inference_seconds': round(sum(row['inference_seconds'] for row in updates), 4),
                    'replay': replayed,
                    'offline_translation_wall_seconds': round(time.monotonic() - case_started, 4),
                    'missing_translation_input': not replayed['outputs'],
                    'status': status,
                }
            except Exception as exc:
                row = {
                    'id': frozen['id'], 'partition': 'development',
                    'audio_sha256': frozen.get('audio_sha256'),
                    'duration_seconds': frozen['duration_seconds'],
                    'english_reference': frozen['english_reference'],
                    'policy_alignment_status': policy_case['transcript_alignment_status'],
                    'protocol_status': policy_case['protocol_status'],
                    'boundary_conflicts': policy_case['boundary_conflicts'],
                    'canonical_source_unit_count': len(policy_case['units']),
                    'eof_flush_unit_count': policy_case['eof_flush_count'],
                    'asr_cost_update_count': len(updates),
                    'asr_cost_inference_seconds': round(sum(row['inference_seconds'] for row in updates), 4),
                    'replay': None,
                    'offline_translation_wall_seconds': round(time.monotonic() - case_started, 4),
                    'missing_translation_input': True,
                    'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc),
                }
                stage['errors'].append({'id': frozen['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            stage['cases'].append(row)
            report['completed_cases'] = len(stage['cases'])
            write_json(POLICY_A_OUTPUT_PATH, report)
        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['aggregate'] = {
            'case_count': len(stage['cases']), 'error_count': len(stage['errors']),
            'forced_fail_count': sum(row['status'].startswith('forced_fail') for row in stage['cases']),
            'source_unit_count': sum(row['canonical_source_unit_count'] for row in stage['cases']),
            'eof_flush_unit_count': sum(row['eof_flush_unit_count'] for row in stage['cases']),
            'strict_pre_eof_unit_count': sum(row.get('strict_pre_eof_unit_count', 0) for row in stage['cases']),
            'asr_cost_update_count': sum(row['asr_cost_update_count'] for row in stage['cases']),
            'asr_cost_inference_seconds': round(sum(row['asr_cost_inference_seconds'] for row in stage['cases']), 4),
            'translation_wall_seconds': round(sum(row['offline_translation_wall_seconds'] for row in stage['cases']), 4),
            'measured_translation_seconds': round(sum((row.get('replay') or {}).get('measured_translation_seconds', 0) for row in stage['cases']), 4),
            'measured_serial_compute_seconds': round(sum((row.get('replay') or {}).get('measured_serial_compute_seconds', 0) for row in stage['cases']), 4),
            'capacity_exceeded_case_ids': [row['id'] for row in stage['cases'] if (row.get('replay') or {}).get('would_exceed_live_queue_capacity')],
        }
        report['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(POLICY_A_OUTPUT_PATH, report)
        return report
    finally:
        gemma.translate_batch = original
        owned_process = gemma._process
        owner_matches = gemma._owner_pid == os.getpid() if owned_process is not None else True
        gemma.shutdown()
        process_exit_code = owned_process.poll() if owned_process is not None else None
        stage['cleanup'] = {
            'shutdown_called': True,
            'owner_matches_this_evaluator': owner_matches,
            'owned_process_pid': owned_process.pid if owned_process is not None else None,
            'owned_model_unloaded': gemma._process is None and (owned_process is None or process_exit_code is not None),
            'owned_process_released': owned_process is None or process_exit_code is not None,
            'owned_process_exit_code': process_exit_code,
            'port_accepting_connections_after_shutdown': port_accepting_connections(PORT),
        }
        write_json(POLICY_A_OUTPUT_PATH, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', help='Run pure schedule/grouping checks without loading a model.')
    parser.add_argument('--asr-report', type=Path, help='Completed, validated development streaming-ASR report.')
    parser.add_argument('--stage', choices=('fixed_8s_packet_baseline', 'la2_stable_commit_batches'), help='Run exactly one pending translation stage; candidate stage requires a validated existing baseline report.')
    parser.add_argument('--policy-a', action='store_true', help='Run the separately approved sentence-terminal plus EOF replay candidate.')
    parser.add_argument('--boundary-replay', type=Path, default=BOUNDARY_REPLAY_PATH, help='Reviewed frozen policy replay artifact for --policy-a.')
    parser.add_argument('--run', action='store_true', help='Run one authorized 20-case translation stage.')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print('Streaming translation schedule checks passed.')
        return 0
    if args.policy_a:
        if not args.run or args.stage:
            parser.error('--policy-a requires --run and cannot be combined with --stage.')
        result = run_policy_a(args.boundary_replay.resolve(), (args.asr_report or SOURCE_ASR_PATH).resolve())
        print(json.dumps({
            'status': result['status'],
            'stage_status': result['stage']['status'],
            'cases': len(result['stage']['cases']),
            'errors': len(result['stage']['errors']),
            'report': str(POLICY_A_OUTPUT_PATH.relative_to(ROOT)),
        }, ensure_ascii=False))
        return 0 if result['status'] == 'complete' else 1
    if not args.run or not args.asr_report or not args.stage:
        parser.error('--asr-report, --stage, and --run are required unless --policy-a is selected.')
    result = run(args.asr_report.resolve(), args.stage)
    print(json.dumps({
        'status': result['status'],
        'stage': args.stage,
        'stage_status': result['stages'][args.stage]['status'],
        'baseline_cases': len(result['stages']['fixed_8s_packet_baseline']['cases']),
        'la2_cases': len(result['stages']['la2_stable_commit_batches']['cases']),
        'baseline_errors': len(result['stages']['fixed_8s_packet_baseline']['errors']),
        'la2_errors': len(result['stages']['la2_stable_commit_batches']['errors']),
        'report': str(OUTPUT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
