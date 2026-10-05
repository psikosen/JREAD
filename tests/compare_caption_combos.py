#!/usr/bin/env python3
"""Bounded, replayable caption context/glossary/boundary comparison.

Runs only when invoked directly. It never downloads models and uses a private
loopback port so the app's own Gemma listener is neither reused nor stopped.
"""
from __future__ import annotations

import hashlib
import inspect
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jread import engine, gemma, subtitles  # noqa: E402
from jread.server import reconcile_overlap, source_context  # noqa: E402

CASES_PATH = ROOT / 'tests/fixtures/caption-eval-cases.json'
GLOSSARY_PATH = ROOT / 'jread/caption-glossary.json'
VISIBLE_GLOSSARY_PATH = ROOT / 'tests/fixtures/caption-glossary.json'
MODEL_LABEL = 'Gemma 4 E4B'
PRESET = 'gemma4'
PORT = 18767
CONDITIONS = ('baseline', 'glossary', 'context', 'context_glossary')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def make_cues(case_id: str, segments: list[dict], translations: list[str], media_offset: float = 0.0) -> list[dict]:
    cues = []
    if len(segments) != len(translations):
        raise ValueError(f'{case_id}: received {len(translations)} translations for {len(segments)} segments')
    for index, (source, translation) in enumerate(zip(segments, translations, strict=True)):
        for sub_index, cue in enumerate(subtitles.assemble(source['start'] + media_offset, source['end'] + media_offset, translation)):
            cues.append(cue | {
                'id': f'{case_id}:s{index}:c{sub_index}',
                'source_text': source['text'],
                'source_segment_index': index,
                'source_segment_start': source['start'],
                'source_segment_end': source['end'],
                'media_time_offset': media_offset,
            })
    return cues


def baseline_accept(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], list[dict]]:
    combined = [dict(cue) for cue in existing]
    accepted = []
    for cue in incoming:
        duplicate = any(old['text'] == cue['text'] and old['start'] < cue['end'] and old['end'] > cue['start'] for old in combined)
        if not duplicate:
            added = dict(cue)
            combined.append(added)
            accepted.append(added)
    return combined, accepted


def accept_boundary(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], list[dict]]:
    combined, changed = reconcile_overlap(existing, incoming)
    return combined, changed


def source_text(case: dict) -> str:
    return ' '.join(segment['text'] for segment in case['source_segments'] if segment.get('text', '').strip())




def reset_case_histories(histories: dict[str, str], case: dict) -> dict[str, str]:
    """Apply session or per-case source-history boundaries before translation."""
    if case.get('reset_context') or case.get('reset_context_before'):
        return {condition: '' for condition in histories}
    return histories


def fake_translator(text: str, context: str) -> dict:
    """Tiny stand-in used to verify context isolation without model inference."""
    return {'source': text, 'context_received': context}


def check_synthetic_context_resets(corpus: dict) -> dict:
    checked = []
    for session in corpus['sessions']:
        if session.get('kind') != 'synthetic':
            continue
        histories = {'context': 'previous-session-source'}
        for case in session['cases']:
            histories = reset_case_histories(histories, case)
            source = source_text(case)
            result = fake_translator(source, histories['context'])
            if case.get('reset_context') and result['context_received']:
                raise AssertionError(f"{case['id']}: prior synthetic source leaked through its reset_context flag")
            checked.append({'case_id': case['id'], 'context_received': result['context_received']})
            histories['context'] = source_context(histories['context'], [{'source_text': source}])
    return {'status': 'pass', 'fake_translator_cases': len(checked), 'cases': checked}


def condition_context(condition: str) -> bool:
    return condition in ('context', 'context_glossary')


def condition_glossary(condition: str) -> bool:
    return condition in ('glossary', 'context_glossary')


def run_main() -> dict:
    corpus = json.loads(CASES_PATH.read_text())
    glossary_data = json.loads(GLOSSARY_PATH.read_text())
    visible_glossary = json.loads(VISIBLE_GLOSSARY_PATH.read_text())
    entries = glossary_data.get('entries', [])
    cases_total = sum(len(session['cases']) for session in corpus['sessions'])
    if cases_total > 24:
        raise RuntimeError(f'Bounded sweep allows at most 24 cases; found {cases_total}.')
    if not gemma.ready(PRESET) or not gemma.runtime_ready():
        raise RuntimeError('Pinned Gemma E4B and runtime must already be installed and verified; this runner never downloads them.')

    # Isolate this runner from the app's fixed port 18766. The gemma module owns
    # the process it starts here and its shutdown method terminates only that PID.
    gemma.PORT = PORT
    gemma.BASE = f'http://{gemma.HOST}:{PORT}'
    if not hasattr(gemma, '_eval_original_translate_batch'):
        gemma._eval_original_translate_batch = gemma.translate_batch
    original_translate_batch = gemma._eval_original_translate_batch

    result = {
        'schema_version': 1,
        'status': 'running',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved Japanese transcript replay with four unique deterministic E4B translation conditions and two deterministic boundary policies. Not real-live latency or an accuracy percentage.',
        'conditions': {
            'baseline': {'glossary': False, 'context': False},
            'glossary': {'glossary': True, 'context': False},
            'context': {'glossary': False, 'context': True},
            'context_glossary': {'glossary': True, 'context': True},
            'boundary_off': 'Existing exact English text plus time-overlap duplicate suppression.',
            'boundary_on': 'jread.server.reconcile_overlap over the same translation output; no extra model call.'
        },
        'counterbalance': 'For case index i, execution order is CONDITIONS rotated left by i modulo 4; each condition maintains its own successful source-only history within a session.',
        'model': {
            'preset': PRESET,
            'label': MODEL_LABEL,
            'provenance': gemma.PRESETS[PRESET],
            'device': 'CPU',
            'threads': 8,
            'runtime': json.loads(gemma.RUNTIME_MANIFEST.read_text()),
            'isolated_port': PORT,
            'temperature': 0,
            'warmup': 'One synthetic non-corpus translation, excluded from all recorded timings and outputs.'
        },
        'prompt_protocol': {
            'builder': 'jread.gemma._messages',
            'source_sha256': hashlib.sha256(inspect.getsource(gemma._messages).encode()).hexdigest(),
            'runner_sha256': sha256(Path(__file__).resolve()),
            'request_fingerprint': 'SHA-256 of the exact serialized chat messages passed to the translation call; includes deterministic source delimiter.',
        },
        'frozen_inputs': {
            'corpus_path': str(CASES_PATH.relative_to(ROOT)),
            'corpus_sha256': sha256(CASES_PATH),
            'jread_glossary_path': str(GLOSSARY_PATH.relative_to(ROOT)),
            'jread_glossary_sha256': sha256(GLOSSARY_PATH),
            'evaluation_glossary_path': str(VISIBLE_GLOSSARY_PATH.relative_to(ROOT)),
            'evaluation_glossary_sha256': sha256(VISIBLE_GLOSSARY_PATH),
            'corpus_cases': cases_total,
            'glossary_entry_count': len(entries),
        },
        'warmup': {'excluded': True},
        'reset_context_regression': check_synthetic_context_resets(corpus),
        'sessions': [],
        'errors': [],
    }
    output_path = ROOT / 'test-results/caption-combos.json'
    write_json(output_path, result)

    try:
        warm_start = time.monotonic()
        original_translate_batch([{'start': 0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        result['warmup'] = {'excluded': True, 'elapsed_seconds_excluded': round(time.monotonic() - warm_start, 3)}

        global_case_index = 0
        for session in corpus['sessions']:
            histories = {condition: '' for condition in CONDITIONS}
            session_result = {
                'id': session['id'],
                'kind': session.get('kind'),
                'reset_context': True,
                'cases': [],
                'boundary_runs': {
                    condition: {
                        'boundary_off': {'accepted_cues': [], 'case_updates': []},
                        'boundary_on': {'accepted_cues': [], 'case_updates': []},
                    } for condition in CONDITIONS
                }
            }
            boundary_state = {
                (condition, enabled): []
                for condition in CONDITIONS
                for enabled in (False, True)
            }
            for case in session['cases']:
                histories = reset_case_histories(histories, case)
                order = list(CONDITIONS)
                rotation = global_case_index % len(order)
                order = order[rotation:] + order[:rotation]
                case_result = {
                    'id': case['id'],
                    'provenance': case['provenance'],
                    'source_segments': case['source_segments'],
                    'media_start': case.get('media_start', 0.0),
                    'execution_order': order,
                    'conditions': {},
                    'boundary': {},
                    'checks': case.get('checks', {}),
                    'uncertainty': case.get('uncertainty', []),
                }
                if case.get('overlap_candidates'):
                    case_result['overlap_candidates'] = case['overlap_candidates']
                for condition in order:
                    context = histories[condition] if condition_context(condition) else ''
                    glossary = entries if condition_glossary(condition) else None
                    call_log = []

                    def measured(segments, context='', preset='gemma4', glossary=None):
                        started = time.monotonic()
                        prompt_sha = ''
                        try:
                            messages = gemma._messages(segments, context, glossary)
                            prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                            return original_translate_batch(segments, context=context, preset=preset, glossary=glossary)
                        finally:
                            call_log.append({
                                'elapsed_seconds': round(time.monotonic() - started, 3),
                                'segment_count': len(segments),
                                'source_texts': [item['text'] for item in segments],
                                'context_sent': context,
                                'glossary_enabled': bool(glossary),
                                'prompt_sha256': prompt_sha,
                            })

                    gemma.translate_batch = measured
                    started = time.monotonic()
                    try:
                        translated = list(engine.gemma_segments(
                            case['source_segments'],
                            preset=PRESET,
                            context=context,
                            glossary=glossary,
                        ))
                        elapsed = time.monotonic() - started
                        translations = [item.text for item in translated]
                        media_offset = case.get('media_start', 0.0)
                        cues = make_cues(case['id'], case['source_segments'], translations, media_offset)
                        condition_result = {
                            'status': 'ok',
                            'translation_seconds': round(elapsed, 3),
                            'batch_calls': call_log,
                            'history_used': context,
                            'segments': [
                                {'index': i, 'source': source, 'translation': text,
                                 'start': source['start'] + media_offset, 'end': source['end'] + media_offset,
                                 'source_start': source['start'], 'source_end': source['end'],
                                 'media_time_offset': media_offset,
                                 'segment_id': f"{case['id']}:s{i}"}
                                for i, (source, text) in enumerate(zip(case['source_segments'], translations, strict=True))
                            ],
                            'generated_cues': cues,
                        }
                    except Exception as exc:
                        elapsed = time.monotonic() - started
                        condition_result = {
                            'status': 'error',
                            'translation_seconds': round(elapsed, 3),
                            'batch_calls': call_log,
                            'history_used': context,
                            'error_type': type(exc).__name__,
                            'error': str(exc),
                            'segments': [],
                            'generated_cues': [],
                        }
                        result['errors'].append({'session_id': session['id'], 'case_id': case['id'], 'condition': condition, 'error_type': type(exc).__name__, 'error': str(exc)})
                    finally:
                        gemma.translate_batch = original_translate_batch
                    case_result['conditions'][condition] = condition_result
                    if condition_result['status'] == 'ok' and condition_context(condition):
                        packet_sources = [{'source_text': segment['text']} for segment in case['source_segments'] if segment.get('text', '').strip()]
                        histories[condition] = source_context(histories[condition], packet_sources)

                # Boundary conditions reuse each model condition's single set of translations.
                for condition in CONDITIONS:
                    generated = case_result['conditions'][condition]['generated_cues']
                    for enabled, label in ((False, 'boundary_off'), (True, 'boundary_on')):
                        existing = boundary_state[(condition, enabled)]
                        combined, changed = accept_boundary(existing, generated) if enabled else baseline_accept(existing, generated)
                        boundary_state[(condition, enabled)] = combined
                        update = {
                            'case_id': case['id'],
                            'accepted_cues': changed,
                            'assembled_cue_count': len(generated),
                            'accepted_cue_count': len(changed),
                            'combined_cue_count': len(combined),
                        }
                        case_result['boundary'][condition + '_' + label] = update
                        session_result['boundary_runs'][condition][label]['case_updates'].append(update)
                session_result['cases'].append(case_result)
                global_case_index += 1
                # Keep results recoverable if inference is interrupted after this case.
                result['sessions'] = result['sessions'] + ([session_result] if not result['sessions'] or result['sessions'][-1]['id'] != session['id'] else [])
                result['sessions'][-1] = session_result
                result['completed_cases'] = global_case_index
                write_json(output_path, result)
            for condition in CONDITIONS:
                for enabled, label in ((False, 'boundary_off'), (True, 'boundary_on')):
                    session_result['boundary_runs'][condition][label]['accepted_cues'] = boundary_state[(condition, enabled)]
            result['sessions'][-1] = session_result
            write_json(output_path, result)
        result['prompt_equivalence_checks'] = check_prompt_equivalence(result, entries)
        result['status'] = 'complete_with_errors' if result['errors'] else 'complete'
        result['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        result['total_translation_seconds'] = round(sum(
            condition.get('translation_seconds', 0)
            for session in result['sessions'] for case in session['cases']
            for condition in case['conditions'].values()
        ), 3)
        write_json(output_path, result)
    finally:
        gemma.translate_batch = original_translate_batch
        gemma.shutdown()
    return result



def check_prompt_equivalence(result: dict, entries: list[dict]) -> dict:
    checks = []
    checked = mismatches = 0
    for session in result['sessions']:
        for case in session['cases']:
            source = ' '.join(segment['text'] for segment in case['source_segments'])
            context_data = case['conditions'].get('context', {})
            history = context_data.get('history_used', '')
            for left, right, applicable in (
                ('baseline', 'glossary', not any(alias in source for entry in entries for alias in entry['aliases'])),
                ('context', 'context_glossary', not any(alias in history + ' ' + source for entry in entries for alias in entry['aliases'])),
                ('baseline', 'context', not history),
            ):
                if not applicable:
                    continue
                left_result = case['conditions'].get(left, {})
                right_result = case['conditions'].get(right, {})
                if left_result.get('status') != 'ok' or right_result.get('status') != 'ok':
                    continue
                left_calls = left_result.get('batch_calls', [])
                right_calls = right_result.get('batch_calls', [])
                left_hashes = [call['prompt_sha256'] for call in left_calls]
                right_hashes = [call['prompt_sha256'] for call in right_calls]
                equal = left_hashes == right_hashes
                checked += 1
                mismatches += int(not equal)
                checks.append({'session_id': session['id'], 'case_id': case['id'],
                               'conditions': [left, right], 'expected_same_prompt': True,
                               'same_batch_count': len(left_hashes) == len(right_hashes),
                               'same_prompt_hashes': equal,
                               'left_prompt_sha256': left_hashes, 'right_prompt_sha256': right_hashes})
    return {'checks_run': checked, 'mismatches': mismatches, 'status': 'pass' if not mismatches else 'mismatch', 'checks': checks}


def reconcile_saved_results() -> dict:
    path = ROOT / 'test-results/caption-combos.json'
    result = json.loads(path.read_text())
    if result.get('status') not in ('complete', 'complete_with_errors'):
        raise RuntimeError('Saved model sweep is incomplete; boundary policies can be reconciled after all cases finish.')
    corpus = json.loads(CASES_PATH.read_text())
    if result.get('frozen_inputs', {}).get('corpus_sha256') != sha256(CASES_PATH):
        raise RuntimeError('Saved results do not match the current frozen corpus; refusing to rebuild cues.')
    source_sessions = {session['id']: {case['id']: case for case in session['cases']} for session in corpus['sessions']}
    helper_source = inspect.getsource(reconcile_overlap)
    helper_sha = hashlib.sha256(helper_source.encode()).hexdigest()
    for session in result['sessions']:
        conditions = tuple(result['conditions'][name] for name in CONDITIONS)
        session['boundary_runs'] = {condition: {
            'boundary_off': {'accepted_cues': [], 'case_updates': []},
            'boundary_on': {'accepted_cues': [], 'case_updates': []},
        } for condition in CONDITIONS}
        state = {(condition, enabled): [] for condition in CONDITIONS for enabled in (False, True)}
        for case in session['cases']:
            source_case = source_sessions[session['id']][case['id']]
            case['media_start'] = source_case.get('media_start', 0.0)
            case['source_segments'] = source_case['source_segments']
            case['boundary'] = {}
            for condition in CONDITIONS:
                condition_result = case['conditions'].get(condition, {})
                if condition_result.get('status') == 'ok':
                    translations = [segment['translation'] for segment in condition_result.get('segments', [])]
                    condition_result['generated_cues'] = make_cues(case['id'], case['source_segments'], translations, case.get('media_start', 0.0))
                    for index, segment in enumerate(condition_result.get('segments', [])):
                        source = case['source_segments'][index]
                        segment.update(start=source['start'] + case.get('media_start', 0.0),
                                       end=source['end'] + case.get('media_start', 0.0),
                                       source_start=source['start'], source_end=source['end'],
                                       media_time_offset=case.get('media_start', 0.0))
                generated = condition_result.get('generated_cues', [])
                for enabled, label in ((False, 'boundary_off'), (True, 'boundary_on')):
                    existing = state[(condition, enabled)]
                    combined, changed = accept_boundary(existing, generated) if enabled else baseline_accept(existing, generated)
                    state[(condition, enabled)] = combined
                    update = {'case_id': case['id'], 'accepted_cues': changed,
                              'assembled_cue_count': len(generated), 'accepted_cue_count': len(changed),
                              'combined_cue_count': len(combined)}
                    case['boundary'][condition + '_' + label] = update
                    session['boundary_runs'][condition][label]['case_updates'].append(update)
        for condition in CONDITIONS:
            for enabled, label in ((False, 'boundary_off'), (True, 'boundary_on')):
                session['boundary_runs'][condition][label]['accepted_cues'] = state[(condition, enabled)]
    result['cue_timing'] = 'For packet-relative source segments, media_start is added to produce absolute cue times. Live/held-out cues already use absolute fixture times; synthetic cues remain session-relative.'
    result['boundary_reconciliation'] = {
        'status': 'complete',
        'helper': 'jread.server.reconcile_overlap',
        'helper_source_sha256': helper_sha,
        'boundary_off': 'Exact English text plus temporal overlap duplicate suppression.',
        'boundary_on': 'Final pure helper result with baseline exact-English overlap suppression; uses stored translations and adds no model calls.',
        'reconciled_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    write_json(path, result)
    return result



def run_synthetic_reset_repair() -> dict:
    path = ROOT / 'test-results/caption-combos.json'
    archive = ROOT / 'test-results/caption-combos-pre-synthetic-reset-repair.json'
    result = json.loads(path.read_text())
    corpus = json.loads(CASES_PATH.read_text())
    glossary_data = json.loads(GLOSSARY_PATH.read_text())
    entries = glossary_data.get('entries', [])
    regression = check_synthetic_context_resets(corpus)
    if not archive.is_file():
        raise RuntimeError('The pre-repair deterministic report must be archived before targeted repair.')
    if result.get('frozen_inputs', {}).get('corpus_sha256') != sha256(CASES_PATH):
        raise RuntimeError('Saved results do not match the frozen corpus; refusing targeted repair.')
    if result.get('frozen_inputs', {}).get('jread_glossary_sha256') != sha256(GLOSSARY_PATH):
        raise RuntimeError('Saved results do not match the frozen glossary; refusing targeted repair.')
    current_prompt_hash = hashlib.sha256(inspect.getsource(gemma._messages).encode()).hexdigest()
    if result.get('prompt_protocol', {}).get('source_sha256') != current_prompt_hash:
        raise RuntimeError('The prompt builder changed after the deterministic sweep; refusing a non-comparable repair.')
    if result.get('model', {}).get('provenance') != gemma.PRESETS[PRESET]:
        raise RuntimeError('Model provenance changed after the deterministic sweep; refusing a non-comparable repair.')
    if not gemma.ready(PRESET) or not gemma.runtime_ready():
        raise RuntimeError('Pinned Gemma E4B and runtime must remain installed and verified; repair never downloads them.')
    synthetic_session = next(session for session in result['sessions'] if session['id'] == 'synthetic_diagnostics')
    if len(synthetic_session['cases']) != 5 or not all(case.get('reset_context') for case in next(s for s in corpus['sessions'] if s['id'] == 'synthetic_diagnostics')['cases']):
        raise RuntimeError('Expected five per-case reset synthetic diagnostics.')

    gemma.PORT = PORT
    gemma.BASE = f'http://{gemma.HOST}:{PORT}'
    original_translate = gemma.translate_batch
    started_at = datetime.now(timezone.utc).isoformat()
    repair = {
        'status': 'running',
        'reason': 'The corpus marks each synthetic diagnostic reset_context=true; the initial runner only handled reset_context_before.',
        'archive': str(archive.relative_to(ROOT)),
        'scope': 'Only the context and context+glossary requests for the five synthetic cases are reissued. Existing baseline/glossary requests, all real transcript requests, ASR results, frozen corpus/glossary/model/runtime are preserved.',
        'conditions_repaired': ['context', 'context_glossary'],
        'case_ids': [case['id'] for case in synthetic_session['cases']],
        'corpus_sha256': sha256(CASES_PATH),
        'glossary_sha256': sha256(GLOSSARY_PATH),
        'prompt_builder_source_sha256': current_prompt_hash,
        'model_provenance': gemma.PRESETS[PRESET],
        'runner_source_sha256': sha256(Path(__file__).resolve()),
        'source_fidelity_limit': 'The Japanese source remains the saved transcript; ASR/source uncertainty is preserved per case and is not corrected by this context reset repair.',
        'corpus_scoring_limit': 'Do not infer a corpus accuracy percentage from these checks; several source cases are uncertain and synthetic diagnostics are authored invariants rather than human-certified audio ground truth.',
        'isolated_port': PORT,
        'started_at_utc': started_at,
        'fake_translator_reset_regression': regression,
        'previous_synthetic_context_outputs_are_stale': True,
        'no_retry_policy': 'Each affected condition is invoked once. Case failures are recorded as errors and are not retried.',
    }
    result['targeted_repair'] = repair
    result['status'] = 'repair_running'
    write_json(path, result)
    try:
        warm_start = time.monotonic()
        original_translate([{'start': 0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        repair['warmup'] = {'excluded': True, 'elapsed_seconds_excluded': round(time.monotonic() - warm_start, 3)}
        for case, corpus_case in zip(synthetic_session['cases'], next(s for s in corpus['sessions'] if s['id'] == 'synthetic_diagnostics')['cases'], strict=True):
            if case['id'] != corpus_case['id']:
                raise RuntimeError('Synthetic case order changed between saved result and frozen corpus.')
            for condition in ('context', 'context_glossary'):
                condition_result = case['conditions'][condition]
                context = ''
                glossary = entries if condition == 'context_glossary' else None
                call_log = []

                def measured(segments, context='', preset='gemma4', glossary=None):
                    started = time.monotonic()
                    prompt_sha = ''
                    try:
                        messages = gemma._messages(segments, context, glossary)
                        prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                        return original_translate(segments, context=context, preset=preset, glossary=glossary)
                    finally:
                        call_log.append({
                            'elapsed_seconds': round(time.monotonic() - started, 3),
                            'segment_count': len(segments),
                            'source_texts': [item['text'] for item in segments],
                            'context_sent': context,
                            'glossary_enabled': bool(glossary),
                            'prompt_sha256': prompt_sha,
                        })

                gemma.translate_batch = measured
                started = time.monotonic()
                try:
                    translated = list(engine.gemma_segments(
                        corpus_case['source_segments'], preset=PRESET,
                        context=context, glossary=glossary,
                    ))
                    translations = [item.text for item in translated]
                    if len(translations) != len(corpus_case['source_segments']):
                        raise ValueError('Translation count did not match synthetic source segments.')
                    media_offset = corpus_case.get('media_start', 0.0)
                    cues = make_cues(case['id'], corpus_case['source_segments'], translations, media_offset)
                    condition_result.update({
                        'status': 'ok',
                        'translation_seconds': round(time.monotonic() - started, 3),
                        'batch_calls': call_log,
                        'history_used': '',
                        'reset_context_applied': True,
                        'segments': [
                            {'index': index, 'source': source, 'translation': text,
                             'start': source['start'] + media_offset, 'end': source['end'] + media_offset,
                             'source_start': source['start'], 'source_end': source['end'],
                             'media_time_offset': media_offset,
                             'segment_id': f"{case['id']}:s{index}"}
                            for index, (source, text) in enumerate(zip(corpus_case['source_segments'], translations, strict=True))
                        ],
                        'generated_cues': cues,
                        'error_type': None,
                        'error': None,
                    })
                except Exception as exc:
                    condition_result.update({
                        'status': 'error',
                        'translation_seconds': round(time.monotonic() - started, 3),
                        'batch_calls': call_log,
                        'history_used': '',
                        'reset_context_applied': True,
                        'error_type': type(exc).__name__,
                        'error': str(exc),
                        'segments': [],
                        'generated_cues': [],
                    })
                finally:
                    gemma.translate_batch = original_translate
                write_json(path, result)
        # Replace prior errors only for these repaired case/condition pairs.
        repaired_keys = {(case['id'], condition) for case in synthetic_session['cases'] for condition in ('context', 'context_glossary')}
        result['errors'] = [error for error in result.get('errors', [])
                            if (error.get('case_id'), error.get('condition')) not in repaired_keys]
        for case in synthetic_session['cases']:
            for condition in ('context', 'context_glossary'):
                value = case['conditions'][condition]
                if value['status'] == 'error':
                    result['errors'].append({'session_id': synthetic_session['id'], 'case_id': case['id'], 'condition': condition,
                                             'error_type': value['error_type'], 'error': value['error']})
        result['prompt_equivalence_checks'] = check_prompt_equivalence(result, entries)
        result['total_translation_seconds'] = round(sum(
            condition.get('translation_seconds', 0)
            for session in result['sessions'] for case in session['cases']
            for condition in case['conditions'].values() if condition.get('status') == 'ok'
        ), 3)
        repair['status'] = 'complete_with_errors' if any(case['conditions'][condition]['status'] == 'error' for case in synthetic_session['cases'] for condition in ('context', 'context_glossary')) else 'complete'
        repair['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        repair['requests_attempted'] = sum(len(case['conditions'][condition].get('batch_calls', [])) for case in synthetic_session['cases'] for condition in ('context', 'context_glossary'))
        result['status'] = 'complete_with_errors' if result['errors'] else 'complete'
        write_json(path, result)
        result = reconcile_saved_results()
        return result
    finally:
        gemma.translate_batch = original_translate
        gemma.shutdown()


def run_asr() -> dict:
    audio_path = ROOT / 'tests/fixtures/japanese-speech.wav'
    reference_path = ROOT / 'tests/fixtures/japanese-speech.json'
    silence_path = ROOT / 'tests/fixtures/silence.wav'
    output_path = ROOT / 'test-results/caption-asr-combos.json'
    meta = {
        'schema_version': 1,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Optional Whisper medium CPU transcript replay on the existing 10.44 second FLEURS Japanese fixture. Four ASR hint conditions use 8 second chunks with 1 second overlap. Raw transcript chunks and timestamps are retained; no CER is reported because overlap alignment is not certified.',
        'conditions': {'baseline': 'no glossary hotwords or recognized context', 'glossary': 'JREAD show glossary hotwords, no context', 'context': 'previous Whisper recognized source only, no glossary hotwords', 'context_glossary': 'both hints'},
        'model': {'name': 'faster-whisper medium', 'revision': engine.PRESETS['medium'][0], 'device': 'cpu', 'cpu_threads': 4, 'num_workers': 1},
        'fixture': {'audio_path': str(audio_path.relative_to(ROOT)), 'audio_sha256': sha256(audio_path), 'reference_path': str(reference_path.relative_to(ROOT)), 'reference_sha256': sha256(reference_path), 'reference': json.loads(reference_path.read_text())['reference']},
        'chunking': {'chunk_seconds': 8.0, 'overlap_seconds': 1.0},
        'results': {},
        'limitations': ['Transcripts are ASR outputs from a short public-speech fixture, not anime dialogue.', 'Overlap may duplicate or alter words; raw chunk timing is preserved and CER is omitted rather than claiming a potentially misleading alignment.', 'The show-specific glossary is an out-of-domain control here, not an expected benefit.', 'No live capture latency or sustained throughput is measured.'],
    }
    if not engine.model_ready('medium'):
        meta['status'] = 'skipped_model_not_already_installed'
        write_json(output_path, meta)
        return meta
    if not audio_path.is_file():
        meta['status'] = 'skipped_audio_fixture_missing'
        write_json(output_path, meta)
        return meta
    try:
        samples = engine.read_audio(audio_path)
        model = engine.load('medium', 'cpu')
        glossary_settings = {'caption_glossary': 'aikatsu_pripara'}
        for condition in CONDITIONS:
            history = ''
            chunks = []
            started_all = time.monotonic()
            chunk_start = 0.0
            chunk_index = 0
            while chunk_start < len(samples) / 16000:
                chunk_end = min(len(samples) / 16000, chunk_start + 8.0)
                left, right = round(chunk_start * 16000), round(chunk_end * 16000)
                audio = samples[left:right]
                hints_on = condition_glossary(condition)
                context_on = condition_context(condition)
                settings = glossary_settings if hints_on else {'caption_glossary': 'none'}
                hints = engine.asr_hints(settings, 'ja', history if context_on else '')
                start = time.monotonic()
                segments, _ = model.transcribe(audio, language='ja', task='transcribe', vad_filter=True,
                                               beam_size=5, condition_on_previous_text=False, **hints)
                rows = []
                for segment in segments:
                    rel_start = max(0.0, float(segment.start))
                    rel_end = min(chunk_end - chunk_start, float(segment.end))
                    text = segment.text.strip()
                    if text and rel_start < rel_end:
                        rows.append({'start': chunk_start + rel_start, 'end': chunk_start + rel_end, 'text': text})
                chunks.append({'chunk_index': chunk_index, 'start': chunk_start, 'end': chunk_end,
                               'inference_seconds': round(time.monotonic() - start, 3),
                               'context_used': history if context_on else '', 'hints': hints,
                               'segments': rows})
                if context_on:
                    # History comes only from the actual recognized Japanese source.
                    history = (history + ' ' + ' '.join(row['text'] for row in rows)).strip()[-150:]
                chunk_index += 1
                if chunk_end >= len(samples) / 16000:
                    break
                chunk_start = chunk_end - 1.0
            meta['results'][condition] = {'status': 'ok', 'elapsed_seconds': round(time.monotonic() - started_all, 3), 'chunks': chunks}
        meta['status'] = 'complete'
    except Exception as exc:
        meta['status'] = 'complete_with_error'
        meta['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    if silence_path.exists():
        try:
            silence = engine.read_audio(silence_path)
            segments, _ = model.transcribe(silence, language='ja', task='transcribe', vad_filter=True, beam_size=5, condition_on_previous_text=False)
            meta['silence_control'] = {'status': 'ok', 'recognized_segments': [{'start': s.start, 'end': s.end, 'text': s.text.strip()} for s in segments if s.text.strip()]}
        except Exception as exc:
            meta['silence_control'] = {'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc)}
    else:
        meta['silence_control'] = {'status': 'skipped_no_fixture', 'expected_empty': True}
    write_json(output_path, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconcile-only', action='store_true', help='Recompute boundary variants from a completed saved sweep without inference.')
    parser.add_argument('--skip-asr', action='store_true', help='Leave the existing optional ASR artifact unchanged.')
    parser.add_argument('--repair-synthetic-resets', action='store_true', help='Rerun only synthetic context-enabled conditions after per-case resets.')
    args = parser.parse_args()
    if args.repair_synthetic_resets:
        result = run_synthetic_reset_repair()
        print(json.dumps({'main_status': result['status'], 'repair_status': result['targeted_repair']['status'],
                          'requests_attempted': result['targeted_repair'].get('requests_attempted'),
                          'errors': result['errors'], 'prompt_equivalence': result['prompt_equivalence_checks']['status']}, ensure_ascii=False))
        return 0
    if args.reconcile_only:
        result = reconcile_saved_results()
        print(json.dumps({'main_status': result['status'], 'completed_cases': result.get('completed_cases'),
                          'boundary_helper_sha256': result['boundary_reconciliation']['helper_source_sha256']}, ensure_ascii=False))
        return 0
    result = run_main()
    result = reconcile_saved_results()
    asr_status = 'unchanged_skipped' if args.skip_asr else run_asr()['status']
    print(json.dumps({'main_status': result['status'], 'completed_cases': result.get('completed_cases'),
                      'total_translation_seconds': result.get('total_translation_seconds'),
                      'prompt_equivalence': result.get('prompt_equivalence_checks', {}).get('status'),
                      'asr_status': asr_status}, ensure_ascii=False))
    return 0 if result['status'] in ('complete', 'complete_with_errors') else 1


if __name__ == '__main__':
    raise SystemExit(main())
