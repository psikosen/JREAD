#!/usr/bin/env python3
"""Development-only translation comparison for the frozen Japanese accuracy corpus."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jread import engine, gemma  # noqa: E402

CORPUS_PATH = ROOT / 'tests/fixtures/accuracy-corpus.json'
OUTPUT_PATH = ROOT / 'test-results/accuracy-translation-development.json'
ASR_PATH = ROOT / 'test-results/accuracy-speech-development.json'
PRESET = 'gemma4'
PORT = 18767
PRECISION_INSTRUCTION = (
    'Preserve grammatical roles, clause relationships, geographic terms and units precisely. '
    "Do not broaden a named place or turn a person's role into an organization. "
    'Leave genuine ambiguity unresolved rather than inventing a specific identity.'
)
FEWSHOT_DEMOS = [
    ('研究員として田中を補佐し、実験を管理した。', 'Serving as a researcher, they assisted Tanaka and managed the experiments.'),
    ('この箱に赤い玉が入っているはずはありません。', 'This box cannot contain a red ball.'),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def frozen_inputs() -> tuple[dict, list[dict]]:
    corpus = json.loads(CORPUS_PATH.read_text())
    cases = [case for case in corpus['cases'] if case.get('partition') == 'development']
    if len(cases) != 20:
        raise RuntimeError(f'Expected only 20 development cases; found {len(cases)}.')
    if any(case.get('partition') != 'development' for case in cases):
        raise RuntimeError('Held-out cases must not enter the development evaluation.')
    return corpus, cases


def model_info() -> dict:
    if not gemma.ready(PRESET) or not gemma.runtime_ready():
        raise RuntimeError('Pinned Gemma E4B and runtime must already be installed and verified; this evaluator never downloads them.')
    gemma.PORT = PORT
    gemma.BASE = f'http://{gemma.HOST}:{PORT}'
    return {
        'preset': PRESET,
        'provenance': gemma.PRESETS[PRESET],
        'device': 'CPU',
        'threads': 8,
        'runtime': json.loads(gemma.RUNTIME_MANIFEST.read_text()),
        'isolated_port': PORT,
        'temperature': 0,
        'prompt_builder': 'jread.gemma._messages',
        'prompt_builder_source_sha256': hashlib.sha256(inspect.getsource(gemma._messages).encode()).hexdigest(),
    }


def call_log_wrapper(original):
    calls = []

    def measured(segments, context='', preset='gemma4', glossary=None):
        started = time.monotonic()
        prompt_sha = ''
        try:
            messages = gemma._messages(segments, context, glossary)
            prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            return original(segments, context=context, preset=preset, glossary=glossary)
        finally:
            calls.append({
                'elapsed_seconds': round(time.monotonic() - started, 3),
                'segment_count': len(segments),
                'context_sent': context,
                'glossary_enabled': bool(glossary),
                'prompt_sha256': prompt_sha,
            })
    return measured, calls


def initial_report(corpus: dict, cases: list[dict]) -> dict:
    return {
        'schema_version': 1,
        'status': 'running_reference_ceiling',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Only the frozen 20 development rows are translated. The pristine Japanese reference pass measures a text-input ceiling; it does not measure ASR or end-to-end accuracy.',
        'dataset_provenance': {
            'dataset': corpus.get('dataset'),
            'revision': corpus.get('revision'),
            'split': corpus.get('source_split'),
            'license': corpus.get('license'),
            'attribution_url': corpus.get('attribution_url'),
            'corpus_path': str(CORPUS_PATH.relative_to(ROOT)),
            'corpus_sha256': sha256(CORPUS_PATH),
            'selection': corpus.get('selection'),
            'manifest_metadata': corpus.get('metadata'),
            'development_case_count': len(cases),
            'included_partitions': ['development'],
            'heldout_rows_used': 0,
        },
        'model': model_info(),
        'warmup': {'excluded': True, 'description': 'One synthetic greeting sent before measured cases.'},
        'stages': {
            'pristine_reference': {'status': 'running', 'cases': [], 'errors': []},
            'full_clip_asr': {'status': 'pending_extension_report', 'cases': [], 'errors': []},
        },
        'interpretation_limits': [
            'Translation output is not automatically graded; human semantic review is required before any accuracy claim.',
            'ASR source fidelity and translation fidelity are separate. Japanese transcript CER is evidence about the source, not a translation score.',
            'These are general read-speech development cases, not anime dialogue or live-latency evidence.',
        ],
    }


def run_reference_stage() -> dict:
    corpus, cases = frozen_inputs()
    result = initial_report(corpus, cases)
    write_json(OUTPUT_PATH, result)
    original = gemma.translate_batch
    try:
        warm_start = time.monotonic()
        original([{'start': 0.0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        result['warmup']['elapsed_seconds_excluded'] = round(time.monotonic() - warm_start, 3)
        stage = result['stages']['pristine_reference']
        for case in cases:
            duration = float(case['duration_seconds'])
            source = {'start': 0.0, 'end': duration, 'text': case['japanese_reference']}
            measured, calls = call_log_wrapper(original)
            gemma.translate_batch = measured
            started = time.monotonic()
            try:
                segments = list(engine.gemma_segments([source], preset=PRESET, context='', glossary=None))
                translations = [segment.text for segment in segments]
                if len(translations) != 1:
                    raise ValueError(f"Expected one whole-clip translation; got {len(translations)}.")
                row = {
                    'id': case['id'],
                    'partition': 'development',
                    'source_provenance': {
                        'japanese_reference': case['japanese_reference'],
                        'english_reference': case['english_reference'],
                        'audio_sha256': case.get('audio_sha256'),
                        'duration_seconds': duration,
                    },
                    'input_segments': [source],
                    'output_segments': [{'start': 0.0, 'end': duration, 'text': translations[0]}],
                    'context_used': '',
                    'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls,
                    'status': 'ok',
                }
            except Exception as exc:
                row = {
                    'id': case['id'],
                    'partition': 'development',
                    'source_provenance': {'japanese_reference': case['japanese_reference'],
                                          'english_reference': case['english_reference'],
                                          'audio_sha256': case.get('audio_sha256'),
                                          'duration_seconds': duration},
                    'input_segments': [source],
                    'output_segments': [],
                    'context_used': '',
                    'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls,
                    'status': 'error',
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                }
                stage['errors'].append({'id': case['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            finally:
                gemma.translate_batch = original
            stage['cases'].append(row)
            result['completed_reference_cases'] = len(stage['cases'])
            write_json(OUTPUT_PATH, result)
        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['total_translation_seconds'] = round(sum(row['translation_seconds'] for row in stage['cases']), 3)
        result['status'] = 'reference_complete_asr_pending'
        result['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT_PATH, result)
        return result
    finally:
        gemma.translate_batch = original
        gemma.shutdown()


def run_full_clip_asr_stage() -> dict:
    if not ASR_PATH.is_file():
        raise RuntimeError(f'ASR report is not ready: {ASR_PATH.relative_to(ROOT)}')
    corpus, cases = frozen_inputs()
    result = json.loads(OUTPUT_PATH.read_text())
    current_corpus_sha = sha256(CORPUS_PATH)
    if result.get('dataset_provenance', {}).get('corpus_sha256') != current_corpus_sha:
        raise RuntimeError('Development corpus changed since the reference pass; refusing a mismatched ASR comparison.')
    asr = json.loads(ASR_PATH.read_text())
    asr_corpus_sha = (asr.get('corpus', {}).get('manifest_sha256')
                      or asr.get('manifest_sha256') or asr.get('manifest', {}).get('sha256'))
    if asr_corpus_sha and asr_corpus_sha != current_corpus_sha:
        raise RuntimeError('ASR report references a different accuracy manifest.')
    asr_cases = {str(item['id']): item for item in asr.get('cases', []) if item.get('partition') == 'development'}
    if set(asr_cases) != {str(case['id']) for case in cases}:
        raise RuntimeError('ASR report must contain exactly the 20 development cases.')
    stage = {'status': 'running', 'source_report_path': str(ASR_PATH.relative_to(ROOT)),
             'source_report_sha256': sha256(ASR_PATH), 'cases': [], 'errors': [],
             'source_fidelity_note': 'Per-case ASR CER/transcript are stored as source-quality evidence and are not translation scores.'}
    result['stages']['full_clip_asr'] = stage
    result['status'] = 'running_full_clip_asr'
    write_json(OUTPUT_PATH, result)
    original = gemma.translate_batch
    gemma.PORT = PORT
    gemma.BASE = f'http://{gemma.HOST}:{PORT}'
    try:
        for case in cases:
            source_case = asr_cases[str(case['id'])]
            full_clip = source_case.get('full_clip') or {}
            source_segments = full_clip.get('segments') or []
            started = time.monotonic()
            row_base = {
                'id': case['id'],
                'partition': 'development',
                'source_provenance': {
                    'japanese_reference': case['japanese_reference'],
                    'english_reference': case['english_reference'],
                    'audio_sha256': case.get('audio_sha256'),
                    'duration_seconds': case['duration_seconds'],
                    'asr_transcript': full_clip.get('transcript'),
                    'asr_cer': full_clip.get('cer'),
                    'asr_segments': source_segments,
                },
                'input_segments': source_segments,
                'context_used': '',
                'glossary_used': False,
            }
            measured, calls = call_log_wrapper(original)
            gemma.translate_batch = measured
            try:
                if not source_segments:
                    raise ValueError('ASR report has no full-clip source segments.')
                segments = list(engine.gemma_segments(source_segments, preset=PRESET, context='', glossary=None))
                outputs = [{'start': segment.start, 'end': segment.end, 'text': segment.text} for segment in segments]
                if len(outputs) != len(source_segments):
                    raise ValueError(f"Expected {len(source_segments)} translated segments; got {len(outputs)}.")
                row = row_base | {'output_segments': outputs, 'translation_seconds': round(time.monotonic() - started, 3), 'batch_calls': calls, 'status': 'ok'}
            except Exception as exc:
                row = row_base | {'output_segments': [], 'translation_seconds': round(time.monotonic() - started, 3),
                                  'batch_calls': calls, 'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc)}
                stage['errors'].append({'id': case['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            finally:
                gemma.translate_batch = original
            stage['cases'].append(row)
            result['completed_asr_cases'] = len(stage['cases'])
            write_json(OUTPUT_PATH, result)
        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['total_translation_seconds'] = round(sum(row['translation_seconds'] for row in stage['cases']), 3)
        result['status'] = 'complete_with_errors' if stage['errors'] or result['stages']['pristine_reference']['errors'] else 'complete'
        result['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT_PATH, result)
        return result
    finally:
        gemma.translate_batch = original
        gemma.shutdown()


def run_candidate_asr_stage(asr_path: Path, stage_key: str, join_segments: bool = False) -> dict:
    corpus, cases = frozen_inputs()
    result = json.loads(OUTPUT_PATH.read_text())
    if stage_key in result.get('stages', {}):
        raise RuntimeError(f'Stage {stage_key} already exists; candidate stages are never overwritten.')
    if result.get('stages', {}).get('pristine_reference', {}).get('status') != 'complete':
        raise RuntimeError('The pristine-reference stage must be complete before adding a candidate ASR stage.')
    if result.get('stages', {}).get('full_clip_asr', {}).get('status') != 'complete':
        raise RuntimeError('The original full-clip ASR stage must be complete before adding a candidate stage.')
    corpus_sha = sha256(CORPUS_PATH)
    if result.get('dataset_provenance', {}).get('corpus_sha256') != corpus_sha:
        raise RuntimeError('Development corpus changed since the original stages; refusing a mismatched candidate comparison.')

    asr = json.loads(asr_path.read_text())
    if asr.get('status') != 'complete':
        raise RuntimeError(f"Candidate ASR report must be complete; got {asr.get('status')!r}.")
    asr_corpus_sha = asr.get('corpus', {}).get('manifest_sha256')
    if asr_corpus_sha != corpus_sha:
        raise RuntimeError('Candidate ASR report manifest SHA does not match the frozen accuracy corpus.')
    report_cases = asr.get('cases')
    if not isinstance(report_cases, list) or len(report_cases) != 20:
        raise RuntimeError('Candidate ASR report must contain exactly 20 development cases.')
    report_ids = [str(item.get('id')) for item in report_cases]
    expected_ids = {str(case['id']) for case in cases}
    if len(set(report_ids)) != 20 or set(report_ids) != expected_ids:
        raise RuntimeError('Candidate ASR report IDs must match the 20 development IDs exactly, with no duplicates.')
    asr_cases = {str(item['id']): item for item in report_cases}
    for case in cases:
        source = asr_cases[str(case['id'])]
        if source.get('partition') != 'development':
            raise RuntimeError(f"Candidate ASR row {case['id']} is not marked development.")
        if source.get('audio_sha256') != case.get('audio_sha256'):
            raise RuntimeError(f"Candidate ASR audio hash does not match development row {case['id']}.")
        if source.get('japanese_reference') != case.get('japanese_reference') or source.get('english_reference') != case.get('english_reference'):
            raise RuntimeError(f"Candidate ASR references do not match development row {case['id']}.")
        if not isinstance(source.get('full_clip', {}).get('segments'), list) or not source['full_clip']['segments']:
            raise RuntimeError(f"Candidate ASR row {case['id']} has no full-clip segments.")

    frozen_model = result.get('model', {})
    current_model = model_info()
    if frozen_model.get('preset') != current_model['preset'] or frozen_model.get('provenance') != current_model['provenance']:
        raise RuntimeError('Gemma model provenance changed since the original stages; refusing a mismatched candidate comparison.')
    if frozen_model.get('prompt_builder_source_sha256') != current_model['prompt_builder_source_sha256']:
        raise RuntimeError('Gemma prompt builder changed since the original stages; refusing a mismatched candidate comparison.')

    report_sha = sha256(asr_path)
    stage = {
        'status': 'running',
        'stage_key': stage_key,
        'asr_model': asr.get('model'),
        'source_report_path': str(asr_path.resolve().relative_to(ROOT)),
        'source_report_sha256': report_sha,
        'source_manifest_sha256': asr_corpus_sha,
        'cases': [],
        'errors': [],
        'context_policy': 'Reset to empty for every recording; no show glossary.',
        'translation_input_policy': (
            'Evaluation-only: concatenate all nonempty turbo ASR segment texts with one ASCII space into a single source segment spanning the earliest start to latest end. This does not change production segmentation.'
            if join_segments else
            'Translate turbo ASR segments separately through production engine.gemma_segments, preserving ASR segment timestamps.'
        ),
        'warmup': {'excluded': True, 'description': 'One synthetic greeting before candidate measurements.'},
        'timing_note': (
            'One joined request per recording after excluded warmup. Translation wall time is coarse evaluation timing, not live latency; source spans over 8 or 12 seconds are flagged. The original full_clip_asr stage had no explicit warmup, so stage totals are not a matched speed comparison.'
            if join_segments else
            'Candidate stage uses an excluded warmup. The original full_clip_asr stage had no explicit warmup, so its first case includes llama-server startup; those stage totals are not a matched speed comparison.'
        ),
        'model': current_model,
    }
    result['stages'][stage_key] = stage
    result['status'] = f'running_{stage_key}'
    result['timing_comparability'] = {
        'pristine_reference': 'Explicit excluded warmup; measured after the isolated server was warm.',
        'full_clip_asr': 'No explicit warmup; first case timing includes isolated llama-server startup.',
        stage_key: 'Explicit excluded warmup; candidate stage timing is post-startup.',
        'warning': 'Do not compare these stage wall times as a pure model-speed ranking.',
    }
    write_json(OUTPUT_PATH, result)

    original = gemma.translate_batch
    try:
        warm_start = time.monotonic()
        original([{'start': 0.0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        stage['warmup']['elapsed_seconds_excluded'] = round(time.monotonic() - warm_start, 3)
        for case in cases:
            source_case = asr_cases[str(case['id'])]
            full_clip = source_case['full_clip']
            asr_segments = full_clip['segments']
            nonempty = [segment for segment in asr_segments
                        if isinstance(segment.get('text'), str) and segment['text'].strip()]
            if join_segments:
                joined_text = ' '.join(segment['text'].strip() for segment in nonempty)
                joined_source = ([{'start': min(float(segment['start']) for segment in nonempty),
                                  'end': max(float(segment['end']) for segment in nonempty),
                                  'text': joined_text}] if nonempty else [])
                source_segments = joined_source
            else:
                joined_text = None
                source_segments = asr_segments
            started = time.monotonic()
            row_base = {
                'id': case['id'],
                'partition': 'development',
                'source_provenance': {
                    'japanese_reference': case['japanese_reference'],
                    'english_reference': case['english_reference'],
                    'audio_sha256': case.get('audio_sha256'),
                    'duration_seconds': case['duration_seconds'],
                    'asr_transcript': full_clip.get('transcript'),
                    'asr_cer': full_clip.get('cer'),
                    'asr_segments': asr_segments,
                    'joined_source_text': joined_text,
                    'joined_source_span_seconds': (round(source_segments[0]['end'] - source_segments[0]['start'], 3)
                                                   if join_segments and source_segments else None),
                    'joined_source_span_over_8s': bool(join_segments and source_segments and source_segments[0]['end'] - source_segments[0]['start'] > 8),
                    'joined_source_span_over_12s': bool(join_segments and source_segments and source_segments[0]['end'] - source_segments[0]['start'] > 12),
                },
                'input_segments': source_segments,
                'context_used': '',
                'glossary_used': False,
            }
            measured, calls = call_log_wrapper(original)
            gemma.translate_batch = measured
            try:
                outputs = list(engine.gemma_segments(source_segments, preset=PRESET, context='', glossary=None))
                output_segments = [{'start': segment.start, 'end': segment.end, 'text': segment.text} for segment in outputs]
                if len(output_segments) != len(source_segments):
                    raise ValueError(f"Expected {len(source_segments)} translations; received {len(output_segments)}.")
                row = row_base | {
                    'output_segments': output_segments,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls,
                    'status': 'ok',
                }
            except Exception as exc:
                row = row_base | {
                    'output_segments': [],
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls,
                    'status': 'error',
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                }
                stage['errors'].append({'id': case['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            finally:
                gemma.translate_batch = original
            stage['cases'].append(row)
            result[f'completed_{stage_key}_cases'] = len(stage['cases'])
            write_json(OUTPUT_PATH, result)

        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['total_translation_seconds'] = round(sum(row['translation_seconds'] for row in stage['cases']), 3)
        if join_segments:
            stage['joined_span_counts'] = {
                'over_8_seconds': sum(row['source_provenance']['joined_source_span_over_8s'] for row in stage['cases']),
                'over_12_seconds': sum(row['source_provenance']['joined_source_span_over_12s'] for row in stage['cases']),
                'cases_with_joined_source': sum(bool(row['input_segments']) for row in stage['cases']),
            }
        all_errors = any(value.get('errors') for value in result['stages'].values())
        result['status'] = 'complete_with_errors' if all_errors else 'complete'
        result['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT_PATH, result)
        return result
    finally:
        gemma.translate_batch = original
        gemma.shutdown()


def run_precision_prompt_stage() -> dict:
    corpus, cases = frozen_inputs()
    baseline = json.loads(OUTPUT_PATH.read_text())
    corpus_sha = sha256(CORPUS_PATH)
    if baseline.get('dataset_provenance', {}).get('corpus_sha256') != corpus_sha:
        raise RuntimeError('Development corpus changed since the baseline reference pass.')
    if baseline.get('stages', {}).get('pristine_reference', {}).get('status') != 'complete':
        raise RuntimeError('The baseline pristine-reference stage must be complete first.')
    model = model_info()
    frozen_model = baseline.get('model', {})
    if frozen_model.get('preset') != model['preset'] or frozen_model.get('provenance') != model['provenance']:
        raise RuntimeError('Gemma model provenance changed since the baseline reference pass.')
    if frozen_model.get('prompt_builder_source_sha256') != model['prompt_builder_source_sha256']:
        raise RuntimeError('Base prompt builder changed since the baseline reference pass.')
    output_path = ROOT / 'test-results/accuracy-precise-prompt-development.json'
    if output_path.exists():
        raise RuntimeError(f'Candidate prompt report already exists: {output_path.relative_to(ROOT)}')

    original_translate = gemma.translate_batch
    original_messages = gemma._messages

    def precise_messages(segments, context, glossary=None):
        messages = original_messages(segments, context, glossary)
        messages[0]['content'] += ' ' + PRECISION_INSTRUCTION
        return messages

    stage = {
        'status': 'running',
        'stage_key': 'precision_prompt_pristine_reference',
        'cases': [],
        'errors': [],
        'prompt_addition': PRECISION_INSTRUCTION,
        'prompt_addition_sha256': hashlib.sha256(PRECISION_INSTRUCTION.encode()).hexdigest(),
        'base_prompt_builder_source_sha256': model['prompt_builder_source_sha256'],
        'effective_prompt_definition_sha256': hashlib.sha256(
            (model['prompt_builder_source_sha256'] + '\n' + PRECISION_INSTRUCTION).encode()
        ).hexdigest(),
        'context_policy': 'One pristine Japanese reference segment per recording; context reset to empty; no glossary.',
        'warmup': {'excluded': True, 'description': 'One synthetic greeting before candidate measurements.'},
        'interpretation_limit': 'This is a prompt-candidate comparison on pristine text, not end-to-end ASR accuracy or human-certified scoring.',
        'model': model,
    }
    report = {
        'schema_version': 1,
        'status': 'running',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_provenance': baseline['dataset_provenance'],
        'model': model,
        'baseline_report_path': str(OUTPUT_PATH.relative_to(ROOT)),
        'baseline_report_sha256_before_run': sha256(OUTPUT_PATH),
        'stages': {stage['stage_key']: stage},
        'scope': 'Only the frozen 20 development cases, pristine Japanese reference text. The sole prompt change is the recorded general instruction; no examples, names, or English references were added to the prompt.',
    }
    write_json(output_path, report)
    gemma._messages = precise_messages
    try:
        warm_start = time.monotonic()
        original_translate([{'start': 0.0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        stage['warmup']['elapsed_seconds_excluded'] = round(time.monotonic() - warm_start, 3)
        for case in cases:
            duration = float(case['duration_seconds'])
            source = {'start': 0.0, 'end': duration, 'text': case['japanese_reference']}
            measured, calls = call_log_wrapper(original_translate)
            gemma.translate_batch = measured
            started = time.monotonic()
            try:
                segments = list(engine.gemma_segments([source], preset=PRESET, context='', glossary=None))
                outputs = [segment.text for segment in segments]
                if len(outputs) != 1:
                    raise ValueError(f'Expected one whole-clip translation; received {len(outputs)}.')
                row = {
                    'id': case['id'], 'partition': 'development',
                    'source_provenance': {
                        'japanese_reference': case['japanese_reference'],
                        'english_reference': case['english_reference'],
                        'audio_sha256': case.get('audio_sha256'),
                        'duration_seconds': duration,
                    },
                    'input_segments': [source],
                    'output_segments': [{'start': 0.0, 'end': duration, 'text': outputs[0]}],
                    'context_used': '', 'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls, 'status': 'ok',
                }
            except Exception as exc:
                row = {
                    'id': case['id'], 'partition': 'development',
                    'source_provenance': {
                        'japanese_reference': case['japanese_reference'],
                        'english_reference': case['english_reference'],
                        'audio_sha256': case.get('audio_sha256'),
                        'duration_seconds': duration,
                    },
                    'input_segments': [source], 'output_segments': [],
                    'context_used': '', 'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls, 'status': 'error',
                    'error_type': type(exc).__name__, 'error': str(exc),
                }
                stage['errors'].append({'id': case['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            finally:
                gemma.translate_batch = original_translate
            stage['cases'].append(row)
            report['completed_cases'] = len(stage['cases'])
            write_json(output_path, report)
        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['total_translation_seconds'] = round(sum(row['translation_seconds'] for row in stage['cases']), 3)
        report['status'] = stage['status']
        report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output_path, report)
        return report
    finally:
        gemma.translate_batch = original_translate
        gemma._messages = original_messages
        gemma.shutdown()


def run_fewshot_prompt_stage() -> dict:
    corpus, cases = frozen_inputs()
    baseline = json.loads(OUTPUT_PATH.read_text())
    corpus_sha = sha256(CORPUS_PATH)
    if baseline.get('dataset_provenance', {}).get('corpus_sha256') != corpus_sha:
        raise RuntimeError('Development corpus changed since the baseline reference pass.')
    if baseline.get('stages', {}).get('pristine_reference', {}).get('status') != 'complete':
        raise RuntimeError('The baseline pristine-reference stage must be complete first.')
    model = model_info()
    frozen_model = baseline.get('model', {})
    if frozen_model.get('preset') != model['preset'] or frozen_model.get('provenance') != model['provenance']:
        raise RuntimeError('Gemma model provenance changed since the baseline reference pass.')
    if frozen_model.get('prompt_builder_source_sha256') != model['prompt_builder_source_sha256']:
        raise RuntimeError('Base prompt builder changed since the baseline reference pass.')
    output_path = ROOT / 'test-results/accuracy-fewshot-prompt-development.json'
    if output_path.exists():
        raise RuntimeError(f'Candidate prompt report already exists: {output_path.relative_to(ROOT)}')

    original_translate = gemma.translate_batch
    original_messages = gemma._messages
    demo_turns = []
    for japanese, english in FEWSHOT_DEMOS:
        demo_segment = {'start': 0.0, 'end': 3.0, 'text': japanese}
        formatted_user = original_messages([demo_segment], '', None)[1]
        demo_turns.extend([
            formatted_user,
            {'role': 'assistant', 'content': json.dumps(
                {'translations': [{'id': 0, 'text': english}]}, ensure_ascii=False, separators=(',', ':'))},
        ])

    def fewshot_messages(segments, context, glossary=None):
        messages = original_messages(segments, context, glossary)
        messages[1:1] = demo_turns
        return messages

    base_system = original_messages([{'start': 0.0, 'end': 3.0, 'text': '確認'}], '', None)[0]['content']
    candidate_system = fewshot_messages([{'start': 0.0, 'end': 3.0, 'text': '確認'}], '', None)[0]['content']
    if candidate_system != base_system:
        raise RuntimeError('Few-shot candidate must preserve the original system prompt exactly.')

    serialized_demos = json.dumps(FEWSHOT_DEMOS, ensure_ascii=False, separators=(',', ':'))
    stage = {
        'status': 'running',
        'stage_key': 'fewshot_prompt_pristine_reference',
        'cases': [],
        'errors': [],
        'demonstrations': [
            {'source': japanese, 'translation': english,
             'segment': {'id': 0, 'start': 0.0, 'end': 3.0},
             'context': '', 'format': 'Existing _messages source-data user turn followed by strict translation JSON assistant turn.'}
            for japanese, english in FEWSHOT_DEMOS
        ],
        'system_prompt_unchanged': True,
        'base_prompt_builder_source_sha256': model['prompt_builder_source_sha256'],
        'fewshot_content_sha256': hashlib.sha256(serialized_demos.encode()).hexdigest(),
        'effective_prompt_definition_sha256': hashlib.sha256(
            (model['prompt_builder_source_sha256'] + '\n' + serialized_demos).encode()
        ).hexdigest(),
        'context_policy': 'Pristine Japanese reference, empty context, no glossary; demonstrations are fixed generic role/negation examples, not corpus answers.',
        'tuning_note': 'Development-set prompt tuning only. These 20 development rows are not an independent test of this prompt candidate.',
        'warmup': {'excluded': True, 'description': 'One synthetic greeting before candidate measurements.'},
        'model': model,
    }
    report = {
        'schema_version': 1,
        'status': 'running',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_provenance': baseline['dataset_provenance'],
        'model': model,
        'baseline_report_path': str(OUTPUT_PATH.relative_to(ROOT)),
        'baseline_report_sha256_before_run': sha256(OUTPUT_PATH),
        'scope': 'Only the frozen 20 development cases, pristine Japanese reference text. The original system prompt is unchanged; only the two recorded generic user/assistant demonstrations are added. No held-out rows were read.',
        'stages': {stage['stage_key']: stage},
    }
    write_json(output_path, report)
    gemma._messages = fewshot_messages
    try:
        warm_start = time.monotonic()
        original_translate([{'start': 0.0, 'end': 0.8, 'text': 'こんにちは。'}], context='', preset=PRESET)
        stage['warmup']['elapsed_seconds_excluded'] = round(time.monotonic() - warm_start, 3)
        for case in cases:
            duration = float(case['duration_seconds'])
            source = {'start': 0.0, 'end': duration, 'text': case['japanese_reference']}
            measured, calls = call_log_wrapper(original_translate)
            gemma.translate_batch = measured
            started = time.monotonic()
            provenance = {
                'japanese_reference': case['japanese_reference'],
                'english_reference': case['english_reference'],
                'audio_sha256': case.get('audio_sha256'),
                'duration_seconds': duration,
            }
            try:
                segments = list(engine.gemma_segments([source], preset=PRESET, context='', glossary=None))
                outputs = [segment.text for segment in segments]
                if len(outputs) != 1:
                    raise ValueError(f'Expected one whole-clip translation; received {len(outputs)}.')
                row = {
                    'id': case['id'], 'partition': 'development',
                    'source_provenance': provenance, 'input_segments': [source],
                    'output_segments': [{'start': 0.0, 'end': duration, 'text': outputs[0]}],
                    'context_used': '', 'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls, 'status': 'ok',
                }
            except Exception as exc:
                row = {
                    'id': case['id'], 'partition': 'development',
                    'source_provenance': provenance, 'input_segments': [source],
                    'output_segments': [], 'context_used': '', 'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls, 'status': 'error',
                    'error_type': type(exc).__name__, 'error': str(exc),
                }
                stage['errors'].append({'id': case['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            finally:
                gemma.translate_batch = original_translate
            stage['cases'].append(row)
            report['completed_cases'] = len(stage['cases'])
            write_json(output_path, report)
        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['total_translation_seconds'] = round(sum(row['translation_seconds'] for row in stage['cases']), 3)
        report['status'] = stage['status']
        report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output_path, report)
        return report
    finally:
        gemma.translate_batch = original_translate
        gemma._messages = original_messages
        gemma.shutdown()


def run_reasoning_stage() -> dict:
    corpus, cases = frozen_inputs()
    baseline = json.loads(OUTPUT_PATH.read_text())
    corpus_sha = sha256(CORPUS_PATH)
    baseline_stage = baseline.get('stages', {}).get('pristine_reference', {})
    if baseline.get('dataset_provenance', {}).get('corpus_sha256') != corpus_sha:
        raise RuntimeError('Development corpus changed since the baseline reference pass.')
    if baseline_stage.get('status') != 'complete' or len(baseline_stage.get('cases', [])) != 20:
        raise RuntimeError('The 20-case pristine-reference baseline must be complete first.')
    model = model_info()
    frozen_model = baseline.get('model', {})
    if frozen_model.get('preset') != model['preset'] or frozen_model.get('provenance') != model['provenance']:
        raise RuntimeError('Gemma model provenance changed since the baseline reference pass.')
    if frozen_model.get('prompt_builder_source_sha256') != model['prompt_builder_source_sha256']:
        raise RuntimeError('The original prompt builder changed since the baseline reference pass.')
    output_path = ROOT / 'test-results/accuracy-gemma-reasoning-development.json'
    if output_path.exists():
        raise RuntimeError(f'Reasoning report already exists: {output_path.relative_to(ROOT)}')

    reasoning_budget = 128
    effort = 'medium'
    original_translate = gemma.translate_batch
    original_request = gemma._request
    actual_server_options = []
    reasoning_calls = []
    original_popen = gemma.subprocess.Popen

    def scoped_popen(args, *popen_args, **popen_kwargs):
        if not args or str(args[0]) != str(gemma.SERVER_PATH):
            return original_popen(args, *popen_args, **popen_kwargs)
        argv = list(args)
        argv[argv.index('--reasoning-budget') + 1] = str(reasoning_budget)
        argv[argv.index('--chat-template-kwargs') + 1] = '{"enable_thinking":true}'
        argv.extend(['--reasoning-format', 'deepseek', '--reasoning-effort', effort])
        actual_server_options[:] = [
            {'name': '--reasoning-budget', 'value': str(reasoning_budget)},
            {'name': '--chat-template-kwargs', 'value': '{"enable_thinking":true}'},
            {'name': '--reasoning-format', 'value': 'deepseek'},
            {'name': '--reasoning-effort', 'value': effort},
        ]
        return original_popen(argv, *popen_args, **popen_kwargs)

    def reasoning_request(path, payload=None, timeout=gemma.REQUEST_TIMEOUT):
        if path != '/v1/chat/completions':
            return original_request(path, payload, timeout=timeout)
        request_payload = dict(payload)
        request_payload['chat_template_kwargs'] = {'enable_thinking': True}
        request_payload['reasoning_effort'] = effort
        prompt_sha = hashlib.sha256(json.dumps(
            request_payload.get('messages', []), ensure_ascii=False, separators=(',', ':')
        ).encode()).hexdigest()
        response = original_request(path, request_payload, timeout=timeout)
        choice = (response.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        chain = message.get('reasoning_content')
        is_string = isinstance(chain, str)
        token_count = None
        char_count = len(chain) if is_string else 0
        if is_string:
            tokenized = original_request('/tokenize', {
                'content': chain, 'add_special': False, 'parse_special': True,
            }, timeout=timeout)
            tokens = tokenized.get('tokens')
            if not isinstance(tokens, list):
                raise ValueError('Runtime tokenizer returned no reasoning token list.')
            token_count = len(tokens)
            message.pop('reasoning_content', None)
        reasoning_calls.append({
            'prompt_sha256': prompt_sha,
            'reasoning_content_present': 'reasoning_content' in message or is_string,
            'reasoning_content_is_string': is_string,
            'reasoning_nonempty': bool(chain) if is_string else False,
            'reasoning_char_count': char_count,
            'reasoning_token_count': token_count,
            'finish_reason': choice.get('finish_reason'),
            'assistant_role': message.get('role'),
            'tool_calls_present': bool(message.get('tool_calls') or message.get('function_call')),
            'final_content_is_string': isinstance(message.get('content'), str),
        })
        return response

    stage = {
        'status': 'running', 'stage_key': 'bounded_reasoning_pristine_reference',
        'cases': [], 'errors': [], 'model': model,
        'prompt_policy': 'Original frozen E4B translation prompt, no context, glossary, examples, or extra instructions.',
        'reasoning_request': {
            'server_reasoning_budget_tokens': reasoning_budget,
            'enable_thinking': True,
            'reasoning_effort': effort,
            'reasoning_format': 'deepseek',
            'reasoning_chain_storage': 'Never stored or printed; only tokenizer count and Unicode character count are recorded. reasoning_content is removed in this test wrapper before the existing strict translation JSON validator runs.',
        },
        'warmup': {'excluded': True, 'description': 'One synthetic Japanese translation control; also verifies nonempty bounded reasoning and valid final strict JSON.'},
        'interpretation_limit': 'This is a 20-case development-only test of E4B with bounded thinking enabled; semantic review is required and this is not independent evidence.',
    }
    report = {
        'schema_version': 1, 'status': 'running',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Only 20 frozen pristine Japanese development references; no ASR, held-out rows, glossary, context, prompt additions, or examples.',
        'dataset_provenance': baseline['dataset_provenance'],
        'baseline_report_path': str(OUTPUT_PATH.relative_to(ROOT)),
        'baseline_report_sha256': sha256(OUTPUT_PATH),
        'baseline_passes_from_frozen_review': 17,
        'runtime': {
            'model': model, 'isolated_port': PORT, 'threads': 8,
            'device': 'CPU', 'gpu_layers': 0,
            'server_binary_sha256': sha256(gemma.SERVER_PATH),
            'runtime_manifest': json.loads(gemma.RUNTIME_MANIFEST.read_text()),
            'runtime_server_options': actual_server_options,
        },
        'evaluator_source_sha256': sha256(ROOT / 'tests/evaluate_translation.py'),
        'stages': {stage['stage_key']: stage},
    }
    write_json(output_path, report)
    old_port, old_base = gemma.PORT, gemma.BASE
    try:
        gemma.PORT = PORT
        gemma.BASE = f'http://{gemma.HOST}:{PORT}'
        gemma.shutdown()
        gemma.subprocess.Popen = scoped_popen
        gemma._request = reasoning_request

        control_source = {'start': 0.0, 'end': 0.8, 'text': 'こんにちは。'}
        control_measured, control_calls = call_log_wrapper(original_translate)
        gemma.translate_batch = control_measured
        control_started = time.monotonic()
        try:
            translations = control_measured([control_source], context='', preset=PRESET, glossary=None)
            if len(translations) != 1:
                raise ValueError('Control did not return exactly one translation.')
            control_call = reasoning_calls[-1] if reasoning_calls else {}
            report['control'] = {
                'status': 'ok', 'input': control_source,
                'final_translation': translations[0],
                'translation_seconds_excluded': round(time.monotonic() - control_started, 3),
                'batch_calls': control_calls,
                'protocol': control_call,
                'server_options_observed': list(actual_server_options),
            }
            if not actual_server_options:
                raise ValueError('Control server did not start with the requested bounded-thinking options.')
            if not control_call.get('reasoning_content_is_string') or not control_call.get('reasoning_nonempty'):
                raise ValueError('Control did not expose a nonempty reasoning_content string; thinking was not verified.')
            if control_call.get('reasoning_token_count') is None or control_call['reasoning_token_count'] > reasoning_budget:
                raise ValueError('Control reasoning token count is missing or exceeds the configured budget.')
        except Exception as exc:
            report['control'] = report.get('control', {}) | {
                'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc),
                'translation_seconds_excluded': round(time.monotonic() - control_started, 3),
                'batch_calls': control_calls,
                'server_options_observed': list(actual_server_options),
            }
            stage['status'] = 'control_failed'
            stage['errors'].append({'phase': 'control', 'error_type': type(exc).__name__, 'error': str(exc)})
            report['status'] = 'control_failed_no_corpus_requests'
            report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(output_path, report)
            return report

        reasoning_calls.clear()
        stage['runtime_server_options'] = list(actual_server_options)
        report['runtime']['runtime_server_options'] = list(actual_server_options)
        stage['reasoning_evidence'] = 'Control produced a nonempty reasoning_content string, runtime /tokenize counted no more than 128 tokens, and the final response passed the unchanged strict translation JSON validator after test-only removal of that string.'
        gemma.translate_batch = original_translate
        original = original_translate
        for case in cases:
            duration = float(case['duration_seconds'])
            source = {'start': 0.0, 'end': duration, 'text': case['japanese_reference']}
            measured, calls = call_log_wrapper(original)
            gemma.translate_batch = measured
            started = time.monotonic()
            telemetry_start = len(reasoning_calls)
            provenance = {
                'japanese_reference': case['japanese_reference'],
                'english_reference': case['english_reference'],
                'audio_sha256': case.get('audio_sha256'),
                'duration_seconds': duration,
            }
            try:
                segments = list(engine.gemma_segments([source], preset=PRESET, context='', glossary=None))
                outputs = [segment.text for segment in segments]
                if len(outputs) != 1:
                    raise ValueError(f'Expected one whole-clip translation; received {len(outputs)}.')
                row = {
                    'id': case['id'], 'partition': 'development',
                    'source_provenance': provenance, 'input_segments': [source],
                    'output_segments': [{'start': 0.0, 'end': duration, 'text': outputs[0]}],
                    'context_used': '', 'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls,
                    'reasoning_calls': reasoning_calls[telemetry_start:],
                    'status': 'ok',
                }
            except Exception as exc:
                row = {
                    'id': case['id'], 'partition': 'development',
                    'source_provenance': provenance, 'input_segments': [source],
                    'output_segments': [], 'context_used': '', 'glossary_used': False,
                    'translation_seconds': round(time.monotonic() - started, 3),
                    'batch_calls': calls, 'reasoning_calls': reasoning_calls[telemetry_start:],
                    'status': 'error', 'error_type': type(exc).__name__, 'error': str(exc),
                }
                stage['errors'].append({'id': case['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
            finally:
                gemma.translate_batch = original
            stage['cases'].append(row)
            report['completed_cases'] = len(stage['cases'])
            write_json(output_path, report)
        stage['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        stage['total_translation_seconds'] = round(sum(row['translation_seconds'] for row in stage['cases']), 3)
        report['status'] = stage['status']
        report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output_path, report)
        return report
    finally:
        gemma.translate_batch = original_translate
        gemma._request = original_request
        gemma.subprocess.Popen = original_popen
        gemma.shutdown()
        gemma.PORT, gemma.BASE = old_port, old_base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--references-only', action='store_true', help='Run the 20 pristine Japanese reference translations.')
    parser.add_argument('--asr-only', action='store_true', help='Run only the 20 full-clip ASR segment translations from the extension report.')
    parser.add_argument('--asr-report', type=Path, help='Append a candidate full-clip ASR stage from this completed report.')
    parser.add_argument('--stage-key', help='Unique stage key required with --asr-report.')
    parser.add_argument('--join-asr-segments', action='store_true', help='Join nonempty ASR text into one evaluation-only segment per recording; requires --asr-report.')
    parser.add_argument('--precision-prompt', action='store_true', help='Run the separate 20-case general precision-instruction prompt experiment.')
    parser.add_argument('--fewshot-prompt', action='store_true', help='Run the one authorized two-demonstration prompt experiment on pristine development text.')
    parser.add_argument('--bounded-reasoning', action='store_true', help='Run one excluded thinking control and then the authorized 20-case E4B bounded-reasoning development experiment.')
    args = parser.parse_args()
    if args.asr_report:
        if not args.stage_key or args.precision_prompt or args.fewshot_prompt or args.bounded_reasoning or args.references_only or args.asr_only:
            parser.error('--asr-report requires --stage-key and cannot be combined with other run modes.')
        result = run_candidate_asr_stage(args.asr_report.resolve(), args.stage_key, join_segments=args.join_asr_segments)
    elif args.stage_key:
        parser.error('--stage-key requires --asr-report.')
    elif args.join_asr_segments:
        parser.error('--join-asr-segments requires --asr-report and --stage-key.')
    elif args.precision_prompt:
        if args.references_only or args.asr_only or args.fewshot_prompt or args.bounded_reasoning:
            parser.error('--precision-prompt cannot be combined with other run modes.')
        result = run_precision_prompt_stage()
    elif args.fewshot_prompt:
        if args.references_only or args.asr_only or args.bounded_reasoning:
            parser.error('--fewshot-prompt cannot be combined with other run modes.')
        result = run_fewshot_prompt_stage()
    elif args.bounded_reasoning:
        if args.references_only or args.asr_only:
            parser.error('--bounded-reasoning cannot be combined with other run modes.')
        result = run_reasoning_stage()
    elif args.asr_only:
        if args.references_only:
            parser.error('--asr-only cannot be combined with --references-only.')
        result = run_full_clip_asr_stage()
    else:
        result = run_reference_stage()
    if args.precision_prompt:
        stage = result['stages']['precision_prompt_pristine_reference']
        summary = {'status': result['status'], 'stage_status': stage['status'],
                   'cases': len(stage['cases']), 'errors': len(stage['errors'])}
    elif args.fewshot_prompt:
        stage = result['stages']['fewshot_prompt_pristine_reference']
        summary = {'status': result['status'], 'stage_status': stage['status'],
                   'cases': len(stage['cases']), 'errors': len(stage['errors'])}
    elif args.bounded_reasoning:
        stage = result['stages']['bounded_reasoning_pristine_reference']
        summary = {'status': result['status'], 'stage_status': stage['status'],
                   'control': result.get('control', {}).get('status'),
                   'cases': len(stage['cases']), 'errors': len(stage['errors'])}
    elif args.asr_report:
        stage = result['stages'][args.stage_key]
        summary = {'status': result['status'], 'stage_status': stage['status'],
                   'stage_key': args.stage_key, 'cases': len(stage['cases']), 'errors': len(stage['errors'])}
    else:
        summary = {
            'status': result['status'],
            'reference_status': result['stages']['pristine_reference']['status'],
            'reference_cases': len(result['stages']['pristine_reference']['cases']),
            'asr_status': result['stages']['full_clip_asr']['status'],
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
