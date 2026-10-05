#!/usr/bin/env python3
"""Translate saved Qwen development EOF transcripts; no audio/model recognition.

One whole-clip text unit per recording, with deliberately unaligned [0,duration]
bounds. This is a text diagnostic, not a streaming-caption test.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import evaluate_streaming_translation as shared

ROOT = shared.ROOT
BASELINE = ROOT / 'test-results/accuracy-asr-reconciliation-development.json'
DEFAULT_MODEL = '0.6b'
ASR_PRESETS = {
    '0.6b': dict(source='accuracy-qwen-asr-development.json',
        sha256='fd57e960903f53547627cd6fcd2a001240b7b0ecea4705562d3aaae21759fed9',
        repo='Qwen/Qwen3-ASR-0.6B', revision='5eb144179a02acc5e5ba31e748d22b0cf3e303b0',
        output='accuracy-qwen-translation-development.json'),
    '1.7b': dict(source='accuracy-qwen-asr-1.7b-development.json',
        sha256='10a3ba67483e9745760c9bca12d7b4c380872178bcaa557106d3457cc1e65427',
        repo='Qwen/Qwen3-ASR-1.7B', revision='7278e1e70fe206f11671096ffdd38061171dd6e5',
        output='accuracy-qwen-1.7b-translation-development.json'),
}
PRESET = 'gemma4'
gemma, engine = shared.gemma, shared.engine


def load_inputs(model_size=DEFAULT_MODEL):
    preset = ASR_PRESETS[model_size]
    asr_path = ROOT / 'test-results' / preset['source']
    manifest, frozen = shared.load_development_cases()
    if shared.sha256(asr_path) != preset['sha256']:
        raise ValueError('Qwen ASR artifact differs from the reviewed full-development screening.')
    asr = json.loads(asr_path.read_text())
    if asr['model']['repo'] != preset['repo'] or asr['model']['revision'] != preset['revision']:
        raise ValueError('ASR model provenance differs from the selected pinned preset.')
    if (asr['status'] != 'complete' or asr['mode'] != 'full-development'
            or asr['partition'] != 'development' or asr['heldout_audio_opened'] != 0
            or asr['completed_cases'] != 20 or asr['worker_exit_code'] != 0
            or not asr['owned_gpu_process_released']):
        raise ValueError('Require all20 completed development ASR cases and confirmed process cleanup.')
    if asr['corpus']['sha256'] != shared.sha256(shared.CORPUS_PATH):
        raise ValueError('Frozen corpus changed after ASR.')
    if [c['id'] for c in asr['cases']] != [c['id'] for c in frozen]:
        raise ValueError('ASR must contain exactly the frozen20 development IDs in order.')
    for source, case in zip(asr['cases'], frozen, strict=True):
        for key in ('partition', 'audio_sha256', 'duration_seconds', 'japanese_reference', 'english_reference'):
            if source[key] != case[key]:
                raise ValueError(f'ASR case provenance differs from frozen case {case["id"]}: {key}.')
    baseline = json.loads(BASELINE.read_text())
    return frozen, asr, baseline


def source_unit(case):
    duration = case['duration_seconds']
    full = case['full_clip']
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not 0 < duration <= 30):
        raise ValueError('Whole-clip duration must be finite and within the frozen30-second limit.')
    if full.get('status') != 'ok' or not isinstance(full.get('text'), str) or not full['text'].strip():
        raise ValueError('ASR has missing/bad output; preserve this case as a failure without translation.')
    return {'start': 0.0, 'end': float(duration), 'text': full['text']}


def translate_unit(source):
    original = gemma.translate_batch
    measured, calls = shared.EVALUATOR.call_log_wrapper(original)
    messages = gemma._messages([source], '', None)
    try:
        gemma.translate_batch = measured
        result = list(engine.gemma_segments([source], preset=PRESET, context='', glossary=None))
        if len(result) != 1 or len(calls) != 1 or not result[0].text.strip():
            raise ValueError('Expected exactly one nonempty English output and one call per full clip.')
        expected_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if calls[0]['context_sent'] != '' or calls[0]['glossary_enabled'] or calls[0]['prompt_sha256'] != expected_hash:
            raise ValueError('Actual translation prompt differed from the recorded source-only prompt.')
        return dict(english=result[0].text, translation_calls=calls,
                    translation_prompt=messages, translation_prompt_sha256=expected_hash)
    finally:
        gemma.translate_batch = original


def run(model_size=DEFAULT_MODEL):
    preset = ASR_PRESETS[model_size]
    asr_path = ROOT / 'test-results' / preset['source']
    output = ROOT / 'test-results' / preset['output']
    if output.exists():
        raise ValueError('Preserve the existing Qwen translation artifact; refusing overwrite.')
    frozen, asr, baseline = load_inputs(model_size)
    if shared.port_accepting_connections(shared.PORT):
        raise ValueError('Private evaluation port18767 is occupied; refusing competing model work.')
    model = shared.EVALUATOR.model_info()
    for key in ('preset', 'provenance', 'prompt_builder_source_sha256', 'device', 'threads', 'runtime'):
        if model[key] != baseline['model'][key]:
            raise ValueError(f'Normal E4B translator differs from the reconciliation comparison: {key}.')
    report = dict(schema_version=1, status='running', created_at_utc=datetime.now(timezone.utc).isoformat(),
        partition='development', heldout_rows_used=0, asr_model_size=model_size,
        corpus=dict(path=str(shared.CORPUS_PATH.relative_to(ROOT)), sha256=shared.sha256(shared.CORPUS_PATH),
                    case_ids=[c['id'] for c in frozen]),
        source_asr=dict(path=str(asr_path.relative_to(ROOT)), sha256=preset['sha256'], model=asr['model'],
                        flags=asr['flags'], aggregate=asr['aggregate'],
                        controls_passed=asr['controls_passed'], silence_control=asr['silence_control']),
        comparison=dict(path=str(BASELINE.relative_to(ROOT)), sha256=shared.sha256(BASELINE)), model=model,
        runner=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=shared.sha256(Path(__file__)),
                    evaluator_sha256=shared.sha256(shared.EVALUATOR_PATH),
                    shared_helper_sha256=shared.sha256(Path(shared.__file__)),
                    engine_gemma_segments_sha256=hashlib.sha256(inspect.getsource(engine.gemma_segments).encode()).hexdigest()),
        translation=dict(method='Original engine.gemma_segments and original Gemma prompt/parser.',
                         context='', glossary=None, whole_clip_units=True, word_aligned=False,
                         timestamps='[0,duration] are container bounds only, not caption or word timings.',
                         references='Stored for independent review only; never supplied to translation.'),
        warmup=dict(excluded=True, source='こんにちは。'),
        stage=dict(status='running', cases=[], errors=[]),
        limits=['EOF-only text diagnostic, not a streaming or live-latency experiment.',
                'All20 cases remain in the denominator, including any source or translation failure.',
                'ASR silence hallucination is retained; no VAD, filtering, correction, or retries.',
                'Meaning accuracy requires independent review; no automatic score is assigned.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    stage = report['stage']
    try:
        started = time.monotonic()
        gemma.translate_batch([{'start': 0.0, 'end': .8, 'text': 'こんにちは。'}],
                              context='', preset=PRESET, glossary=None)
        report['warmup']['elapsed_seconds'] = time.monotonic() - started
        for case in asr['cases']:
            row = {k: case[k] for k in ('id', 'partition', 'audio_sha256', 'duration_seconds',
                                        'japanese_reference', 'english_reference')}
            row['asr'] = case['full_clip']
            started = time.monotonic()
            try:
                source = source_unit(case)
                row['translation_input'] = source
                row.update(translate_unit(source), status='ok')
            except Exception as error:
                row.update(status='error', english=None, error=f'{type(error).__name__}: {error}')
                stage['errors'].append({'id': case['id'], 'error': row['error']})
            row['translation_seconds'] = time.monotonic() - started
            stage['cases'].append(row)
            shared.write_json(output, report)
        stage['aggregate'] = dict(case_count=len(stage['cases']),
            error_count=len(stage['errors']), output_count=sum(c['status'] == 'ok' for c in stage['cases']),
            translation_seconds=sum(c['translation_seconds'] for c in stage['cases']),
            meaning_accuracy='Pending independent frozen-rubric review.')
        stage['status'] = report['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
    except BaseException as error:
        stage['status'] = report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        process = gemma._process
        owned = process is None or gemma._owner_pid == os.getpid()
        gemma.shutdown()
        stage['cleanup'] = dict(owner_matches_this_evaluator=owned,
            owned_process_pid=process.pid if process else None,
            owned_process_released=process is None or process.poll() is not None,
            port_accepting_connections_after_shutdown=shared.port_accepting_connections(shared.PORT))
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        shared.write_json(output, report)
    print(json.dumps({'status': report['status'], 'report': str(output.relative_to(ROOT))}))


def self_test():
    assert DEFAULT_MODEL == '0.6b'
    assert ASR_PRESETS[DEFAULT_MODEL]['output'] == 'accuracy-qwen-translation-development.json'
    assert ASR_PRESETS['1.7b']['output'] != ASR_PRESETS[DEFAULT_MODEL]['output']
    case = dict(duration_seconds=2.5, full_clip={'status': 'ok', 'text': 'これは音声だけです。'},
                japanese_reference='REFERENCE MUST NOT REACH MODEL')
    source = source_unit(case)
    assert source == {'start': 0.0, 'end': 2.5, 'text': 'これは音声だけです。'}
    original = gemma.translate_batch
    seen = []
    def fake(segments, context='', preset='gemma4', glossary=None):
        seen.append((segments, context, preset, glossary))
        return ['Only the spoken text.']
    try:
        gemma.translate_batch = fake
        result = translate_unit(source)
        assert result['english'] == 'Only the spoken text.'
        assert seen == [([source], '', PRESET, None)]
        assert gemma.translate_batch is fake
    finally:
        gemma.translate_batch = original
    for bad in ({'status': 'bad_output', 'text': 'unfinished'}, {'status': 'ok', 'text': ''}):
        try:
            source_unit({**case, 'full_clip': bad})
        except ValueError:
            pass
        else:
            raise AssertionError('Bad or empty ASR must not be translated.')
    print('Qwen translation self-test passed: source-only single-unit mapping and failed-input handling; no inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--model-size', choices=tuple(ASR_PRESETS), default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.validate_only:
        frozen, _, _ = load_inputs(args.model_size)
        print(json.dumps({'validated_development_cases': len(frozen), 'source_asr_sha256': ASR_PRESETS[args.model_size]['sha256'],
                          'inference_performed': False}))
    else:
        run(args.model_size)
