#!/usr/bin/env python3
"""Frozen development-only candidate selection; four serial, non-overwriting stages."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import evaluate_streaming_combinations as shared

translation = shared.translation
gemma = translation.gemma
ROOT = translation.ROOT
PROTOCOL = ROOT / 'tests/fixtures/accuracy-ensemble-protocol.json'
CONTROLS = ROOT / 'tests/fixtures/accuracy-ensemble-controls.json'
ASR = ROOT / 'test-results/accuracy-streaming-turbo-speech-development.json'
REPLAY = ROOT / 'test-results/accuracy-streaming-turbo-boundary-replay.json'
STAGES = ('generate-4b', 'generate-2b', 'judge-4b', 'judge-2b')
E2B = {
    'label': 'Gemma 4 E2B', 'directory': str(Path.home() / '.cache/jread-experiments/gemma4-e2b'),
    'alias': 'jread-eval-gemma4-e2b', 'repo': 'google/gemma-4-E2B-it-qat-q4_0-gguf',
    'revision': '675cff42a74c774d6cb76f76d8eacb49b48c9b93',
    'filename': 'gemma-4-E2B_q4_0-it.gguf',
    'sha256': 'fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634',
    'size': 3349516256,
}
PRESETS = {'4b': 'gemma4', '2b': 'eval-gemma4-e2b'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def output_path(stage):
    return ROOT / f'test-results/accuracy-ensemble-{stage}-development.json'


def inputs():
    protocol = json.loads(PROTOCOL.read_text())
    controls = json.loads(CONTROLS.read_text())
    frozen, asr, replay, units, _ = shared.validated_inputs(ASR, REPLAY)
    baseline_path = ROOT / protocol['baseline_path']
    if translation.sha256(baseline_path) != protocol['baseline_sha256']:
        raise ValueError('Frozen Turbo English baseline changed.')
    baseline = json.loads(baseline_path.read_text())
    rows = baseline['stage']['cases']
    if (baseline['status'] != 'complete' or len(frozen) != 20
            or [c['id'] for c in rows] != [c['id'] for c in frozen]
            or baseline['source_asr']['sha256'] != translation.sha256(ASR)
            or baseline['boundary_replay']['sha256'] != translation.sha256(REPLAY)):
        raise ValueError('Require the complete, matching, frozen20 Turbo baseline.')
    current_prompt = hashlib.sha256(inspect.getsource(gemma._messages).encode()).hexdigest()
    if current_prompt != baseline['model']['prompt_builder_source_sha256']:
        raise ValueError('Original Gemma prompt changed versus baseline.')
    cases = []
    for case, row in zip(frozen, rows, strict=True):
        outs = row['replay']['outputs']
        calls = [c for u in row['replay']['updates'] for c in u['translation_calls']]
        canonical = units[str(case['id'])]
        if (row['status'] != 'ok' or row['audio_sha256'] != case['audio_sha256']
                or len(outs) != 1 or len(calls) != 1 or len(canonical['units']) != 1
                or canonical['protocol_status'] != 'pass' or calls[0]['context_sent'] != ''
                or calls[0]['glossary_enabled'] or outs[0]['source_text'] != canonical['units'][0]['text']):
            raise ValueError('Each baseline case must be one successful source-only canonical unit.')
        source = dict(start=outs[0]['source_start_seconds'], end=outs[0]['source_end_seconds'],
                      text=outs[0]['source_text'])
        if digest(gemma._messages([source], '', None)) != calls[0]['prompt_sha256']:
            raise ValueError('Original source-only prompt differs from actual saved baseline call.')
        cases.append(dict(id=case['id'], audio_sha256=case['audio_sha256'], source=source,
            baseline=dict(status='ok', text=outs[0]['translation'],
                inference_seconds=row['replay']['measured_translation_seconds'], reused=True),
            review_only={k: case[k] for k in ('japanese_reference', 'english_reference')}))
    if len(controls) != 6 or len({c['id'] for c in controls}) != 6:
        raise ValueError('Require six separate frozen controls.')
    for c in controls:
        if not c['source'].strip() or not set(c['review_only']['acceptable']) <= set(c['candidates']):
            raise ValueError('Invalid control candidates or review-only keys.')
    if (len(protocol['pools']) != 10 or len(protocol['candidates']) != 8
            or {p['id'] for p in protocol['pools']} != {f'p{i:02}' for i in range(1, 11)}):
        raise ValueError('Require the frozen ten-pool/eight-candidate matrix.')
    for pool in protocol['pools']:
        if (len(set(pool['candidates'])) != len(pool['candidates'])
                or not set(pool['candidates']) <= set(protocol['candidates'])
                or pool['fallback'] not in pool['candidates'] or pool['judge'] not in PRESETS):
            raise ValueError('Invalid pool definition.')
    return protocol, controls, cases, baseline


def generation_messages(source, spec, protocol):
    messages = gemma._messages([source], '', None)
    if spec['prompt'] != 'original':
        messages[0]['content'] += ' ' + protocol[spec['prompt'] + '_instruction']
    return messages


def eligible(source, pool, candidates):
    """Exact duplicates share one choice; source-derived opaque IDs hide origins."""
    by_text = {}
    excluded = []
    for key in pool['candidates']:
        candidate = candidates[key]
        if candidate['status'] != 'ok':
            excluded.append(key)
            continue
        text = candidate['text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Successful candidate must contain nonempty text.')
        if text not in by_text:
            by_text[text] = dict(id='c' + digest([source, text])[:12], text=text, origins=[])
        by_text[text]['origins'].append(key)
    rows = list(by_text.values())
    rows.sort(key=lambda r: digest([source, pool['id'], r['origins'][0]]))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Opaque candidate ID collision.')
    return rows, excluded


def judge_messages(source, choices, protocol):
    data = dict(source=source, candidates=[{k: row[k] for k in ('id', 'text')} for row in choices])
    return [{'role': 'system', 'content': protocol['judge_system']},
            {'role': 'user', 'content': 'Select from these quoted data only.\n'
             + json.dumps(data, ensure_ascii=False, separators=(',', ':'))}]


def payload(model, messages, temperature, seed, max_tokens, allowed=None):
    text_schema = {'type': 'string'}
    if allowed is not None:
        text_schema['enum'] = allowed
    schema = {'type': 'object', 'properties': {'translations': {'type': 'array', 'minItems': 1, 'maxItems': 1,
        'items': {'type': 'object', 'properties': {'id': {'type': 'integer'}, 'text': text_schema},
                  'required': ['id', 'text'], 'additionalProperties': False}}},
        'required': ['translations'], 'additionalProperties': False}
    result = dict(model=gemma.PRESETS[PRESETS[model]]['alias'], messages=messages, stream=False,
        temperature=temperature, max_tokens=max_tokens, cache_prompt=False,
        chat_template_kwargs={'enable_thinking': False}, reasoning_effort='none', tools=[], tool_choice='none',
        response_format={'type': 'json_schema', 'json_schema': {'name': 'subtitle_translations', 'strict': True, 'schema': schema}})
    if seed is not None:
        result['seed'] = seed
    return result


def complete(model, messages, temperature, seed, max_tokens, allowed=None):
    request = payload(model, messages, temperature, seed, max_tokens, allowed)
    result = dict(status='error', request=request, prompt_sha256=digest(messages),
                  request_sha256=digest(request), inference_seconds=0.0, startup_seconds=0.0)
    try:
        with gemma._lock:
            started = time.monotonic()
            gemma._start(PRESETS[model])
            result['startup_seconds'] = time.monotonic() - started
            started = time.monotonic()
            try:
                result['raw_response'] = gemma._request('/v1/chat/completions', request)
            finally:
                result['inference_seconds'] = time.monotonic() - started
        text = gemma._translations(result['raw_response'], 1)[0]
        if allowed is not None and text not in allowed:
            raise ValueError('Judge returned an unknown ID or a rewrite.')
        result.update(status='ok', text=text)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def choose_pool(source, pool, candidates, protocol, call=complete):
    choices, excluded = eligible(source, pool, candidates)
    generation_cost = sum(candidates[k]['inference_seconds'] for k in pool['candidates'])
    result = dict(status='error', candidate_mapping=choices, excluded_invalid_candidates=excluded,
                  generation_seconds=generation_cost, orders=[], primary=None, order_stable=None)
    if not choices:
        result['error'] = 'No valid generated candidate; case remains a failed selection.'
        return result
    if len(choices) == 1:
        chosen = choices[0]
        result.update(status='ok', zero_cost_selection=True, order_sensitive=False,
            primary=dict(candidate_id=chosen['id'], text=chosen['text']),
            order_stable=dict(candidate_id=chosen['id'], text=chosen['text'], fallback=False),
            primary_sequential_seconds=generation_cost, stable_sequential_seconds=generation_cost)
        return result
    allowed = [c['id'] for c in choices]
    for rows in (choices, list(reversed(choices))):
        response = call(pool['judge'], judge_messages(source, rows, protocol),
                        0, None, protocol['judge_max_tokens'], allowed)
        result['orders'].append(dict(presented_ids=[c['id'] for c in rows], result=response))
    first, second = [order['result'] for order in result['orders']]
    result['primary_sequential_seconds'] = generation_cost + first['inference_seconds']
    result['stable_sequential_seconds'] = generation_cost + sum(o['result']['inference_seconds'] for o in result['orders'])
    lookup = {c['id']: c for c in choices}
    if first['status'] == 'ok' and first['text'] in lookup:
        result['primary'] = dict(candidate_id=first['text'], text=lookup[first['text']]['text'])
    if any(r['status'] != 'ok' or r.get('text') not in lookup for r in (first, second)):
        result['error'] = 'At least one judge order failed; retain primary if valid but stable selection fails.'
        return result
    sensitive = first['text'] != second['text']
    fallback = candidates[pool['fallback']]
    if sensitive and fallback['status'] != 'ok':
        result['error'] = 'Order-sensitive choice has an invalid original fallback.'
        return result
    chosen = next(c for c in choices if pool['fallback'] in c['origins']) if sensitive else lookup[first['text']]
    result.update(status='ok', zero_cost_selection=False, order_sensitive=sensitive,
                  order_stable=dict(candidate_id=chosen['id'], text=chosen['text'], fallback=sensitive))
    return result


def provenance(protocol):
    return dict(runner=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=translation.sha256(Path(__file__))),
        protocol=dict(path=str(PROTOCOL.relative_to(ROOT)), sha256=translation.sha256(PROTOCOL), contents=protocol),
        controls=dict(path=str(CONTROLS.relative_to(ROOT)), sha256=translation.sha256(CONTROLS)),
        corpus_sha256=translation.sha256(translation.CORPUS_PATH),
        sources={str(p.relative_to(ROOT)): translation.sha256(p) for p in (ASR, REPLAY, ROOT / protocol['baseline_path'])},
        helpers={str(Path(p).relative_to(ROOT)): translation.sha256(Path(p)) for p in
                 (shared.__file__, translation.__file__, translation.EVALUATOR_PATH, gemma.__file__)})


def load_candidates(cases, current_provenance):
    by_id = {c['id']: {'e4_original': c['baseline']} for c in cases}
    dependencies = {}
    for model in ('4b', '2b'):
        path = output_path('generate-' + model)
        report = json.loads(path.read_text())
        if (report['status'] not in ('complete', 'complete_with_errors')
                or report['provenance'] != current_provenance or report['stage'] != 'generate-' + model
                or [c['id'] for c in report['cases']] != list(by_id)
                or not report['cleanup']['owned_process_released']):
            raise ValueError('Candidate artifact is incomplete or has different provenance.')
        expected = {key for key, spec in current_provenance['protocol']['contents']['candidates'].items()
                    if spec['model'] == model and not spec.get('reused')}
        for row, source in zip(report['cases'], cases, strict=True):
            if (row['source'] != source['source'] or row['audio_sha256'] != source['audio_sha256']
                    or set(row['candidates']) != expected):
                raise ValueError('Candidate source changed between generation and selection.')
            by_id[row['id']].update(row['candidates'])
        dependencies[str(path.relative_to(ROOT))] = translation.sha256(path)
    return by_id, dependencies


def run(stage):
    protocol, controls, cases, baseline = inputs()
    path = output_path(stage)
    if path.exists():
        raise ValueError(f'Preserve existing artifact; refusing overwrite: {path}')
    if translation.port_accepting_connections(protocol['port']):
        raise ValueError('Private evaluation port is occupied; refusing concurrent model work.')
    model = stage.rsplit('-', 1)[1]
    prior_preset = gemma.PRESETS.get(PRESETS['2b'])
    gemma.PRESETS[PRESETS['2b']] = E2B.copy()  # Process-local experiment, no production settings or files.
    gemma.PORT = protocol['port']
    gemma.BASE = f'http://{gemma.HOST}:{gemma.PORT}'
    report = None
    artifact_created = False
    try:
        if not gemma.ready(PRESETS[model]) or not gemma.runtime_ready():
            raise ValueError('Selected pinned model/runtime must already be verified; evaluator never downloads.')
        if gemma.PRESETS['gemma4'] != baseline['model']['provenance']:
            raise ValueError('E4B model provenance changed versus reused baseline.')
        runtime = json.loads(gemma.RUNTIME_MANIFEST.read_text())
        if runtime != baseline['model']['runtime']:
            raise ValueError('Runtime provenance changed versus reused baseline.')
        evidence = provenance(protocol)
        all_candidates, dependencies = load_candidates(cases, evidence) if stage.startswith('judge') else ({}, {})
        report = dict(schema_version=1, status='running', stage=stage, partition='development', heldout_rows_used=0,
            created_at_utc=datetime.now(timezone.utc).isoformat(), provenance=evidence, dependencies=dependencies,
            model=dict(config=gemma.PRESETS[PRESETS[model]], runtime=runtime, device='CPU', threads=8,
                runtime_binary_sha256=translation.sha256(gemma.SERVER_PATH),
                context_tokens=4096, port=gemma.PORT, thinking=False),
            cases=[], controls=[], errors=[],
            limits=['Text-only frozen20 development diagnostic; no heldout, audio inference, or live performance.',
                'References and grades are review-only and never passed to generation or selection.',
                'All20 cases remain in each configuration denominator; invalid outputs are failures, without retries.',
                'Candidate selection cannot recover information absent from Japanese ASR.',
                'Timings are sequential measured inference, including reused baseline work; no parallel-speed estimate.'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            artifact_created = True
            json.dump(report, stream, ensure_ascii=False, indent=2)
        report['warmup'] = complete(model, gemma._messages([{'start': 0, 'end': 1, 'text': 'こんにちは。'}], ''),
                                    0, None, protocol['generation_max_tokens'])
        report['warmup']['excluded'] = True
        if report['warmup']['status'] != 'ok':
            raise ValueError('Warmup failed; preserve artifact and do not retry.')
        for case in cases:
            row = copy.deepcopy(case)
            if stage.startswith('generate'):
                row['candidates'] = {}
                for key, spec in protocol['candidates'].items():
                    if spec['model'] != model or spec.get('reused'):
                        continue
                    response = complete(model, generation_messages(case['source'], spec, protocol),
                        spec['temperature'], spec.get('seed'), protocol['generation_max_tokens'])
                    row['candidates'][key] = response
                    if response['status'] != 'ok':
                        report['errors'].append(dict(id=case['id'], candidate=key, error=response['error']))
                    translation.write_json(path, report | {'in_progress_case': row})
            else:
                row['pools'] = {}
                for pool in protocol['pools']:
                    if pool['judge'] != model:
                        continue
                    selection = choose_pool(case['source']['text'], pool, all_candidates[case['id']], protocol)
                    row['pools'][pool['id']] = selection
                    if selection['status'] != 'ok':
                        report['errors'].append(dict(id=case['id'], pool=pool['id'], error=selection['error']))
                    translation.write_json(path, report | {'in_progress_case': row})
            report['cases'].append(row)
            translation.write_json(path, report)
        if stage.startswith('judge'):
            for control in controls:
                pool = dict(id='control-' + control['id'], candidates=list(control['candidates']), judge=model,
                            fallback=next(iter(control['candidates'])))
                candidates = {k: dict(status='ok', text=v, inference_seconds=0.0) for k, v in control['candidates'].items()}
                result = choose_pool(control['source'], pool, candidates, protocol)
                report['controls'].append(dict(control, result=result, excluded_from_20_case_denominator=True))
                if result['status'] != 'ok':
                    report['errors'].append(dict(control=control['id'], error=result['error']))
                translation.write_json(path, report)
        report['status'] = 'complete_with_errors' if report['errors'] else 'complete'
        report['aggregate'] = dict(case_count=len(report['cases']), error_count=len(report['errors']),
                                   meaning_accuracy='Pending independent review; no automatic semantic grade.')
        if stage.startswith('judge'):
            report['aggregate']['pools'] = {p['id']: dict(case_count=20,
                primary_outputs=sum(c['pools'][p['id']]['primary'] is not None for c in report['cases']),
                stable_outputs=sum(c['pools'][p['id']]['order_stable'] is not None for c in report['cases']),
                order_sensitive=sum(c['pools'][p['id']].get('order_sensitive', False) for c in report['cases']),
                primary_sequential_seconds=sum(c['pools'][p['id']].get('primary_sequential_seconds', c['pools'][p['id']]['generation_seconds']) for c in report['cases']),
                stable_sequential_seconds=sum(c['pools'][p['id']].get('stable_sequential_seconds', c['pools'][p['id']]['generation_seconds']) for c in report['cases']))
                for p in protocol['pools'] if p['judge'] == model}
        else:
            report['aggregate']['generation_seconds'] = sum(v['inference_seconds'] for c in report['cases'] for v in c['candidates'].values())
        report['aggregate']['new_inference_seconds_excluding_warmup'] = (
            sum(v['inference_seconds'] for c in report['cases'] for v in c.get('candidates', {}).values())
            + sum(o['result']['inference_seconds'] for c in report['cases'] for p in c.get('pools', {}).values() for o in p['orders'])
            + sum(o['result']['inference_seconds'] for c in report['controls'] for o in c['result']['orders']))
    except BaseException as error:
        if report is not None:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        process = gemma._process
        owner = process is None or gemma._owner_pid == os.getpid()
        gemma.shutdown()
        if prior_preset is None:
            gemma.PRESETS.pop(PRESETS['2b'], None)
        else:
            gemma.PRESETS[PRESETS['2b']] = prior_preset
        if report is not None and artifact_created:
            report['cleanup'] = dict(owner_matches_this_evaluator=owner,
                owned_process_pid=process.pid if process else None,
                owned_process_released=process is None or process.poll() is not None,
                port_accepting_connections_after_shutdown=translation.port_accepting_connections(gemma.PORT))
            report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            translation.write_json(path, report)
    print(json.dumps(dict(status=report['status'], report=str(path.relative_to(ROOT)))))


def self_test():
    from unittest.mock import patch
    protocol = json.loads(PROTOCOL.read_text())
    source = '今日は行きません。'
    candidates = {k: dict(status='ok', text=t, inference_seconds=i + 1) for i, (k, t) in enumerate(
        [('original', 'I will not go today.'), ('literal', 'I will not go today.'), ('bad', 'I will go today.')])}
    pool = dict(id='example', candidates=list(candidates), judge='4b', fallback='original')
    seen = []
    def faithful(model, messages, temperature, seed, limit, allowed):
        data = json.loads(messages[1]['content'].split('\n', 1)[1])
        assert set(data) == {'source', 'candidates'} and data['source'] == source
        assert all(set(c) == {'id', 'text'} for c in data['candidates'])
        seen.append(data['candidates'])
        chosen = next(c['id'] for c in data['candidates'] if c['text'] == candidates['original']['text'])
        return dict(status='ok', text=chosen, inference_seconds=.5)
    result = choose_pool(source, pool, candidates, protocol, faithful)
    assert result['status'] == 'ok' and not result['order_sensitive']
    assert seen[1] == list(reversed(seen[0])) and len(seen[0]) == 2
    assert result['primary_sequential_seconds'] == 6.5 and result['stable_sequential_seconds'] == 7
    def first(*args):
        data = json.loads(args[1][1]['content'].split('\n', 1)[1])
        return dict(status='ok', text=data['candidates'][0]['id'], inference_seconds=1)
    biased = choose_pool(source, pool, candidates, protocol, first)
    assert biased['order_sensitive'] and biased['order_stable']['fallback']
    assert biased['order_stable']['text'] == candidates['original']['text']
    candidates['bad'] = dict(status='error', error='truncated', inference_seconds=3)
    single = choose_pool(source, pool, candidates, protocol, lambda *a: (_ for _ in ()).throw(AssertionError('No judge call expected')))
    assert single['zero_cost_selection'] and single['excluded_invalid_candidates'] == ['bad']
    assert single['primary_sequential_seconds'] == 6
    candidates['literal']['text'] = 'Different.'
    failed = choose_pool(source, pool, candidates, protocol, lambda *a: dict(status='error', error='bad JSON', inference_seconds=2))
    assert failed['primary'] is None and failed['order_stable'] is None and failed['status'] == 'error'
    response = {'choices': [{'finish_reason': 'length', 'message': {'role': 'assistant', 'content': '{"translations":[{"id":0,"text":"c1"}]}'}}]}
    try:
        gemma._translations(response, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Truncated JSON must remain a failed candidate.')
    valid_response = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
        'content': '{"translations":[{"id":0,"text":"a rewritten sentence"}]}'}}]}
    with patch.object(gemma, '_start'), patch.object(gemma, '_request', return_value=valid_response):
        bad_id = complete('4b', [{'role': 'user', 'content': 'data'}], 0, None, 64, ['c123'])
        assert bad_id['status'] == 'error' and 'unknown ID' in bad_id['error']
    sample = protocol['candidates']['e4_sample17']
    messages = generation_messages({'start': 0, 'end': 1, 'text': source}, sample, protocol)
    assert messages == gemma._messages([{'start': 0, 'end': 1, 'text': source}], '', None)
    request = payload('4b', messages, sample['temperature'], sample['seed'], 1024)
    assert request['temperature'] == .6 and request['seed'] == 17 and request['cache_prompt'] is False
    print('Ensemble self-test passed: hidden origins, deduplication, reversed orders, fallback, costs and failures; no inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.validate_only:
        protocol, controls, cases, _ = inputs()
        print(json.dumps(dict(development_cases=len(cases), controls=len(controls), pools=len(protocol['pools']),
                              provenance=provenance(protocol), inference_performed=False)))
    elif args.stage:
        run(args.stage)
    else:
        parser.error('Choose --self-test, --validate-only or one explicit --stage.')
