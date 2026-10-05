#!/usr/bin/env python3
"""Isolated judge-schema fix; reuse frozen candidates, never alter or regenerate them."""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import evaluate_translation_ensemble as base

ROOT, gemma = base.ROOT, base.gemma
ORIGINAL_PAYLOAD = base.payload


def fixed_payload(model, messages, temperature, seed, max_tokens, allowed=None):
    request = ORIGINAL_PAYLOAD(model, messages, temperature, seed, max_tokens, allowed)
    if allowed is not None:
        request['response_format']['json_schema']['schema']['properties']['translations']['items']['properties']['id']['enum'] = [0]
    return request


def validated():
    protocol, controls, cases, baseline = base.inputs()
    original_provenance = base.provenance(protocol)
    candidates, dependencies = base.load_candidates(cases, original_provenance)
    judge_path = base.output_path('judge-4b')
    previous = json.loads(judge_path.read_text())
    if (previous['status'] not in ('complete', 'complete_with_errors')
            or previous['provenance'] != original_provenance
            or [c['id'] for c in previous['cases']] != [c['id'] for c in cases]
            or not previous['cleanup']['owned_process_released']):
        raise ValueError('Require completed original E4B judging with matching frozen provenance.')
    dependencies[str(judge_path.relative_to(ROOT))] = base.translation.sha256(judge_path)
    return protocol, controls, cases, baseline, candidates, dependencies, previous


def raw_control_result(control, selection):
    """Score actual protocol-valid choices, never the known-correct fallback."""
    mapping = {c['id']: c['origins'] for c in selection['candidate_mapping']}
    acceptable = set(control['review_only']['acceptable'])
    results = []
    for order in selection['orders']:
        response = order['result']
        selected = response.get('text') if response['status'] == 'ok' else None
        results.append(dict(protocol_valid=response['status'] == 'ok', selected_id=selected,
            acceptable=response['status'] == 'ok' and bool(acceptable.intersection(mapping[selected]))))
    if len(results) != 2:
        raise ValueError('Each synthetic control must have both actual judge calls.')
    return results


def run(pool_id=None):
    protocol, controls, cases, baseline, candidates, dependencies, previous = validated()
    pool = next((p for p in protocol['pools'] if p['id'] == pool_id and p['judge'] == '4b'), None) if pool_id else None
    if pool_id and pool is None:
        raise ValueError('Select exactly one frozen pool using the E4B judge.')
    suffix = f'{pool_id}-development' if pool else 'controls'
    output = ROOT / f'test-results/accuracy-ensemble-id0-{suffix}.json'
    if output.exists():
        raise ValueError('Preserve existing fixed-schema result; refusing overwrite.')
    if base.translation.port_accepting_connections(protocol['port']):
        raise ValueError('Private evaluation port is occupied; refusing model contention.')
    if not gemma.ready('gemma4') or not gemma.runtime_ready():
        raise ValueError('Verified E4B/runtime must already be installed; no downloads.')
    if gemma.PRESETS['gemma4'] != baseline['model']['provenance'] or json.loads(gemma.RUNTIME_MANIFEST.read_text()) != baseline['model']['runtime']:
        raise ValueError('Model/runtime differs from frozen original inference.')
    if base.translation.sha256(gemma.SERVER_PATH) != previous['model']['runtime_binary_sha256']:
        raise ValueError('Runtime binary differs from original judge inference.')
    gate = None
    if pool:
        gate_path = ROOT / 'test-results/accuracy-ensemble-id0-controls.json'
        gate_report = json.loads(gate_path.read_text())
        if (gate_report['status'] != 'complete' or not gate_report['aggregate']['all_controls_passed']
                or gate_report['aggregate']['protocol_valid_calls'] != 12
                or gate_report['runner']['sha256'] != base.translation.sha256(Path(__file__))
                or gate_report['original_artifacts'] != dependencies
                or not gate_report['cleanup']['owned_process_released']):
            raise ValueError('The same frozen adapter must pass all twelve raw synthetic choices before the development screen.')
        gate = dict(path=str(gate_path.relative_to(ROOT)), sha256=base.translation.sha256(gate_path))
    gemma.PORT = protocol['port']
    gemma.BASE = f'http://{gemma.HOST}:{gemma.PORT}'
    report = dict(schema_version=1, status='running', mode='single-pool-development' if pool else 'controls-only',
        pool=pool, created_at_utc=datetime.now(timezone.utc).isoformat(), heldout_rows_used=0,
        runner=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=base.translation.sha256(Path(__file__))),
        original_provenance=base.provenance(protocol), original_artifacts=dependencies, control_gate=gate,
        patch=dict(scope='Judge requests only', change='translations.items.properties.id adds enum:[0]',
                   unchanged=['prompts', 'candidate labels/order/text', 'temperature0', '64-token judge limit', 'strict parser']),
        model=previous['model'], controls=[], cases=[], errors=[],
        limits=['Controls use raw choices in each order; known-correct fallback is excluded from control scoring.',
                'No candidate generation, parser repair, retries, heldout data or production changes.',
                'Schema validity does not establish semantic reliability or remove position bias.'])
    created = False
    original = base.payload
    try:
        with output.open('x') as stream:
            created = True
            json.dump(report, stream, ensure_ascii=False, indent=2)
        base.payload = fixed_payload
        report['warmup'] = base.complete('4b', gemma._messages([{'start': 0, 'end': 1, 'text': 'こんにちは。'}], ''),
                                         0, None, protocol['generation_max_tokens'])
        report['warmup']['excluded'] = True
        if report['warmup']['status'] != 'ok':
            raise ValueError('Warmup failed; no retries.')
        if pool:
            previous_cases = {c['id']: c for c in previous['cases']}
            for case in cases:
                result = base.choose_pool(case['source']['text'], pool, candidates[case['id']], protocol)
                old = previous_cases[case['id']]['pools'][pool_id]
                if (result['candidate_mapping'] != old['candidate_mapping']
                        or [o['presented_ids'] for o in result['orders']] != [o['presented_ids'] for o in old['orders']]):
                    raise ValueError('Candidate mapping/order changed from original pool.')
                report['cases'].append(dict(case, result=result))
                if result['status'] != 'ok':
                    report['errors'].append(dict(id=case['id'], error=result['error']))
                base.translation.write_json(output, report)
        else:
            for control in controls:
                control_pool = dict(id='control-' + control['id'], candidates=list(control['candidates']),
                                    judge='4b', fallback=next(iter(control['candidates'])))
                choices = {k: dict(status='ok', text=v, inference_seconds=0) for k, v in control['candidates'].items()}
                result = base.choose_pool(control['source'], control_pool, choices, protocol)
                old = next(c['result'] for c in previous['controls'] if c['id'] == control['id'])
                if (result['candidate_mapping'] != old['candidate_mapping']
                        or [o['presented_ids'] for o in result['orders']] != [o['presented_ids'] for o in old['orders']]):
                    raise ValueError('Synthetic candidate mapping/order changed from original controls.')
                report['controls'].append(dict(control, result=result, raw_orders=raw_control_result(control, result)))
                if result['status'] != 'ok':
                    report['errors'].append(dict(control=control['id'], error=result['error']))
                base.translation.write_json(output, report)
        report['status'] = 'complete_with_errors' if report['errors'] else 'complete'
        selections = [c['result'] for c in report['cases'] + report['controls']]
        calls = [o for s in selections for o in s['orders']]
        report['aggregate'] = dict(case_count=len(report['cases']), control_count=len(report['controls']),
            raw_judge_calls=len(calls), protocol_valid_calls=sum(o['result']['status'] == 'ok' for o in calls),
            new_inference_seconds_excluding_warmup=sum(o['result']['inference_seconds'] for o in calls),
            first_position_valid_choices=sum(o['result']['status'] == 'ok' and o['result']['text'] == o['presented_ids'][0] for o in calls),
            order_sensitive_valid_pairs=sum(bool(s.get('order_sensitive')) for s in selections),
            valid_judged_pairs=sum(s['status'] == 'ok' and bool(s['orders']) for s in selections))
        if pool:
            report['aggregate'].update(denominator=20, meaning_accuracy='Pending frozen-rubric review.',
                primary_outputs=sum(s['primary'] is not None for s in selections),
                stable_outputs=sum(s['order_stable'] is not None for s in selections),
                primary_sequential_seconds=sum(s.get('primary_sequential_seconds', s['generation_seconds']) for s in selections),
                stable_sequential_seconds=sum(s.get('stable_sequential_seconds', s['generation_seconds']) for s in selections))
        else:
            report['aggregate'].update(raw_control_denominator=12,
                acceptable_raw_choices=sum(o['acceptable'] for c in report['controls'] for o in c['raw_orders']),
                all_controls_passed=all(o['acceptable'] for c in report['controls'] for o in c['raw_orders']))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        base.payload = original
        process = gemma._process
        owned = process is None or gemma._owner_pid == os.getpid()
        gemma.shutdown()
        if created:
            report['cleanup'] = dict(owner_matches_this_evaluator=owned, owned_process_pid=process.pid if process else None,
                owned_process_released=process is None or process.poll() is not None,
                port_accepting_connections_after_shutdown=base.translation.port_accepting_connections(gemma.PORT))
            report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            base.translation.write_json(output, report)
    print(json.dumps(dict(status=report['status'], report=str(output.relative_to(ROOT)))))


def self_test():
    messages = [{'role': 'user', 'content': 'quoted source'}]
    before = ORIGINAL_PAYLOAD('4b', messages, 0, None, 64, ['opaque1', 'opaque2'])
    expected = copy.deepcopy(before)
    expected['response_format']['json_schema']['schema']['properties']['translations']['items']['properties']['id']['enum'] = [0]
    assert fixed_payload('4b', messages, 0, None, 64, ['opaque1', 'opaque2']) == expected
    assert before == ORIGINAL_PAYLOAD('4b', messages, 0, None, 64, ['opaque1', 'opaque2'])
    assert fixed_payload('4b', messages, 0, None, 1024) == ORIGINAL_PAYLOAD('4b', messages, 0, None, 1024)
    response = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
        'content': '{"translations":[{"id":7,"text":"opaque1"}]}'}}]}
    try:
        gemma._translations(response, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('The parser must continue rejecting nonzero envelope IDs.')
    control = dict(review_only={'acceptable': ['correct']})
    result = dict(candidate_mapping=[dict(id='a', origins=['correct']), dict(id='b', origins=['wrong'])],
        order_stable=dict(text='correct fallback'),
        orders=[dict(result=dict(status='ok', text='b')), dict(result=dict(status='error'))])
    assert not any(o['acceptable'] for o in raw_control_result(control, result))
    print('ID0 adapter self-test passed: only judge enum changed; strict parser retained; fallback excluded from raw control score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--controls-only', action='store_true')
    mode.add_argument('--pool', choices=['p01', 'p02', 'p03', 'p06', 'p07', 'p09', 'p10'])
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.validate_only:
        _, controls, cases, _, _, dependencies, _ = validated()
        print(json.dumps(dict(development_cases=len(cases), controls=len(controls), original_artifacts=dependencies,
                              runner_sha256=base.translation.sha256(Path(__file__)), inference_performed=False)))
    else:
        run(args.pool)
