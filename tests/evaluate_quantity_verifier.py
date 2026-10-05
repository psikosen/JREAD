#!/usr/bin/env python3
"""Fresh paired comparison of XCOMET alone and a conservative quantity veto."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import evaluate_caption_verifiers as prior

ROOT = Path(__file__).resolve().parents[1]
BASE = prior.BASE / 'quantity-v2'
PROTOCOL = ROOT / 'tests/fixtures/accuracy-quantity-verifier-v2.json'


def preferred(good_score, bad_score, good_status, bad_status):
    """No thresholds: act only on a comparable consistent/mismatch contrast."""
    if (good_status, bad_status) == ('consistent', 'mismatch'):
        return True, True
    if (good_status, bad_status) == ('mismatch', 'consistent'):
        return False, True
    return good_score > bad_score, False


def summarize(cases):
    families = {}
    for case in cases:
        counts = families.setdefault(case['family'], {'passed': 0, 'total': 0})
        counts['passed'] += case['passed_pairs']
        counts['total'] += 4
    total = 4 * len(cases)
    passed = sum(c['passed_pairs'] for c in cases)
    return {'passed_pairs': passed, 'total_pairs': total, 'fraction': passed / total,
            'all_pairs_pass_groups': sum(c['passed_pairs'] == 4 for c in cases),
            'total_groups': len(cases), 'families': families,
            'faithful_variants': {str(i): {'passed': sum(p['pass_'] for c in cases for p in c['pairs'] if p['good_index'] == i),
                                         'total': 2 * len(cases)} for i in range(2)}, 'cases': cases}


def analyze(groups, requests, rows):
    from quantity_guard import check
    by_id = {r['request_id']: r for r in rows}
    guarded, seconds = {}, 0.0
    for request in requests:
        start = time.perf_counter()
        guarded[request['request_id']] = check(request['source'], request['text'])
        seconds += time.perf_counter() - start
        if guarded[request['request_id']].get('status') not in ('consistent', 'mismatch', 'abstain'):
            raise ValueError('Unknown guard verdict.')
    baseline_cases, combined_cases = [], []
    improved = regressed = overrides = 0
    for group in groups:
        baseline_pairs, combined_pairs = [], []
        for pair in group['pairs']:
            good, bad = pair['good_request'], pair['bad_request']
            g, b = by_id[good]['score'], by_id[bad]['score']
            original = g > b
            chosen, override = preferred(g, b, guarded[good]['status'], guarded[bad]['status'])
            improved += not original and chosen
            regressed += original and not chosen
            overrides += override
            evidence = {'good_score': g, 'bad_score': b, 'good_guard': guarded[good]['status'],
                        'bad_guard': guarded[bad]['status'], 'rule_applied': override}
            baseline_pairs.append(pair | {'pass_': original, 'evidence': evidence})
            combined_pairs.append(pair | {'pass_': chosen, 'evidence': evidence})
        for dest, pairs in ((baseline_cases, baseline_pairs), (combined_cases, combined_pairs)):
            dest.append({'id': group['id'], 'family': group['family'], 'pairs': pairs,
                         'passed_pairs': sum(p['pass_'] for p in pairs)})
    good_ids = {p['good_request'] for g in groups for p in g['pairs']}
    bad_ids = {p['bad_request'] for g in groups for p in g['pairs']}
    faithful = dict(Counter(guarded[i]['status'] for i in good_ids))
    incorrect = dict(Counter(guarded[i]['status'] for i in bad_ids))
    baseline, combined = summarize(baseline_cases), summarize(combined_cases)
    gate = (combined['fraction'] >= .9 and all(c['passed'] / c['total'] >= .75 for c in combined['families'].values())
            and improved > 0 and regressed == 0 and faithful.get('mismatch', 0) == 0)
    return {'baseline': baseline, 'combined': combined, 'improved_pairs': improved,
            'regressed_pairs': regressed, 'rule_applied_pairs': overrides,
            'guard_status_counts': {'faithful': faithful, 'incorrect': incorrect},
            'guard_seconds': seconds, 'guarded_candidates': guarded,
            'further_study_gate': gate, 'caption_accuracy_claim': False, 'production_eligible': False}


def run(phase):
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'frozen' or protocol['runner_sha256'] != prior.sha(__file__):
        raise ValueError('Require frozen runner/protocol.')
    for binding in protocol['bindings'] + protocol['model']['bindings']:
        if prior.sha(binding['path']) != binding['sha256']:
            raise ValueError(f'Binding changed: {binding["path"]}')
    fixture = prior.read_bound(protocol['fixture'])
    review = prior.read_bound(protocol['fixture_review'])
    if review['status'] != 'accepted' or review['fixture_sha256'] != protocol['fixture']['sha256']:
        raise ValueError('Require independent bilingual review and root acceptance of exact fixture bytes.')
    cases = fixture['cases']
    if len(cases) != 32 or sorted(Counter(c['family'] for c in cases).values()) != [8] * 4:
        raise ValueError('Expected 32 groups in four balanced strata.')
    if len({c['ja'] for c in cases}) != 32 or len({c['id'] for c in cases}) != 32:
        raise ValueError('Duplicate source/group.')
    for case in cases:
        candidates = case['good'] + [v['text'] for v in case['bad']]
        if len(case['good']) != 2 or len(case['bad']) != 2 or len(set(candidates)) != 4:
            raise ValueError('Expected four distinct candidates per source.')
    requests, groups = prior.requests_for(cases, 'xcomet')
    smoke = prior.read_bound(protocol['reused_technical_smoke'])
    smoke_protocol = prior.read_bound(protocol['reused_technical_protocol'])
    if (smoke['status'] != 'complete' or smoke.get('model') != 'xcomet' or smoke.get('phase') != 'smoke'
            or not smoke['cleanup']['owned_worker_released']):
        raise ValueError('Reusable same-helper technical smoke did not succeed/release.')
    if (smoke['protocol']['sha256'] != protocol['reused_technical_protocol']['sha256'] or
            smoke_protocol['models']['xcomet'] != protocol['model'] or
            smoke['inputs'] != protocol['warmup_requests']):
        raise ValueError('Reused smoke model, protocol or input binding differs.')
    smoke_requests = prior.read_bound(protocol['warmup_requests'])
    prior.checked_rows(smoke_requests, prior.read_bound(smoke['worker']), 'xcomet', False)
    warmup = smoke_requests[0]
    if warmup['request_id'] in {r['request_id'] for r in requests}:
        raise ValueError('Warmup overlaps fresh candidates.')
    all_requests = [warmup] + requests
    if phase == 'screen':
        preceding = json.loads((BASE / 'validate/report.json').read_text())
        if (preceding['status'] != 'complete' or preceding.get('phase') != 'validate' or
                preceding['protocol']['sha256'] != prior.sha(PROTOCOL) or
                not preceding['cleanup']['owned_worker_released']):
            raise ValueError('Require frozen token preflight before scoring.')
        if prior.read_bound(preceding['inputs']) != all_requests:
            raise ValueError('Token preflight did not cover these exact requests.')
        prior.checked_rows(all_requests, prior.read_bound(preceding['worker']), 'xcomet', True)
    output = BASE / phase
    output.mkdir(parents=True, exist_ok=False)
    input_path, worker_path = output / 'requests.json', output / 'worker.json'
    prior.write(input_path, all_requests)
    report = {'status': 'running', 'phase': phase, 'protocol': prior.binding(PROTOCOL),
              'inputs': prior.binding(input_path), 'expected_requests': len(all_requests),
              'group_ids': [g['id'] for g in groups], 'heldout_rows_used': 0, 'production_changed': False,
              'created_at_utc': datetime.now(timezone.utc).isoformat()}
    prior.write(output / 'report.json', report)
    env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', USE_TF='0',
               HF_HOME=str(prior.BASE / 'xcomet/hf-cache'), XDG_CACHE_HOME=str(prior.BASE / 'xcomet/cache'))
    model = protocol['model']
    command = [model['python'], model['helper'], '--input', str(input_path), '--output', str(worker_path)]
    if phase == 'validate':
        command.append('--validate-only')
    child = None
    with (prior.BASE / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prior.free_ports()
            with (output / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                child.wait(timeout=1800)
            if child.returncode != 0:
                raise RuntimeError('Worker failed; preserve log and partial output.')
            raw = json.loads(worker_path.read_text())
            rows = prior.checked_rows(all_requests, raw, 'xcomet', phase == 'validate')
            report.update(status='complete', worker=prior.binding(worker_path), model_load_seconds=raw['model_load_seconds'])
            if phase == 'screen':
                report['results'] = analyze(groups, requests, rows[1:])
                report['scoring_seconds'] = sum(r['seconds'] for r in rows[1:])
                report['warmup'] = rows[0]
        except BaseException as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            if worker_path.exists():
                report['partial_worker'] = prior.binding(worker_path)
        finally:
            if child is not None and child.poll() is None:
                prior.signal_group(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    prior.signal_group(child.pid, signal.SIGKILL)
                    child.wait()
            descendants = prior.group_members(child.pid) if child else []
            if descendants:
                prior.signal_group(child.pid, signal.SIGKILL)
                report.update(status='failed', cleanup_error='Owned descendants survived worker; killed.')
            remaining = prior.group_members(child.pid) if child else []
            report['cleanup'] = {'owned_worker_pid': child.pid if child else None,
                                 'worker_exit_code': child.returncode if child else None,
                                 'owned_worker_released': (child is None or child.poll() is not None) and not remaining,
                                 'descendants_detected_after_worker_exit': descendants, 'remaining_group_members': remaining}
            if not report['cleanup']['owned_worker_released']:
                report.update(status='failed', cleanup_error='Owned process group remains after cleanup.')
            report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            if report['status'] != 'complete' and 'results' in report:
                report['results']['further_study_gate'] = False
            prior.write(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json')}))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['validate', 'screen'])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert preferred(.1, .9, 'consistent', 'mismatch') == (True, True)
        assert preferred(.9, .1, 'mismatch', 'consistent') == (False, True)
        for g, b in [('abstain', 'mismatch'), ('consistent', 'abstain'), ('mismatch', 'mismatch')]:
            assert preferred(.1, .9, g, b) == (False, False)
            assert preferred(.5, .5, g, b) == (False, False)
        print('Pair override, fallback, abstention and tie checks passed; no model load.')
    elif args.phase:
        raise SystemExit(run(args.phase))
    else:
        parser.error('Require --phase or --self-test')
