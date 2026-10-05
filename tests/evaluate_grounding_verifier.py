#!/usr/bin/env python3
"""Paired directional grounding screen; synthetic preference only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import evaluate_caption_verifiers as prior

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/media/psikosen/newvolume/jread-verifier/grounding-v1')
PROTOCOL = ROOT / 'tests/fixtures/accuracy-grounding-verifier.json'


def requests_for(cases):
    requests, groups = [], []
    for case in cases:
        candidates = case['good'] + [v['text'] for v in case['bad']]
        group = {'id': case['id'], 'family': case['family'], 'candidates': [], 'pairs': []}
        ids = []
        for text in candidates:
            pair = []
            for direction, premise, hypothesis in (
                    ('ja_en', case['ja'], text), ('en_ja', text, case['ja'])):
                rid = prior.request_id(case['id'], direction, premise, hypothesis)
                pair.append(rid)
                requests.append({'request_id': rid, 'premise': premise, 'hypothesis': hypothesis})
            ids.append(pair)
            group['candidates'].append({'forward': pair[0], 'reverse': pair[1]})
        for good in range(2):
            for bad in range(2, 4):
                group['pairs'].append({'good_index': good, 'bad_index': bad})
        groups.append(group)
    requests.sort(key=lambda r: r['request_id'])
    if len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('Duplicate public request ID.')
    return requests, groups


def validate_cases(data, expected, strata=None):
    cases = data.get('cases', [])
    if len(cases) != expected or len({c['id'] for c in cases}) != expected:
        raise ValueError(f'Expected {expected} unique groups.')
    if strata and (set(Counter(c['family'] for c in cases).values()) != {8} or
                   set(Counter(c['family'] for c in cases)) != set(strata)):
        raise ValueError('Main fixture must have eight groups in each of four frozen strata.')
    for c in cases:
        texts = c['good'] + [v['text'] for v in c['bad']]
        if (len(c['good']) != 2 or len(c['bad']) != 2 or len(set(texts)) != 4 or
                not c['ja'].strip() or not all(isinstance(x, str) and x.strip() for x in texts) or
                not c.get('propositions')):
            raise ValueError('Require four distinct reviewed candidates and proposition evidence per group.')
    return cases


def checked_rows(requests, raw, validate_only):
    if raw.get('cuda_initialized') is not False:
        raise ValueError('Worker must explicitly report no CUDA initialization.')
    if raw.get('error') is not None or raw.get('unprocessed_ids'):
        raise ValueError('Worker reports an error or unprocessed requests.')
    if validate_only:
        if raw.get('model_loaded') is not False or raw.get('inference_performed') is not False or raw.get('model_load_seconds') != 0:
            raise ValueError('Validate must remain model-free with zero load time.')
        if any(raw.get(k) is not None for k in ('model_device', 'model_parameter_dtype', 'inference_dtype', 'loading_info')):
            raise ValueError('Model-free validation must not claim actual model or inference precision.')
    else:
        if (raw.get('model_loaded') is not True or raw.get('inference_performed') is not True or
                raw.get('model_device') != 'cpu' or raw.get('model_parameter_dtype') != 'torch.float32' or
                raw.get('inference_dtype') != 'torch.float32' or raw.get('torch_threads') != 4):
            raise ValueError('Require completed CPU FP32 four-thread model inference.')
        if raw.get('loading_info') != {k: [] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}:
            raise ValueError('Require native checkpoint load with no weight differences.')
        if not isinstance(raw.get('model_load_seconds'), (int, float)) or not math.isfinite(raw['model_load_seconds']) or raw['model_load_seconds'] < 0:
            raise ValueError('Require finite nonnegative model load time.')
    rows = raw.get('rows', [])
    if raw.get('status') != 'complete' or len(rows) != len(requests) or [r.get('request_id') for r in rows] != [r['request_id'] for r in requests]:
        raise ValueError('Incomplete, duplicate, reordered or unknown worker rows.')
    for row in rows:
        if (row.get('status') != 'ok' or not isinstance(row.get('input_tokens'), int) or row['input_tokens'] <= 0 or
                row.get('token_limit') != 2048 or row['input_tokens'] > 2048 or row.get('truncated') is not False):
            raise ValueError('Worker error, missing token evidence, or truncated/over-limit request.')
        if validate_only:
            continue
        vals = [row.get('supported_probability'), *row.get('logits', [])]
        if (len(vals) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals) or
                not 0 <= vals[0] <= 1 or not isinstance(row.get('seconds'), (int, float)) or
                not math.isfinite(row['seconds']) or row['seconds'] < 0):
            raise ValueError('Missing/nonfinite score, logits, or inference time.')
    return rows


def summarize(groups, rows, direction='both'):
    by_id = {r['request_id']: r for r in rows}
    scored, pairs = {}, []
    for g in groups:
        for i, ids in enumerate(g['candidates']):
            vals = [by_id[ids['forward']]['supported_probability']]
            if direction == 'both':
                vals.append(by_id[ids['reverse']]['supported_probability'])
            scored[(g['id'], i)] = min(vals)
        gpairs = []
        for p in g['pairs']:
            good, bad = scored[(g['id'], p['good_index'])], scored[(g['id'], p['bad_index'])]
            item = p | {'good_score': good, 'bad_score': bad, 'pass_': good > bad}
            gpairs.append(item)
            pairs.append((g, item))
        g['pair_results'] = gpairs
    total, passed = len(pairs), sum(p['pass_'] for _, p in pairs)
    strata = {}
    for g, p in pairs:
        s = strata.setdefault(g['family'], {'passed': 0, 'total': 0})
        s['passed'] += p['pass_']; s['total'] += 1
    variants = {str(i): {'passed': sum(p['pass_'] for g, p in pairs if p['good_index'] == i),
                         'total': len(groups) * 2} for i in range(2)}
    return {'passed_pairs': passed, 'total_pairs': total, 'fraction': passed / total,
            'all_pairs_pass_groups': sum(all(p['pass_'] for p in g['pair_results']) for g in groups),
            'total_groups': len(groups), 'strata': strata, 'good_variants': variants,
            'pair_details': [{'id': g['id'], 'family': g['family'], **p} for g, p in pairs],
            'selected_candidate_scores': {f'{gid}:{i}': score for (gid, i), score in scored.items()}}


def compare_to_baseline(current, baseline, groups):
    saved = {c['id']: c for c in baseline['cases']}
    if set(saved) != {g['id'] for g in groups}:
        raise ValueError('Baseline and grounding fixture group IDs differ.')
    old = {(c['id'], p['good_index'], p['bad_index']): p['pass_'] for c in baseline['cases'] for p in c['pairs']}
    now = {(p['id'], p['good_index'], p['bad_index']): p['pass_'] for p in current['pair_details']}
    if set(old) != set(now):
        raise ValueError('Baseline pair denominator differs.')
    return {'baseline_passed_pairs': sum(old.values()), 'grounding_passed_pairs': sum(now.values()),
            'improved_pairs': sum(not old[k] and now[k] for k in old),
            'regressed_pairs': sum(old[k] and not now[k] for k in old), 'total_pairs': len(old)}


def run(phase):
    p = json.loads(PROTOCOL.read_text())
    if p['status'] != 'frozen' or p['runner_sha256'] != prior.sha(__file__):
        raise ValueError('Require frozen protocol and unchanged runner hash.')
    for b in p['bindings'] + p['model']['bindings']:
        if prior.sha(b['path']) != b['sha256']:
            raise ValueError(f'Frozen helper/runtime binding changed: {b["path"]}')
    main = validate_cases(prior.read_bound(p['fixture']), 32,
                          ('date_time', 'money_measurement', 'equivalent_or_unsupported', 'other_meaning'))
    controls = validate_cases(prior.read_bound(p['controls']), 6)
    for key in ('fixture_review', 'controls_review'):
        review = prior.read_bound(p[key])
        bound = p['fixture'] if key == 'fixture_review' else p['controls']
        if review.get('status') != 'accepted' or review.get('fixture_sha256') != bound['sha256']:
            raise ValueError(f'Require accepted exact bytes for {key}.')
    baseline = prior.read_bound(p['baseline_report'])
    if baseline.get('status') != 'complete' or not baseline.get('cleanup', {}).get('owned_worker_released'):
        raise ValueError('Require a complete released quantity baseline report.')
    baseline_ref = baseline['protocol']
    if prior.sha(baseline_ref['path']) != baseline_ref['sha256']:
        raise ValueError('Saved quantity baseline protocol binding changed.')
    baseline_proto = json.loads(Path(baseline_ref['path']).read_text())
    if baseline_proto['fixture']['sha256'] != p['fixture']['sha256']:
        raise ValueError('Saved quantity baseline was not scored on these exact main fixture bytes.')
    req_main, groups_main = requests_for(main)
    req_ctrl, groups_ctrl = requests_for(controls)
    if {g['id'] for g in groups_main} & {g['id'] for g in groups_ctrl}:
        raise ValueError('Main and control group IDs overlap.')
    all_requests = sorted(req_main + req_ctrl, key=lambda r: r['request_id'])
    if len(all_requests) != 304 or len({r['request_id'] for r in all_requests}) != 304:
        raise ValueError('Expected 304 complete directional requests.')
    requests = {'validate': all_requests, 'smoke': req_ctrl, 'screen': req_main}[phase]
    groups = {'validate': groups_main + groups_ctrl, 'smoke': groups_ctrl, 'screen': groups_main}[phase]
    if phase in ('smoke', 'screen'):
        previous = prior.read_bound(prior.binding(BASE / 'validate/report.json'))
        if (previous['status'] != 'complete' or previous['phase'] != 'validate' or
                previous['protocol']['sha256'] != prior.sha(PROTOCOL) or
                not previous['cleanup']['owned_worker_released']):
            raise ValueError('Require successful same-protocol complete model-free token preflight.')
        validated = prior.read_bound(previous['inputs'])
        validate_worker = prior.read_bound(previous['worker'])
        checked_rows(all_requests, validate_worker, True)
        if validated != all_requests:
            raise ValueError('Token preflight did not cover the exact complete 304-request set.')
        by_id = {r['request_id']: r for r in validated}
        if any(by_id.get(r['request_id']) != r for r in requests):
            raise ValueError('Token preflight did not cover exact public request objects.')
    warmup = None
    if phase == 'screen':
        smoke = json.loads((BASE / 'smoke/report.json').read_text())
        if smoke['status'] != 'complete' or not smoke['cleanup']['owned_worker_released'] or smoke['protocol']['sha256'] != prior.sha(PROTOCOL):
            raise ValueError('Require successful same-protocol smoke and released worker.')
        smoke_inputs = prior.read_bound(smoke['inputs'])
        smoke_worker = prior.read_bound(smoke['worker'])
        checked_rows(req_ctrl, smoke_worker, False)
        if smoke_inputs != req_ctrl:
            raise ValueError('Smoke worker did not cover the exact 48 control requests.')
        warmup = smoke_inputs[0]
        if by_id.get(warmup['request_id']) != warmup:
            raise ValueError('Warmup is not covered by exact token preflight.')
        requests = [warmup] + requests
    out = BASE / phase
    out.mkdir(parents=True, exist_ok=False)
    ip, wp = out / 'requests.json', out / 'worker.json'
    prior.write(ip, requests)
    report = {'status': 'running', 'phase': phase, 'protocol': prior.binding(PROTOCOL),
              'inputs': prior.binding(ip), 'expected_requests': len(requests),
              'group_ids': [g['id'] for g in groups], 'heldout_rows_used': 0, 'production_changed': False,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'accuracy_scope': 'Synthetic grounding preference diagnostic; no caption-accuracy claim.'}
    prior.write(out / 'report.json', report)
    env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', USE_TF='0',
               HF_HOME=str(BASE / 'hf-cache'), XDG_CACHE_HOME=str(BASE / 'cache'))
    model, child, lock = p['model'], None, None
    try:
        lock = (BASE.parent / 'evaluation.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior.free_ports()
        with (out / 'worker.log').open('x') as log:
            child = subprocess.Popen([model['python'], model['helper'], '--input', str(ip), '--output', str(wp)] +
                                     (['--validate-only'] if phase == 'validate' else []), env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child.wait(timeout=1800)
        if child.returncode:
            raise RuntimeError('Worker failed; raw worker output and log preserved.')
        raw = json.loads(wp.read_text())
        rows = checked_rows(requests, raw, phase == 'validate')
        report.update(status='complete', worker=prior.binding(wp), model_load_seconds=raw['model_load_seconds'])
        if phase != 'validate':
            measured = rows[1:] if warmup else rows
            report['results'] = {'primary': summarize(groups, measured, 'both'),
                                 'forward_only_diagnostic': summarize(groups, measured, 'forward')}
            if phase == 'screen':
                for name, value in (('xcomet_alone', baseline['results']['baseline']),
                                    ('quantity_combination', baseline['results']['combined'])):
                    report['results'][name] = compare_to_baseline(report['results']['primary'], value, groups_main)
                    report['results'][name + '_forward_diagnostic'] = compare_to_baseline(report['results']['forward_only_diagnostic'], value, groups_main)
                primary = report['results']['primary']
                report['further_study_gate'] = (primary['fraction'] >= .90 and
                    all(s['passed'] / s['total'] >= .75 for s in primary['strata'].values()) and
                    primary['good_variants']['1']['passed'] / primary['good_variants']['1']['total'] >= .90)
                report['further_study_gate_only_for_fresh_independent_followup'] = True
            report['scoring_seconds'] = sum(r['seconds'] for r in measured)
            if warmup: report['warmup'] = rows[0]
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if wp.exists(): report['partial_worker'] = prior.binding(wp)
    finally:
        if child is not None and child.poll() is None:
            prior.signal_group(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                prior.signal_group(child.pid, signal.SIGKILL); child.wait()
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
        if phase == 'screen' and report['status'] != 'complete': report['further_study_gate'] = False
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        try: prior.write(out / 'report.json', report)
        finally:
            if lock is not None: lock.close()
    print(json.dumps({'status': report['status'], 'report': str(out / 'report.json')}))
    return 0 if report['status'] == 'complete' else 1


def self_test():
    case = {'id': 'mock', 'family': 'mock', 'ja': '例', 'good': ['正', '良'],
            'bad': [{'text': '誤'}, {'text': '違'}], 'propositions': ['mock']}
    requests, groups = requests_for([case])
    assert len(requests) == 8 and all(set(r) == {'request_id', 'premise', 'hypothesis'} for r in requests)
    rows = [{'request_id': r['request_id'], 'status': 'ok', 'input_tokens': 4, 'token_limit': 2048,
             'truncated': False, 'supported_probability': .9 if (r['premise'] if r['premise'] != '例' else r['hypothesis']) in ('正', '良') else .1,
             'logits': [1., 0.], 'seconds': .01} for r in requests]
    assert summarize(groups, rows)['passed_pairs'] == 4
    ids = groups[0]['candidates'][0]
    next(r for r in rows if r['request_id'] == ids['reverse'])['supported_probability'] = .1
    assert summarize(groups, rows)['passed_pairs'] == 2  # reverse direction is required
    next(r for r in rows if r['request_id'] == ids['reverse'])['supported_probability'] = .9
    bad_id = groups[0]['candidates'][2]['forward']
    bad_value = next(r['supported_probability'] for r in rows if r['request_id'] == bad_id)
    for rid in ids.values(): next(r for r in rows if r['request_id'] == rid)['supported_probability'] = bad_value
    assert summarize(groups, rows)['passed_pairs'] == 2  # ties fail
    # Strict floating-point comparison preserves margins below the display precision.
    close = rows.copy()
    for r in close: r['supported_probability'] = .5
    for i in (0, 1):
        for rid in groups[0]['candidates'][i].values():
            next(r for r in close if r['request_id'] == rid)['supported_probability'] = .5000000000000001
    assert summarize(groups, close)['passed_pairs'] == 4
    raw = {'status': 'complete', 'rows': rows, 'model_loaded': True, 'inference_performed': True,
           'model_load_seconds': 1., 'cuda_initialized': False, 'model_device': 'cpu',
           'model_parameter_dtype': 'torch.float32', 'inference_dtype': 'torch.float32', 'torch_threads': 4,
           'loading_info': {k: [] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}}
    assert len(checked_rows(requests, raw, False)) == 8
    for bad in (rows[:-1], rows + [rows[0]]):
        try: checked_rows(requests, raw | {'rows': bad}, False)
        except ValueError: pass
        else: raise AssertionError('Incomplete/duplicate denominator accepted.')
    preflight = raw | {'model_loaded': False, 'inference_performed': False, 'model_load_seconds': 0,
                      'model_device': None, 'model_parameter_dtype': None, 'inference_dtype': None, 'loading_info': None}
    assert len(checked_rows(requests, preflight, True)) == 8
    assert summarize(groups, rows)['total_pairs'] == 4
    try: checked_rows(requests, raw | {'rows': list(reversed(rows))}, False)
    except ValueError: pass
    else: raise AssertionError('Reordered worker rows accepted.')
    for change in ({'model_parameter_dtype': 'torch.float16'}, {'inference_dtype': 'torch.bfloat16'},
                   {'model_device': 'cuda:0'}, {'torch_threads': 8}, {'loading_info': None}):
        try: checked_rows(requests, raw | change, False)
        except ValueError: pass
        else: raise AssertionError('Invalid runtime or loading evidence accepted.')
    print('Pure self-test passed: public schema, directional minimum, ties and exact row coverage; no model load.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['validate', 'smoke', 'screen'])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: self_test()
    elif args.phase: raise SystemExit(run(args.phase))
    else: parser.error('Require --phase or --self-test.')
