#!/usr/bin/env python3
"""Frozen synthetic bilingual checker screen; no translation generation or training."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone

from evaluate_reazon import free_ports, sha, write

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/media/psikosen/newvolume/jread-verifier')
PROTOCOL = ROOT / 'tests/fixtures/accuracy-caption-verifiers.json'
QUESTION = 'Which English translation preserves all of the Japanese meaning without adding or changing facts?'


def binding(path):
    return dict(path=str(path), sha256=sha(path))


def read_bound(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError(f'Frozen binding changed: {path}')
    return json.loads(path.read_text())


def validate_fixture(data):
    cases, smoke = data['cases'], data['smoke_cases']
    if len(cases) != 28 or len(smoke) != 6 or sorted(Counter(r['family'] for r in cases).values()) != [4] * 7:
        raise ValueError('Require fixed 28 groups, seven balanced families, and six separate smoke groups.')
    if len({r['id'] for r in cases + smoke}) != 34 or len({r['ja'] for r in cases + smoke}) != 34:
        raise ValueError('Duplicate ID/source across groups or phases.')
    for row in cases + smoke:
        texts = row['good'] + [v['text'] for v in row['bad']]
        if len(row['good']) != 2 or len(row['bad']) != 2 or len(set(texts)) != 4:
            raise ValueError('Require two distinct good and two distinct bad candidates.')
        if not all(isinstance(v, str) and v.strip() for v in texts + [row['ja']]):
            raise ValueError('Empty source/candidate.')
        if not row['propositions'] or not all(v['error'] for v in row['bad']):
            raise ValueError('Missing semantic review evidence.')


def request_id(*parts):
    return 'r_' + hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:20]


def requests_for(cases, model):
    requests, groups = [], []
    for case in cases:
        candidates = case['good'] + [v['text'] for v in case['bad']]
        group = dict(id=case['id'], family=case['family'], pairs=[])
        if model == 'xcomet':
            ids = [request_id(case['id'], text) for text in candidates]
            requests.extend(sorted((dict(request_id=rid, source=case['ja'], text=text)
                                    for rid, text in zip(ids, candidates)), key=lambda r: r['request_id']))
        for good in range(2):
            for bad in range(2, 4):
                pair = dict(good_index=good, bad_index=bad)
                if model == 'xcomet':
                    pair.update(good_request=ids[good], bad_request=ids[bad])
                else:
                    pair['orders'] = []
                    for indices in ((good, bad), (bad, good)):
                        options = [candidates[i] for i in indices]
                        rid = request_id(case['id'], options)
                        requests.append(dict(request_id=rid, source=case['ja'], options=options))
                        pair['orders'].append(dict(request_id=rid, expected=indices.index(good)))
                group['pairs'].append(pair)
        groups.append(group)
    assert len({r['request_id'] for r in requests}) == len(requests)
    return requests, groups


def score(groups, rows, model):
    by_id = {r['request_id']: r for r in rows}
    results, family = [], {}
    for group in groups:
        pairs = []
        for pair in group['pairs']:
            if model == 'xcomet':
                good, bad = (by_id.get(pair[key], {}) for key in ('good_request', 'bad_request'))
                valid = all(r.get('status') == 'ok' and isinstance(r.get('score'), (int, float))
                            and math.isfinite(r['score']) for r in (good, bad))
                passed = valid and good['score'] > bad['score']
                evidence = dict(good_score=good.get('score'), bad_score=bad.get('score'))
            else:
                ordered = [(by_id.get(o['request_id'], {}), o['expected']) for o in pair['orders']]
                choices = [r.get('choice') for r, _ in ordered]
                passed = all(r.get('status') == 'ok' and r.get('choice') == expected and
                             r['scores'][expected] > r['scores'][1 - expected] for r, expected in ordered)
                evidence = dict(choices=choices, expected=[e for _, e in ordered],
                                correct_orders=sum(r.get('status') == 'ok' and r.get('choice') == e
                                                   and r['scores'][e] > r['scores'][1 - e] for r, e in ordered),
                                order_stable=all(r.get('status') == 'ok' for r, _ in ordered)
                                and choices[0] in (0, 1) and choices[1] == 1 - choices[0])
            pairs.append(pair | dict(pass_=bool(passed), evidence=evidence))
        correct = sum(p['pass_'] for p in pairs)
        family.setdefault(group['family'], dict(passed=0, total=0))
        family[group['family']]['passed'] += correct
        family[group['family']]['total'] += len(pairs)
        results.append(dict(id=group['id'], family=group['family'], passed_pairs=correct,
                            all_pairs_pass=correct == 4, pairs=pairs))
    total = sum(len(r['pairs']) for r in results)
    correct = sum(r['passed_pairs'] for r in results)
    summary = dict(passed_pairs=correct, total_pairs=total, pair_fraction=correct / total,
                all_pairs_pass_groups=sum(r['all_pairs_pass'] for r in results), total_groups=len(results),
                families=family, cases=results,
                further_study_gate=correct / total >= .9 and all(f['passed'] / f['total'] >= .75 for f in family.values()),
                meaning_accuracy_claim=False, production_eligible=False)
    summary['faithful_variants'] = {str(i): dict(
        passed=sum(p['pass_'] for r in results for p in r['pairs'] if p['good_index'] == i),
        total=sum(p['good_index'] == i for r in results for p in r['pairs'])) for i in range(2)}
    if model == 'laya':
        summary['option_orders'] = dict(
            consistent_pairs=sum(p['evidence']['order_stable'] for r in results for p in r['pairs']),
            total_pairs=total,
            correct_directional_choices=sum(p['evidence']['correct_orders'] for r in results for p in r['pairs']),
            total_directional_choices=2 * total)
    return summary


def checked_rows(requests, raw, model, tokenize_only):
    if raw.get('cuda_initialized') is not False:
        raise ValueError('CPU-only worker must explicitly report no CUDA initialization.')
    if tokenize_only and (raw.get('model_loaded') is not False or raw.get('inference_performed') is not False
                          or raw.get('model_load_seconds') != 0):
        raise ValueError('Tokenizer preflight must explicitly remain model-free.')
    if not tokenize_only and (raw.get('model_loaded') is not True or raw.get('inference_performed') is not True):
        raise ValueError('Missing actual model execution evidence.')
    if not tokenize_only and (raw.get('model_device') != 'cpu' or
                              raw.get('model_parameter_dtype') != 'torch.float32' or
                              raw.get('inference_dtype') != 'torch.float32' or
                              raw.get('torch_threads') != 4 or
                              (model == 'laya' and raw.get('amp_enabled') is not False)):
        raise ValueError('Worker does not match frozen CPU FP32 four-thread execution.')
    rows = raw.get('rows', [])
    if len(rows) != len(requests) or [r['request_id'] for r in rows] != [r['request_id'] for r in requests]:
        raise ValueError('Missing, duplicate, or reordered worker rows.')
    for row in rows:
        if row.get('status') != 'ok' or not isinstance(row.get('input_tokens'), int) or row['input_tokens'] <= 0:
            raise ValueError('Worker error or missing no-truncation token count.')
        if row.get('truncated') is not False or not isinstance(row.get('token_limit'), int) or row['input_tokens'] > row['token_limit']:
            raise ValueError('Missing no-truncation evidence or over-limit input.')
        if model == 'laya' and (row.get('state_tokens_dropped') != 0 or
                               len(row.get('option_tokens_per_candidate', [])) != 2 or
                               any(n > 48 for n in row['option_tokens_per_candidate'])):
            raise ValueError('Laya source/option truncation or missing option audit.')
        if tokenize_only:
            continue
        if not isinstance(row.get('seconds'), (int, float)) or not math.isfinite(row['seconds']) or row['seconds'] < 0:
            raise ValueError('Invalid inference time.')
        values = row.get('scores', []) if model == 'laya' else [row.get('score')]
        if len(values) != (2 if model == 'laya' else 1) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError('Missing or nonfinite scores.')
        if model == 'laya' and (row.get('choice') not in (0, 1) or
                               not all(0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-4):
            raise ValueError('Invalid choice/probabilities.')
    if raw.get('status') != 'complete':
        raise ValueError('Worker incomplete.')
    return rows


def group_members(pgid):
    """Only our new process group; exclude already-dead zombies."""
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == pgid:
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def run(args):
    protocol = json.loads(args.protocol.read_text())
    if protocol['status'] != 'frozen' or protocol['runner_sha256'] != sha(__file__) or protocol['question'] != QUESTION:
        raise ValueError('Require frozen protocol, runner and question.')
    fixture = read_bound(protocol['fixture']); validate_fixture(fixture)
    review = read_bound(protocol['fixture_review'])
    if review['status'] != 'accepted' or review['fixture_sha256'] != protocol['fixture']['sha256']:
        raise ValueError('Require root bilingual acceptance of these exact fixture bytes.')
    config = protocol['models'][args.model]
    for item in protocol['bindings'] + config['bindings']:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'Pinned code/runtime/model manifest changed: {item["path"]}')
    if args.phase != 'validate':
        previous_phase = 'smoke' if args.phase == 'screen' else 'validate'
        previous_path = BASE / 'runs' / f'{args.model}-{previous_phase}-v1/report.json'
        previous = json.loads(previous_path.read_text())
        if previous['status'] != 'complete' or not previous['cleanup']['owned_worker_released'] or previous['protocol']['sha256'] != sha(args.protocol):
            raise ValueError('Require successful preceding technical stage and released worker from this protocol.')
    cases = fixture['smoke_cases'] if args.phase == 'smoke' else fixture['cases']
    if args.phase == 'validate':
        cases = fixture['smoke_cases'] + fixture['cases']
    requests, groups = requests_for(cases, args.model)
    # A separate previously screened control warms the full screen, never a new diagnostic case.
    if args.phase == 'screen':
        warmup = requests_for(fixture['smoke_cases'][:1], args.model)[0][0]
        requests = [warmup] + requests
    output = BASE / 'runs' / f'{args.model}-{args.phase}-v1'
    output.mkdir(parents=True, exist_ok=False)
    inputs, worker_path, report_path = output / 'requests.json', output / 'worker.json', output / 'report.json'
    write(inputs, requests)
    report = dict(status='running', model=args.model, phase=args.phase, protocol=binding(args.protocol),
                  inputs=binding(inputs), expected_requests=len(requests), heldout_rows_used=0,
                  group_ids=[g['id'] for g in groups], expected_pairs=4 * len(groups),
                  created_at_utc=datetime.now(timezone.utc).isoformat(), production_changed=False,
                  accuracy_scope='Synthetic preference diagnostic only; no end-to-end caption accuracy.')
    write(report_path, report)
    child = None
    env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD')}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', USE_TF='0',
               LAYA_CPU_AMP='off',
               HF_HOME=str(BASE / args.model / 'hf-cache'), XDG_CACHE_HOME=str(BASE / args.model / 'cache'))
    command = [config['python'], config['helper'], '--input', str(inputs), '--output', str(worker_path)]
    if args.phase == 'validate':
        command.append('--validate-only')
    with (BASE / 'evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free_ports()
            with (output / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                child.wait(timeout=1800)
            if child.returncode != 0:
                raise RuntimeError('Worker failed; preserve and inspect raw output/log.')
            raw = json.loads(worker_path.read_text())
            rows = checked_rows(requests, raw, args.model, args.phase == 'validate')
            report.update(status='complete', worker=binding(worker_path), model_load_seconds=raw.get('model_load_seconds'))
            if args.phase != 'validate':
                report['scores'] = score(groups, rows, args.model)
                report['further_study_gate'] = args.phase == 'screen' and report['scores'].pop('further_study_gate')
                report['scores'].pop('further_study_gate', None)
                measured = rows[1:] if args.phase == 'screen' else rows
                report['scoring_seconds'] = sum(r['seconds'] for r in measured)
                report['warmup'] = rows[0] if args.phase == 'screen' else None
        except BaseException as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            if worker_path.exists():
                report['partial_worker'] = binding(worker_path)
        finally:
            if child is not None and child.poll() is None:
                signal_group(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_group(child.pid, signal.SIGKILL); child.wait()
            descendants = group_members(child.pid) if child else []
            if descendants:
                signal_group(child.pid, signal.SIGKILL)
                report.update(status='failed', cleanup_error='Owned descendants survived the worker; sent SIGKILL.')
            remaining = group_members(child.pid) if child else []
            report['cleanup'] = dict(owned_worker_pid=child.pid if child else None,
                                    worker_exit_code=child.returncode if child else None,
                                    owned_worker_released=(child is None or child.poll() is not None) and not remaining,
                                    descendants_detected_after_worker_exit=descendants,
                                    remaining_group_members=remaining)
            report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            write(report_path, report)
    print(json.dumps(dict(status=report['status'], report=str(report_path))))
    return 0 if report['status'] == 'complete' else 1


def self_test():
    case = dict(id='demo', family='roles', ja='テスト。', good=['Good one', 'Good paraphrase'],
                bad=[dict(text='Bad one'), dict(text='Bad two')])
    for model in ('laya', 'xcomet'):
        requests, groups = requests_for([case], model)
        assert len(requests) == (8 if model == 'laya' else 4)
        assert all(set(r) == {'request_id', 'source', 'options' if model == 'laya' else 'text'} for r in requests)
        rows = []
        for r in requests:
            row = dict(request_id=r['request_id'], status='ok', input_tokens=10, token_limit=1024,
                       truncated=False, state_tokens_dropped=0, option_tokens_per_candidate=[3, 3], seconds=.1)
            if model == 'laya':
                index = next(i for i, text in enumerate(r['options']) if text.startswith('Good'))
                row.update(choice=index, scores=[.9 if i == index else .1 for i in range(2)])
            else:
                row['score'] = .9 if r['text'].startswith('Good') else .1
            rows.append(row)
        raw = dict(status='complete', rows=rows, model_loaded=True, inference_performed=True, cuda_initialized=False,
                   model_device='cpu', model_parameter_dtype='torch.float32', inference_dtype='torch.float32',
                   torch_threads=4, amp_enabled=False)
        checked_rows(requests, raw, model, False)
        assert score(groups, rows, model)['passed_pairs'] == 4
        assert score(groups, [], model)['passed_pairs'] == 0
        for row in rows:
            row.update(dict(choice=0, scores=[.5, .5]) if model == 'laya' else dict(score=.5))
        assert score(groups, rows, model)['passed_pairs'] == 0
        try:
            checked_rows(requests, raw | dict(rows=rows[:-1]), model, False)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete denominator accepted.')
        preflight = raw | dict(model_loaded=False, inference_performed=False, model_load_seconds=0)
        checked_rows(requests, preflight, model, True)
        for invalid in (preflight | dict(model_loaded=True), preflight | dict(cuda_initialized=True)):
            try:
                checked_rows(requests, invalid, model, True)
            except ValueError:
                pass
            else:
                raise AssertionError('Model/CUDA use in tokenizer preflight accepted.')
        rows[0]['truncated'] = True
        try:
            checked_rows(requests, preflight, model, True)
        except ValueError:
            pass
        else:
            raise AssertionError('Truncated input accepted.')
    print('Pure CPU self-test passed: pair order, no label fields, missing rows and ties; no subprocess/model load.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--model', choices=['laya', 'xcomet'])
    parser.add_argument('--phase', choices=['validate', 'smoke', 'screen'])
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.model and args.phase:
        sys.exit(run(args))
    else:
        parser.error('Require --self-test or --model and --phase.')
