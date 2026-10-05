#!/usr/bin/env python3
"""Validate a finished adapter evaluation and freeze a condition-blind review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import review_translation_adapter as review

CONDITIONS = {(0, False), (0, True), (.6, False), (.6, True)}
EVALUATION_FIELDS = ('max_new_tokens', 'primary_temperature', 'secondary_temperature',
                     'secondary_top_p', 'secondary_top_k', 'secondary_seed')
RUBRIC_FIELDS = ('required_propositions', 'acceptable_variants_and_ambiguities', 'major_errors_to_reject')


def snapshot(bindings, report_path):
    files = {}
    for name, (path, expected) in bindings.items():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'Frozen {name} bytes changed.')
        files[name] = raw
    raw_report = report_path.read_bytes()
    return files, json.loads(raw_report), hashlib.sha256(raw_report).hexdigest()


def validate(files, report, bindings):
    protocol, manifest, rubric = (json.loads(files[k]) for k in ('protocol', 'manifest', 'rubric'))
    if (protocol.get('status') != 'frozen' or manifest.get('status') != 'frozen'
            or rubric.get('status') != 'frozen_before_generation'):
        raise ValueError('Protocol, data and pre-generation rubric must be frozen.')
    if (report.get('status') != 'complete' or report.get('mode') != 'evaluate'
            or report.get('owned_worker_released') is not True or report.get('worker_exit_code') != 0):
        raise ValueError('Evaluation must complete and release its owned worker.')
    if (report.get('runner_sha256') != bindings['runner'][1]
            or protocol.get('runner_sha256') != bindings['runner'][1]
            or report.get('protocol', {}).get('sha256') != bindings['protocol'][1]
            or report.get('protocol', {}).get('contents') != protocol
            or report.get('data', {}).get('manifest_sha256') != bindings['manifest'][1]
            or report.get('data', {}).get('manifest') != manifest
            or protocol.get('data', {}).get('manifest_sha256') != bindings['manifest'][1]
            or protocol.get('data', {}).get('data_sha256') != bindings['data'][1]
            or protocol.get('data', {}).get('validation_rubric_sha256') != bindings['rubric'][1]
            or manifest.get('data_sha256') != bindings['data'][1]
            or Path(manifest.get('data_path', '')).resolve() != bindings['data'][0]
            or Path(report.get('data', {}).get('path', '')).resolve() != bindings['data'][0]):
        raise ValueError('Report, protocol and frozen input bindings disagree.')
    settings = protocol.get('evaluation', {})
    if (any(k not in settings for k in EVALUATION_FIELDS)
            or report.get('evaluation') != {k: settings[k] for k in EVALUATION_FIELDS}):
        raise ValueError('Reported decoding settings differ from the frozen protocol.')
    if report.get('evaluation_warmup', {}).get('excluded') is not True:
        raise ValueError('Evaluation must explicitly exclude its separate warmup.')
    data = [json.loads(line) for line in files['data'].splitlines() if line.strip()]
    if len({r['id'] for r in data}) != len(data):
        raise ValueError('Frozen data contains duplicate row IDs.')
    rows = {r['id']: r for r in data if r['partition'] == 'validation'}
    cases = report.get('cases', [])
    if len(rows) != 40 or len(cases) != 40 or {c['id'] for c in cases} != set(rows):
        raise ValueError('Evaluation must cover exactly all40 frozen validation rows once.')
    groups = {}
    errors = 0
    for case in cases:
        row = rows[case['id']]
        if any(case.get(k) != v for k, v in row.items()) or row.get('review_status') != 'accepted':
            raise ValueError('Saved case source, variant or metadata differs from its frozen row.')
        group = groups.setdefault(row['group_id'], {})
        if row['variant'] not in ('clean', 'whisper') or row['variant'] in group:
            raise ValueError('Every group requires one clean and one Whisper validation row.')
        group[row['variant']] = row
        outputs = case.get('outputs', [])
        if len(outputs) != 4 or {(o['temperature'], o['adapter_enabled']) for o in outputs} != CONDITIONS:
            raise ValueError('Every validation row must retain all four conditions.')
        for output in outputs:
            seconds, temperature = output.get('seconds'), output.get('temperature')
            if (type(output.get('adapter_enabled')) is not bool or type(temperature) not in (int, float)
                    or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                    or type(output.get('seed')) is not int or output['seed'] != settings['secondary_seed']
                    or output.get('status') not in ('ok', 'error')):
                raise ValueError('Invalid condition, seed, timing or unfinished output.')
            # The frozen runner labels non-EOS/capped responses as errors. Preserve those
            # failures; do not silently repair a report that inconsistently calls them ok.
            if output['status'] == 'ok' and (not isinstance(output.get('text'), str)
                    or not output['text'].strip() or output.get('ended_with_eos') is not True
                    or output.get('reached_token_cap') is not False):
                raise ValueError('Successful output must have nonempty, completed English text.')
            errors += output['status'] == 'error'
    if len(groups) != 20 or any(set(g) != {'clean', 'whisper'} for g in groups.values()):
        raise ValueError('Exactly20 paired source groups are required.')
    rubric_cases = rubric.get('cases', [])
    if len(rubric_cases) != 20 or {r['group_id'] for r in rubric_cases} != set(groups):
        raise ValueError('Rubric must cover exactly the20 validation source groups.')
    for rule in rubric_cases:
        pair = groups[rule['group_id']]
        if (rule.get('source') != pair['clean']['source']
                or pair['whisper'].get('clean_source') != pair['clean']['source']
                or pair['whisper']['target'] != pair['clean']['target']):
            raise ValueError('Rubric clean source or paired-row intent changed.')
        if set(rule) - {'group_id', 'source', *RUBRIC_FIELDS}:
            raise ValueError('Unrecognized per-case rubric field; do not silently omit it.')
        for field in RUBRIC_FIELDS:
            value = rule.get(field, [])
            if (not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value)
                    or (field == 'required_propositions' and not value)):
                raise ValueError('Invalid or missing frozen rubric propositions.')
    summary = report.get('evaluation_summary', {})
    expected = dict(coverage_complete=True, expected_rows=40, expected_outputs=160,
                    observed_outputs=160, valid_outputs=160 - errors, error_outputs=errors)
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('Reported evaluation coverage/error counts disagree with actual outputs.')
    if not isinstance(rubric.get('global_rule'), str) or not rubric['global_rule'].strip():
        raise ValueError('Frozen rubric needs its global grading rule.')
    return rubric


def opaque(prefix, *parts):
    return prefix + hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def prepare(bindings_path, report_path, output):
    bindings = review.load_bindings(bindings_path)
    report_path = report_path.resolve()
    files, report, report_sha = snapshot(bindings, report_path)
    rubric = validate(files, report, bindings)
    groups = {}
    for rule in rubric['cases']:
        groups[rule['group_id']] = dict(group_id=opaque('g_', rule['group_id']), source=rule['source'],
            **{k: rule.get(k, []) for k in RUBRIC_FIELDS}, candidates={})
    entries = []
    for case in report['cases']:
        group = groups[case['group_id']]
        for actual in case['outputs']:
            text = actual.get('text') if actual['status'] == 'ok' else None
            cid = opaque('c_', group['group_id'], actual['status'], text)
            group['candidates'][cid] = dict(id=cid, text=text, status=actual['status'])
            entries.append(dict(row_id=case['id'], source_group_id=case['group_id'],
                blind_group_id=group['group_id'], source=case['source'], variant=case['variant'],
                temperature=actual['temperature'], adapter_enabled=actual['adapter_enabled'],
                seconds=actual['seconds'], status=actual['status'], candidate_id=cid, output=actual))
    public_groups = []
    for group in sorted(groups.values(), key=lambda g: g['group_id']):
        public_groups.append({**group, 'candidates': sorted(group['candidates'].values(), key=lambda c: c['id'])})
    packet = dict(schema_version=1, scope='Whole-example meaning review across20 Japanese source groups. Exact duplicate captions within each group share one grade; error outputs remain failures.',
                  rule=rubric['global_rule'], rubric_sha256=bindings['rubric'][1], groups=public_groups)
    inputs = {f'{name}_sha256': expected for name, (_, expected) in bindings.items()}
    inputs.update(evaluation_report=str(report_path), evaluation_report_sha256=report_sha,
                  preparer_sha256=review.sha(Path(__file__)), scorer_helper_sha256=review.sha(Path(review.__file__)),
                  bindings={k: dict(path=str(p), sha256=h) for k, (p, h) in bindings.items()})
    # A new directory plus exclusive files preserves earlier packets, including failed attempts.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    packet_path, key_path = output / 'packet.json', output / 'private-key.json'
    review.write_new(packet_path, packet)
    key = dict(schema_version=1, packet_sha256=review.sha(packet_path), inputs=inputs, entries=entries,
        warmup_excluded=True, latency_scope='GPU generation plus parsing; separate warmup excluded, errors included. Not CPU-history or browser latency.')
    review.validate_packet_key(packet, key, packet_path, key_path, bindings)
    if review.sha(report_path) != report_sha:
        raise ValueError('Evaluation report changed while preparing review files.')
    review.write_new(key_path, key)
    key_path.chmod(0o600)
    return dict(status='complete', packet=str(packet_path), packet_sha256=review.sha(packet_path),
                private_key=str(key_path), private_key_sha256=review.sha(key_path),
                groups=20, real_output_slots=len(entries), unique_candidates=sum(len(g['candidates']) for g in public_groups))


def self_test():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        def write(name, value):
            path = root / name
            review.write_new(path, value)
            return path
        data, rules, cases = [], [], []
        for i in range(20):
            source, gid = f'日本語{i}', f'group{i}'
            rules.append(dict(group_id=gid, source=source, required_propositions=['Required meaning.'],
                acceptable_variants_and_ambiguities=['Allowed paraphrase.'], major_errors_to_reject=['Reversed polarity.']))
            for variant in ('clean', 'whisper'):
                row = dict(id=f'{gid}_{variant}', group_id=gid, source=source, target='English.',
                    partition='validation', variant=variant, review_status='accepted', clean_source=source)
                data.append(row)
                outputs = [dict(temperature=t, adapter_enabled=a, seed=17, seconds=.1, status='ok', text='English.',
                    ended_with_eos=True, reached_token_cap=False) for t, a in sorted(CONDITIONS)]
                cases.append(dict(row, outputs=outputs))
        cases[0]['outputs'][0].update(status='error', text=None, error='Truncated response.',
                                    ended_with_eos=False, reached_token_cap=True)
        cases[1]['outputs'][0]['text'] = 'English'  # Exact text dedup must preserve punctuation differences.
        data_path = root / 'data.jsonl'
        data_path.write_text(''.join(json.dumps(r) + '\n' for r in data))
        runner = root / 'runner.py'; runner.write_text('# Frozen runner\n')
        rubric = write('rubric.json', dict(status='frozen_before_generation', global_rule='Preserve meaning.', cases=rules))
        manifest = write('manifest.json', dict(status='frozen', data_path=str(data_path), data_sha256=review.sha(data_path)))
        evaluation = dict(max_new_tokens=256, primary_temperature=0, secondary_temperature=.6,
                          secondary_top_p=.95, secondary_top_k=64, secondary_seed=17)
        protocol = write('protocol.json', dict(status='frozen', runner_sha256=review.sha(runner),
            evaluation=evaluation, data=dict(manifest_sha256=review.sha(manifest),
                data_sha256=review.sha(data_path), validation_rubric_sha256=review.sha(rubric))))
        bindings_path = write('bindings.json', {k: dict(path=str(p), sha256=review.sha(p)) for k, p in
            dict(data=data_path, runner=runner, rubric=rubric, manifest=manifest, protocol=protocol).items()})
        report = dict(status='complete', mode='evaluate', owned_worker_released=True, worker_exit_code=0,
            runner_sha256=review.sha(runner), protocol=dict(sha256=review.sha(protocol), contents=review.read(protocol)),
            data=dict(manifest=review.read(manifest), manifest_sha256=review.sha(manifest), path=str(data_path)),
            evaluation=evaluation, evaluation_warmup={'excluded': True}, cases=cases,
            evaluation_summary=dict(coverage_complete=True, expected_rows=40, expected_outputs=160,
                observed_outputs=160, valid_outputs=159, error_outputs=1))
        report_path = write('report.json', report)
        result = prepare(bindings_path, report_path, root / 'review')
        assert result['real_output_slots'] == 160 and result['unique_candidates'] == 22
        public = review.read(Path(result['packet']))
        assert all(g['acceptable_variants_and_ambiguities'] and g['major_errors_to_reject'] for g in public['groups'])
        assert all(set(c) == {'id', 'text', 'status'} for g in public['groups'] for c in g['candidates'])
        bindings = review.load_bindings(bindings_path)
        files, actual, _ = snapshot(bindings, report_path)
        def rejects(changed):
            try:
                validate(files, changed, bindings)
            except ValueError:
                return
            raise AssertionError('Corrupted or incomplete evaluation was accepted.')
        changed = json.loads(json.dumps(actual)); changed['cases'].pop(); rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['cases'][0]['outputs'].pop(); rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['cases'][0]['source'] = 'Mutated source'; rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['cases'][0]['variant'] = 'whisper'; rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['owned_worker_released'] = False; rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['evaluation'].pop('secondary_top_k'); rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['cases'][0]['outputs'][0]['seed'] = 43; rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['cases'][1]['outputs'][1]['ended_with_eos'] = False; rejects(changed)
        changed = json.loads(json.dumps(actual)); changed['cases'][1]['outputs'][1]['reached_token_cap'] = True; rejects(changed)
        key_path, packet_path = Path(result['private_key']), Path(result['packet'])
        key = review.read(key_path)
        key['entries'][0]['output']['seed'] = 43
        try:
            review.validate_packet_key(public, key, packet_path, key_path, bindings)
        except ValueError:
            pass
        else:
            raise AssertionError('Mutated private raw-output snapshot was accepted.')
        # Legacy v1 entries have no full output snapshot and remain supported.
        for entry in key['entries']:
            entry.pop('output')
        review.validate_packet_key(public, key, packet_path, key_path, bindings)
        rubric.write_text('{}')
        try:
            review.load_bindings(bindings_path)
        except ValueError:
            pass
        else:
            raise AssertionError('Frozen rubric mutation was accepted.')
    print('Packet prep self-test passed: coverage, exact dedup, error retention, full rubric, source/variant mutation, complete settings/seeds, inconsistent successful truncation, private snapshot mutation, legacy keys, and frozen-file mutation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, help='Five path/SHA256 bindings; omitted retains frozen v1 defaults.')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path, help='New directory for packet.json and private-key.json.')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not args.report or not args.output:
        parser.error('Provide --report and --output (and --bindings for v2).')
    else:
        print(json.dumps(prepare(args.bindings, args.report, args.output), indent=2))


if __name__ == '__main__':
    main()
