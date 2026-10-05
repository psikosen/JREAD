#!/usr/bin/env python3
"""Join frozen blind grades to the E4B adapter pilot's private condition key."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('/media/psikosen/newvolume/jread-finetune')
FROZEN_FILES = {
    'protocol': (ROOT / 'tests/fixtures/accuracy-adapter-protocol.json', '06390bd0859d63b6f602403fa602b083238a2550f69c2da99dbb84a46fb625c8'),
    'manifest': (EXPERIMENT / 'data/pilot-v1/data-manifest.json', '6c274cedf9997192bbf1a8b3794c242b03e6cca6cd4632f6e0bdedfd2aac5e11'),
    'data': (EXPERIMENT / 'data/pilot-v1/pilot.jsonl', 'cd9f4851ba14fdc984746000d702677ce78349d5e4907e95b4011b7caa60d66d'),
    'rubric': (EXPERIMENT / 'data/covost2-ja-en-source-check/validation-rubric.json', '8bdba55aecfeb6524987edff7c2efbd8faa3fcc2670b9cce5193f422573860a9'),
    'runner': (ROOT / 'tests/train_translation_adapter.py', '4f08756bcab33c873b0068a3ebcbb1458a43fbb79114e0cd6e02bc643dbabf68'),
}
FROZEN_NAMES = frozenset(FROZEN_FILES)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_bindings(path: Path | None = None):
    """Load explicit v2 inputs or retain the exact v1 defaults."""
    if path is None:
        return dict(FROZEN_FILES)
    raw = read(path)
    if not isinstance(raw, dict) or set(raw) != FROZEN_NAMES:
        raise ValueError('Bindings must contain exactly protocol, manifest, data, rubric, and runner.')
    bindings = {}
    for name, item in raw.items():
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError(f'Binding {name} must contain exactly path and sha256.')
        value = item['path']
        expected = item['sha256']
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Binding {name} requires a nonempty path.')
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError(f'Binding {name} requires a lowercase SHA256.')
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = path.parent / source
        if source.is_symlink():
            raise ValueError(f'Binding {name} must reference a regular file, not a symlink.')
        source = source.resolve()
        if not source.is_file() or sha(source) != expected:
            raise ValueError(f'Binding {name} path is absent, a symlink, or does not match its SHA256.')
        bindings[name] = (source, expected)
    return bindings


def write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def summary(states):
    return {name: sum(state == name for state in states) for name in ('pass', 'fail', 'unclear')}


def validate_grades(packet_sha, candidate_statuses, grades):
    if grades.get('packet_sha256') != packet_sha:
        raise ValueError('Grades refer to a different blinded packet.')
    rows = grades.get('grades', [])
    ids = [row.get('candidate_id') for row in rows]
    if len(ids) != len(candidate_statuses) or len(set(ids)) != len(ids) or set(ids) != set(candidate_statuses):
        raise ValueError('Grades must cover every unique packet candidate exactly once.')
    by_candidate = {row['candidate_id']: row for row in rows}
    for cid, grade in by_candidate.items():
        if grade.get('status') not in ('pass', 'fail', 'unclear') or not isinstance(grade.get('reason'), str) or not grade['reason'].strip():
            raise ValueError(f'Invalid or unevidenced grade for {cid}.')
        if candidate_statuses[cid] == 'error' and grade['status'] != 'fail':
            raise ValueError(f'Missing/error output {cid} must remain a failure.')
    return by_candidate


def validate_packet_key(packet, key, packet_path: Path, key_path: Path, frozen_files=None):
    frozen_files = FROZEN_FILES if frozen_files is None else frozen_files
    packet_hash = sha(packet_path)
    if key.get('packet_sha256') != packet_hash:
        raise ValueError('Private key is not bound to these exact packet bytes.')
    if key.get('warmup_excluded') is not True or not key.get('latency_scope'):
        raise ValueError('Private key lacks evaluation timing scope or warmup exclusion.')
    groups = packet.get('groups', [])
    if len(groups) != 20 or len({g.get('group_id') for g in groups}) != 20:
        raise ValueError('Blind packet must contain exactly20 unique opaque groups.')
    group_ids = {g['group_id'] for g in groups}
    candidate_ids = set()
    for group in groups:
        if not group.get('source') or not group.get('required_propositions'):
            raise ValueError(f'Packet group {group["group_id"]} lacks frozen Japanese/rubric content.')
        for candidate in group.get('candidates', []):
            cid = candidate.get('id')
            if not cid or cid in candidate_ids:
                raise ValueError('Candidate IDs must be present and globally unique.')
            if candidate.get('status') not in ('ok', 'error'):
                raise ValueError(f'Invalid packet candidate status: {cid}.')
            if candidate['status'] == 'ok' and (not isinstance(candidate.get('text'), str) or not candidate['text'].strip()):
                raise ValueError(f'Successful candidate has no English text: {cid}.')
            candidate_ids.add(cid)
    entries = key.get('entries', [])
    if len(entries) != 160:
        raise ValueError('Private key must preserve all160 row/condition slots.')
    row_ids = {entry.get('row_id') for entry in entries}
    if len(row_ids) != 40:
        raise ValueError('Private key must contain exactly40 unique validation rows.')
    expected_conditions = {(0, False), (0, True), (.6, False), (.6, True)}
    per_row = {}
    slots = set()
    for entry in entries:
        gid, cid, row_id = entry.get('blind_group_id'), entry.get('candidate_id'), entry.get('row_id')
        if gid not in group_ids or cid not in candidate_ids:
            raise ValueError('Private slot points to a group or candidate absent from the packet.')
        if entry.get('variant') not in ('clean', 'whisper'):
            raise ValueError('Private slot has an unexpected validation variant.')
        pair = (entry.get('temperature'), entry.get('adapter_enabled'))
        if pair not in expected_conditions:
            raise ValueError(f'Unexpected adapter evaluation condition: {pair}.')
        slot = (row_id, pair)
        if slot in slots:
            raise ValueError(f'Duplicate row/condition slot: {slot}.')
        slots.add(slot)
        per_row.setdefault(row_id, set()).add(pair)
        seconds = entry.get('seconds')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Invalid GPU timing for {row_id}.')
        if entry.get('status') not in ('ok', 'error'):
            raise ValueError(f'Unfinished evaluation output for {row_id}.')
    if any(conditions != expected_conditions for conditions in per_row.values()):
        raise ValueError('Every validation row must retain all four evaluation conditions.')
    if len(slots) != 160 or len(per_row) != 40:
        raise ValueError('Incomplete row/condition coverage.')
    if {entry['blind_group_id'] for entry in entries} != group_ids:
        raise ValueError('Private entries do not cover all20 opaque groups.')
    if {entry['candidate_id'] for entry in entries} != candidate_ids:
        raise ValueError('A candidate is orphaned or absent from all160 private slots.')
    for entry in entries:
        group = next(g for g in groups if g['group_id'] == entry['blind_group_id'])
        candidate = next(c for c in group['candidates'] if c['id'] == entry['candidate_id'])
        if entry['status'] == 'error' and candidate['status'] != 'error':
            raise ValueError('Private-key error output was hidden as a successful candidate.')
        if entry['status'] == 'ok' and candidate['status'] != 'ok':
            raise ValueError('Private-key successful output points to an error candidate.')
    frozen = {f'{name}_sha256': frozen_files[name][1]
              for name in ('protocol', 'manifest', 'rubric')}
    if any(key.get('inputs', {}).get(name) != value for name, value in frozen.items()):
        raise ValueError('Private key is not bound to the frozen protocol/data/rubric hashes.')
    for path_field, hash_field in (('evaluation_report', 'evaluation_report_sha256'),):
        path = key['inputs'].get(path_field)
        expected = key['inputs'].get(hash_field)
        if not path or not expected or sha(Path(path)) != expected:
            raise ValueError('Finished evaluation report changed since blind packet creation.')
    for name, (path, expected) in frozen_files.items():
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Frozen {name} input changed since the pilot: {path}.')
    report = read(Path(key['inputs']['evaluation_report']))
    if report.get('status') != 'complete' or not report.get('owned_worker_released') or report.get('worker_exit_code') != 0:
        raise ValueError('Evaluation did not complete and release its worker.')
    manifest = read(frozen_files['manifest'][0])
    if (report.get('protocol', {}).get('sha256') != frozen_files['protocol'][1]
            or report.get('data', {}).get('manifest_sha256') != frozen_files['manifest'][1]
            or report.get('runner_sha256') != frozen_files['runner'][1]
            or Path(manifest.get('data_path', '')).resolve() != frozen_files['data'][0]
            or manifest.get('data_sha256') != frozen_files['data'][1]):
        raise ValueError('Evaluation report is not bound to the supplied protocol, manifest, data, and runner.')
    cases = {case['id']: case for case in report['cases']}
    if len(cases) != 40 or len(report['cases']) != 40 or set(cases) != row_ids:
        raise ValueError('Evaluation and key do not cover identical validation rows.')
    candidates = {c['id']: c for group in groups for c in group['candidates']}
    for entry in entries:
        case = cases[entry['row_id']]
        matches = [o for o in case['outputs'] if o['temperature'] == entry['temperature']
                   and o['adapter_enabled'] == entry['adapter_enabled']]
        if len(matches) != 1 or case['group_id'] != entry['source_group_id'] or case['variant'] != entry['variant']:
            raise ValueError('Private entry points to the wrong saved group/variant/condition.')
        actual, candidate = matches[0], candidates[entry['candidate_id']]
        if 'output' in entry and entry['output'] != actual:
            raise ValueError('Private entry changes the complete saved output snapshot.')
        if actual['seconds'] != entry['seconds'] or actual['status'] != entry['status'] or candidate.get('text') != (actual.get('text') if actual['status'] == 'ok' else None):
            raise ValueError('Private entry changes a saved output, status or latency.')
    return candidate_ids, group_ids


def score(packet_path: Path, key_path: Path, grades_path: Path, bindings_path: Path | None = None):
    packet, key, grades = read(packet_path), read(key_path), read(grades_path)
    frozen_files = load_bindings(bindings_path)
    candidate_ids, group_ids = validate_packet_key(packet, key, packet_path, key_path, frozen_files)
    packet_hash = sha(packet_path)
    if grades.get('packet_sha256') != packet_hash:
        raise ValueError('Grades refer to a different blinded packet.')
    packet_candidates = {c['id']: c for group in packet['groups'] for c in group['candidates']}
    by_candidate = validate_grades(packet_hash, {cid: item['status'] for cid, item in packet_candidates.items()}, grades)

    results = {}
    details = {}
    for variant in ('clean', 'whisper'):
        for temperature, adapter in sorted({(e['temperature'], e['adapter_enabled']) for e in key['entries']}):
            label = f"{variant}/t{temperature:g}_adapter_{'on' if adapter else 'off'}"
            entries = [e for e in key['entries'] if e['variant'] == variant
                       and e['temperature'] == temperature and e['adapter_enabled'] == adapter]
            if len(entries) != 20 or len({e['blind_group_id'] for e in entries}) != 20:
                raise ValueError(f'{label} does not have20 paired-group observations.')
            states, times = [], []
            rows = []
            for entry in entries:
                grade = by_candidate[entry['candidate_id']]['status'] if entry['status'] == 'ok' else 'fail'
                states.append(grade)
                times.append(entry['seconds'])
                rows.append(dict(group_id=entry['blind_group_id'], status=grade,
                                 candidate_id=entry['candidate_id'], seconds=entry['seconds']))
            results[label] = dict(variant=variant, temperature=temperature, adapter_enabled=adapter,
                denominator=20, grades=summary(states), gpu_generation_seconds={
                    'calls': len(times), 'total': sum(times), 'mean': statistics.mean(times),
                    'median': statistics.median(times)}, cases=rows)
            details[(variant, temperature, adapter)] = {row['group_id']: row['status'] for row in rows}
    paired = {}
    for variant in ('clean', 'whisper'):
        for temperature in (0, .6):
            off = details[(variant, temperature, False)]
            on = details[(variant, temperature, True)]
            paired[f'{variant}/t{temperature:g}'] = dict(denominator=20,
                gains=[g for g in off if off[g] != 'pass' and on[g] == 'pass'],
                regressions=[g for g in off if off[g] == 'pass' and on[g] != 'pass'],
                pass_to_unclear=[g for g in off if off[g] == 'pass' and on[g] == 'unclear'],
                unclear_to_pass=[g for g in off if off[g] == 'unclear' and on[g] == 'pass'])
    return dict(schema_version=1, status='complete_review_scored', packet_sha256=packet_hash,
        key_sha256=sha(key_path), grades_sha256=sha(grades_path), reviewer=grades.get('reviewer'),
        frozen_input_sha256={name: expected for name, (_, expected) in frozen_files.items()},
        denominator_groups=20, candidate_count=len(candidate_ids), condition_scores=results,
        matched_adapter_changes=paired,
        timing_note='All20 clean or Whisper-input validation calls per condition are included, including malformed/error calls. Timings are measured GPU generation plus parsing; separate warmup is excluded. Not a CPU-history or browser-latency comparison.')


def self_test():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        entries = {}
        for name in sorted(FROZEN_NAMES):
            source = root / f'{name}.input'
            source.write_text(name)
            entries[name] = {'path': str(source), 'sha256': sha(source)}
        bindings_path = root / 'bindings.json'
        bindings_path.write_text(json.dumps(entries))
        loaded = load_bindings(bindings_path)
        assert set(loaded) == FROZEN_NAMES
        assert all(path == (root / f'{name}.input').resolve() for name, (path, _) in loaded.items())
        assert set(load_bindings()) == FROZEN_NAMES

        def rejects_bindings(value):
            bindings_path.write_text(json.dumps(value))
            try:
                load_bindings(bindings_path)
                return False
            except ValueError:
                return True

        assert rejects_bindings({**entries, 'unsupported': {'path': 'x', 'sha256': '0' * 64}})
        assert rejects_bindings({name: value for name, value in entries.items() if name != 'runner'})
        malformed = dict(entries); malformed['protocol'] = {'path': str(root / 'protocol.input'), 'sha256': 'bad'}
        assert rejects_bindings(malformed)
        malformed = dict(entries); malformed['data'] = {'path': str(root / 'data.input'), 'sha256': entries['data']['sha256'], 'extra': True}
        assert rejects_bindings(malformed)

    candidate_statuses = {'c1': 'ok', 'c2': 'error'}
    grades = {'packet_sha256': 'p', 'grades': [{'candidate_id': 'c1', 'status': 'pass', 'reason': 'faithful'},
        {'candidate_id': 'c2', 'status': 'fail', 'reason': 'missing'}]}
    joined = validate_grades('p', candidate_statuses, grades)
    assert summary([g['status'] for g in joined.values()]) == {'pass': 1, 'fail': 1, 'unclear': 0}
    def rejects(changed):
        try:
            validate_grades('p', candidate_statuses, changed)
            return False
        except ValueError:
            return True
    assert rejects({**grades, 'packet_sha256': 'wrong'})
    assert rejects({'packet_sha256': 'p', 'grades': grades['grades'][:1]})
    assert rejects({'packet_sha256': 'p', 'grades': grades['grades'] + [grades['grades'][0]]})
    assert rejects({'packet_sha256': 'p', 'grades': [grades['grades'][0], {'candidate_id': 'c2', 'status': 'pass', 'reason': 'oops'}]})
    print('Adapter grade-join self-test passed: v1 defaults, explicit bindings, malformed/unsupported binding rejection, packet binding, exact coverage, duplicate/missing grades, and error-fail rule.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--key', type=Path)
    parser.add_argument('--grades', type=Path)
    parser.add_argument('--bindings', type=Path,
                        help='Optional JSON object binding protocol, manifest, data, rubric, and runner paths to SHA256 values; omitted uses frozen v1 inputs.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.packet, args.key, args.grades, args.output)):
        parser.error('Provide --packet, --key, --grades, and --output.')
    result = score(args.packet, args.key, args.grades, args.bindings)
    write_new(args.output, result)
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'packet_sha256': result['packet_sha256'], 'candidates': result['candidate_count']}))


if __name__ == '__main__':
    main()
