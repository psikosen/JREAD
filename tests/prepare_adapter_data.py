#!/usr/bin/env python3
"""Freeze reviewed pilot data; exclude evaluation overlap without printing its text."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'tests/fixtures/accuracy-corpus.json'
CORPUS_SHA = '66825d56bfa271a58037a07ef68a59c1998c07f65b7740833b9e30d2d44cd578'


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if unicodedata.category(c)[0] in 'LN')


def similar(left, right):
    if left == right:
        return True
    # Conservative lexical guard, not a claim of semantic/paraphrase deduplication.
    if min(len(left), len(right)) < 12:
        return False
    if min(len(left), len(right)) / max(len(left), len(right)) < .75:
        return False
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= .85


def audit(rows, excluded):
    ids, groups, prepared = set(), {}, []
    for row in rows:
        for key in ('id', 'group_id', 'source', 'target', 'partition', 'variant'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Missing nonempty {key}')
        if row['id'] in ids:
            raise ValueError(f"Duplicate row id: {row['id']}")
        ids.add(row['id'])
        if row['partition'] not in ('train', 'validation'):
            raise ValueError('Only independent train/validation rows are accepted')
        if row['variant'] not in ('clean', 'whisper'):
            raise ValueError('Unsupported variant')
        if row.get('review_status') != 'accepted' or not isinstance(row.get('provenance'), dict) or not row['provenance']:
            raise ValueError(f"Missing review/provenance: {row['id']}")
        previous = groups.setdefault(row['group_id'], row['partition'])
        if previous != row['partition']:
            raise ValueError('Clean/noisy variants of a source cross partitions')
        if row['variant'] == 'whisper':
            if any(not re.fullmatch('[0-9a-f]{64}', row.get(key, ''))
                   for key in ('audio_sha256', 'asr_report_sha256')):
                raise ValueError('Whisper rows require real audio and inference provenance')
            if not row.get('clean_source'):
                raise ValueError('Whisper rows require the separately verified clean source')
        sources = [normalized(row['source'])]
        if row.get('clean_source'):
            sources.append(normalized(row['clean_source']))
        target = normalized(row['target'])
        if not all(sources) or not target:
            raise ValueError('Text must contain letters or numbers')
        for case in excluded:
            if (any(similar(source, case['source']) for source in sources)
                    or similar(target, case['target'])
                    or row.get('audio_sha256') in case['audio_hashes']):
                raise ValueError(f"Evaluation overlap rejected for training-data row {row['id']}")
        prepared.append((row, sources, target))
    clean = {row['group_id']: row for row in rows if row['variant'] == 'clean'}
    reports = {}
    for row in rows:
        if row['variant'] != 'whisper':
            continue
        authority = clean.get(row['group_id'])
        if (not authority or normalized(authority['source']) != normalized(row['clean_source'])
                or authority['target'] != row['target']):
            raise ValueError('Every Whisper row must retain its reviewed clean source/target group')
        report_path = Path(row['provenance'].get('asr_report', ''))
        key = str(report_path), row['asr_report_sha256']
        if key not in reports:
            raw = report_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != key[1]:
                raise ValueError('ASR report hash changed')
            report = json.loads(raw)
            if report.get('status') != 'complete':
                raise ValueError('ASR source run is incomplete')
            reports[key] = report['cases']
        matches = [case for case in reports[key] if case['group_id'] == row['group_id']]
        if len(matches) != 1 or any(matches[0].get(k) != v for k, v in {
                'transcript': row['source'], 'source': row['clean_source'],
                'audio_sha256': row['audio_sha256']}.items()):
            raise ValueError('Whisper row does not match its saved inference record')
        if digest(matches[0]['audio_path']) != row['audio_sha256']:
            raise ValueError('Whisper source audio hash changed')
    for index, (row, sources, target) in enumerate(prepared):
        for other, other_sources, other_target in prepared[:index]:
            if row['partition'] == other['partition']:
                continue
            if (any(similar(a, b) for a in sources for b in other_sources)
                    or similar(target, other_target)
                    or (row.get('audio_sha256') and row.get('audio_sha256') == other.get('audio_sha256'))):
                raise ValueError(f"Train/validation overlap: {row['id']} / {other['id']}")
    return {
        'rows': len(rows), 'partitions': dict(Counter(row['partition'] for row in rows)),
        'variants': dict(Counter(row['variant'] for row in rows)),
        'source_groups': len(groups), 'excluded_evaluation_cases': len(excluded),
        'evaluation_text_emitted': False, 'evaluation_audio_opened': False,
        'exact_and_near_overlap_found': False,
        'method': 'NFKC casefold letters/numbers; exact equality or SequenceMatcher >=0.85 for >=12 characters and length ratio >=0.75; source, clean_source and target; available audio hashes. Lexical only, not semantic deduplication or pretraining-contamination detection.',
    }


def self_test():
    row = dict(id='a', group_id='a', source='妹は昨日駅で赤い傘を買いました',
               target='My younger sister bought a red umbrella at the station yesterday.',
               partition='train', variant='clean', review_status='accepted', provenance={'kind': 'self-test'})
    audit([row], [])
    excluded = [dict(source=normalized(row['source']), target='unrelated', audio_hashes=[])]
    for rows, cases in (([row], excluded), ([row, dict(row, id='b', partition='validation')], []),
                        ([dict(row, variant='whisper')], []), ([dict(row, review_status='pending')], [])):
        try:
            audit(rows, cases)
        except ValueError:
            pass
        else:
            raise AssertionError('Expected unsafe data to be rejected')
    assert similar(normalized('Ａ test!'), normalized('a TEST'))
    assert similar('abcdefghijklmnop', 'abcdefxhijklmnop')
    with tempfile.TemporaryDirectory() as folder:
        audio = Path(folder) / 'clip.wav'
        audio.write_bytes(b'self-test-placeholder-not-inference')
        report = Path(folder) / 'asr.json'
        report.write_text(json.dumps({'status': 'complete', 'cases': [{
            'group_id': row['group_id'], 'source': row['source'], 'transcript': row['source'] + '。',
            'audio_path': str(audio), 'audio_sha256': digest(audio)}]}))
        noisy = dict(row, id='a-whisper', variant='whisper', source=row['source'] + '。',
                     clean_source=row['source'], audio_sha256=digest(audio),
                     asr_report_sha256=digest(report), provenance={'asr_report': str(report)})
        audit([row, noisy], [])
        for bad in (dict(noisy, source='別の保存されていない発話'),
                    dict(noisy, asr_report_sha256='0' * 64), dict(noisy, group_id='missing-clean')):
            try:
                audit([row, bad], [])
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched saved ASR provenance must be rejected')
    print('adapter data guard self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input or not args.output:
        parser.error('--input and --output are required')
    if digest(CORPUS) != CORPUS_SHA:
        raise ValueError('Frozen evaluation manifest changed')
    corpus = json.loads(CORPUS.read_text())
    if len(corpus['cases']) != 70:
        raise ValueError('Must protect all 20 development and 50 heldout cases')
    excluded = [dict(source=normalized(c['japanese_reference']), target=normalized(c['english_reference']),
                     audio_hashes=[c[k] for k in ('audio_sha256', 'download_sha256') if c.get(k)])
                for c in corpus['cases']]
    raw_input = args.input.read_bytes()
    rows = [json.loads(line) for line in raw_input.decode('utf-8').splitlines() if line.strip()]
    report = audit(rows, excluded)
    if not all(report['partitions'].get(partition) for partition in ('train', 'validation')):
        raise ValueError('Both independent partitions are required')
    args.output.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)
    path = args.output / 'pilot.jsonl'
    manifest = args.output / 'data-manifest.json'
    if path.exists() or manifest.exists():
        raise ValueError('Refusing to overwrite an existing frozen pilot')
    with path.open('x') as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    manifest_payload = json.dumps({
        'version': 1, 'status': 'frozen', 'data_path': str(path.resolve()), 'data_sha256': digest(path),
        'reviewed_input_sha256': hashlib.sha256(raw_input).hexdigest(), 'evaluation_manifest_sha256': CORPUS_SHA,
        'preparer_sha256': digest(__file__), 'audit': report,
        'review_note': 'Model-assisted bilingual data review, not native-speaker certification.',
        'integrity_note': 'Exclusive creation; manifest written last. Consumers must recheck hashes; mount permissions alone do not guarantee immutability.',
    }, indent=2) + '\n'
    with manifest.open('x') as file:
        file.write(manifest_payload)
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps(report))


if __name__ == '__main__':
    main()
