#!/usr/bin/env python3
"""Export blind candidate grades and score frozen ensemble selections; no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'tests/fixtures/accuracy-ensemble-protocol.json'
CONTROLS_PATH = ROOT / 'tests/fixtures/accuracy-ensemble-controls.json'
RUBRIC_PATH = ROOT / 'tests/fixtures/accuracy-development-rubric.json'
CORPUS_PATH = ROOT / 'tests/fixtures/accuracy-corpus.json'
BASELINE_PATH = ROOT / 'test-results/accuracy-streaming-turbo-none-translation-development.json'
PACKET_PATH = ROOT / 'test-results/accuracy-ensemble-review-packet.json'
KEY_PATH = ROOT / 'test-results/accuracy-ensemble-private-review-key.json'
SCORE_PATH = ROOT / 'test-results/accuracy-ensemble-selection-scores.json'
GRADE_PATH = ROOT / 'test-results/accuracy-ensemble-candidate-grades.json'
FROZEN_CORPUS_SHA256 = '66825d56bfa271a58037a07ef68a59c1998c07f65b7740833b9e30d2d44cd578'
FROZEN_RUBRIC_SHA256 = 'ce7514d3e12dded03e3c2ce0894cb0e3750720d0034aec0f63d45256cf3a89e9'
JUDGE_REPORTS = {
    '4b': ROOT / 'test-results/accuracy-ensemble-judge-4b-development.json',
    '2b': ROOT / 'test-results/accuracy-ensemble-judge-2b-development.json',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def output_id(case_id, text: str) -> str:
    return 'out_' + digest([case_id, text])[:16]


def completed_report(path: Path, stage: str, provenance: dict, case_ids: list[int]):
    report = read_json(path)
    if (report.get('status') not in ('complete', 'complete_with_errors')
            or report.get('stage') != stage or report.get('provenance') != provenance
            or [row.get('id') for row in report.get('cases', [])] != case_ids
            or len(report['cases']) != 20):
        raise ValueError(f'{path.name} is incomplete, inconsistent, or not the frozen20 stage.')
    cleanup = report.get('cleanup', {})
    if (not cleanup.get('owned_process_released')
            or cleanup.get('port_accepting_connections_after_shutdown') is not False):
        raise ValueError(f'{path.name} has no confirmed owned-process cleanup.')
    return report


def load_artifacts():
    protocol = read_json(PROTOCOL_PATH)
    rubric = read_json(RUBRIC_PATH)
    controls = read_json(CONTROLS_PATH)
    baseline = read_json(BASELINE_PATH)
    expected_ids = [row['id'] for row in rubric['cases']]
    if (len(expected_ids) != 20 or len(set(expected_ids)) != 20
            or protocol['case_count'] != 20 or len(protocol['pools']) != 10
            or len(controls) != 6 or baseline.get('status') != 'complete'
            or sha(BASELINE_PATH) != protocol['baseline_sha256']
            or sha(CORPUS_PATH) != FROZEN_CORPUS_SHA256
            or sha(RUBRIC_PATH) != FROZEN_RUBRIC_SHA256):
        raise ValueError('Frozen protocol, rubric, controls, or original E4B baseline mismatch.')
    report_paths = {
        'generate-4b': ROOT / 'test-results/accuracy-ensemble-generate-4b-development.json',
        'generate-2b': ROOT / 'test-results/accuracy-ensemble-generate-2b-development.json',
        'judge-4b': JUDGE_REPORTS['4b'], 'judge-2b': JUDGE_REPORTS['2b'],
    }
    first = read_json(report_paths['generate-4b'])
    provenance = first['provenance']
    if provenance['protocol']['sha256'] != sha(PROTOCOL_PATH):
        raise ValueError('Frozen protocol changed since generation.')
    if provenance['controls']['sha256'] != sha(CONTROLS_PATH):
        raise ValueError('Frozen controls changed since generation.')
    if provenance.get('corpus_sha256') != FROZEN_CORPUS_SHA256:
        raise ValueError('Frozen corpus differs from generation provenance.')
    if provenance['sources'].get(str(BASELINE_PATH.relative_to(ROOT))) != sha(BASELINE_PATH):
        raise ValueError('Saved original E4B baseline differs from generation provenance.')
    reports = {'generate-4b': completed_report(report_paths['generate-4b'], 'generate-4b', provenance, expected_ids)}
    reports['generate-2b'] = completed_report(report_paths['generate-2b'], 'generate-2b', provenance, expected_ids)
    reports['judge-4b'] = completed_report(report_paths['judge-4b'], 'judge-4b', provenance, expected_ids)
    reports['judge-2b'] = completed_report(report_paths['judge-2b'], 'judge-2b', provenance, expected_ids)
    generation_paths = {name: sha(report_paths[name]) for name in ('generate-4b', 'generate-2b')}
    for stage in ('judge-4b', 'judge-2b'):
        if reports[stage].get('dependencies') != {
            str(report_paths['generate-4b'].relative_to(ROOT)): generation_paths['generate-4b'],
            str(report_paths['generate-2b'].relative_to(ROOT)): generation_paths['generate-2b'],
        }:
            raise ValueError(f'{stage} did not consume these exact frozen generation reports.')
        if len(reports[stage].get('controls', [])) != 6:
            raise ValueError(f'{stage} is missing its six selector controls.')
    if any(report['aggregate']['case_count'] != 20 for report in reports.values()):
        raise ValueError('Every stage must retain all20 cases.')

    baseline_by_id = {row['id']: row for row in baseline['stage']['cases']}
    rubric_by_id = {row['id']: row for row in rubric['cases']}
    corpus = read_json(CORPUS_PATH)
    corpus_by_id = {row['id']: row for row in corpus['cases']}
    candidates_by_id = {}
    source_by_id = {}
    for case_id in expected_ids:
        b = baseline_by_id[case_id]
        if b['status'] != 'ok' or len(b['replay']['outputs']) != 1:
            raise ValueError(f'Original baseline is not one successful output for {case_id}.')
        base = b['replay']['outputs'][0]
        source_by_id[case_id] = dict(start=base['source_start_seconds'], end=base['source_end_seconds'],
                                     text=base['source_text'])
        candidates_by_id[case_id] = {
            'e4_original': {'status': 'ok', 'text': base['translation'],
                            'inference_seconds': b['replay']['measured_translation_seconds'], 'reused': True}
        }
    specs = protocol['candidates']
    for model in ('4b', '2b'):
        stage_rows = reports['generate-' + model]['cases']
        expected = {key for key, spec in specs.items() if spec['model'] == model and not spec.get('reused')}
        for row in stage_rows:
            if set(row.get('candidates', {})) != expected:
                raise ValueError(f'{model} generation case {row["id"]} omits a frozen candidate.')
            for key, result in row['candidates'].items():
                candidates_by_id[row['id']][key] = {
                    'status': result.get('status', 'error'), 'text': result.get('text'),
                    'error': result.get('error'), 'inference_seconds': result.get('inference_seconds', 0.0),
                    'reused': False,
                }
    # Keep every stage bound to the same ASR text, audio, and canonical timings.
    for stage, report in reports.items():
        for row in report['cases']:
            if row.get('audio_sha256') != corpus_by_id[row['id']]['audio_sha256'] or row.get('source') != source_by_id[row['id']]:
                raise ValueError(f'{stage} source/audio identity changed for case {row["id"]}.')
    # Selections must be exact members of the candidate set in their frozen pool.
    for pool in protocol['pools']:
        stage = 'judge-' + pool['judge']
        for row in reports[stage]['cases']:
            selection = row['pools'].get(pool['id'])
            if selection is None:
                raise ValueError(f'Missing frozen pool {pool["id"]} for case {row["id"]}.')
            allowed = {candidates_by_id[row['id']][k]['text'] for k in pool['candidates']
                       if candidates_by_id[row['id']][k]['status'] == 'ok'}
            for field in ('primary', 'order_stable'):
                selected = selection.get(field)
                if selected is not None and selected.get('text') not in allowed:
                    raise ValueError(f'{field} selection is not a candidate in {pool["id"]}, case {row["id"]}.')
    return protocol, controls, rubric_by_id, corpus_by_id, reports, candidates_by_id, provenance


def build_exports(protocol, controls, rubric_by_id, corpus_by_id, reports, candidates_by_id, provenance, artifact_hashes=None):
    packet_cases = []
    private_cases = {}
    candidate_ids = set()
    for case_id in [row['id'] for row in corpus_by_id.values() if row['id'] in rubric_by_id]:
        candidates = candidates_by_id[case_id]
        by_text = {}
        invalid = []
        for key, item in candidates.items():
            if item['status'] != 'ok' or not isinstance(item.get('text'), str) or not item['text'].strip():
                invalid.append({'candidate_key': key, 'status': item['status'], 'error': item.get('error')})
                continue
            text = item['text']
            by_text.setdefault(text, {'output_id': output_id(case_id, text), 'english': text})
        outputs = sorted(by_text.values(), key=lambda row: row['output_id'])
        for row in outputs:
            if row['output_id'] in candidate_ids:
                raise ValueError('Opaque output ID collision.')
            candidate_ids.add(row['output_id'])
        raw = corpus_by_id[case_id]
        if case_id not in rubric_by_id:
            raise ValueError('Missing frozen rubric case.')
        packet_cases.append({
            'case_id': case_id,
            'japanese_reference': raw['japanese_reference'],
            'rubric': {k: rubric_by_id[case_id][k] for k in (
                'required_propositions', 'required_entities', 'numbers_dates',
                'polarity_agent_constraints', 'guide_discrepancies', 'reference_caveats') if k in rubric_by_id[case_id]},
            'outputs': outputs,
        })
        text_to_id = {row['english']: row['output_id'] for row in outputs}
        origins = {}
        candidate_output_ids = {}
        candidate_times = {}
        for key, item in candidates.items():
            if item['status'] == 'ok' and isinstance(item.get('text'), str) and item['text'].strip():
                candidate_output_ids[key] = text_to_id[item['text']]
                origins.setdefault(text_to_id[item['text']], []).append(key)
            else:
                candidate_output_ids[key] = None
            candidate_times[key] = item.get('inference_seconds', 0.0)
        pool_rows = {}
        for pool in protocol['pools']:
            pool_rows[pool['id']] = {
                key: (text_to_id.get(candidates[key].get('text')) if candidates[key]['status'] == 'ok' else None)
                for key in pool['candidates']
            }
        actual = {}
        for stage in ('judge-4b', 'judge-2b'):
            row = next(c for c in reports[stage]['cases'] if c['id'] == case_id)
            for pool_id, selection in row['pools'].items():
                actual[pool_id] = {
                    'primary_output_id': text_to_id.get((selection.get('primary') or {}).get('text')),
                    'stable_output_id': text_to_id.get((selection.get('order_stable') or {}).get('text')),
                    'status': selection.get('status'),
                    'order_sensitive': selection.get('order_sensitive', False),
                    'stable_used_fallback': (selection.get('order_stable') or {}).get('fallback', False),
                    'generation_seconds': selection.get('generation_seconds', 0.0),
                    'primary_sequential_seconds': selection.get('primary_sequential_seconds', 0.0),
                    'stable_sequential_seconds': selection.get('stable_sequential_seconds', 0.0),
                }
        private_cases[str(case_id)] = {
            'candidate_origins': origins,
            'candidate_output_ids': candidate_output_ids,
            'candidate_generation_seconds': candidate_times,
            'invalid_or_missing_candidates': invalid,
            'pool_candidate_output_ids': pool_rows,
            'actual_selections': actual,
        }
    packet_controls = []
    private_controls = {}
    for control in controls:
        by_text = {}
        for name, text in control['candidates'].items():
            by_text.setdefault(text, {'output_id': output_id('control-' + control['id'], text), 'english': text})
        outputs = sorted(by_text.values(), key=lambda row: row['output_id'])
        packet_controls.append({'control_id': control['id'], 'japanese_source': control['source'], 'outputs': outputs})
        private_controls[control['id']] = {
            'acceptable_origins': control['review_only']['acceptable'],
            'output_origins': {row['output_id']: [k for k, text in control['candidates'].items() if text == row['english']]
                               for row in outputs},
            'judge_orders': {},
        }
        for judge in ('4b', '2b'):
            report = reports['judge-' + judge]
            row = next(c for c in report['controls'] if c['id'] == control['id'])
            mapping = row['result']['candidate_mapping']
            id_to_origins = {candidate['id']: candidate['origins'] for candidate in mapping}
            raw = []
            for order_index, order in enumerate(row['result']['orders']):
                result = order['result']
                selected_origins = id_to_origins.get(result.get('text'), []) if result.get('status') == 'ok' else []
                raw.append({
                    'order_index': order_index,
                    'selected_origins': selected_origins,
                    'judge_status': result.get('status'),
                    'correct': result.get('status') == 'ok' and bool(set(selected_origins) & set(control['review_only']['acceptable'])),
                    'successful': result.get('status') == 'ok' and result.get('text') in id_to_origins,
                    'selected_output_id': next((o['output_id'] for o in outputs
                        if set(private_controls[control['id']]['output_origins'][o['output_id']]) & set(selected_origins)), None),
                })
            private_controls[control['id']]['judge_orders'][judge] = raw
    packet = {
        'schema_version': 1,
        'review_type': 'blinded exact-text candidate grading; selector outputs withheld',
        'partition': 'development', 'denominator': 20,
        'review_instructions': 'Grade every unique English output against only the Japanese reference and frozen rubric. Mark pass/fail/unclear and give concise evidence. Accept materially faithful paraphrases. Do not infer model identity, candidate order, selector preference, or preferred writing style. Controls are separate synthetic diagnostics and are not part of the20-case accuracy denominator.',
        'cases': packet_cases,
        'synthetic_controls': packet_controls,
        'limits': ['Exact English duplicates within one case are shown once.',
                   'No candidate origin, model name, prompt variant, pool, selected choice, or English reference is included.',
                   'All20 cases appear even if some generation candidates failed; an empty output list is not a pass.'],
    }
    if artifact_hashes is None:
        artifact_hashes = {
            'generation_reports': {f'test-results/accuracy-ensemble-generate-{m}-development.json':
                sha(ROOT / f'test-results/accuracy-ensemble-generate-{m}-development.json') for m in ('4b', '2b')},
            'judge_reports': {f'test-results/accuracy-ensemble-judge-{m}-development.json':
                sha(JUDGE_REPORTS[m]) for m in ('4b', '2b')},
        }
    key = {
        'schema_version': 1,
        'packet_sha256': digest(packet),
        'provenance': {
            'packet_inputs': {str(path.relative_to(ROOT)): sha(path) for path in (
                PROTOCOL_PATH, CONTROLS_PATH, RUBRIC_PATH, CORPUS_PATH, BASELINE_PATH)},
            **artifact_hashes,
            'frozen_run_provenance': provenance,
        },
        'cases': private_cases,
        'controls': private_controls,
    }
    return packet, key


def write_new(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def grade_summary(statuses):
    return {state: sum(status == state for status in statuses) for state in ('pass', 'fail', 'unclear')}


def score(packet, key, grades):
    if grades.get('packet_sha256') != key['packet_sha256']:
        raise ValueError('Grade file is for a different blinded review packet.')
    provenance = key['provenance']
    current_files = dict(provenance.get('packet_inputs', {}))
    current_files.update(provenance.get('generation_reports', {}))
    current_files.update(provenance.get('judge_reports', {}))
    frozen = provenance.get('frozen_run_provenance', {})
    runner = frozen.get('runner', {})
    if runner.get('path') and runner.get('sha256'):
        current_files[runner['path']] = runner['sha256']
    for section in ('sources', 'helpers'):
        current_files.update(frozen.get(section, {}))
    for relpath, expected_sha in current_files.items():
        actual_path = ROOT / relpath
        if not actual_path.is_file() or sha(actual_path) != expected_sha:
            raise ValueError(f'Frozen key provenance changed since export: {relpath}.')
    grade_list = grades.get('cases', [])
    grade_ids = [row.get('case_id') for row in grade_list]
    if len(grade_ids) != 20 or len(set(grade_ids)) != 20 or set(grade_ids) != {int(case_id) for case_id in key['cases']}:
        raise ValueError('Review grades must include exactly all20 case IDs.')
    grade_rows = {row['case_id']: row for row in grade_list}
    by_case_grade = {}
    for case_id, row in grade_rows.items():
        candidate_grades = {c['output_id']: c for c in row.get('outputs', [])}
        grade_output_ids = [c.get('output_id') for c in row.get('outputs', [])]
        if len(grade_output_ids) != len(set(grade_output_ids)):
            raise ValueError(f'Duplicate output grade rows for case {case_id}.')
        expected = {o['output_id'] for case in packet['cases'] if case['case_id'] == case_id for o in case['outputs']}
        if set(candidate_grades) != expected:
            raise ValueError(f'Every unique output in case {case_id} must receive a grade.')
        if any(c.get('status') not in ('pass', 'fail', 'unclear') or not c.get('evidence') for c in candidate_grades.values()):
            raise ValueError(f'Invalid or unevidenced output grade for case {case_id}.')
        by_case_grade[str(case_id)] = candidate_grades

    protocol = read_json(PROTOCOL_PATH)
    scores_by_pool = {}
    for pool in protocol['pools']:
        coverage, primary, stable, baseline_states = [], [], [], []
        selection_details = []
        invalid_count = 0
        for case_id in grade_rows:
            private = key['cases'][str(case_id)]
            pool_ids = private['pool_candidate_output_ids'][pool['id']]
            output_ids = [out_id for out_id in pool_ids.values() if out_id is not None]
            invalid_count += sum(out_id is None for out_id in pool_ids.values())
            grades_for_case = by_case_grade[str(case_id)]
            candidate_states = [grades_for_case[out_id]['status'] for out_id in set(output_ids)]
            if 'pass' in candidate_states:
                coverage_status = 'pass'
            elif 'unclear' in candidate_states:
                coverage_status = 'unclear'
            else:
                coverage_status = 'fail'
            coverage.append(coverage_status)
            actual = private['actual_selections'].get(pool['id'], {})
            chosen = actual.get('primary_output_id')
            chosen_status = grades_for_case[chosen]['status'] if chosen in grades_for_case else 'fail'
            primary.append(chosen_status)
            chosen_stable = actual.get('stable_output_id')
            stable_status = grades_for_case[chosen_stable]['status'] if chosen_stable in grades_for_case else 'fail'
            stable.append(stable_status)
            baseline_id = private['candidate_output_ids'].get('e4_original')
            baseline_status = grades_for_case[baseline_id]['status'] if baseline_id in grades_for_case else 'fail'
            baseline_states.append(baseline_status)
            selection_details.append({'case_id': case_id, 'candidate_coverage': coverage_status,
                'primary_status': chosen_status, 'stable_status': stable_status,
                'baseline_original_e4_status': baseline_status,
                'primary_selection_error': chosen is None,
                'stable_selection_error': chosen_stable is None,
                'order_sensitive': actual.get('order_sensitive', False),
                'stable_used_fallback': actual.get('stable_used_fallback', False),
                'valid_unique_candidates': len(set(output_ids)),
                'generation_seconds_including_reused_baseline': actual.get('generation_seconds', 0.0),
                'primary_sequential_seconds': actual.get('primary_sequential_seconds', 0.0),
                'stable_sequential_seconds': actual.get('stable_sequential_seconds', 0.0)})
        pool_times = [key['cases'][str(case_id)]['actual_selections'][pool['id']] for case_id in grade_rows]
        primary_times = [row['primary_sequential_seconds'] for row in pool_times]
        stable_times = [row['stable_sequential_seconds'] for row in pool_times]
        generation_times = [row['generation_seconds'] for row in pool_times]
        gains = [case_id for case_id, current, base in zip(grade_rows, primary, baseline_states)
                 if current == 'pass' and base == 'fail']
        regressions = [case_id for case_id, current, base in zip(grade_rows, primary, baseline_states)
                       if current == 'fail' and base == 'pass']
        stable_gains = [case_id for case_id, current, base in zip(grade_rows, stable, baseline_states)
                        if current == 'pass' and base == 'fail']
        stable_regressions = [case_id for case_id, current, base in zip(grade_rows, stable, baseline_states)
                              if current == 'fail' and base == 'pass']
        scores_by_pool[pool['id']] = {
            'judge_family': pool['judge'], 'candidate_keys': pool['candidates'],
            'denominator': 20, 'invalid_or_missing_candidate_outputs': invalid_count,
            'oracle_candidate_coverage': grade_summary(coverage),
            'actual_raw_primary_selection': grade_summary(primary),
            'actual_order_stable_selection': grade_summary(stable),
            'baseline_original_e4': grade_summary(baseline_states),
            'primary_gains_vs_baseline_case_ids': gains,
            'primary_regressions_vs_baseline_case_ids': regressions,
            'stable_gains_vs_baseline_case_ids': stable_gains,
            'stable_regressions_vs_baseline_case_ids': stable_regressions,
            'primary_pass_to_unclear_case_ids': [i for i, current, base in zip(grade_rows, primary, baseline_states) if base == 'pass' and current == 'unclear'],
            'primary_unclear_to_pass_case_ids': [i for i, current, base in zip(grade_rows, primary, baseline_states) if base == 'unclear' and current == 'pass'],
            'sequential_timing_seconds': {
                'generation_including_reused_baseline': {'total': sum(generation_times), 'mean_per_case': statistics.mean(generation_times), 'median_per_case': statistics.median(generation_times)},
                'primary_including_one_judge': {'total': sum(primary_times), 'mean_per_case': statistics.mean(primary_times), 'median_per_case': statistics.median(primary_times)},
                'stable_including_two_judge_orders': {'total': sum(stable_times), 'mean_per_case': statistics.mean(stable_times), 'median_per_case': statistics.median(stable_times)},
            },
            'oracle_minus_primary_pass_cases': sum(s == 'pass' for s in coverage)-sum(s == 'pass' for s in primary),
            'order_sensitive_case_count': sum(d['order_sensitive'] for d in selection_details),
            'stable_fallback_case_count': sum(d['stable_used_fallback'] for d in selection_details),
            'cases': selection_details,
        }
    # Individual generation-condition scores, including reused E4B original and unavailable outputs.
    all_candidate_keys = list(read_json(PROTOCOL_PATH)['candidates'])
    individual = {}
    for candidate_key in all_candidate_keys:
        states, times, missing = [], [], 0
        for case_id in grade_rows:
            private = key['cases'][str(case_id)]
            oid = private['candidate_output_ids'].get(candidate_key)
            times.append(private['candidate_generation_seconds'].get(candidate_key, 0.0))
            if oid is None:
                states.append('fail'); missing += 1
            else:
                states.append(by_case_grade[str(case_id)][oid]['status'])
        individual[candidate_key] = {'denominator': 20, 'reviewed_candidate_outputs': grade_summary(states),
            'missing_or_invalid_candidate_count': missing,
            'measured_generation_seconds': {'total': sum(times), 'mean_per_case': statistics.mean(times),
                                            'median_per_case': statistics.median(times)}}
    controls = {}
    for control_id, data in key['controls'].items():
        controls[control_id] = {}
        for judge, rows in data['judge_orders'].items():
            choices = []
            for row in rows:
                choices.append({'order_index': row['order_index'], 'correct': row['correct'],
                    'status': 'pass' if row['correct'] else 'fail',
                    'selected_output_id': row['selected_output_id'],
                    'judge_status': row['judge_status']})
            controls[control_id][judge] = {
                'raw_order_choices': choices,
                'raw_order_pass_count': sum(item['correct'] for item in choices),
                'raw_order_count': len(choices),
                'orders_agree': len(choices) == 2 and all(item['successful'] for item in rows)
                    and len({item['selected_output_id'] for item in choices}) == 1,
                'note': 'Scores each raw judge order; does not use the stable selection or its correct-first-candidate fallback.',
            }
    return {
        'schema_version': 1, 'status': 'complete_review_scored',
        'packet_sha256': key['packet_sha256'], 'grade_artifact_sha256': sha(GRADE_PATH),
        'denominator_per_pool': 20, 'individual_candidate_scores': individual, 'pool_scores': scores_by_pool,
        'synthetic_controls_excluded_from_denominator': controls,
        'interpretation': 'Oracle coverage is whether any generated candidate passes after blinded review, not practical selector accuracy. Primary uses the first shuffled judge output. Order-stable uses runner policy and is reported separately; when judges disagree it may choose the original fallback and must not be treated as proof of judge quality. Invalid or missing candidates/selections remain failures in the20-case denominator.',
    }


def self_test():
    # Exact-text grouping is case-local and origin labels never enter blind rows.
    outputs = {}
    origins = {}
    for origin, text in [('e4_original', 'A faithful answer.'), ('e2_natural', 'A faithful answer.'), ('e4_literal', 'A wrong answer.')]:
        oid = output_id(42, text)
        outputs.setdefault(oid, {'output_id': oid, 'english': text})
        origins.setdefault(oid, []).append(origin)
    assert len(outputs) == 2 and len(origins[next(k for k,v in outputs.items() if v['english']=='A faithful answer.')]) == 2
    assert all(not any(word in json.dumps(row) for word in ('e4_original', 'e2_natural', 'e4_literal')) for row in outputs.values())
    # Per-pool scoring retains a failed/missing selected result and independent oracle coverage.
    assert grade_summary(['pass'] + ['fail']*18 + ['unclear']) == {'pass': 1, 'fail': 18, 'unclear': 1}
    # Raw synthetic controls are checked against acceptable origin keys, never a fallback field.
    acceptable = {'less', 'fewer'}
    raw_orders = [{'selected_origins': ['less'], 'correct': True}, {'selected_origins': ['opposite'], 'correct': False}]
    assert [bool(set(row['selected_origins']) & acceptable) for row in raw_orders] == [True, False]
    # Exercise the actual export path when both raw and stable selector outputs are absent.
    protocol = {'pools': [{'id': 'p01', 'candidates': ['e4_original'], 'judge': '4b', 'fallback': 'e4_original'}]}
    controls = [{'id': 'roles', 'source': '娘が父親を駅まで車で送りました。',
                 'candidates': {'faithful': 'The daughter drove her father to the station.',
                               'reversed': 'The father drove his daughter to the station.'},
                 'review_only': {'acceptable': ['faithful']}}]
    rubric = {7: {'required_propositions': ['The daughter drove her father to the station.']}}
    corpus = {7: {'id': 7, 'japanese_reference': '娘が父親を駅まで車で送りました。'}}
    failed_selection = {'status': 'error', 'candidate_mapping': [], 'primary': None,
                        'order_stable': None, 'orders': [], 'order_sensitive': False}
    control_map = [{'id': 'a', 'origins': ['faithful']}, {'id': 'b', 'origins': ['reversed']}]
    raw_orders = [{'result': {'status': 'ok', 'text': 'a'}}, {'result': {'status': 'ok', 'text': 'b'}}]
    control_report = [{'id': 'roles', 'result': {'candidate_mapping': control_map, 'orders': raw_orders}}]
    reports = {
        'judge-4b': {'cases': [{'id': 7, 'pools': {'p01': failed_selection}}], 'controls': control_report},
        'judge-2b': {'cases': [{'id': 7, 'pools': {}}], 'controls': control_report},
    }
    fake_candidates = {7: {'e4_original': {'status': 'ok', 'text': 'The daughter drove her father to the station.',
                                            'inference_seconds': 1.0}}}
    packet, key = build_exports(protocol, controls, rubric, corpus, reports, fake_candidates, {},
        artifact_hashes={'generation_reports': {}, 'judge_reports': {}})
    selected = key['cases']['7']['actual_selections']['p01']
    assert selected['primary_output_id'] is None and selected['stable_output_id'] is None
    blind = json.dumps(packet, ensure_ascii=False)
    assert not any(word in blind for word in ('e4_original', 'e2_natural', 'primary_output_id', 'order_stable'))
    raw_control = key['controls']['roles']['judge_orders']['4b']
    assert [row['correct'] for row in raw_control] == [True, False]
    assert raw_control[0]['selected_output_id'] != raw_control[1]['selected_output_id']
    print('Review-export self-test passed: exact deduplication, origin blinding, failed primary/stable export, and independent raw-order control grading.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.export:
        protocol, controls, rubric, corpus, reports, candidates, provenance = load_artifacts()
        packet, key = build_exports(protocol, controls, rubric, corpus, reports, candidates, provenance)
        write_new(PACKET_PATH, packet)
        write_new(KEY_PATH, key)
        print(json.dumps({'status': 'exported', 'packet': str(PACKET_PATH.relative_to(ROOT)),
                          'private_key': str(KEY_PATH.relative_to(ROOT)),
                          'packet_sha256': key['packet_sha256'],
                          'unique_english_outputs': sum(len(c['outputs']) for c in packet['cases']),
                          'case_denominator': 20, 'inference_performed': False}))
        return
    if args.score:
        packet, key, grades = read_json(PACKET_PATH), read_json(KEY_PATH), read_json(GRADE_PATH)
        if digest(packet) != key['packet_sha256']:
            raise ValueError('Blind review packet changed after export.')
        result = score(packet, key, grades)
        write_new(SCORE_PATH, result)
        print(json.dumps({'status': result['status'], 'scores': str(SCORE_PATH.relative_to(ROOT)),
                          'pools': len(result['pool_scores']), 'inference_performed': False}))
        return
    parser.error('Choose --self-test, --export or --score.')


if __name__ == '__main__':
    main()
