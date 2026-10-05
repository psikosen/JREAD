#!/usr/bin/env python3
"""Compute development-only candidate-pool oracle ceilings from review JSONs."""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test-results/accuracy-oracle-ceilings.json'
EXPECTED = 20


def load_review(filename: str) -> tuple[Path, dict]:
    path = ROOT / 'test-results' / filename
    data = json.loads(path.read_text())
    if data.get('status') != 'complete' or data.get('partition') != 'development':
        raise ValueError(f'{filename} is not a completed development-only review.')
    return path, data


def pass_ids(review: dict, *, stage: str | None = None, status_field: str = 'status') -> tuple[set[int], set[int]]:
    rows = review['stages'][stage]['cases'] if stage else review['cases']
    if len(rows) != EXPECTED or any(row.get('partition') not in (None, 'development') for row in rows):
        raise ValueError(f'{stage or "review"} must contain exactly 20 development rows.')
    ids = [int(row['id']) for row in rows]
    if len(set(ids)) != EXPECTED:
        raise ValueError(f'{stage or "review"} contains duplicate IDs.')
    passed = {int(row['id']) for row in rows if row.get(status_field) == 'pass'}
    failed = {int(row['id']) for row in rows if row.get(status_field) == 'fail'}
    if passed | failed != set(ids) or passed & failed:
        raise ValueError(f'{stage or "review"} contains an unexpected case status.')
    return passed, failed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(path_name: str, *, stage: str | None = None, status_field: str = 'status') -> tuple[dict, set[int], set[int]]:
    path, review = load_review(path_name)
    passed, failed = pass_ids(review, stage=stage, status_field=status_field)
    info = {
        'review_path': str(path.relative_to(ROOT)),
        'review_sha256': sha256(path),
        'review_type': review.get('review_type'),
        'pass_count': len(passed),
        'fail_count': len(failed),
        'pass_ids': sorted(passed),
        'fail_ids': sorted(failed),
    }
    return info, passed, failed


def pool(candidates: dict[str, tuple[dict, set[int], set[int]]]) -> dict:
    pass_sets = {name: value[1] for name, value in candidates.items()}
    fail_sets = {name: value[2] for name, value in candidates.items()}
    denominators = [passed | failed for passed, failed in ((value[1], value[2]) for value in candidates.values())]
    if not denominators or any(ids != denominators[0] for ids in denominators):
        raise ValueError('Candidates in a pool do not have identical 20-case denominators.')
    all_ids = denominators[0]
    union = set.union(*pass_sets.values())
    intersection = set.intersection(*pass_sets.values())
    names = list(pass_sets)
    pairwise = {}
    for first, second in itertools.combinations(names, 2):
        overlap = pass_sets[first] & pass_sets[second]
        pairwise[f'{first} ∩ {second}'] = {'count': len(overlap), 'ids': sorted(overlap)}
    baseline_name = names[0]
    baseline_failures = sorted(fail_sets[baseline_name])
    recovery = {
        str(case_id): [name for name in names[1:] if case_id in pass_sets[name]]
        for case_id in baseline_failures
    }
    recovery_unresolved = [int(case_id) for case_id, alternatives in recovery.items() if not alternatives]

    target_subsets = []
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            subset_union = set.union(*(pass_sets[name] for name in subset))
            if len(subset_union) >= 18:
                target_subsets.append({'candidates': list(subset), 'pass_count': len(subset_union), 'pass_ids': sorted(subset_union)})
        if target_subsets:
            break

    return {
        'denominator': len(all_ids),
        'case_ids': sorted(all_ids),
        'candidate_counts': {name: candidates[name][0]['pass_count'] for name in names},
        'oracle_union': {'pass_count': len(union), 'fail_count': len(all_ids - union), 'pass_ids': sorted(union), 'remaining_ids': sorted(all_ids - union)},
        'intersection_all_candidates': {'pass_count': len(intersection), 'pass_ids': sorted(intersection)},
        'pairwise_pass_intersections': pairwise,
        'baseline_candidate': baseline_name,
        'baseline_fail_ids': baseline_failures,
        'baseline_fail_recovered_by_alternatives': recovery,
        'baseline_fail_unrecovered_by_alternatives': recovery_unresolved,
        'can_oracle_reach_18_of_20': len(union) >= 18,
        'smallest_candidate_subsets_reaching_18': target_subsets,
    }


def main() -> None:
    clean_specs = {
        'e4b_pristine_reference': ('accuracy-development-review.json', 'pristine_reference', 'status'),
        'e4b_precision_prompt': ('accuracy-precise-prompt-review.json', None, 'status'),
        'e4b_fewshot_prompt': ('accuracy-fewshot-prompt-review.json', None, 'status'),
        'e4b_bounded_reasoning': ('accuracy-gemma-reasoning-review.json', None, 'status'),
        'nllb_600m_reference': ('accuracy-nllb-600m-review.json', None, 'status'),
    }
    audio_specs = {
        'e4b_medium_asr_full_clip': ('accuracy-development-review.json', 'full_clip_asr', 'end_to_end_status'),
        'e4b_turbo_asr_segmented': ('accuracy-turbo-review.json', 'turbo_full_clip_asr', 'status'),
        'e4b_turbo_asr_joined': ('accuracy-turbo-review.json', 'turbo_joined_clip_asr', 'status'),
        'whisper_medium_direct_translation': ('accuracy-whisper-direct-review.json', None, 'status'),
    }

    clean = {name: candidate(path, stage=stage, status_field=field) for name, (path, stage, field) in clean_specs.items()}
    audio = {name: candidate(path, stage=stage, status_field=field) for name, (path, stage, field) in audio_specs.items()}
    clean_info = {name: value[0] for name, value in clean.items()}
    audio_info = {name: value[0] for name, value in audio.items()}
    report = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete',
        'scope': 'Development-only perfect-selector upper bounds from existing semantic review pass/fail IDs. No model inference, held-out rows, source answers, references, or raw translation artifacts are read by this script.',
        'method': 'Within each comparable pool, a case counts as an oracle pass if any candidate has a frozen-review status of pass. An oracle fail is a case where every candidate fails. This assumes a perfect case-level selector that knows the review labels; it does not measure a practical reranker.',
        'pools': {
            'clean_japanese_reference_input': {
                'input_comparability': 'All five candidates were translated from the same 20 pristine Japanese development references, one whole-clip text input per case; no audio/ASR.',
                'baseline_is_first_candidate': True,
                'candidates': clean_info,
                'summary': pool(clean),
            },
            'actual_audio_input': {
                'input_comparability': 'All four candidates use the same 20 development recordings. Medium/Turbo candidates use ASR plus E4B translation; direct Whisper translates audio to English without the same intermediate ASR transcript.',
                'baseline_is_first_candidate': True,
                'candidates': audio_info,
                'summary': pool(audio),
            },
        },
        'limits': [
            'All reviewed scores are model-assisted, single-run development judgments, not human-certified accuracy estimates.',
            'Oracle ceilings assume access to the correct per-case candidate after the fact; no selector was trained or evaluated.',
            'Turbo segmented and joined modes share the same turbo ASR transcript, so they are correlated candidates, not independent votes.',
            'Turbo joined mode assigns one coarse outer timestamp span; 10/20 spans exceed 8 seconds and 2/20 exceed 12 seconds, so it is not a live-timing-equivalent candidate.',
            'ASR/source fidelity findings are separate from English translation fidelity; audio pool pass IDs use end-to-end review labels.',
            'Review labels are treated as frozen inputs; audited label corrections require regenerating this report from the corrected review artifacts.',
        ],
    }
    report['script_sha256'] = sha256(Path(__file__))
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        key: {'oracle_pass_count': value['summary']['oracle_union']['pass_count'],
              'remaining_ids': value['summary']['oracle_union']['remaining_ids'],
              'can_reach_18': value['summary']['can_oracle_reach_18_of_20']}
        for key, value in report['pools'].items()
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
