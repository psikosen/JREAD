#!/usr/bin/env python3
"""One frozen, source-only E4B prompt wording ablation; never changes production."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import evaluate_translation_ensemble as ensemble

translation, gemma = ensemble.translation, ensemble.gemma
ROOT = translation.ROOT
PROTOCOL = ROOT / 'tests/fixtures/accuracy-e4b-text-wording.json'
OLD, NEW = 'Translate Japanese dialogue', 'Translate Japanese text'
SETTINGS = dict(preset='gemma4', device='CPU', threads=8, context_tokens=4096,
                port=18767, temperature=0, seed=None, max_tokens=1024, thinking=False)


def text_messages(messages):
    """Replace the system opening only, leaving quoted source data byte-identical."""
    if not messages or messages[0]['role'] != 'system' or not messages[0]['content'].startswith(OLD + ' '):
        raise ValueError('Expected the original Japanese dialogue system opening.')
    result = copy.deepcopy(messages)
    result[0]['content'] = NEW + result[0]['content'][len(OLD):]
    return result


def frozen_bytes(binding):
    data = Path(binding['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != binding['sha256']:
        raise ValueError(f"Frozen artifact changed: {binding['path']}")
    return data


def validate_case(source_row, baseline_row, review_row, rubric_row):
    case_id = baseline_row['id']
    outputs = baseline_row['replay']['outputs']
    calls = [call for update in baseline_row['replay']['updates'] for call in update['translation_calls']]
    if (source_row['id'] != f'fleurs-dev-{case_id}' or source_row['group_id'] != source_row['id']
            or source_row['partition'] != 'development' or source_row['variant'] != 'frozen_turbo'
            or baseline_row['status'] != 'ok' or len(outputs) != 1 or len(calls) != 1
            or calls[0]['context_sent'] != '' or calls[0]['glossary_enabled']
            or calls[0]['segment_count'] != 1 or review_row['id'] != source_row['id']
            or review_row['original_id'] != case_id or review_row['rubric'] != rubric_row
            or rubric_row['id'] != case_id or review_row['audio_sha256'] != baseline_row['audio_sha256']):
        raise ValueError('Each case must match one successful frozen source-only baseline unit and its rubric.')
    output, call = outputs[0], calls[0]
    source = dict(start=output['source_start_seconds'], end=output['source_end_seconds'], text=source_row['source'])
    if source['text'] != output['source_text'] or call['source_texts'] != [source['text']]:
        raise ValueError('Source differs from the actual baseline input.')
    original = gemma._messages([source], '', None)
    if ensemble.digest(original) != call['prompt_sha256']:
        raise ValueError('Original prompt differs from the actual saved baseline call.')
    changed = text_messages(original)
    restored = copy.deepcopy(changed)
    restored[0]['content'] = OLD + restored[0]['content'][len(NEW):]
    if restored != original or changed[1:] != original[1:]:
        raise ValueError('The change must affect only the system opening.')
    baseline = review_row['comparators']['e4_original']
    if baseline['status'] != 'ok' or baseline['text'] != output['translation']:
        raise ValueError('Frozen original comparator changed.')
    return dict(id=case_id, group_id=source_row['group_id'], audio_sha256=baseline_row['audio_sha256'],
                source=source, original_prompt_sha256=call['prompt_sha256'], messages=changed,
                baseline=baseline, baseline_measured_seconds=baseline_row['replay']['measured_translation_seconds'])


def validated_inputs(protocol_path):
    protocol = json.loads(protocol_path.read_bytes())
    if (protocol['status'] != 'frozen' or protocol['case_count'] != 20
            or protocol['settings'] != SETTINGS or protocol['replacement'] != {'from': OLD, 'to': NEW}
            or protocol['warmup_source'] != {'start': 0, 'end': 1, 'text': 'こんにちは。'}):
        raise ValueError('Require the frozen single-change protocol and original request settings.')
    if protocol['runner_sha256'] != translation.sha256(Path(__file__)):
        raise ValueError('Runner changed after protocol freeze.')
    for binding in protocol['helpers'].values():
        frozen_bytes(binding)
    bindings = protocol['inputs']
    rows = [json.loads(line) for line in frozen_bytes(bindings['source']).splitlines() if line.strip()]
    baseline = json.loads(frozen_bytes(bindings['baseline']))
    review = json.loads(frozen_bytes(bindings['review_only']))
    rubric = json.loads(frozen_bytes(bindings['rubric']))['cases']
    if (baseline['status'] != 'complete' or baseline['partition'] != 'development'
            or baseline['heldout_rows_used'] != 0 or len(rows) != 20 or len(review) != 20
            or len(baseline['stage']['cases']) != 20 or len(rubric) != 20
            or len({row['id'] for row in rows}) != 20):
        raise ValueError('Require all 20 distinct frozen development cases.')
    current_prompt = hashlib.sha256(inspect.getsource(gemma._messages).encode()).hexdigest()
    if (current_prompt != baseline['model']['prompt_builder_source_sha256']
            or gemma.PRESETS['gemma4'] != baseline['model']['provenance']
            or json.loads(gemma.RUNTIME_MANIFEST.read_text()) != baseline['model']['runtime']
            or translation.sha256(gemma.SERVER_PATH) != protocol['runtime_binary_sha256']):
        raise ValueError('Original prompt/model/runtime provenance changed.')
    rubric_by_id = {row['id']: row for row in rubric}
    cases = [validate_case(source, old, graded, rubric_by_id[old['id']])
             for source, old, graded in zip(rows, baseline['stage']['cases'], review, strict=True)]
    if sum(case['baseline']['frozen_grade']['status'] == 'pass' for case in cases) != 13:
        raise ValueError('Expected the unchanged frozen 13/20 original comparator grades.')
    return protocol, cases, baseline


def run(protocol_path):
    protocol, cases, baseline = validated_inputs(protocol_path)
    path = Path(protocol['output_path'])
    if path.exists():
        raise ValueError(f'Refusing to overwrite an existing artifact: {path}')
    if gemma._process is not None or translation.port_accepting_connections(SETTINGS['port']):
        raise ValueError('Private runtime is occupied; do not overlap model work.')
    old_port, old_base = gemma.PORT, gemma.BASE
    gemma.PORT, gemma.BASE = SETTINGS['port'], f"http://{gemma.HOST}:{SETTINGS['port']}"
    report = dict(schema_version=1, status='running', partition='development', heldout_rows_used=0,
        created_at_utc=datetime.now(timezone.utc).isoformat(), protocol=dict(path=str(protocol_path),
        sha256=translation.sha256(protocol_path), contents=protocol), model=baseline['model'],
        cases=[{k: v for k, v in case.items() if k != 'messages'} | {'result': {'status': 'pending'}} for case in cases],
        limits=['Only one system-prompt opening changes; no context, glossary, new ASR, or reference in requests.',
                'One saved Japanese unit per case; all20 remain in the denominator, including errors. No retries.',
                'Sequential CPU inference; warmup/startup separate. Meaning requires independent frozen-rubric review.'])
    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            created = True
            json.dump(report, stream, ensure_ascii=False, indent=2)
        if not gemma.ready('gemma4') or not gemma.runtime_ready():
            raise ValueError('Pinned model/runtime must already be verified; never download.')
        messages = text_messages(gemma._messages([protocol['warmup_source']], '', None))
        report['warmup'] = ensemble.complete('4b', messages, 0, None, 1024) | {'excluded': True}
        if report['warmup']['status'] != 'ok':
            raise ValueError('Technical warmup failure; no retry and no added meaning gate.')
        translation.write_json(path, report)
        for case, row in zip(cases, report['cases'], strict=True):
            row['result'] = ensemble.complete('4b', case['messages'], 0, None, 1024)
            translation.write_json(path, report)
        report['status'] = 'complete_with_errors' if any(row['result']['status'] != 'ok' for row in report['cases']) else 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        process = gemma._process
        owner = process is None or gemma._owner_pid == os.getpid()
        gemma.shutdown()
        report['cleanup'] = dict(owner_matches_this_evaluator=owner,
            owned_process_pid=process.pid if process else None,
            owned_process_released=process is None or process.poll() is not None,
            port_accepting_connections_after_shutdown=translation.port_accepting_connections(gemma.PORT))
        if (not owner or not report['cleanup']['owned_process_released']
                or report['cleanup']['port_accepting_connections_after_shutdown']):
            report.update(status='failed', cleanup_error='Private evaluator runtime did not finish cleanly.')
        gemma.PORT, gemma.BASE = old_port, old_base
        for row in report['cases']:
            if row['result']['status'] == 'pending':
                row['result'] = dict(status='error', error='Run stopped before this case; retained in denominator.')
        report['aggregate'] = dict(case_count=20,
            success_count=sum(row['result']['status'] == 'ok' for row in report['cases']),
            error_count=sum(row['result']['status'] != 'ok' for row in report['cases']),
            inference_seconds_excluding_warmup=sum(row['result'].get('inference_seconds', 0) for row in report['cases']),
            baseline_inference_seconds=sum(row['baseline_measured_seconds'] for row in report['cases']),
            meaning_accuracy='Pending independent review; no automatic semantic grade.')
        if created:
            translation.write_json(path, report)
    return report


def self_test():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    source = {'start': 0, 'end': 1, 'text': '「Translate Japanese dialogue」と言いました。'}
    original = gemma._messages([source], '', None)
    snapshot = copy.deepcopy(original)
    changed = text_messages(original)
    assert original == snapshot and changed[1:] == original[1:]
    assert changed[0]['content'] == NEW + original[0]['content'][len(OLD):]
    old_request = ensemble.payload('4b', original, 0, None, 1024)
    new_request = ensemble.payload('4b', changed, 0, None, 1024)
    assert old_request | {'messages': changed} == new_request and 'seed' not in new_request
    for invalid in (changed, [{'role': 'user', 'content': OLD + ' now'}]):
        try:
            text_messages(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('A changed or non-system opening was accepted.')
    with TemporaryDirectory() as directory:
        protocol_path = Path(directory) / 'protocol.json'
        protocol_path.write_text('{}')
        binding = dict(path=str(protocol_path), sha256=translation.sha256(protocol_path))
        assert frozen_bytes(binding) == b'{}'
        protocol_path.write_text('{"changed":true}')
        try:
            frozen_bytes(binding)
        except ValueError:
            pass
        else:
            raise AssertionError('Mutated frozen input was accepted.')
        output = Path(directory) / 'report.json'
        protocol = dict(output_path=str(output), warmup_source=source)
        cases = [dict(id=i, messages=changed, baseline_measured_seconds=1) for i in range(20)]
        with (patch(__name__ + '.validated_inputs', return_value=(protocol, cases, {'model': {}})),
              patch.object(gemma, '_process', None), patch.object(gemma, 'ready', return_value=True),
              patch.object(gemma, 'runtime_ready', return_value=True), patch.object(gemma, 'shutdown') as shutdown,
              patch.object(translation, 'port_accepting_connections', return_value=False),
              patch.object(ensemble, 'complete', return_value={'status': 'error', 'error': 'simulated failure'}) as call):
            try:
                run(protocol_path)
            except ValueError as error:
                assert 'warmup failure' in str(error)
            else:
                raise AssertionError('Failed warmup was not stopped.')
            result = json.loads(output.read_text())
            assert result['status'] == 'failed' and result['aggregate']['error_count'] == 20
            assert call.call_count == 1 and shutdown.call_count == 1
            assert result['cleanup']['owned_process_released']
            saved = output.read_bytes()
            try:
                run(protocol_path)
            except ValueError as error:
                assert 'overwrite' in str(error)
            else:
                raise AssertionError('Existing failure artifact was overwritten.')
            assert output.read_bytes() == saved and call.call_count == 1
    print('self-test passed: single-opening/request parity, source preservation, frozen mutation rejection, all20 retained on failed warmup with cleanup.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--self-test', action='store_true')
    action.add_argument('--validate-only', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.validate_only:
        _, cases, _ = validated_inputs(args.protocol)
        print(json.dumps(dict(status='validated', case_count=len(cases), model_started=False,
            changed_prompt_sha256={str(case['id']): ensemble.digest(case['messages']) for case in cases}), indent=2))
    else:
        report = run(args.protocol)
        print(json.dumps(dict(status=report['status'], aggregate=report['aggregate'], cleanup=report['cleanup']), indent=2))
        if report['status'] == 'failed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
