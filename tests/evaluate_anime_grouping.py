#!/usr/bin/env python3
"""Frozen38-call E4B wording/grouping diagnostic on one correlated anime clip."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import evaluate_e4b_text_wording as wording

ensemble, translation, gemma = wording.ensemble, wording.translation, wording.gemma
PROTOCOL = wording.ROOT / 'tests/fixtures/accuracy-anime-grouping.json'
CONDITIONS = ('original_dialogue', 'neutral_text')


def validate_rows(rows, manifest):
    if (manifest['status'] != 'inputs_frozen_before_translation' or manifest['group_count'] != 1
            or manifest['segment_inputs'] != 18 or manifest['whole_clip_inputs'] != 1
            or len(rows) != 19 or [row['id'] for row in rows] != manifest['expected_ids']
            or len({row['id'] for row in rows}) != 19
            or len({row['group_id'] for row in rows}) != 1
            or len({row['audio_sha256'] for row in rows}) != 1
            or any(row['partition'] != 'anime_exploratory' for row in rows)):
        raise ValueError('Require all18 segments and one whole input from one frozen anime group.')
    parts, whole = rows[:-1], rows[-1]
    if (any(row['variant'] != 'segment' or row['segment_index'] != index
            for index, row in enumerate(parts, 1)) or whole['variant'] != 'whole_clip'
            or whole['component_ids'] != [row['id'] for row in parts] or whole['join_separator'] != '\n'
            or whole['source'] != '\n'.join(row['source'] for row in parts)):
        raise ValueError('Whole input must be the exact newline-joined original18 segments, in order.')
    if any(not whole['start'] <= row['start'] < row['end'] <= whole['end'] for row in parts):
        raise ValueError('Original segment timestamps must remain within the whole clip.')


def call_order(rows):
    return [{'input_id': row['id'], 'condition': condition} for row in rows for condition in CONDITIONS]


def messages_for(row, condition):
    original = gemma._messages([dict(start=row['start'], end=row['end'], text=row['source'])], '', None)
    if condition == 'original_dialogue':
        return original
    if condition == 'neutral_text':
        return wording.text_messages(original)
    raise ValueError('Unknown wording condition.')


def validated_inputs(protocol_path):
    protocol = json.loads(protocol_path.read_bytes())
    if (protocol['status'] != 'frozen' or protocol['call_count'] != 38
            or protocol['settings'] != wording.SETTINGS or protocol['conditions'] != list(CONDITIONS)
            or protocol['warmup_source'] != {'start': 0, 'end': 1, 'source': 'こんにちは。'}):
        raise ValueError('Require the frozen38-call original/neutral protocol.')
    if protocol['runner_sha256'] != translation.sha256(Path(__file__)):
        raise ValueError('Runner changed after freeze.')
    for binding in protocol['helpers'].values():
        wording.frozen_bytes(binding)
    manifest = json.loads(wording.frozen_bytes(protocol['input_manifest']))
    rows = [json.loads(line) for line in wording.frozen_bytes(manifest['input']).splitlines() if line.strip()]
    for name in ('asr_sources', 'asr_report'):
        wording.frozen_bytes(manifest[name])
    validate_rows(rows, manifest)
    if protocol['call_order'] != call_order(rows):
        raise ValueError('Frozen request order changed.')
    if (gemma.PRESETS['gemma4'] != protocol['model']
            or json.loads(wording.frozen_bytes(protocol['runtime_manifest'])) != protocol['runtime']
            or translation.sha256(gemma.SERVER_PATH) != protocol['runtime_binary_sha256']
            or hashlib.sha256(inspect.getsource(gemma._messages).encode()).hexdigest() != protocol['original_prompt_builder_sha256']):
        raise ValueError('Original model/runtime/prompt provenance changed.')
    by_id = {row['id']: row for row in rows}
    calls = []
    for index, entry in enumerate(protocol['call_order'], 1):
        row = by_id[entry['input_id']]
        messages = messages_for(row, entry['condition'])
        calls.append(dict(sequence=index, **entry, variant=row['variant'], messages=messages,
                          prompt_sha256=ensemble.digest(messages)))
    return protocol, rows, calls


def run(protocol_path):
    protocol, rows, calls = validated_inputs(protocol_path)
    path, port = Path(protocol['output_path']), protocol['settings']['port']
    if path.exists():
        raise ValueError(f'Refusing to overwrite artifact: {path}')
    if gemma._process is not None or translation.port_accepting_connections(port):
        raise ValueError('Private runtime occupied; all model runs must be serial.')
    old_port, old_base = gemma.PORT, gemma.BASE
    gemma.PORT, gemma.BASE = port, f'http://{gemma.HOST}:{port}'
    report = dict(schema_version=1, status='running', partition='anime_exploratory', group_count=1,
        heldout_rows_used=0, created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=dict(path=str(protocol_path), sha256=translation.sha256(protocol_path), contents=protocol),
        inputs=rows, calls=[{k: v for k, v in call.items() if k != 'messages'} | {'result': {'status': 'pending'}} for call in calls],
        limits=['One correlated60s group;38 requests are not38 independent accuracy examples.',
                'Raw frozen ASR only, no references, glossary, context, new ASR, or production change.',
                'Joined English is a whole-clip translation, not timestamp-aligned subtitles.',
                'No independent Japanese reference; no end-to-end accuracy percentage or live-speed claim.'])
    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            created = True
            json.dump(report, stream, ensure_ascii=False, indent=2)
        if not gemma.ready('gemma4') or not gemma.runtime_ready():
            raise ValueError('Require already verified local model/runtime; never download.')
        report['warmup'] = ensemble.complete('4b', messages_for(protocol['warmup_source'], 'original_dialogue'), 0, None, 1024) | {'excluded': True}
        if report['warmup']['status'] != 'ok':
            raise ValueError('Technical warmup failure; no retry or meaning gate.')
        translation.write_json(path, report)
        for call, result in zip(calls, report['calls'], strict=True):
            result['result'] = ensemble.complete('4b', call['messages'], 0, None, 1024)
            translation.write_json(path, report)
        report['status'] = 'complete_with_errors' if any(call['result']['status'] != 'ok' for call in report['calls']) else 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        process = gemma._process
        owner = process is None or gemma._owner_pid == os.getpid()
        gemma.shutdown()
        report['cleanup'] = dict(owner_matches_this_evaluator=owner, owned_process_pid=process.pid if process else None,
            owned_process_released=process is None or process.poll() is not None,
            port_accepting_connections_after_shutdown=translation.port_accepting_connections(port))
        if not owner or not report['cleanup']['owned_process_released'] or report['cleanup']['port_accepting_connections_after_shutdown']:
            report.update(status='failed', cleanup_error='Private runtime did not finish cleanly.')
        gemma.PORT, gemma.BASE = old_port, old_base
        for call in report['calls']:
            if call['result']['status'] == 'pending':
                call['result'] = dict(status='error', error='Run stopped before this call; retained in denominator.')
        report['aggregate'] = dict(call_count=38, group_count=1,
            error_count=sum(call['result']['status'] != 'ok' for call in report['calls']),
            inference_seconds_excluding_warmup=sum(call['result'].get('inference_seconds', 0) for call in report['calls']))
        if created:
            translation.write_json(path, report)
    return report


def self_test():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    rows = [dict(id=str(i), group_id='one', audio_sha256='audio', partition='anime_exploratory',
                 variant='segment', segment_index=i, start=i, end=i + .5, source=f'文{i}。') for i in range(1, 19)]
    rows.append(dict(id='whole', group_id='one', audio_sha256='audio', partition='anime_exploratory',
        variant='whole_clip', start=0, end=20, component_ids=[r['id'] for r in rows], join_separator='\n', source='\n'.join(r['source'] for r in rows)))
    manifest = dict(status='inputs_frozen_before_translation', group_count=1, segment_inputs=18,
                    whole_clip_inputs=1, expected_ids=[row['id'] for row in rows])
    validate_rows(rows, manifest)
    assert len(call_order(rows)) == 38
    assert messages_for(rows[0], 'original_dialogue')[1:] == messages_for(rows[0], 'neutral_text')[1:]
    for mutation in ('join', 'order', 'coverage'):
        changed = copy.deepcopy(rows)
        if mutation == 'join':
            changed[-1]['source'] += '追加'
        elif mutation == 'order':
            changed[0], changed[1] = changed[1], changed[0]
        else:
            changed.pop()
        try:
            validate_rows(changed, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid {mutation}.')
    with TemporaryDirectory() as directory:
        protocol_path, output = Path(directory) / 'protocol.json', Path(directory) / 'report.json'
        protocol_path.write_text('{}')
        protocol = dict(output_path=str(output), settings=wording.SETTINGS,
                        warmup_source=dict(start=0, end=1, source='こんにちは。'))
        calls = [entry | {'messages': [], 'variant': 'segment'} for entry in call_order(rows)]
        with (patch(__name__ + '.validated_inputs', return_value=(protocol, rows, calls)),
              patch.object(gemma, '_process', None), patch.object(gemma, 'ready', return_value=True),
              patch.object(gemma, 'runtime_ready', return_value=True), patch.object(gemma, 'shutdown') as shutdown,
              patch.object(translation, 'port_accepting_connections', return_value=False),
              patch.object(ensemble, 'complete', return_value={'status': 'error', 'error': 'simulated'}) as call):
            try:
                run(protocol_path)
            except ValueError as error:
                assert 'warmup failure' in str(error)
            else:
                raise AssertionError('Warmup failure was not stopped.')
            report = json.loads(output.read_text())
            assert report['status'] == 'failed' and report['aggregate']['error_count'] == 38
            assert call.call_count == 1 and shutdown.call_count == 1 and report['cleanup']['owned_process_released']
    print('self-test passed: join/order/coverage rejection, same source in both prompts, all38 failure retention and owned cleanup.')


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
        _, rows, calls = validated_inputs(args.protocol)
        print(json.dumps(dict(status='validated', input_count=len(rows), call_count=len(calls), group_count=1, model_started=False)))
    else:
        report = run(args.protocol)
        print(json.dumps(dict(status=report['status'], aggregate=report['aggregate'], cleanup=report['cleanup']), indent=2))
        if report['status'] == 'failed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
