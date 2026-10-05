"""Controlled recognition/boundary and previous-source-context comparisons."""
import argparse
import hashlib
import inspect
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import evaluate_streaming_translation as translation
import replay_streaming_boundaries as boundary

ROOT = translation.ROOT
BASELINE = ROOT / 'test-results/accuracy-streaming-repaired-translation-development.json'
CONTROLS = ROOT / 'tests/fixtures/accuracy-context-controls.json'
POLICY = 'sentence_terminal_eof'


def previous_sentence(text):
    ends = sorted(boundary.sentence_boundaries(text))
    if not ends:
        return ''
    sentence = text[ends[-2] if len(ends) > 1 else 0:ends[-1]].strip()
    # Keep a complete sentence; clipping a long antecedent would change this test.
    return sentence if len(sentence) <= 150 else ''


class SourceContext:
    """Per-recording source-only history; never includes generated English."""
    def __init__(self, original, enabled):
        self.original = original
        self.enabled = enabled
        self.previous = ''
        self.calls = []

    def __call__(self, segments, context='', preset='gemma4', glossary=None):
        actual = self.previous if self.enabled else ''
        messages = translation.gemma._messages(segments, actual, glossary)
        self.calls.append({
            'context_sent': actual,
            'prompt_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
            'source_texts': [s['text'] for s in segments],
        })
        result = self.original(segments, context=actual, preset=preset, glossary=glossary)
        if self.enabled:
            self.previous = previous_sentence(''.join(s['text'] for s in segments))
        return result


def validated_inputs(asr_path, replay_path):
    manifest, frozen = translation.load_development_cases()
    asr, by_id, _ = translation.validate_asr_report(asr_path, manifest, frozen)
    replay = json.loads(replay_path.read_text())
    provenance = replay['provenance']
    if (replay.get('status') != 'complete'
            or provenance.get('partition') != 'development'
            or provenance.get('manifest_sha256') != translation.sha256(translation.CORPUS_PATH)
            or provenance.get('asr_report_sha256') != translation.sha256(asr_path)):
        raise ValueError('Replay provenance does not match the completed development ASR.')
    recomputed = boundary.replay_policy(POLICY, asr['cases'])
    if recomputed != replay['policies'][POLICY]['cases']:
        raise ValueError('Saved canonical units differ from an independent replay.')
    candidates = {str(c['id']): c for c in recomputed}
    updates = {}
    for case in frozen:
        key = str(case['id'])
        row = candidates[key]
        raw = by_id[key]['streaming']['updates']
        grouped = translation.group_policy_units(row['events'], row['units'], key)
        updates[key] = []
        for index, (event, source, units) in enumerate(zip(row['events'], raw, grouped, strict=True)):
            if event['emitted_units'] != units or event['update_index'] != index:
                raise ValueError('Duplicate or misassigned source units.')
            if event['audio_end_seconds'] != source['audio_end_seconds'] or event['inference_seconds'] != source['inference_seconds']:
                raise ValueError('Replay changed an ASR observation clock or cost.')
            for unit in units:
                if not 0 <= unit['source_start_seconds'] < unit['source_end_seconds'] <= unit['emitted_at_audio_seconds'] <= case['duration_seconds'] + 1e-4:
                    raise ValueError('Noncausal or invalid source unit clocks.')
            updates[key].append({k: event[k] for k in ('audio_end_seconds', 'inference_seconds', 'emitted_units')})
    return frozen, asr, replay, candidates, updates


def aggregate(cases):
    outputs = [out for case in cases for out in (case.get('replay') or {}).get('outputs', [])]
    latencies = [out['simulated_latency_after_source_end_seconds'] for out in outputs]
    return {
        'case_count': len(cases),
        'error_count': sum(c['status'] == 'error' for c in cases),
        'forced_fail_count': sum(c['status'].startswith('forced_fail') for c in cases),
        'output_units': len(outputs),
        'measured_asr_seconds': sum((c.get('replay') or {}).get('measured_asr_seconds', 0) for c in cases),
        'measured_translation_seconds': sum((c.get('replay') or {}).get('measured_translation_seconds', 0) for c in cases),
        'translation_calls': sum(len(u['translation_calls']) for c in cases for u in (c.get('replay') or {}).get('updates', [])),
        'nonempty_context_calls': sum(bool(call['context_sent']) for c in cases for u in (c.get('replay') or {}).get('updates', []) for call in u['translation_calls']),
        'simulated_caption_delay_seconds': {'median': statistics.median(latencies) if latencies else None, 'max': max(latencies) if latencies else None},
        'first_english_before_eof_cases': sum(bool((c.get('replay') or {}).get('outputs')) and c['replay']['outputs'][0]['scheduled_translation_complete_audio_seconds'] < c['duration_seconds'] for c in cases),
        'meaning_accuracy': 'Requires independent semantic review; no automatic pass score.',
    }


def run(asr_name, context_name):
    prefix = 'accuracy-streaming' if asr_name == 'medium' else f'accuracy-streaming-{asr_name}'
    asr_path = ROOT / f'test-results/{prefix}-speech-development.json'
    replay_path = ROOT / f'test-results/{prefix}-boundary-replay.json'
    output = ROOT / f'test-results/accuracy-streaming-{asr_name}-{context_name}-translation-development.json'
    if output.exists():
        raise ValueError(f'Preserve existing result: {output}')
    if translation.port_accepting_connections(translation.PORT):
        raise ValueError('Isolated translation port is occupied; refusing contention.')
    controls = json.loads(CONTROLS.read_text()) if context_name == 'previous-source' else []
    if not isinstance(controls, list) or any(not all(key in c for key in ('id', 'previous_source', 'current_source')) for c in controls):
        raise ValueError('Invalid context diagnostic fixture.')
    frozen, asr, replay, candidates, updates = validated_inputs(asr_path, replay_path)
    model = translation.EVALUATOR.model_info()
    baseline = json.loads(BASELINE.read_text())
    for key in ('preset', 'provenance', 'prompt_builder_source_sha256', 'device', 'threads', 'runtime'):
        if model[key] != baseline['model'][key]:
            raise ValueError(f'Translator changed versus baseline: {key}')
    if baseline['translation_policy']['engine_gemma_segments_source_sha256'] != hashlib.sha256(inspect.getsource(translation.engine.gemma_segments).encode()).hexdigest():
        raise ValueError('Production translation grouping changed versus baseline.')
    report = {
        'schema_version': 1, 'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'partition': 'development', 'heldout_rows_used': 0,
        'runner_sha256': translation.sha256(Path(__file__)),
        'source_asr': {'path': str(asr_path.relative_to(ROOT)), 'sha256': translation.sha256(asr_path), 'model': asr['model']},
        'boundary_replay': {'path': str(replay_path.relative_to(ROOT)), 'sha256': translation.sha256(replay_path), 'selected_policy': POLICY, 'summary': replay['policies'][POLICY]['summary']},
        'corpus': {'manifest_sha256': translation.sha256(translation.CORPUS_PATH), 'case_ids': [c['id'] for c in frozen]},
        'baseline': {'path': str(BASELINE.relative_to(ROOT)), 'sha256': translation.sha256(BASELINE)},
        'model': model,
        'translation_policy': {'context': context_name, 'thinking': 'off', 'temperature': 0, 'glossary': 'none',
            'context_rule': 'For previous-source only: one preceding completed recognized Japanese sentence, at most150 characters; otherwise empty. Reset each recording. Current-call units stay in their existing batch.',
            'grouping': 'Reuse engine.gemma_segments and per-update canonical source grouping unchanged.',
            'references': 'Review-only; never passed to the translator.',
            'latency': 'Serial reconstruction from offline measured ASR and translation calls; not browser latency.'},
        'stage': {'status': 'running', 'cases': [], 'errors': []},
    }
    stage = report['stage']
    translation.write_json(output, report)
    original = translation.gemma.translate_batch
    try:
        warm = time.monotonic()
        original([{'start': 0., 'end': .8, 'text': 'こんにちは。'}], preset='gemma4')
        report['warmup'] = {'excluded': True, 'seconds': time.monotonic() - warm}
        for case in frozen:
            key = str(case['id'])
            policy = candidates[key]
            row = {k: case[k] for k in ('id', 'partition', 'duration_seconds', 'audio_sha256', 'japanese_reference', 'english_reference')}
            row.update(policy_alignment_status=policy['transcript_alignment_status'], protocol_status=policy['protocol_status'],
                       canonical_source_unit_count=len(policy['units']), boundary_conflicts=policy['boundary_conflicts'])
            adapter = SourceContext(original, context_name == 'previous-source')
            try:
                result = translation.simulate_case('la2_stable_commit_batches', updates[key], adapter)
                logged = [call for update in result['updates'] for call in update['translation_calls']]
                for call, actual in zip(logged, adapter.calls, strict=True):
                    call.update(actual)  # Record the prompt/context actually sent, not the outer empty-context adapter.
                if len(result['outputs']) != len(policy['units']):
                    raise ValueError('Canonical source units did not map exactly once to output.')
                valid = policy['protocol_status'] == 'pass' and policy['final_raw_text_equals_fused_text'] and bool(result['outputs'])
                row.update(replay=result, status='ok' if valid else 'forced_fail_source_coverage')
            except Exception as error:
                row.update(status='error', error=f'{type(error).__name__}: {error}', replay=None, attempted_calls=adapter.calls)
                stage['errors'].append({'id': case['id'], 'error': row['error']})
            finally:
                translation.gemma.translate_batch = original
            stage['cases'].append(row)
            translation.write_json(output, report)
            print(json.dumps({'variant': f'{asr_name}/{context_name}', 'completed_cases': len(stage['cases']), 'id': case['id']}), flush=True)
        stage['aggregate'] = aggregate(stage['cases'])
        stage['status'] = report['status'] = 'complete_with_errors' if stage['errors'] else 'complete'
        if context_name == 'previous-source':
            report['context_controls'] = {'fixture_sha256': translation.sha256(CONTROLS), 'cases': [], 'excluded_from_development_accuracy': True}
            for control in controls:
                source = [{'start': 0., 'end': 1., 'text': control['current_source']}]
                adapter = SourceContext(original, True)
                adapter.previous = previous_sentence(control['previous_source'])
                try:
                    without = original(source, context='', preset='gemma4')
                    with_context = adapter(source)
                    result = dict(control, baseline=without, contextual=with_context, actual_calls=adapter.calls, status='ok')
                except Exception as error:
                    result = dict(control, status='error', error=f'{type(error).__name__}: {error}')
                    report['status'] = 'complete_with_errors'
                report['context_controls']['cases'].append(result)
                translation.write_json(output, report)
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        translation.gemma.translate_batch = original
        process = translation.gemma._process
        owned = process is None or translation.gemma._owner_pid == os.getpid()
        translation.gemma.shutdown()
        stage['cleanup'] = {'owner_matches_this_evaluator': owned,
            'owned_process_pid': process.pid if process else None,
            'owned_process_released': process is None or process.poll() is not None,
            'port_accepting_connections_after_shutdown': translation.port_accepting_connections(translation.PORT)}
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        translation.write_json(output, report)


def self_test():
    seen = []
    def fake(segments, **kwargs):
        seen.append(kwargs['context'])
        return ['ENGLISH MUST NEVER BE CONTEXT'] * len(segments)
    def source(text):
        return [{'start': 0., 'end': 1., 'text': text}]
    translator = SourceContext(fake, True)
    translator(source('美咲は鍵を探しています。'))
    translator(source('まだ見つかりません。'), context='FUTURE OR ENGLISH MUST NOT ENTER')
    assert seen == ['', '美咲は鍵を探しています。']
    SourceContext(fake, True)(source('新しい話です。'))
    assert seen[-1] == ''
    SourceContext(fake, False)(source('文です。'), context='Earlier batch context must also be disabled')
    assert seen[-1] == ''
    assert previous_sentence('最初。最後。未完成') == '最後。'
    assert previous_sentence('あ' * 151 + '。') == ''
    assert previous_sentence('未完成') == ''
    print('Previous-source causality, reset, incomplete-sentence and length checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--asr', choices=('medium', 'turbo', 'kotoba'), default='turbo')
    parser.add_argument('--context', choices=('none', 'previous-source'), default='none')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        result = run(args.asr, args.context)
        print(json.dumps({'status': result['status'], **result['stage']['aggregate']}))
