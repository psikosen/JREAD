#!/usr/bin/env python3
"""Translate frozen ASR top-three candidates with exact-request cache reuse."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import evaluate_translation_ensemble as ensemble  # noqa: E402

translation = ensemble.shared.translation
gemma = ensemble.gemma
PROTOCOL = ROOT / 'tests/fixtures/accuracy-asr-recovery-translation.json'
LOCK = Path('/media/psikosen/newvolume/jread-verifier/evaluation.lock')
ASR_REPORT = Path('/media/psikosen/newvolume/jread-asr-recovery/asr/report.json')
CACHE_PACKET = Path('/media/psikosen/newvolume/jread-asr-recovery/data/translation-cache-v1.json')
EXPECTED_SETTINGS = {
    'preset': 'gemma4', 'device': 'CPU', 'threads': 8, 'context_tokens': 4096,
    'port': 18767, 'temperature': 0, 'seed': None, 'max_tokens': 1024, 'thinking': False,
}
MODEL_SLOT = '4b'
RANKS = (0, 1, 2)
TIMEOUT_SECONDS = 1800
ASR_PROTOCOL = ROOT / 'tests/fixtures/accuracy-asr-recovery.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def payload_bytes(request):
    return json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def binding(item):
    path = Path(item['path'])
    actual = sha(path)
    if actual != item['sha256']:
        raise ValueError(f'Frozen input changed: {path}')
    return json.loads(path.read_text())


def model_record():
    manifest = json.loads(gemma.RUNTIME_MANIFEST.read_text())
    return {
        'preset': 'gemma4',
        'provenance': gemma.PRESETS['gemma4'],
        'runtime': manifest,
        'runtime_binary_sha256': sha(gemma.SERVER_PATH),
        'prompt_builder_source_sha256': hashlib.sha256(inspect.getsource(gemma._messages).encode()).hexdigest(),
    }


def request_for(source, settings):
    messages = gemma._messages([source], '', None)
    request = ensemble.payload(MODEL_SLOT, messages, settings['temperature'], settings['seed'],
                              settings['max_tokens'])
    encoded = payload_bytes(request)
    return messages, request, encoded, hashlib.sha256(encoded).hexdigest()


def cache_row(row, expected_source, settings, model):
    if set(row['source']) != {'start', 'end', 'text'} or row['source'] != expected_source:
        raise ValueError('Cached baseline source differs from its frozen source object.')
    if not isinstance(row.get('text'), str) or not row['text'].strip():
        raise ValueError('Cached baseline translation must be nonempty.')
    seconds = row.get('inference_seconds')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Cached baseline timing is invalid.')
    prov = row.get('provenance')
    if not valid_provenance(prov):
        raise ValueError('Cached baseline provenance is incomplete.')
    _, expected, encoded, digest = request_for(expected_source, settings)
    if (row.get('request_sha256') != digest or row.get('request') != expected
            or payload_bytes(row['request']) != encoded):
        raise ValueError('Cached baseline is not the exact current source-only Gemma request.')
    if model is None:
        raise AssertionError('unreachable')
    return digest, encoded


def valid_provenance(prov):
    if not isinstance(prov, dict):
        return False
    digest = prov.get('sha256')
    return (isinstance(prov.get('path'), str) and isinstance(digest, str) and len(digest) == 64
            and all(ch in '0123456789abcdef' for ch in digest))


def valid_seconds(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def prepare(protocol_path=PROTOCOL):
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get('status') != 'frozen' or protocol.get('runner_sha256') != sha(__file__):
        raise ValueError('Require frozen protocol bound to this runner.')
    if protocol.get('settings') != EXPECTED_SETTINGS:
        raise ValueError('The fixed E4B CPU8 generation settings changed.')
    if Path(protocol['asr_report']['path']) != ASR_REPORT or Path(protocol['cache_packet']['path']) != CACHE_PACKET:
        raise ValueError('Unexpected ASR report or cache packet path.')
    asr = binding(protocol['asr_report'])
    cache = binding(protocol['cache_packet'])
    for item in protocol.get('bindings', []):
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'Frozen model/runtime/helper binding changed: {item["path"]}')

    current_model = model_record()
    if cache.get('status') != 'frozen_public_translation_cache' or cache.get('model') != current_model:
        raise ValueError('Public cache packet does not match the current pinned Gemma model/runtime.')
    cache_settings = {'temperature': 0, 'seed': None, 'max_tokens': 1024,
                      'context': '', 'glossary': False, 'thinking': False}
    if cache.get('settings') != cache_settings:
        raise ValueError('Public cache packet used different original request settings.')

    cases = asr.get('cases')
    validate_asr_report(asr, protocol, cases)
    ids = [case.get('id') for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(set(ids)) != 20:
        raise ValueError('ASR report must contain20 unique string case IDs.')
    baselines = cache.get('baselines')
    entries = cache.get('entries')
    if (not isinstance(baselines, list) or len(baselines) != 20 or not isinstance(entries, list)
            or len(entries) != 87):
        raise ValueError('Require20 original baselines and the frozen87-entry public request cache.')
    baseline_by_id = {row.get('id'): row for row in baselines}
    if set(baseline_by_id) != set(ids) or len(baseline_by_id) != 20:
        raise ValueError('ASR IDs and saved baseline IDs differ.')

    cache_by_hash = {}
    cache_bytes = {}
    for row in entries:
        request = row.get('request')
        if not isinstance(request, dict) or not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Public request cache contains an invalid entry.')
        encoded = payload_bytes(request)
        digest = hashlib.sha256(encoded).hexdigest()
        if (row.get('request_sha256') != digest or digest in cache_by_hash
                or not valid_seconds(row.get('inference_seconds'))):
            raise ValueError('Public request cache has a wrong hash, duplicate, or invalid timing.')
        prov = row.get('provenance')
        if not valid_provenance(prov):
            raise ValueError('Public request cache provenance is incomplete.')
        cache_by_hash[digest] = row
        cache_bytes[digest] = encoded

    provenance_files = {}
    for row in [*baselines, *entries]:
        prov = row['provenance']
        path = Path(prov['path'])
        expected = prov['sha256']
        if path not in provenance_files:
            provenance_files[path] = sha(path)
        if provenance_files[path] != expected:
            raise ValueError(f'Cached output provenance file changed: {path}')

    for case_id, row in baseline_by_id.items():
        source = row.get('source')
        if not isinstance(source, dict) or set(source) != {'start', 'end', 'text'}:
            raise ValueError(f'Baseline source for {case_id} is malformed.')
        base_hash, base_bytes = cache_row(row, source, protocol['settings'], current_model)
        if base_hash in cache_by_hash and cache_bytes[base_hash] != base_bytes:
            raise ValueError('Baseline/cache request hash collision.')

    cache_packet = {'baselines': baseline_by_id, 'entries': cache_by_hash,
                    'entry_bytes': cache_bytes, 'model': current_model}
    plan = build_plan(cases, baseline_by_id, cache_packet, protocol['settings'])
    out_dir = Path(protocol['output_dir'])
    return protocol, asr, cache, current_model, plan, out_dir


def validate_asr_report(report, translation_protocol, cases):
    """Bind only the audio-only ASR artifact and its owned worker evidence."""
    if report.get('status') not in ('complete', 'complete_with_errors') or report.get('phase') != 'asr':
        raise ValueError('Require a completed main ASR phase, retaining per-case failures if any.')
    if not isinstance(cases, list) or len(cases) != 20 or report.get('expected_cases') != 20:
        raise ValueError('Require all20 ASR case rows.')
    asr_protocol = json.loads(ASR_PROTOCOL.read_text())
    asr_protocol_hash = sha(ASR_PROTOCOL)
    if (asr_protocol.get('status') != 'frozen' or asr_protocol.get('runner_sha256') != sha(ROOT / 'tests/evaluate_asr_recovery.py')
            or report.get('protocol') != {'path': str(ASR_PROTOCOL), 'sha256': asr_protocol_hash}
            or Path(translation_protocol.get('asr_protocol', {}).get('path', '')) != ASR_PROTOCOL
            or translation_protocol.get('asr_protocol', {}).get('sha256') != asr_protocol_hash):
        raise ValueError('ASR report does not match its currently frozen protocol and runner.')
    expected_packets = {key: asr_protocol[key] for key in ('development_audio', 'controls_audio', 'context_audio')}
    if report.get('packet_bindings') != expected_packets:
        raise ValueError('ASR report packet identities differ from its frozen protocol.')
    worker_binding = report.get('worker')
    if not isinstance(worker_binding, dict):
        raise ValueError('ASR report lacks its worker binding.')
    worker = binding(worker_binding)
    rows = worker.get('rows')
    if (worker.get('phase') != 'asr' or worker.get('status') not in ('complete', 'complete_with_errors')
            or worker.get('model_loaded') is not True or worker.get('inference_performed') is not True
            or worker.get('cuda_initialized') is not True or worker.get('model_device') != 'cuda'
            or worker.get('model_device_index') != '[0]' or worker.get('model_compute_type') != 'int8_float16'
            or not valid_seconds(worker.get('model_load_seconds')) or worker['model_load_seconds'] <= 0
            or not isinstance(rows, list) or len(rows) != 20 or report.get('cases') != rows
            or report.get('worker_exit_code') != 0):
        raise ValueError('ASR worker lacks complete phase, model/load, device or row evidence.')
    cleanup = report.get('cleanup')
    if (not isinstance(cleanup, dict) or cleanup.get('owned_worker_released') is not True
            or cleanup.get('owned_gpu_process_absent') is not True
            or cleanup.get('remaining_group_members') != []
            or cleanup.get('descendants_after_worker_exit') != []):
        raise ValueError('ASR worker cleanup is incomplete.')
    expected_ids = [str(row['id']) for row in cases]
    if ([str(row.get('id')) for row in rows] != expected_ids
            or any(row.get('status') not in ('ok', 'error') for row in rows)):
        raise ValueError('ASR worker rows are missing, duplicated or reordered.')
    return worker


def build_plan(cases, baselines, cache, settings):
    plan, targets, request_rows = [], {}, []
    baseline_requests = {}
    for case_id, base in baselines.items():
        key, encoded = cache_row(base, base['source'], settings, cache['model'])
        if key in baseline_requests and baseline_requests[key]['text'] != base['text']:
            raise ValueError('Conflicting saved baseline outputs for one exact request.')
        baseline_requests[key] = base

    for case in cases:
        case_id = case['id']
        status = case.get('status')
        raw_hypotheses = case.get('raw_hypotheses')
        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = []
        ranked = {}
        for raw in raw_hypotheses:
            rank = raw.get('rank') if isinstance(raw, dict) else None
            if type(rank) is int and rank in RANKS:
                if rank in ranked:
                    raise ValueError(f'Duplicate raw ASR rank {rank} for {case_id}.')
                ranked[rank] = raw
        base = baselines[case_id]
        case_row = {'id': case_id, 'asr_status': status, 'baseline': base, 'candidates': []}
        for rank in RANKS:
            raw = ranked.get(rank)
            text = raw.get('text') if raw else None
            candidate = {'rank': rank, 'candidate_id': None, 'raw_asr_text': text,
                         'raw_hypothesis': raw, 'asr_case_status': status,
                         'source_truncated': bool(raw and (raw.get('potentially_truncated_at_max_length') is True
                                                          or raw.get('truncated') is True)),
                         'request_sha256': None, 'request': None, 'translation_status': 'failed',
                         'translation_text': None, 'result_origin': 'not_generated',
                         'cache_hit': False, 'generated_this_run': False, 'stored_generation': False,
                         'inference_seconds': 0.0, 'startup_seconds': 0.0, 'error': None,
                         'shared_request': False}
            if status not in ('ok', 'error'):
                candidate['error'] = 'ASR case status is not a completed row.'
            elif not isinstance(text, str) or not text.strip():
                candidate['error'] = 'Missing or empty top-three ASR text.'
            else:
                source = dict(start=base['source']['start'], end=base['source']['end'], text=text)
                try:
                    _, request, encoded, digest = request_for(source, settings)
                    candidate.update(candidate_id='c' + digest[:16], request_sha256=digest, request=request)
                    target = {'request': request, 'bytes': encoded, 'digest': digest, 'source': source,
                              'case_id': case_id, 'rank': rank}
                    if digest in baseline_requests and payload_bytes(baseline_requests[digest]['request']) == encoded:
                        target.update(origin='baseline_cache', cached=baseline_requests[digest])
                    elif digest in cache['entries']:
                        entry = cache['entries'][digest]
                        if cache['entry_bytes'][digest] != encoded:
                            raise ValueError('Cache key matches but canonical request bytes differ.')
                        target.update(origin='request_cache', cached=entry)
                    else:
                        target.update(origin='new', cached=None)
                    if digest in targets:
                        if targets[digest]['bytes'] != encoded:
                            raise ValueError('Request digest collision across ASR ranks.')
                        candidate['shared_request'] = True
                    else:
                        targets[digest] = target
                    canonical_target = targets[digest]
                    candidate['result_origin'] = canonical_target['origin']
                    if canonical_target['origin'] in ('baseline_cache', 'request_cache'):
                        cached = canonical_target['cached']
                        candidate.update(translation_status='ok', translation_text=cached['text'],
                            cache_hit=True, stored_generation=True,
                            inference_seconds=cached['inference_seconds'],
                            error=None, cache_provenance=cached['provenance'])
                    elif candidate['shared_request']:
                        candidate['stored_generation'] = True
                    request_rows.append({'case_id': case_id, 'rank': rank,
                                         'candidate_id': candidate['candidate_id'],
                                         'request_sha256': digest, 'request': request,
                                         'result_origin': canonical_target['origin'],
                                         'shared_request': candidate['shared_request']})
                except Exception as error:
                    candidate['error'] = f'{type(error).__name__}: {error}'
            case_row['candidates'].append(candidate)
        plan.append(case_row)
    return {'cases': plan, 'targets': targets, 'requests': request_rows,
            'baseline_requests': baseline_requests}


def public_plan(plan):
    return [{'case_id': c['id'], 'ranks': [
        {k: row.get(k) for k in ('rank', 'candidate_id', 'raw_asr_text', 'raw_hypothesis',
                                'request_sha256', 'request', 'result_origin', 'error')}
        for row in c['candidates']]} for c in plan['cases']]


def port_accepting(port):
    with socket.socket() as probe:
        probe.settimeout(.2)
        return probe.connect_ex((gemma.HOST, port)) == 0


def proc_row(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        tail = raw.rsplit(')', 1)[1].split()
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return {'state': tail[0], 'ppid': int(tail[1]), 'pgrp': int(tail[2]),
                'start': int(tail[19]), 'argv': [part.decode(errors='replace') for part in argv],
                'cmdline': b' '.join(argv).decode(errors='replace')}
    except (OSError, ValueError, IndexError):
        return None


def group_members(pgid):
    found = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        row = proc_row(int(path.parent.name))
        if row and row['state'] != 'Z' and row['pgrp'] == pgid:
            found.append(int(path.parent.name))
    return sorted(found)


def descendants(pid):
    rows = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        child = int(path.parent.name)
        row = proc_row(child)
        if row and row['state'] != 'Z':
            rows[child] = row
    out, frontier = [], [pid]
    while frontier:
        parent = frontier.pop()
        children = [child for child, row in rows.items() if row['ppid'] == parent]
        out.extend(children)
        frontier.extend(children)
    return out


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def release_owned(child, out_dir):
    server_path = out_dir / 'owned-gemma-process.json'
    server_ref = owned_server_ref(server_path)
    worker_before = group_members(child.pid) if child else []
    server_before = group_members(server_ref['pgid']) if server_ref else []
    if child and child.poll() is None:
        signal_group(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if server_ref:
                cleanup_server(server_ref, signal.SIGTERM)
            signal_group(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    server_ref = owned_server_ref(server_path) or server_ref
    if server_ref:
        cleanup_server(server_ref, signal.SIGTERM)
        time.sleep(.1)
        if group_members(server_ref['pgid']):
            cleanup_server(server_ref, signal.SIGKILL)
            time.sleep(.1)
    worker_remaining = group_members(child.pid) if child else []
    if child and worker_remaining:
        signal_group(child.pid, signal.SIGKILL)
        time.sleep(.1)
        worker_remaining = group_members(child.pid)
    server_remaining = group_members(server_ref['pgid']) if server_ref else []
    return {'worker_pid': child.pid if child else None,
        'worker_exit_code': child.returncode if child else None,
        'worker_released': child is None or child.poll() is not None,
        'worker_group_before_cleanup': worker_before,
        'worker_group_remaining': worker_remaining,
        'gemma_server_pid': server_ref['pid'] if server_ref else None,
        'gemma_server_group_before_cleanup': server_before,
        'gemma_server_group_remaining': server_remaining,
        'port_18767_accepting': port_accepting(18767),
        'port_18768_accepting': port_accepting(18768)}


def validated_server_ref(path):
    try:
        data = json.loads(Path(path).read_text())
        row = proc_row(int(data['pid']))
        if not row or row['state'] == 'Z' or row['pgrp'] != int(data['pgid']) or row['start'] != data['start_ticks']:
            return None
        if not valid_server_command(row['argv']):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def owned_server_ref(path):
    """Return only the process-group identity created by this fresh worker directory."""
    try:
        data = json.loads(Path(path).read_text())
        pid, pgid, start = int(data['pid']), int(data['pgid']), int(data['start_ticks'])
        if (pid <= 1 or pgid != pid or data.get('executable') != str(gemma.SERVER_PATH)
                or not isinstance(data.get('command'), list) or not data['command']
                or data['command'][0] != str(gemma.SERVER_PATH)):
            return None
        row = proc_row(pid)
        if row and (row['pgrp'] != pgid or row['start'] != start
                    or row['argv'] != data['command']):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def valid_server_command(argv):
    if not isinstance(argv, list) or not argv or argv[0] != str(gemma.SERVER_PATH):
        return False
    expected = {'--model': str(gemma.model_path('gemma4')), '--alias': gemma.PRESETS['gemma4']['alias'],
        '--host': gemma.HOST, '--port': '18767', '--ctx-size': '4096', '--threads': '8',
        '--n-gpu-layers': '0', '--parallel': '1', '--reasoning-budget': '0',
        '--chat-template-kwargs': '{"enable_thinking":false}'}
    try:
        positions = {arg: index for index, arg in enumerate(argv[:-1])}
        if any(flag not in positions or argv[positions[flag] + 1] != value
               for flag, value in expected.items()):
            return False
    except (IndexError, TypeError):
        return False
    return all(flag in argv for flag in ('--jinja', '--no-webui', '--no-slots', '--offline', '--log-disable'))


def cleanup_server(server_ref, signum):
    if not server_ref:
        return
    row = proc_row(server_ref['pid'])
    if (row is None or (row['pgrp'] == server_ref['pgid'] and row['start'] == server_ref['start_ticks'])):
        signal_group(server_ref['pgid'], signum)


def make_worker_record(protocol, asr, cache, model, plan):
    new_count = sum(t['origin'] == 'new' for t in plan['targets'].values())
    return {
        'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': {'path': str(PROTOCOL), 'sha256': sha(PROTOCOL)},
        'inputs': {'asr_report': {'path': str(ASR_REPORT), 'sha256': sha(ASR_REPORT)},
                   'cache_packet': {'path': str(CACHE_PACKET), 'sha256': sha(CACHE_PACKET)}},
        'partition': 'development', 'heldout_rows_used': 0, 'production_changed': False,
        'model': model, 'settings': protocol['settings'],
        'coverage': {'case_count': 20, 'rank_slots': 60, 'new_unique_requests': new_count,
                     'baseline_cache_hits': sum(t['origin'] == 'baseline_cache' for t in plan['targets'].values()),
                     'request_cache_hits': sum(t['origin'] == 'request_cache' for t in plan['targets'].values())},
        'selection_policy': {'primary_rank': 0,
            'alternatives': 'All ranks 0–2 and the saved baseline are retained for later human review; no automatic selection or semantic grading.'},
        'warmup': None, 'model_startup_seconds': 0.0, 'cases': [], 'request_results': [], 'errors': [],
        'limits': ['Development ASR recovery diagnostic; no heldout or production accuracy claim.',
                   'One serial new request per unique exact prompt; no retries or parallel calls.',
                   'References and grades are never read or sent to the model.',
                   'ASR rank scores are preserved as metadata and are not semantic grades.'],
    }


def _term_handler(signum, frame):
    raise KeyboardInterrupt('Owned worker interrupted for cleanup.')


def worker(protocol_path, output_dir):
    old_port, old_base = gemma.PORT, gemma.BASE
    protocol, asr, cache_data, model, plan, out_dir = prepare(protocol_path)
    if Path(output_dir) != out_dir:
        raise ValueError('Worker output directory differs from frozen protocol.')
    worker_path = out_dir / 'worker.json'
    server_path = out_dir / 'owned-gemma-process.json'
    report = make_worker_record(protocol, asr, cache_data, model, plan)
    write(worker_path, report)
    gemma.PORT = protocol['settings']['port']
    gemma.BASE = f'http://{gemma.HOST}:{gemma.PORT}'
    previous_term = signal.signal(signal.SIGTERM, _term_handler)
    process = None
    try:
        has_new_targets = any(target['origin'] == 'new' for target in plan['targets'].values())
        if has_new_targets:
            for port in (gemma.PORT, 18768):
                if port_accepting(port):
                    raise RuntimeError(f'Private model port {port} is occupied; refusing concurrent inference.')
            if not gemma.ready('gemma4') or not gemma.runtime_ready():
                raise RuntimeError('Pinned Gemma E4B weights/runtime are not ready; this runner never downloads.')
            with gemma._lock:
                startup_started = time.perf_counter()
                gemma._start('gemma4')
                report['model_startup_seconds'] = time.perf_counter() - startup_started
                process = gemma._process
                proc = proc_row(process.pid)
                if proc is None or proc['pgrp'] != process.pid:
                    raise RuntimeError('Could not verify the owned Gemma server process group.')
                if not valid_server_command(proc['argv']):
                    raise RuntimeError('The Gemma server command does not match the frozen CPU8 settings.')
                server_ref = {'pid': process.pid, 'pgid': proc['pgrp'], 'start_ticks': proc['start'],
                              'executable': str(gemma.SERVER_PATH), 'command': proc['argv']}
                write(server_path, server_ref)
            messages = gemma._messages([{'start': 0, 'end': 1, 'text': 'こんにちは。'}], '', None)
            report['warmup'] = ensemble.complete(MODEL_SLOT, messages, 0, None, 1024)
            report['warmup']['excluded'] = True
            write(worker_path, report)
            if report['warmup']['status'] != 'ok':
                raise RuntimeError('The technical warmup failed; no candidate calls were made.')
        else:
            report['warmup'] = {'status': 'skipped', 'excluded': True,
                                'reason': 'All unique candidate requests are exact cache hits; no model call needed.'}

        result_by_hash = {}
        for digest, target in plan['targets'].items():
            if target['origin'] in ('baseline_cache', 'request_cache'):
                cached = target['cached']
                result_by_hash[digest] = {
                    'status': 'ok', 'text': cached['text'], 'request': target['request'],
                    'request_sha256': digest, 'result_origin': target['origin'],
                    'cache_hit': True, 'generated_this_run': False, 'stored_generation': True,
                    'inference_seconds': cached['inference_seconds'], 'startup_seconds': 0.0,
                    'provenance': cached['provenance'],
                }
            elif target['origin'] == 'new':
                messages, request, encoded, actual_hash = request_for(target['source'], protocol['settings'])
                if actual_hash != digest or encoded != target['bytes']:
                    raise RuntimeError('Candidate request changed after frozen preflight.')
                response = ensemble.complete(MODEL_SLOT, messages, 0, None, 1024)
                result_by_hash[digest] = response | {'result_origin': 'generated', 'cache_hit': False,
                    'generated_this_run': True, 'stored_generation': True}
                report['request_results'].append(result_by_hash[digest])
                write(worker_path, report)

        for case in plan['cases']:
            case_out = {'id': case['id'], 'asr_status': case['asr_status'],
                        'saved_baseline': case['baseline'], 'candidates': []}
            for candidate in case['candidates']:
                row = dict(candidate)
                digest = row.get('request_sha256')
                if digest and digest in result_by_hash:
                    result = result_by_hash[digest]
                    row.update(translation_status=result['status'], translation_text=result.get('text'),
                        result_origin=result['result_origin'], cache_hit=result['cache_hit'],
                        generated_this_run=result['generated_this_run'], stored_generation=result['stored_generation'],
                        inference_seconds=result['inference_seconds'], startup_seconds=result.get('startup_seconds', 0.0),
                        error=result.get('error'), raw_response=result.get('raw_response'),
                        cache_provenance=result.get('provenance'))
                case_out['candidates'].append(row)
            report['cases'].append(case_out)
            write(worker_path, report)

        failed = [r for c in report['cases'] for r in c['candidates'] if r['translation_status'] != 'ok']
        report.update(status='complete_with_errors' if failed else 'complete',
            candidate_slot_count=sum(len(c['candidates']) for c in report['cases']),
            failed_candidate_slots=len(failed),
            new_inference_seconds_excluding_warmup=sum(r.get('inference_seconds', 0) for r in report['request_results']),
            finished_at_utc=datetime.now(timezone.utc).isoformat())
        if failed:
            report['errors'] = [{'case_id': c['id'], 'rank': r['rank'], 'error': r.get('error')}
                                for c in report['cases'] for r in c['candidates'] if r['translation_status'] != 'ok']
        write(worker_path, report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(worker_path, report)
        raise
    finally:
        process = gemma._process
        server_ref = owned_server_ref(server_path)
        gemma.shutdown()
        if server_ref:
            remaining = group_members(server_ref['pgid'])
            if remaining:
                cleanup_server(server_ref, signal.SIGKILL)
                time.sleep(.1)
                remaining = group_members(server_ref['pgid'])
            proc_after = proc_row(server_ref['pid'])
            write(server_path, server_ref | {'released': not remaining and
                  (proc_after is None or proc_after['state'] == 'Z'),
                  'remaining_group_members': remaining})
        if process is not None:
            report['owned_gemma_server_pid'] = process.pid
            report['owned_gemma_server_released'] = process.poll() is not None
            write(worker_path, report)
        signal.signal(signal.SIGTERM, previous_term)
        gemma.PORT, gemma.BASE = old_port, old_base


def run_parent(protocol_path=PROTOCOL):
    protocol, asr, cache, model, plan, out_dir = prepare(protocol_path)
    if out_dir.exists():
        raise ValueError(f'Preserve existing output; refusing overwrite: {out_dir}')
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = None
    child = None
    report = {'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': {'path': str(Path(protocol_path)), 'sha256': sha(protocol_path)},
              'inputs': {'asr_report': {'path': str(ASR_REPORT), 'sha256': sha(ASR_REPORT)},
                         'cache_packet': {'path': str(CACHE_PACKET), 'sha256': sha(CACHE_PACKET)}},
              'expected_case_count': 20, 'expected_rank_slots': 60,
              'request_plan_sha256': None, 'heldout_rows_used': 0, 'production_changed': False,
              'accuracy_scope': 'Saved-output translation coverage diagnostic only; no semantic grading.'}
    try:
        lock = LOCK.open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (protocol['settings']['port'], 18768):
            if port_accepting(port):
                raise RuntimeError(f'Private model port {port} is occupied; refusing concurrent model work.')
        out_dir.mkdir(parents=True, exist_ok=False)
        requests_path = out_dir / 'requests.json'
        write(requests_path, public_plan(plan))
        report['request_plan_sha256'] = sha(requests_path)
        report['model'] = model
        write(out_dir / 'report.json', report)
        env = {k: v for k, v in os.environ.items()
               if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD')}
        env.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONNOUSERSITE='1')
        with (out_dir / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
                                      '--protocol', str(protocol_path), '--output-dir', str(out_dir)],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                child.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                raise TimeoutError('Worker timed out; owned processes will be terminated and checked.')
        worker_path = out_dir / 'worker.json'
        if worker_path.exists():
            report['worker'] = {'path': str(worker_path), 'sha256': sha(worker_path)}
            raw = json.loads(worker_path.read_text())
            report['worker_status'] = raw.get('status')
            report['new_request_count'] = raw.get('coverage', {}).get('new_unique_requests')
            report['completed_rank_slots'] = raw.get('candidate_slot_count', 0)
            raw_cases = raw.get('cases', [])
            raw_slots = [candidate for case in raw_cases for candidate in case.get('candidates', [])]
            exact_coverage = ([case.get('id') for case in raw_cases] == [case['id'] for case in plan['cases']]
                and len(raw_slots) == 60
                and all([candidate.get('rank') for candidate in case.get('candidates', [])] == [0, 1, 2]
                        for case in raw_cases)
                and raw.get('candidate_slot_count') == 60)
            if raw.get('status') in ('complete', 'complete_with_errors') and exact_coverage:
                report['status'] = raw['status']
            else:
                report.update(status='failed', error='Worker report lacks exact20-case/60-rank coverage.')
        else:
            report.update(status='failed', error=report.get('error', 'Worker report is missing.'))
        if child.returncode != 0 and report['status'] not in ('failed',):
            report.update(status='failed', error='Worker exited unsuccessfully.')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            if out_dir.exists():
                report['cleanup'] = release_owned(child, out_dir)
                cleanup = report['cleanup']
                if (not cleanup['worker_released'] or cleanup['worker_group_remaining']
                        or cleanup['gemma_server_group_remaining'] or cleanup['port_18767_accepting']
                        or cleanup['port_18768_accepting']):
                    report.update(status='failed', error='Owned process or private port remains after cleanup.')
                report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                write(out_dir / 'report.json', report)
        finally:
            if lock is not None:
                lock.close()
    print(json.dumps({'status': report['status'], 'report': str(out_dir / 'report.json')}))
    return 0 if report['status'] in ('complete', 'complete_with_errors') else 1


def validate(protocol_path=PROTOCOL):
    protocol, asr, cache, model, plan, out_dir = prepare(protocol_path)
    if out_dir.exists():
        raise ValueError(f'Output already exists; validation does not overwrite it: {out_dir}')
    print(json.dumps({'status': 'valid', 'case_count': len(plan['cases']),
        'rank_slots': sum(len(c['candidates']) for c in plan['cases']),
        'unique_new_requests': sum(t['origin'] == 'new' for t in plan['targets'].values()),
        'baseline_cache_hits': sum(t['origin'] == 'baseline_cache' for t in plan['targets'].values()),
        'request_cache_hits': sum(t['origin'] == 'request_cache' for t in plan['targets'].values()),
        'heldout_rows_used': 0, 'model_loaded': False, 'inference_performed': False,
        'output_dir': str(out_dir)}))
    return 0


def self_test():
    settings = EXPECTED_SETTINGS
    base = {'start': 0.0, 'end': 2.0, 'text': '元の音声認識。'}
    alt = {'start': 0.0, 'end': 2.0, 'text': '別の音声認識。'}
    _, base_request, base_bytes, base_hash = request_for(base, settings)
    _, alt_request, alt_bytes, alt_hash = request_for(alt, settings)
    assert base_hash != alt_hash and base_bytes != alt_bytes
    fake = {'status': 'ok', 'request': base_request, 'request_sha256': base_hash,
            'text': 'saved original translation', 'inference_seconds': 1.0,
            'provenance': {'path': 'mock', 'sha256': '0' * 64}}
    fake_cache = {base_hash: fake}
    # Cache reuse is based on the full current request; matching Japanese text with a
    # different request must not hit (timestamps/settings are part of the payload).
    _, changed_request, changed_bytes, changed_hash = request_for(
        {'start': 0.0, 'end': 2.0, 'text': '別の音声認識。'}, settings)
    assert changed_hash != base_hash and changed_bytes != base_bytes
    assert changed_hash not in fake_cache
    # Rank mapping stays raw rank0 even when a different rank has a cached translation.
    ranks = [{'rank': 0, 'text': alt['text'], 'request': alt_request, 'request_sha256': alt_hash},
             {'rank': 1, 'text': base['text'], 'request': base_request, 'request_sha256': base_hash},
             {'rank': 2, 'text': alt['text'], 'request': alt_request, 'request_sha256': alt_hash}]
    assert [r['rank'] for r in ranks] == [0, 1, 2]
    assert ranks[0]['request_sha256'] == ranks[2]['request_sha256']
    assert ranks[0]['request_sha256'] not in fake_cache and ranks[1]['request_sha256'] in fake_cache
    baselines = {'case-1': {'id': 'case-1', 'source': base, 'request': base_request,
        'request_sha256': base_hash, 'text': 'saved original translation', 'inference_seconds': 1.0,
        'provenance': {'path': 'mock', 'sha256': '0' * 64}}}
    cases = [{'id': 'case-1', 'status': 'error', 'raw_hypotheses': [
        {'rank': 0, 'text': alt['text'], 'potentially_truncated_at_max_length': True}, {'rank': 1, 'text': base['text']},
        {'rank': 2, 'text': alt['text']}]}]
    built = build_plan(cases, baselines, {'model': {'mock': True}, 'entries': {}, 'entry_bytes': {}}, settings)
    assert len(built['targets']) == 2
    assert [r['candidate_id'] is not None for r in built['cases'][0]['candidates']] == [True, True, True]
    assert built['cases'][0]['candidates'][1]['result_origin'] == 'baseline_cache'
    assert built['cases'][0]['candidates'][0]['shared_request'] is False
    assert built['cases'][0]['candidates'][0]['source_truncated'] is True
    assert built['cases'][0]['candidates'][0]['request'] is not None
    assert built['cases'][0]['candidates'][2]['shared_request'] is True
    assert not valid_provenance(None)
    # The source-exact baseline reuse remains tied to its case ID.
    assert fake['request'] == base_request and fake['request_sha256'] == base_hash
    print('Pure self-test passed: full-payload cache identity, exact baseline reuse, rank0 preservation and duplicate-rank mapping; no checkpoint load.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if args.validate:
        raise SystemExit(validate(args.protocol))
    if args.run:
        raise SystemExit(run_parent(args.protocol))
    if args.worker and args.output_dir:
        worker(args.protocol, args.output_dir)
    else:
        parser.error('Choose --validate, --run, --worker --output-dir, or --self-test.')
