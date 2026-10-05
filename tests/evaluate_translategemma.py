#!/usr/bin/env python3
"""Isolated stock-template TranslateGemma4B readiness/development evaluation."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time

import evaluate_gemma_audio_hf as shared
import train_translation_adapter as pilot

REPO = 'google/translategemma-4b-it'
REVISION = '10042cb0e6e7fdce748996a71dc3dc432a4e0c89'
WEIGHTS = {
    'model-00001-of-00002.safetensors': (4961251752, '6187566ec8d00990103cd9243b306aceaa352697abfa3bb6fcbc5ce7a64ce142'),
    'model-00002-of-00002.safetensors': (3639026128, '40c6a896b2275866e8bf56abc1da85f37a0d1d8fee824ad3008ea3231f4a3b70'),
}
GENERATION = dict(do_sample=False, temperature=0, seed=17, num_beams=1, max_new_tokens=256)
MAX_INPUT_TOKENS = 1792  # Reserve256 output tokens within the documented2K translation context.
GPU_FREE_MIB = 16000
LOCK = pilot.RUNTIME / 'audio-readiness.lock'  # Same lock as the existing HF audio runner.


def bound_bytes(binding):
    path, expected = Path(binding['path']).expanduser(), binding['sha256']
    if not path.is_absolute() or path.is_symlink() or not re.fullmatch(r'[0-9a-f]{64}', expected or ''):
        raise ValueError('Bindings require absolute regular-file paths and lowercase SHA256 values.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Frozen input changed: {path}')
    return raw


def messages(source):
    return [{'role': 'user', 'content': [{'type': 'text', 'source_lang_code': 'ja',
                                        'target_lang_code': 'en', 'text': source}]}]


def validate_rows(raw, expected_ids, stage):
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (not rows or len(rows) > 128 or len(expected_ids) != len(set(expected_ids))
            or [r.get('id') for r in rows] != expected_ids
            or any(not isinstance(r.get('id'), str) or not r['id'] for r in rows)
            or any(not isinstance(r.get('source'), str) or not r['source'].strip() for r in rows)
            or any(r.get('partition') == 'heldout' for r in rows)):
        raise ValueError('Frozen nonempty source rows must exactly cover ordered unique IDs, without heldout rows.')
    if stage in ('readiness', 'development') and len(rows) != {'readiness': 3, 'development': 20}[stage]:
        raise ValueError('Readiness requires3 controls; development requires all20 sources.')
    return rows


def validate_gate(gate, model):
    if not gate:
        raise ValueError('Root meaning review of all3 readiness controls is required before benchmark inference.')
    report = json.loads(bound_bytes(dict(path=gate['report_path'], sha256=gate['report_sha256'])))
    review = json.loads(bound_bytes(dict(path=gate['review_path'], sha256=gate['review_sha256'])))
    cases = report.get('cases', [])
    grades = review.get('cases', [])
    if (report.get('status') != 'complete' or report.get('mode') != 'run'
            or report.get('owned_worker_released') is not True or report.get('worker_exit_code') != 0
            or report.get('owned_gpu_process_released') is not True
            or report.get('protocol', {}).get('stage') != 'readiness'
            or report.get('protocol', {}).get('model') != model
            or report.get('runner_sha256') != pilot.sha(__file__)
            or len(cases) != 3 or len({c['id'] for c in cases}) != 3
            or any(c.get('status') != 'ok' or c.get('ended_with_eos') is not True
                   or not c.get('text', '').strip() for c in cases)
            or review.get('status') != 'complete' or review.get('report_sha256') != gate['report_sha256']
            or len(grades) != 3 or {g['id'] for g in grades} != {c['id'] for c in cases}
            or any(g.get('status') != 'pass' or not g.get('reason', '').strip() for g in grades)):
        raise ValueError('Readiness gate lacks complete nonempty/EOS outputs and all3 evidenced meaning passes.')


def inputs(path, mode):
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get('status') not in (('frozen',) if mode == 'run' else ('draft', 'frozen')):
        raise ValueError('GPU inference requires a technically reviewed frozen protocol.')
    if (protocol.get('stage') not in ('readiness', 'development', 'anime')
            or protocol.get('runner_sha256') != pilot.sha(__file__)
            or protocol.get('generation') != GENERATION
            or protocol.get('runtime_manifest_sha256') != shared.RUNTIME_SHA
            or pilot.sha(pilot.RUNTIME / 'runtime-manifest.json') != shared.RUNTIME_SHA
            or protocol.get('helper_sha256') != {'pilot': pilot.sha(pilot.__file__), 'audio_readiness': pilot.sha(shared.__file__)}
            or protocol.get('model', {}).get('repo_id') != REPO
            or protocol.get('model', {}).get('revision') != REVISION):
        raise ValueError('Runner, helpers, runtime, model or generation settings differ from the frozen protocol.')
    manifest = json.loads(bound_bytes(protocol['input_manifest']))
    if manifest.get('status') != 'inputs_frozen_before_generation':
        raise ValueError('Source inputs must be frozen before generation.')
    binding = manifest['artifacts'][f'{protocol["stage"]}_inputs']
    if {k: protocol['data'][k] for k in ('path', 'sha256')} != binding:
        raise ValueError('Selected input file does not match the source-only manifest.')
    # Hash provenance/reference files without loading their text into inference inputs.
    for binding in [*manifest.get('source_artifacts', {}).values(), *manifest['artifacts'].values()]:
        bound_bytes(binding)
    rows = validate_rows(bound_bytes(protocol['data']), protocol['data']['expected_ids'], protocol['stage'])
    if mode == 'run' and protocol['stage'] != 'readiness':
        validate_gate(protocol.get('readiness_gate'), protocol['model'])
    return protocol, hashlib.sha256(raw).hexdigest(), rows


def verify_model(spec, weights):
    directory = Path(spec['directory'])
    raw = bound_bytes(dict(path=str(directory / 'provenance.json'), sha256=spec['provenance_sha256']))
    metadata = json.loads(raw)
    files = metadata['files']
    marker = dict(line.split('=', 1) for line in (directory / '.verified').read_text().splitlines() if line.strip())
    if (metadata.get('status') != 'verified' or metadata.get('repo_id') != REPO or metadata.get('revision') != REVISION
            or marker != dict(provenance_sha256=spec['provenance_sha256'], repo_revision=REVISION, files_verified=str(len(files)))
            or {name for name in files if name.endswith('.safetensors')} != set(WEIGHTS)
            or not {'config.json', 'generation_config.json', 'chat_template.jinja', 'tokenizer.json',
                    'tokenizer_config.json', 'model.safetensors.index.json'} <= set(files)):
        raise ValueError('Expected the complete verified official TranslateGemma checkpoint and stock template.')
    for name, item in files.items():
        if Path(name).name != name or not item.get('verified'):
            raise ValueError('Invalid or unverified model inventory entry.')
        file = directory / name
        if name in WEIGHTS and (item['size_bytes'], item['sha256']) != WEIGHTS[name]:
            raise ValueError('Official weight shard pin changed.')
        if not weights and name in WEIGHTS:
            continue
        if file.is_symlink() or file.stat().st_size != item['size_bytes'] or pilot.sha(file) != item['sha256']:
            raise ValueError(f'Pinned model file changed: {name}')
    config = json.loads((directory / 'config.json').read_bytes())
    if config.get('model_type') != 'gemma3' or config.get('quantization_config'):
        raise ValueError('Expected unquantized Gemma3 architecture.')
    return metadata


def encode(processor, source):
    chat = messages(source)  # IDs, reference translations, rubric and previous examples never enter chat.
    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    batch = processor.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors='pt')
    count = int(batch['input_ids'].shape[-1])
    if not 0 < count <= MAX_INPUT_TOKENS or any(k in batch for k in ('pixel_values', 'input_features')):
        raise ValueError('Stock-template text-only prompt exceeds reserved context or contains unexpected media.')
    return batch, dict(messages=chat, formatted_prompt=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        input_tokens=count, input_ids_sha256=hashlib.sha256(batch['input_ids'].contiguous().numpy().tobytes()).hexdigest(),
        input_keys=sorted(batch), reference_text_sent=False)


def completed_output(token_ids, eos_ids, raw, text):
    ended = bool(token_ids) and token_ids[-1] in eos_ids
    result = dict(raw_text=raw, text=text.strip(), token_ids=token_ids, new_tokens=len(token_ids),
        ended_with_eos=ended, reached_token_cap=len(token_ids) >= GENERATION['max_new_tokens'])
    if not ended or not result['text']:
        result.update(status='error', error='Missing EOS or empty translated text; retained failure, no retry.')
    else:
        result['status'] = 'ok'
    return result


def require_bf16(dtypes):
    if dtypes != ['torch.bfloat16']:
        raise ValueError(f'Expected all BF16 model parameters; observed {dtypes}. No silent precision change.')


def finish_cases(report):
    for case in report['cases']:
        if case['status'] == 'pending':
            case.update(status='error', error='Worker ended before producing this frozen row.', seconds=None)
    report['coverage'] = dict(expected=len(report['expected_ids']), observed=len(report['cases']),
        exact_ids=[c['id'] for c in report['cases']] == report['expected_ids'],
        valid=sum(c['status'] == 'ok' for c in report['cases']), errors=sum(c['status'] == 'error' for c in report['cases']))


def worker(args):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    report_path = args.output / 'report.json'
    report = json.loads(report_path.read_bytes())
    model = None
    try:
        protocol, digest, rows = inputs(args.protocol, args.mode)
        if digest != report['protocol_sha256']:
            raise ValueError('Protocol changed after parent preflight.')
        report['runtime_versions'] = shared.runtime_versions()
        report['model'] = verify_model(protocol['model'], weights=args.mode == 'run')
        report['weights_verified_this_run'] = args.mode == 'run'
        directory = Path(protocol['model']['directory'])
        processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        report['template_sha256'] = pilot.sha(directory / 'chat_template.jinja')
        torch.set_num_threads(4)
        encoded = [encode(processor, row['source']) for row in rows]
        report['cpu_template_checks'] = [dict(id=row['id'], **evidence) for row, (_, evidence) in zip(rows, encoded, strict=True)]
        if torch.cuda.is_initialized():
            raise ValueError('Template-only preprocessing unexpectedly initialized CUDA.')
        report['cpu_template_cuda_initialized'] = False
        pilot.write(report_path, report)
        if args.mode == 'validate-template':
            report.update(status='complete', cases=[], validation_only=True)
            return
        report['gpu_before_load'] = shared.audio_helpers.gpu_snapshot()
        if not any(g['index'] == 0 and g['free_mib'] >= GPU_FREE_MIB for g in report['gpu_before_load']['gpus']):
            raise ValueError('Need16000MiB free GPU0 memory; other processes are never changed.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        started = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(str(directory), local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
        model.eval(); model.requires_grad_(False)
        if getattr(model, 'is_quantized', False):
            raise ValueError('Initial challenger must remain BF16 without extra quantization.')
        report.update(model_loaded=True, load_seconds=time.perf_counter() - started,
            model_memory_bytes=model.get_memory_footprint(), parameters=sum(p.numel() for p in model.parameters()),
            parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            device_name=torch.cuda.get_device_name(0), torch_cuda_version=torch.version.cuda,
            device_capability=list(torch.cuda.get_device_capability(0)))
        require_bf16(report['parameter_dtypes'])
        config = copy.deepcopy(model.generation_config)
        config.update(do_sample=False, num_beams=1, max_new_tokens=256, use_cache=True,
                      temperature=None, top_p=None, top_k=None)
        eos = config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        report['effective_generation_config'] = config.to_dict()
        for index, row in enumerate(rows):
            torch.cuda.synchronize(); started = time.perf_counter()
            batch, evidence = encode(processor, row['source'])
            batch = {k: v.to('cuda:0') for k, v in batch.items()}
            torch.manual_seed(17)
            report['model_inference_attempted'] = True
            with torch.inference_mode():
                generated = model.generate(**batch, generation_config=config)[0, batch['input_ids'].shape[-1]:].tolist()
            result = completed_output(generated, eos,
                processor.decode(generated, skip_special_tokens=False), processor.decode(generated, skip_special_tokens=True))
            torch.cuda.synchronize()
            report['cases'][index].update(**evidence, **result, seconds=time.perf_counter() - started)
            report['model_inference_performed'] = True
            pilot.write(report_path, report)
        finish_cases(report)
        report.update(status='complete', peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        model = None
        if 'batch' in locals():
            batch = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        report['model_references_released'] = True
        pilot.write(report_path, report)


def self_test():
    rows = [dict(id=f'control{i}', source=f'日本語{i}', target='REFERENCE MUST NOT ENTER CHAT') for i in range(3)]
    raw = b'\n'.join(json.dumps(r).encode() for r in rows)
    assert validate_rows(raw, [r['id'] for r in rows], 'readiness') == rows
    assert 'REFERENCE' not in json.dumps(messages(rows[0]['source']))
    for bad_ids in (['control0', 'control1'], ['control0'] * 3):
        try:
            validate_rows(raw, bad_ids, 'readiness')
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete or repeated coverage was accepted.')
    assert completed_output([1, 2], [2], 'English<EOS>', 'English')['status'] == 'ok'
    assert completed_output([2], [2], '<EOS>', '')['status'] == 'error'
    assert completed_output([1] * 256, [2], 'truncated', 'truncated')['status'] == 'error'
    require_bf16(['torch.bfloat16'])
    try:
        require_bf16(['torch.bfloat16', 'torch.float32'])
    except ValueError:
        pass
    else:
        raise AssertionError('Mixed parameter precision was accepted under an all-BF16 claim.')
    report = dict(expected_ids=['a', 'b'], cases=[dict(id='a', status='ok'), dict(id='b', status='pending')])
    finish_cases(report)
    assert report['coverage'] == dict(expected=2, observed=2, exact_ids=True, valid=1, errors=1)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        model = dict(repo_id=REPO)
        report_path, review_path = root / 'report.json', root / 'review.json'
        readiness = dict(status='complete', mode='run', owned_worker_released=True, worker_exit_code=0,
            owned_gpu_process_released=True,
            protocol=dict(stage='readiness', model=model), runner_sha256=pilot.sha(__file__),
            cases=[dict(id=r['id'], status='ok', ended_with_eos=True, text='English') for r in rows])
        pilot.write(report_path, readiness)
        grades = dict(status='complete', report_sha256=pilot.sha(report_path),
            cases=[dict(id=r['id'], status='pass', reason='Independent meaning review.') for r in rows])
        pilot.write(review_path, grades)
        gate = dict(report_path=str(report_path), report_sha256=pilot.sha(report_path),
                    review_path=str(review_path), review_sha256=pilot.sha(review_path))
        validate_gate(gate, model)
        readiness['owned_gpu_process_released'] = False
        pilot.write(report_path, readiness)
        old_report_sha = gate['report_sha256']
        gate['report_sha256'] = pilot.sha(report_path)
        grades['report_sha256'] = gate['report_sha256']
        pilot.write(review_path, grades)
        gate['review_sha256'] = pilot.sha(review_path)
        try:
            validate_gate(gate, model)
        except ValueError:
            pass
        else:
            raise AssertionError('Unreleased GPU process passed the readiness gate.')
        readiness['owned_gpu_process_released'] = True
        pilot.write(report_path, readiness)
        assert pilot.sha(report_path) == old_report_sha
        gate['report_sha256'] = old_report_sha
        grades['report_sha256'] = old_report_sha
        pilot.write(review_path, grades)
        gate['review_sha256'] = pilot.sha(review_path)
        grades['cases'][0]['status'] = 'fail'
        pilot.write(review_path, grades)
        for expected in (gate['review_sha256'], pilot.sha(review_path)):
            try:
                validate_gate({**gate, 'review_sha256': expected}, model)
            except ValueError:
                pass
            else:
                raise AssertionError('Mutated review or failed readiness meaning was accepted.')
    print('TranslateGemma self-test passed: coverage, source-only chat, EOS-only/truncated failures, missing outputs, BF16 precision, GPU release, review mutation and meaning gate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['self-test', 'validate-template', 'run'])
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == 'self-test':
        self_test(); return
    if not args.protocol or not args.output:
        parser.error('Provide frozen --protocol and a new --output directory.')
    args.protocol, args.output = args.protocol.resolve(), args.output.resolve()
    if args.worker:
        worker(args); return
    protocol, digest, rows = inputs(args.protocol, args.mode)
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = dict(status='starting', mode=args.mode, protocol=protocol, protocol_sha256=digest,
        runner_sha256=pilot.sha(__file__), expected_ids=[r['id'] for r in rows],
        cases=[dict(row, status='pending') for row in rows], cpu=dict(platform=platform.platform(), threads=4),
        model_loaded=False, model_inference_attempted=False, model_inference_performed=False,
        extra_quantization=False, adapter_used=False, training_performed=False, production_modified=False,
        uploads=False, reference_text_sent=False, max_input_tokens=MAX_INPUT_TOKENS, lock_path=str(LOCK),
        load_configuration=dict(dtype='bfloat16', attention='sdpa', device='cuda:0', cpu_threads=4,
                                minimum_free_gpu_mib=GPU_FREE_MIB, quantization=None),
        warmup_performed=False, latency_scope='Serial GPU BF16: tokenizer+transfer+generation+decode, cold first call included; model load separate. Not a matched speedup against historical CPU8 Q4 E4B.')
    pilot.write(report_path, report)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD'))}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        PYTHONNOUSERSITE='1', HF_HOME=str(pilot.RUNTIME / 'hf-cache'), TMPDIR=str(pilot.RUNTIME / 'tmp'),
        XDG_CACHE_HOME=str(pilot.RUNTIME / 'cache'), CUDA_VISIBLE_DEVICES='' if args.mode == 'validate-template' else '0',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    child = None
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with (args.output / 'worker.log').open('x') as log:
                child = subprocess.Popen([str(pilot.RUNTIME / 'venv/bin/python'), str(Path(__file__).resolve()),
                    args.mode, '--protocol', str(args.protocol), '--output', str(args.output), '--worker'],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=pilot.ROOT, start_new_session=True)
                child.wait(timeout=300 if args.mode == 'validate-template' else 3600)
        except BaseException as error:
            report = json.loads(report_path.read_bytes())
            report.update(status='failed', parent_error=f'{type(error).__name__}: {error}')
            pilot.write(report_path, report)
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            report = json.loads(report_path.read_bytes())
            report.update(worker_pid=child.pid if child else None, worker_exit_code=child.returncode if child else None,
                          owned_worker_released=child is None or child.poll() is not None)
            if report['worker_exit_code'] != 0:
                report['status'] = 'failed'
            if args.mode == 'run':
                finish_cases(report)
                try:
                    report['gpu_after'] = shared.audio_helpers.gpu_snapshot()
                    report['owned_gpu_process_released'] = all(p['pid'] != report['worker_pid'] for p in report['gpu_after']['processes'])
                except Exception as error:
                    report['gpu_after_error'] = f'{type(error).__name__}: {error}'
            pilot.write(report_path, report)
    print(json.dumps(dict(status=report['status'], report=str(report_path))))
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
