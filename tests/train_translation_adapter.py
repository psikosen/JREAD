#!/usr/bin/env python3
"""Isolated, frozen-data Gemma E4B QLoRA pilot. Never changes the production model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('/media/psikosen/newvolume/jread-finetune')
RUNTIME = EXPERIMENT / 'runtime'
MODEL = EXPERIMENT / 'model'
REPO = 'google/gemma-4-E4B-it-qat-q4_0-unquantized'
REVISION = '476025a01dbf99361c062bbeca3d6a76bb4c4566'
WEIGHT_SHA = 'ad8ef515194b15ab13b6f98d54d37652b691c9a2c787cdf7ed5f951f5ed2c7fa'
WEIGHT_BYTES = 15882477468
TARGET_MODULES = r'.*language_model\..*\.(q_proj|v_proj)'
HYPERPARAMETERS = dict(rank=8, alpha=16, dropout=.05, learning_rate=1e-4,
    betas=[.9, .999], eps=1e-8, weight_decay=0, max_grad_norm=1, gradient_accumulation=8,
    epochs=1, seed=20261004, max_length=512, scheduler='constant', warmup_steps=0,
    target_modules=TARGET_MODULES)
EVALUATION = dict(max_new_tokens=256, primary_temperature=0, secondary_temperature=.6,
                  secondary_top_p=.95, secondary_top_k=64, secondary_seed=17)
SMOKE = dict(id='independent-readiness-only', group_id='independent-readiness-only',
    source='妹は昨日、駅で赤い傘を買いました。',
    target='My younger sister bought a red umbrella at the station yesterday.',
    partition='train', variant='clean')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def helpers():
    # Import the real prompt/parser without reading or changing production state.
    os.environ['JREAD_DATA_DIR'] = str(RUNTIME / 'isolated-jread-state')
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from jread import gemma
    return gemma


def manifest_rows(args):
    import prepare_adapter_data as guard
    manifest_bytes, data_bytes = args.data_manifest.read_bytes(), args.data.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest['status'] != 'frozen' or manifest['data_sha256'] != hashlib.sha256(data_bytes).hexdigest()
            or Path(manifest['data_path']).resolve() != args.data.resolve()
            or manifest['evaluation_manifest_sha256'] != guard.CORPUS_SHA
            or sha(guard.CORPUS) != guard.CORPUS_SHA
            or manifest['preparer_sha256'] != sha(guard.__file__)
            or manifest['audit']['excluded_evaluation_cases'] != 70
            or manifest['audit']['exact_and_near_overlap_found']):
        raise ValueError('Data manifest is not the matching frozen, exclusion-audited pilot.')
    rows = [json.loads(line) for line in data_bytes.decode('utf-8').splitlines() if line.strip()]
    audit = guard.audit(rows, [])  # Group/variant/split checks; never sends protected evaluation text to a model.
    if audit['partitions'] != manifest['audit']['partitions'] or not all(audit['partitions'].get(p) for p in ('train', 'validation')):
        raise ValueError('Frozen data partitions changed or are empty.')
    return rows, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def prompt(processor, source):
    messages = helpers()._messages([dict(start=0, end=1, text=source)], '', None)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if '<|think|>' in text or '<|channel>thought' in text:
        raise ValueError('Thinking-disabled template unexpectedly contains a thinking channel.')
    ids = processor.tokenizer(text, add_special_tokens=False)['input_ids']
    return messages, text, ids


def assistant_labels(prefix, full):
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('Assistant-only token boundary is not an exact prompt prefix; refuse full-prompt loss.')
    return [-100] * len(prefix) + full[len(prefix):]


def encode(processor, row):
    messages, _, prefix = prompt(processor, row['source'])
    answer = json.dumps({'translations': [{'id': 0, 'text': row['target']}]}, ensure_ascii=False, separators=(',', ':'))
    full_text = processor.apply_chat_template(messages + [{'role': 'assistant', 'content': answer}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    full = processor.tokenizer(full_text, add_special_tokens=False)['input_ids']
    return dict(input_ids=full, attention_mask=[1] * len(full), labels=assistant_labels(prefix, full),
                prompt_length=len(prefix), target_json=answer)


def verify_model(path, weights=True):
    metadata_bytes = (path / 'provenance.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get('revision') != REVISION or metadata.get('repo_id') != REPO:
        raise ValueError('Model metadata differs from pinned official QAT source.')
    if weights:
        marker = dict(line.split('=', 1) for line in (path / '.verified').read_text().splitlines() if line.strip())
        if (metadata.get('status') != 'verified' or marker != dict(provenance_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                repo_revision=REVISION, files_verified='9')):
            raise ValueError('Verified complete model and matching provenance marker are required before GPU work.')
    entries = metadata['files']
    if set(entries) != {'.gitattributes', 'README.md', 'config.json', 'generation_config.json',
                        'processor_config.json', 'tokenizer.json', 'tokenizer_config.json',
                        'chat_template.jinja', 'model.safetensors'}:
        raise ValueError('Model provenance must contain exactly the nine pinned files.')
    for name, item in entries.items():
        if not weights and name.endswith('.safetensors'):
            continue
        file = path / name
        if not item.get('verified') or file.is_symlink() or file.stat().st_size != item['size_bytes'] or sha(file) != item['sha256']:
            raise ValueError(f'Pinned model file failed verification: {name}')
    if weights and (entries['model.safetensors']['size_bytes'] != WEIGHT_BYTES or entries['model.safetensors']['sha256'] != WEIGHT_SHA):
        raise ValueError('Primary QAT model weights do not match the independently pinned hash.')
    return metadata


def tensor_batch(torch, encoded):
    return {k: torch.tensor([encoded[k]], dtype=torch.long, device='cuda:0') for k in ('input_ids', 'attention_mask', 'labels')}


def last_logits(torch, model, encoded):
    model.eval()
    batch = tensor_batch(torch, encoded)
    batch.pop('labels')
    with torch.inference_mode():
        return model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu().clone()


def load_model(args, report):
    import torch
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    from asr_cuda_pilot import gpu_snapshot
    if TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get('gemma4') != TARGET_MODULES:
        raise ValueError('Installed PEFT Gemma4 LM target default differs from the frozen target regex.')
    report['gpu_before_load'] = gpu_snapshot()
    if not any(g['index'] == 0 and g['free_mib'] >= 24000 for g in report['gpu_before_load']['gpus']):
        raise ValueError('Pilot requires at least24000MiB free GPU0 memory; no other process will be changed.')
    torch.set_num_threads(4)
    torch.manual_seed(HYPERPARAMETERS['seed'])
    random.seed(HYPERPARAMETERS['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(str(args.model_dir), local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    if args.mode == 'evaluate':
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=TARGET_MODULES, bias='none', task_type='CAUSAL_LM', modules_to_save=None))
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not names or any('lora_' not in name or 'language_model' not in name or not any(f'.{m}.' in name for m in ('q_proj', 'v_proj')) for name in names):
            raise ValueError('Unexpected trainable parameter outside frozen LM-only LoRA scope.')
        report['trainable_names'] = names
        report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['load_seconds'] = time.monotonic() - started
    report['model_memory_footprint'] = model.get_memory_footprint()
    return torch, model


def optimizer(torch, model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)


def smoke(torch, model, encoded, report):
    from peft import PeftModel
    with model.disable_adapter():
        before = last_logits(torch, model, encoded)
    previous = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    optim = optimizer(torch, model)
    model.train()
    loss = model(**tensor_batch(torch, encoded), use_cache=False).loss
    if not torch.isfinite(loss):
        raise ValueError('Smoke loss is not finite.')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(torch.count_nonzero(g).item() for g in gradients):
        raise ValueError('Smoke gradients are missing, nonfinite or all zero.')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
    optim.step(); optim.zero_grad(set_to_none=True)
    changed = sum(not torch.equal(previous[n], p.detach().cpu()) for n, p in model.named_parameters() if p.requires_grad)
    if not changed:
        raise ValueError('Smoke optimizer did not update an adapter tensor.')
    with model.disable_adapter():
        after = last_logits(torch, model, encoded)
    if not torch.equal(before, after):
        raise ValueError('Adapter-off baseline tensors changed after training.')
    enabled = last_logits(torch, model, encoded)
    with tempfile.TemporaryDirectory(prefix='discard-smoke-adapter-', dir=RUNTIME / 'tmp') as folder:
        model.save_pretrained(folder, safe_serialization=True, save_embedding_layers=False)
        files = {p.name: sha(p) for p in Path(folder).iterdir() if p.is_file()}
        base = model.unload()  # Remove adapters without merging into frozen quantized weights.
        reloaded = PeftModel.from_pretrained(base, folder, is_trainable=False, local_files_only=True)
        if not torch.equal(enabled, last_logits(torch, reloaded, encoded)):
            raise ValueError('Saved/reloaded adapter tensors produce different smoke logits.')
        with reloaded.disable_adapter():
            if not torch.equal(before, last_logits(torch, reloaded, encoded)):
                raise ValueError('Reloaded adapter-off baseline differs from the original tensor.')
    report['smoke'] = dict(loss=float(loss.detach()), updated_adapter_tensors=changed,
        finite_nonzero_gradients=True, baseline_off_exact_tensor_equality=True,
        reload_adapter_on_exact_tensor_equality=True, reload_adapter_off_exact_tensor_equality=True,
        adapter_files_sha256=files, smoke_adapter_discarded=True, readiness_only=True)


def train(torch, model, rows, encoded, report, output):
    indices = [i for i, row in enumerate(rows) if row['partition'] == 'train']
    random.Random(HYPERPARAMETERS['seed']).shuffle(indices)
    report['training_order'] = [rows[i]['id'] for i in indices]
    report['steps'] = []
    model.train()
    optim = optimizer(torch, model)
    for offset in range(0, len(indices), 8):
        group = indices[offset:offset + 8]
        optim.zero_grad(set_to_none=True)
        losses = []
        started = time.monotonic()
        for index in group:
            loss = model(**tensor_batch(torch, encoded[index]), use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; stop without retries.')
            losses.append(float(loss.detach()))
            (loss / len(group)).backward()  # Final partial group uses its actual number of examples.
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        optim.step()
        torch.cuda.synchronize()
        report['steps'].append(dict(step=len(report['steps']) + 1, row_ids=[rows[i]['id'] for i in group],
            mean_loss=sum(losses) / len(losses), grad_norm=float(norm), seconds=time.monotonic() - started))
        write(output / 'report.json', report)
    model.save_pretrained(output / 'adapter', safe_serialization=True, save_embedding_layers=False)
    report['adapter_files'] = {p.name: sha(p) for p in (output / 'adapter').iterdir() if p.is_file()}
    write(output / 'adapter/pilot-provenance.json', {k: report[k] for k in ('model', 'data', 'protocol', 'hyperparameters', 'runner_sha256', 'adapter_files')})


def verify_adapter_files(folder, hashes):
    if not {'adapter_config.json', 'adapter_model.safetensors'} <= set(hashes):
        raise ValueError('Adapter provenance omits required artifact hashes.')
    for name, expected in hashes.items():
        path = folder / name
        if Path(name).name != name or path.is_symlink() or sha(path) != expected:
            raise ValueError(f'Saved adapter artifact changed: {name}')


def evaluation_coverage(rows, cases):
    expected = {row['id'] for row in rows if row['partition'] == 'validation'}
    conditions = {(0, False), (0, True), (.6, False), (.6, True)}
    if len(cases) != len(expected) or {case['id'] for case in cases} != expected:
        raise ValueError('Evaluation omitted or duplicated a frozen validation row.')
    errors = 0
    for case in cases:
        outputs = case['outputs']
        if len(outputs) != 4 or {(v['temperature'], v['adapter_enabled']) for v in outputs} != conditions:
            raise ValueError('Evaluation omitted or duplicated a decoding/adapter condition.')
        if any(v['status'] not in ('ok', 'error') for v in outputs):
            raise ValueError('Evaluation contains an unfinished output.')
        errors += sum(v['status'] == 'error' for v in outputs)
    return dict(coverage_complete=True, expected_rows=len(expected), expected_outputs=len(expected) * 4,
                observed_outputs=len(cases) * 4, valid_outputs=len(cases) * 4 - errors, error_outputs=errors,
                meaning_accuracy_status='not_graded',
                error_policy='Completed malformed/truncated outputs are observed failures retained in every condition denominator; run completion is not accuracy success. Runtime exceptions abort the run.')


def generate_one(torch, model, processor, inputs, temperature, adapter_enabled):
    stop_ids = model.generation_config.eos_token_id
    stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids]
    if processor.tokenizer.convert_tokens_to_ids('<turn|>') not in stop_ids:
        raise ValueError('Generation stop IDs omit the official turn terminator.')
    torch.manual_seed(17)
    kwargs = dict(max_new_tokens=256, do_sample=temperature > 0, eos_token_id=stop_ids,
                  pad_token_id=processor.tokenizer.pad_token_id, use_cache=True)
    if temperature:
        kwargs.update(temperature=.6, top_p=.95, top_k=64)
    torch.cuda.synchronize(); started = time.monotonic()
    with (nullcontext() if adapter_enabled else model.disable_adapter()), torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)[0, inputs['input_ids'].shape[1]:]
    torch.cuda.synchronize()
    tokens = generated.tolist()
    ended = bool(tokens) and tokens[-1] in stop_ids
    raw = processor.tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
    result = dict(adapter_enabled=adapter_enabled, temperature=temperature, seed=17,
        new_tokens=len(tokens), ended_with_eos=ended, reached_token_cap=len(tokens) >= 256,
        raw_text=raw, status='error')
    try:
        response = {'choices': [{'finish_reason': 'stop' if ended else 'length',
            'message': {'role': 'assistant', 'content': raw}}]}
        result.update(text=helpers()._translations(response, 1)[0], status='ok')
    except ValueError as error:
        result['error'] = str(error)
    result['seconds'] = time.monotonic() - started
    return result


def evaluate(torch, model, processor, rows, report, output):
    report['cases'] = []
    model.eval()
    _, _, warm_ids = prompt(processor, SMOKE['source'])
    warm_inputs = dict(input_ids=torch.tensor([warm_ids], device='cuda:0'),
                       attention_mask=torch.ones((1, len(warm_ids)), dtype=torch.long, device='cuda:0'))
    report['evaluation_warmup'] = dict(source=SMOKE['source'], excluded=True,
        result=generate_one(torch, model, processor, warm_inputs, 0, False))
    write(output / 'report.json', report)
    for row in rows:
        if row['partition'] != 'validation':
            continue
        messages, text, ids = prompt(processor, row['source'])
        if len(ids) > 512:
            raise ValueError('Frozen validation prompt exceeds512 tokens; no truncation.')
        case = dict(row, prompt_messages=messages, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), outputs=[])
        inputs = dict(input_ids=torch.tensor([ids], device='cuda:0'), attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device='cuda:0'))
        for temperature in (0, .6):
            for adapter_enabled in (False, True):
                case['outputs'].append(generate_one(torch, model, processor, inputs, temperature, adapter_enabled))
        report['cases'].append(case)
        write(output / 'report.json', report)
    report['evaluation_summary'] = evaluation_coverage(rows, report['cases'])


def worker(args):
    from transformers import AutoProcessor
    report_path = args.output / 'report.json'
    report = json.loads(report_path.read_text())
    report['worker_pid'] = os.getpid()
    model = None
    try:
        if args.mode != 'validate-data':
            report['model'] = verify_model(args.model_dir)
        else:
            report['model'] = dict(repo=REPO, revision=REVISION, tokenizer_only=True)
        processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
        report['template_sha256'] = sha(args.model_dir / 'chat_template.jinja')
        rows, manifest, manifest_sha = ([SMOKE], None, None) if args.mode == 'smoke' else manifest_rows(args)
        if manifest:
            report['data'] = dict(manifest=manifest, manifest_sha256=manifest_sha, path=str(args.data))
        encoded = [encode(processor, row) for row in rows]
        report['token_lengths'] = [dict(id=r['id'], total=len(e['input_ids']), prompt=e['prompt_length']) for r, e in zip(rows, encoded, strict=True)]
        report['overlong_ids'] = [r['id'] for r, e in zip(rows, encoded, strict=True) if len(e['input_ids']) > 512]
        if report['overlong_ids']:
            raise ValueError('Frozen rows exceed512 tokens; rebuild/review the dataset before training. No rows were truncated or filtered.')
        if args.mode == 'validate-data':
            report['status'] = 'complete'
            return
        if args.mode in ('train', 'evaluate'):
            protocol_bytes = args.protocol.read_bytes()
            protocol = json.loads(protocol_bytes)
            if (protocol.get('status') != 'frozen' or protocol.get('hyperparameters') != HYPERPARAMETERS
                    or any(protocol.get('evaluation', {}).get(k) != v for k, v in EVALUATION.items())
                    or protocol.get('data', {}).get('manifest_sha256') != manifest_sha
                    or protocol.get('runner_sha256') != sha(__file__)):
                raise ValueError('Pilot protocol does not match the reviewed frozen hyperparameters/evaluation settings.')
            report['protocol'] = dict(path=str(args.protocol), sha256=hashlib.sha256(protocol_bytes).hexdigest(), contents=protocol)
            if args.mode == 'evaluate':
                trained = json.loads((args.adapter / 'pilot-provenance.json').read_text())
                if any(trained[k] != report[k] for k in ('model', 'data', 'protocol', 'hyperparameters')):
                    raise ValueError('Adapter is not from the same frozen pilot model/data/protocol.')
                verify_adapter_files(args.adapter, trained['adapter_files'])
                report['verified_adapter_files'] = trained['adapter_files']
        write(report_path, report)
        torch, model = load_model(args, report)
        if args.mode == 'smoke':
            smoke(torch, model, encoded[0], report)
        elif args.mode == 'train':
            train(torch, model, rows, encoded, report, args.output)
        else:
            evaluate(torch, model, processor, rows, report, args.output)
        report['status'] = 'complete'
        report['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        del model
        gc.collect()
        if 'torch' in locals():
            torch.cuda.empty_cache()
        report['model_references_released'] = True
        write(report_path, report)


def self_test():
    assert assistant_labels([2, 4], [2, 4, 7, 106]) == [-100, -100, 7, 106]
    for prefix, full in (([2, 4], [2, 8, 7]), ([2], [2]), ([], [7])):
        try:
            assistant_labels(prefix, full)
        except ValueError:
            pass
        else:
            raise AssertionError('Masking must fail closed on an ambiguous assistant boundary.')
    groups = [list(range(10))[i:i + 8] for i in range(0, 10, 8)]
    assert [sum(1 / len(g) for _ in g) for g in groups] == [1, 1]
    cases = [dict(id='v', outputs=[dict(temperature=t, adapter_enabled=a, status='error' if t == .6 and a else 'ok')
                                  for t in (0, .6) for a in (False, True)])]
    assert evaluation_coverage([dict(id='v', partition='validation')], cases)['error_outputs'] == 1
    try:
        evaluation_coverage([dict(id='v', partition='validation')], [])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing evaluation rows must fail coverage checks')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            (path / name).write_bytes(b'independent integrity test')
        hashes = {p.name: sha(p) for p in path.iterdir()}
        verify_adapter_files(path, hashes)
        (path / 'adapter_model.safetensors').write_bytes(b'changed')
        try:
            verify_adapter_files(path, hashes)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed adapter must not be loaded')
    print('Pilot self-test passed: assistant-only mask, rejected boundary mismatch, final partial accumulation normalization.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['self-test', 'validate-data', 'smoke', 'train', 'evaluate'])
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == 'self-test':
        self_test(); return
    if not args.output or (args.mode not in ('smoke',) and (not args.data or not args.data_manifest)):
        parser.error('Provide --output and, except for smoke, frozen --data/--data-manifest.')
    if args.mode in ('train', 'evaluate') and not args.protocol:
        parser.error('Training/evaluation requires the reviewed --protocol.')
    if args.mode == 'evaluate' and not args.adapter:
        parser.error('Evaluation requires --adapter.')
    if args.worker:
        worker(args); return
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = dict(status='starting', mode=args.mode, runner_sha256=sha(__file__),
        hyperparameters=HYPERPARAMETERS, evaluation=EVALUATION, worker_pid=None,
        runtime_manifest_sha256=sha(RUNTIME / 'runtime-manifest.json'),
        gemma_prompt_parser_source_sha256=sha(ROOT / 'jread/gemma.py'),
        no_uploads=True, production_modified=False)
    write(report_path, report)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'LD_PRELOAD'))}
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        PYTHONNOUSERSITE='1', HF_HOME=str(RUNTIME / 'hf-cache'), TMPDIR=str(RUNTIME / 'tmp'),
        XDG_CACHE_HOME=str(RUNTIME / 'cache'), CUDA_VISIBLE_DEVICES='' if args.mode == 'validate-data' else '0',
        JREAD_DATA_DIR=str(RUNTIME / 'isolated-jread-state'), TOKENIZERS_PARALLELISM='false')
    child = None
    try:
        with (args.output / 'worker.log').open('x') as log:
            child = subprocess.Popen([str(RUNTIME / 'venv/bin/python'), str(Path(__file__).resolve()),
                *sys.argv[1:], '--worker'], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
            child.wait(timeout={'validate-data': 300, 'smoke': 900, 'train': 7200, 'evaluate': 3600}[args.mode])
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        report = json.loads(report_path.read_text())
        report['worker_pid'] = child.pid if child else None
        report['owned_worker_released'] = child is None or child.poll() is not None
        report['worker_exit_code'] = child.returncode if child else None
        if report['worker_exit_code'] != 0:
            report['status'] = 'failed'
        write(report_path, report)
    print(json.dumps(dict(status=report['status'], report=str(report_path))))
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
