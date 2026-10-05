#!/usr/bin/env python3
"""Isolated EOF-only Qwen3-ASR diagnostic; smoke by default, never held-out audio.

Run --self-test without a model. Actual inference requires an explicitly prepared
local model/runtime; --full-development is a separate, review-gated invocation.
Official API: https://github.com/QwenLM/Qwen3-ASR
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone

from asr_cuda_pilot import gpu_snapshot
from evaluate_speech import character_error, development_cases
from evaluate_streaming_turbo import write_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
MANIFEST = ROOT / 'tests/fixtures/accuracy-corpus.json'
MANIFEST_SHA = '66825d56bfa271a58037a07ef68a59c1998c07f65b7740833b9e30d2d44cd578'
FIXTURE = ROOT / 'tests/fixtures/japanese-speech.wav'
FIXTURE_SHA = '26b6cd491b5d52cf05d89207cb88499cdaba89749aebc29c8f216e834e3d9071'
EXPERIMENTS = Path.home() / '.cache/jread-experiments'
RUNTIME_DIR = EXPERIMENTS / 'qwen3-asr-0.6b-runtime'
RUNTIME = RUNTIME_DIR / 'venv'
DEFAULT_MODEL = '0.6b'
PRESETS = {
    '0.6b': dict(directory='Qwen3-ASR-0.6B',
        revision='5eb144179a02acc5e5ba31e748d22b0cf3e303b0', min_free_mib=6144,
        weights={'model.safetensors': (1876091704, '79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea')},
        smoke='qwen-asr-smoke.json', development='accuracy-qwen-asr-development.json'),
    '1.7b': dict(directory='Qwen3-ASR-1.7B',
        revision='7278e1e70fe206f11671096ffdd38061171dd6e5', min_free_mib=8192,
        weights={
            'model-00001-of-00002.safetensors': (4220320824, 'a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6'),
            'model-00002-of-00002.safetensors': (478200688, '6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc'),
        }, smoke='qwen-asr-1.7b-smoke.json', development='accuracy-qwen-asr-1.7b-development.json'),
}
MAX_NEW_TOKENS = 512
OFFLINE = dict(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONNOUSERSITE='1')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select_cases(manifest: dict, full: bool) -> list[dict]:
    cases = development_cases(manifest)
    if cases[0]['id'] != 1632:
        raise ValueError('Smoke must use the frozen first development recording, 1632.')
    return cases if full else cases[:1]


def pcm16(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('Audio must be uncompressed mono PCM16 at 16000 Hz.')
        frames = wav.getnframes()
        if not 0 < frames <= 30 * 16000:
            raise ValueError('Diagnostic audio must be nonempty and at most30 seconds.')
        raw = wav.readframes(frames)
        if len(raw) != frames * 2:
            raise ValueError('Truncated PCM16 WAV.')
        return raw, frames / 16000


def validate_model(model_size=DEFAULT_MODEL) -> dict:
    preset = PRESETS[model_size]
    model = EXPERIMENTS / preset['directory']
    if (model / '.verified').read_text().strip() != preset['revision']:
        raise ValueError('Local Qwen model revision does not match the pinned revision.')
    files = {}
    for name, (size, checksum) in preset['weights'].items():
        if (model / name).stat().st_size != size or digest(model / name) != checksum:
            raise ValueError(f'Qwen model weight failed pinned size/SHA256 verification: {name}.')
        files[name] = checksum
    files.update({p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file() and p.name not in files})
    if not {'config.json', 'generation_config.json', 'preprocessor_config.json',
            'tokenizer_config.json', 'vocab.json', 'merges.txt', 'chat_template.json',
            'provenance.json'} <= files.keys():
        raise ValueError('Pinned local model is missing configuration/provenance files.')
    provenance = json.loads((model / 'provenance.json').read_text())
    if provenance['status'] != 'verified' or provenance['revision'] != preset['revision']:
        raise ValueError('Model provenance does not certify the selected pinned revision.')
    for name, expected in provenance['files'].items():
        if files.get(name) != expected['sha256'] or (model / name).stat().st_size != expected['size_bytes']:
            raise ValueError(f'Model sidefile differs from verified provenance: {name}.')
    if model_size == '1.7b':
        index = json.loads((model / 'model.safetensors.index.json').read_text())
        if set(index['weight_map'].values()) != set(preset['weights']):
            raise ValueError('Weight index must reference exactly the two pinned1.7B shards.')
    return files


def validate_inputs(full: bool) -> list[dict]:
    if digest(MANIFEST) != MANIFEST_SHA or digest(FIXTURE) != FIXTURE_SHA:
        raise ValueError('Frozen corpus manifest or warmup fixture hash changed.')
    selected = select_cases(json.loads(MANIFEST.read_text()), full)
    pcm16(FIXTURE)
    for case in selected:
        path = (ROOT / case['audio_path']).resolve()
        if not path.is_relative_to(ROOT / 'test-results/accuracy-audio') or digest(path) != case['audio_sha256']:
            raise ValueError(f"Development audio path/hash mismatch: {case['id']}.")
        _, duration = pcm16(path)
        if abs(duration - case['duration_seconds']) > 1 / 16000:
            raise ValueError('Audio duration differs from the frozen corpus.')
    return selected


def require_gpu(min_free_mib=6144) -> dict:
    snapshot = gpu_snapshot()
    if not any(g['index'] == 0 and g['free_mib'] >= min_free_mib for g in snapshot['gpus']):
        raise ValueError(f'At least{min_free_mib} MiB free on GPU0 is required; no model started.')
    return snapshot


def worker(output: Path, model_size: str) -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = json.loads(output.read_text())
    model = original_generate = None
    try:
        preset = PRESETS[model_size]
        if report['model_size'] != model_size or report['model']['revision'] != preset['revision']:
            raise ValueError('Worker model selection differs from its parent report.')
        if report['runner_sha256'] != digest(SCRIPT):
            raise ValueError('Runner changed after preflight.')
        if Path(sys.prefix).resolve() != RUNTIME.resolve():
            raise ValueError('Worker is not running inside the prepared isolated runtime.')
        if any(digest(ROOT / 'tests' / name) != value
               for name, value in report['helper_source_sha256'].items()):
            raise ValueError('Evaluation helper changed after preflight.')
        if any(os.environ.get(k) != v for k, v in OFFLINE.items()):
            raise ValueError('Worker must run in the declared isolated offline environment.')
        selected = validate_inputs(report['mode'] == 'full-development')
        if validate_model(model_size) != report['model']['local_file_sha256']:
            raise ValueError('Model files changed after preflight.')
        runtime_manifest = RUNTIME_DIR / 'runtime-manifest.json'
        if digest(runtime_manifest) != report['runtime_manifest']['sha256']:
            raise ValueError('Runtime manifest changed after preflight.')
        if digest(RUNTIME_DIR / 'requirements.lock') != report['runtime_lock']['sha256']:
            raise ValueError('Runtime dependency lock changed after preflight.')
        report['gpu_before_load'] = require_gpu(preset['min_free_mib'])
        import numpy as np
        import torch
        import qwen_asr
        from qwen_asr import Qwen3ASRModel
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.manual_seed(0)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('This runtime cannot use CUDA BF16 on the selected GPU.')
        report['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
        for name, version in {'qwen-asr': '0.0.6', 'transformers': '4.57.6',
                              'accelerate': '1.12.0', 'torch': '2.7.1+cu128',
                              'torchaudio': '2.7.1+cu128'}.items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f'Isolated runtime package pin mismatch: {name}.')
        package = Path(qwen_asr.__file__).resolve().parent
        report['qwen_package_source_sha256'] = {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*.py'))}
        report['torch_cuda'] = dict(version=torch.version.cuda, device=torch.cuda.get_device_name(0),
                                    capability=list(torch.cuda.get_device_capability(0)))
        report['status'] = 'loading'
        write_json(output, report)
        started = time.perf_counter()
        model = Qwen3ASRModel.from_pretrained(
            str(EXPERIMENTS / preset['directory']), dtype=torch.bfloat16, device_map='cuda:0',
            max_inference_batch_size=1, max_new_tokens=MAX_NEW_TOKENS,
            local_files_only=True)
        model.model.eval()
        torch.cuda.synchronize()
        report['model']['load_seconds'] = time.perf_counter() - started
        report['gpu_after_load'] = gpu_snapshot()
        prompt = model._build_text_prompt(context='', force_language='Japanese')
        report['prompt'] = dict(text=prompt, sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                context='', language='Japanese')
        report['loaded_generation_config'] = model.model.thinker.generation_config.to_dict()
        original_generate = model.model.generate
        signature = inspect.signature(original_generate)
        default_eos = signature.parameters['eos_token_id'].default
        if not isinstance(default_eos, (int, list, tuple)):
            raise ValueError('Official generate EOS default is unavailable; cannot validate completion.')
        captures = []

        def measured_generate(*args, **kwargs):
            if args or 'input_ids' not in kwargs:
                raise ValueError('Unexpected official generation interface; cannot inspect completion.')
            kwargs.update(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS)
            inputs = kwargs['input_ids']
            if inputs.ndim != 2 or inputs.shape[0] != 1:
                raise ValueError('Only one generation input is allowed per diagnostic call.')
            result = original_generate(**kwargs)
            tail = result.sequences[:, inputs.shape[1]:]
            eos = kwargs.get('eos_token_id', default_eos)
            eos = [eos] if isinstance(eos, int) else list(eos)
            ids = tail[0].tolist()
            # A cap-ending EOS is complete; reaching the cap without EOS is not.
            captures.append(dict(input_tokens=int(inputs.shape[1]), new_tokens=len(ids),
                ended_with_eos=bool(ids and ids[-1] in eos), eos_token_ids=eos,
                reached_token_cap=len(ids) >= MAX_NEW_TOKENS,
                input_ids_sha256=hashlib.sha256(inputs.detach().cpu().numpy().tobytes()).hexdigest(),
                raw_decoded=model.processor.batch_decode(tail, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]))
            return result

        model.model.generate = measured_generate

        def transcribe(samples, silence=False):
            captures.clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.transcribe(audio=(samples, 16000), language='Japanese',
                                      context='', return_time_stamps=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if len(result) != 1 or not isinstance(result[0].text, str):
                raise ValueError('Expected one official transcription result.')
            text = result[0].text
            problems = []
            if len(captures) != 1:
                problems.append('unexpected_generation_count')
            for call in captures:
                if not call['ended_with_eos']:
                    problems.append('truncated_at_token_cap' if call['reached_token_cap'] else 'missing_eos')
            if silence and text.strip():
                problems.append('nonempty_silence')
            if not silence and not text.strip():
                problems.append('empty_speech')
            return dict(text=text, language=result[0].language, inference_seconds=elapsed,
                        generation=[dict(row) for row in captures], status='bad_output' if problems else 'ok',
                        problems=problems, word_timestamps=None)

        def samples(path):
            raw, _ = pcm16(path)
            return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768

        report['status'] = 'running'
        report['warmup'] = dict(excluded=True, path=str(FIXTURE.relative_to(ROOT)),
                                audio_sha256=FIXTURE_SHA, **transcribe(samples(FIXTURE)))
        write_json(output, report)
        report['silence_control'] = dict(duration_seconds=8, expected_empty=True,
            **transcribe(np.zeros(8 * 16000, dtype=np.float32), silence=True))
        write_json(output, report)
        for case in selected:
            result = transcribe(samples(ROOT / case['audio_path']))
            result['cer'] = character_error(case['japanese_reference'], result['text'])
            report['cases'].append(dict(case, full_clip=result))
            report['completed_cases'] = len(report['cases'])
            write_json(output, report)
        errors = sum(c['full_clip']['cer']['errors'] for c in report['cases'])
        chars = sum(c['full_clip']['cer']['reference_characters'] for c in report['cases'])
        report['aggregate'] = dict(case_count=len(selected), errors=errors,
            reference_characters=chars, cer=errors / chars, meaning_accuracy=False,
            inference_seconds=sum(c['full_clip']['inference_seconds'] for c in report['cases']),
            bad_output_case_ids=[c['id'] for c in report['cases'] if c['full_clip']['status'] != 'ok'])
        report['controls_passed'] = report['warmup']['status'] == report['silence_control']['status'] == 'ok'
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if model is not None and original_generate is not None:
            model.model.generate = original_generate
        model = original_generate = None
        gc.collect()
        if 'torch' in locals() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        report['owned_model_references_released'] = True
        write_json(output, report)


def stop_owned(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def run(args) -> None:
    preset = PRESETS[args.model_size]
    output = args.output or ROOT / 'test-results' / preset['development' if args.full_development else 'smoke']
    output = output.resolve()
    report = dict(schema_version=1, status='preflight', started_at_utc=datetime.now(timezone.utc).isoformat(),
        mode='full-development' if args.full_development else 'smoke', model_size=args.model_size,
        partition='development',
        heldout_audio_opened=0, runner_sha256=digest(SCRIPT), completed_cases=0, cases=[],
        official_api='https://github.com/QwenLM/Qwen3-ASR',
        helper_source_sha256={name: digest(ROOT / 'tests' / name) for name in
            ('evaluate_speech.py', 'asr_cuda_pilot.py', 'evaluate_streaming_turbo.py')},
        model=dict(repo='Qwen/' + preset['directory'], revision=preset['revision'],
                   directory=str(EXPERIMENTS / preset['directory']),
                   weights={name: dict(size_bytes=size, sha256=checksum)
                            for name, (size, checksum) in preset['weights'].items()},
                   minimum_free_gpu_mib=preset['min_free_mib'], device='cuda:0',
                   dtype='bfloat16', cpu_threads=4, backend='transformers'),
        flags=dict(max_inference_batch_size=1, max_new_tokens=MAX_NEW_TOKENS,
                   do_sample=False, num_beams=1, language='Japanese', context='', return_time_stamps=False),
        limitations=['EOF recognition only; no translation, alignment, or streaming claims.',
                     'CER is a transcript diagnostic, not meaning accuracy.',
                     'Only explicitly selected development audio is opened; references never enter model prompts.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents reruns from replacing prior evidence.
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    child = None
    try:
        selected = validate_inputs(args.full_development)
        report['development_ids'] = [c['id'] for c in selected]
        report['corpus'] = dict(path=str(MANIFEST.relative_to(ROOT)), sha256=MANIFEST_SHA)
        report['model']['local_file_sha256'] = validate_model(args.model_size)
        runtime_manifest = RUNTIME_DIR / 'runtime-manifest.json'
        report['runtime_manifest'] = dict(path=str(runtime_manifest), sha256=digest(runtime_manifest))
        lock = RUNTIME_DIR / 'requirements.lock'
        report['runtime_lock'] = dict(path=str(lock), sha256=digest(lock))
        runtime_python = RUNTIME / 'bin/python'
        if not runtime_python.is_file():
            raise ValueError('Prepared isolated Qwen runtime Python is missing.')
        report['gpu_before'] = require_gpu(preset['min_free_mib'])
        env = os.environ | OFFLINE
        for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN',
                    'PYTHONPATH', 'PYTHONHOME'):
            env.pop(key, None)
        report['worker_environment'] = OFFLINE
        report['worker_timeout_seconds'] = 1200 if args.full_development else 300
        write_json(output, report)
        child = subprocess.Popen([str(runtime_python), str(SCRIPT), '--worker', '--output', str(output),
                                  '--model-size', args.model_size],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        timeout = False
        try:
            stdout, stderr = child.communicate(timeout=report['worker_timeout_seconds'])
        except subprocess.TimeoutExpired:
            timeout = True
            stop_owned(child)
            stdout, stderr = child.communicate(timeout=10)
        report = json.loads(output.read_text())
        report.update(worker_pid=child.pid, worker_exit_code=child.returncode,
                      worker_stdout_tail=stdout[-3000:], worker_stderr_tail=stderr[-6000:])
        if timeout:
            report.update(status='timed_out', error='Owned Qwen worker exceeded its bounded timeout.')
        elif child.returncode != 0 or report.get('status') != 'complete':
            report.update(status='failed', error=report.get('error') or 'Worker did not finish a complete report.')
    except BaseException as error:
        if child is not None:
            stop_owned(child)
            report = json.loads(output.read_text())
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop_owned(child)
        try:
            report['gpu_after'] = gpu_snapshot()
            report['owned_gpu_process_released'] = child is None or all(
                row['pid'] != child.pid for row in report['gpu_after']['processes'])
        except Exception as error:
            report['cleanup_check_error'] = f'{type(error).__name__}: {error}'
            report['owned_gpu_process_released'] = None
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output, report)
        print(json.dumps(dict(status=report['status'], report=str(output),
                              owned_gpu_process_released=report['owned_gpu_process_released'])))
    if report['status'] != 'complete':
        raise SystemExit(1)


def self_test() -> None:
    default = PRESETS[DEFAULT_MODEL]
    assert DEFAULT_MODEL == '0.6b' and default['directory'] == 'Qwen3-ASR-0.6B'
    assert default['revision'] == '5eb144179a02acc5e5ba31e748d22b0cf3e303b0'
    assert default['min_free_mib'] == 6144 and MAX_NEW_TOKENS == 512
    assert default['smoke'] == 'qwen-asr-smoke.json' and default['development'] == 'accuracy-qwen-asr-development.json'
    assert PRESETS['1.7b']['min_free_mib'] == 8192 and len(PRESETS['1.7b']['weights']) == 2
    assert len({p[k] for p in PRESETS.values() for k in ('smoke', 'development')}) == 4
    cases = [dict(id=1632 + n, partition='development') for n in range(20)]
    manifest = {'cases': [dict(id=1, partition='heldout'), *cases]}
    assert select_cases(manifest, False) == cases[:1]
    assert select_cases(manifest, True) == cases
    try:
        select_cases({'cases': cases[:-1] + [cases[0]]}, True)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate development IDs must fail.')
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'input.wav'
        for channels, rate, frames, valid in [(1, 16000, 16, True), (2, 16000, 16, False),
                                              (1, 8000, 16, False), (1, 16000, 0, False)]:
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((channels, 2, rate, frames, 'NONE', 'not compressed'))
                wav.writeframes(bytes(frames * channels * 2))
            try:
                raw, seconds = pcm16(path)
            except ValueError:
                assert not valid
            else:
                assert valid and len(raw) == 32 and seconds == .001
    print('Qwen runner self-test passed: development selection and strict PCM16 validation; no model loaded.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--full-development', action='store_true', help='Run20 frozen development clips only after smoke review.')
    parser.add_argument('--model-size', choices=tuple(PRESETS), default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.worker:
        if args.output is None:
            parser.error('Internal worker requires an existing parent report.')
        worker(args.output, args.model_size)
    else:
        run(args)
