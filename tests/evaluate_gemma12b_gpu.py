"""E1: replay frozen development translation inputs through Gemma 4 12B on the GPU.

Development partition only. Reuses the exact production prompt builder
(jread.gemma._messages), strict JSON schema and parser, temperature 0, thinking
off. Inputs are copied byte-for-byte from two frozen reports:

- pristine: accuracy-translation-development.json / pristine_reference
  (E4B scored 17/20 on these inputs)
- turbo: accuracy-streaming-turbo-none-translation-development.json
  (repaired sentence-terminal units; E4B scored 13/20)

References are never sent to the model. Output goes to
test-results/accuracy-gemma12b-gpu-development.json for blind review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jread import gemma  # noqa: E402

LLAMA = Path('/media/psikosen/bigboy/ai/muse-glimmer/llama.cpp/build-cuda/bin/llama-server')
MODELS = Path.home() / '.local/share/JREAD/models'
PRESETS = {
    'gemma4-12b': MODELS / 'gemma4-12b/gemma-4-12b-it-qat-q4_0.gguf',
    'gemma4-e4b': MODELS / 'gemma4-e4b/gemma-4-E4B_q4_0-it.gguf',
}
PORT = 18768
PRISTINE = ROOT / 'test-results/accuracy-translation-development.json'
TURBO = ROOT / 'test-results/accuracy-streaming-turbo-none-translation-development.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> dict[str, list[dict]]:
    """Return {condition: [{id, calls: [[text, ...], ...]}]} from frozen reports."""
    pristine = []
    for case in json.loads(PRISTINE.read_text())['stages']['pristine_reference']['cases']:
        pristine.append({'id': case['id'], 'calls': [[s['text'] for s in case['input_segments']]]})
    turbo = []
    for case in json.loads(TURBO.read_text())['stage']['cases']:
        calls = [call['source_texts'] for update in case['replay']['updates'] for call in update['translation_calls']]
        turbo.append({'id': case['id'], 'calls': calls})
    return {'pristine': pristine, 'turbo': turbo}


def post(payload: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def translate(texts: list[str], temperature: float, seed: int | None) -> tuple[list[str], float, dict]:
    segments = [{'start': float(i), 'end': float(i) + 1.0, 'text': t} for i, t in enumerate(texts)]
    messages = gemma._messages(segments, '')
    schema = {'type': 'object', 'properties': {'translations': {'type': 'array', 'minItems': len(segments), 'maxItems': len(segments),
              'items': {'type': 'object', 'properties': {'id': {'type': 'integer'}, 'text': {'type': 'string'}}, 'required': ['id', 'text'], 'additionalProperties': False}}},
              'required': ['translations'], 'additionalProperties': False}
    payload = {'model': 'jread-eval', 'messages': messages, 'stream': False, 'temperature': temperature,
               'max_tokens': 1024, 'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False},
               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'subtitle_translations', 'strict': True, 'schema': schema}}}
    if seed is not None:
        payload['seed'] = seed
    start = time.perf_counter()
    response = post(payload)
    elapsed = time.perf_counter() - start
    return gemma._translations(response, len(segments)), elapsed, response.get('usage', {})


def wait_ready(proc: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'llama-server exited with {proc.returncode}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError('llama-server did not become healthy')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='gemma4-12b', choices=sorted(PRESETS))
    ap.add_argument('--conditions', default='pristine,turbo')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    model = PRESETS[args.preset]
    tag = f"{args.preset}-t{args.temperature:g}" + (f'-s{args.seed}' if args.seed is not None else '')
    out = Path(args.out) if args.out else ROOT / f'test-results/accuracy-{tag}-gpu-development.json'
    inputs = load_inputs()
    log = open(ROOT / f'test-results/{out.stem}.server.log', 'w')
    proc = subprocess.Popen([str(LLAMA), '-m', str(model), '--host', '127.0.0.1', '--port', str(PORT), '-ngl', '99',
                             '-c', '8192', '--parallel', '1', '--no-webui', '--alias', 'jread-eval'],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report = {
        'status': 'running', 'partition': 'development', 'heldout_rows_used': 0,
        'model': {'preset': args.preset, 'path': str(model), 'device': 'CUDA (RTX 5090), -ngl 99',
                  'runtime': str(LLAMA)},
        'decoding': {'temperature': args.temperature, 'seed': args.seed, 'thinking': 'off', 'context': 'none', 'glossary': 'none'},
        'prompt_builder': 'jread.gemma._messages (production, unchanged)',
        'inputs': {'pristine': {'path': str(PRISTINE.relative_to(ROOT)), 'sha256': sha256(PRISTINE)},
                   'turbo': {'path': str(TURBO.relative_to(ROOT)), 'sha256': sha256(TURBO)}},
        'references_sent_to_model': False,
        'conditions': {},
    }
    try:
        wait_ready(proc)
        translate(['こんにちは。'], args.temperature, args.seed)  # warmup, excluded
        for condition in args.conditions.split(','):
            rows, total = [], 0.0
            for case in inputs[condition]:
                outputs, errors = [], []
                for texts in case['calls']:
                    try:
                        english, elapsed, usage = translate(texts, args.temperature, args.seed)
                        total += elapsed
                        outputs.append({'source_texts': texts, 'translations': english, 'elapsed_seconds': round(elapsed, 3), 'usage': usage})
                    except Exception as error:  # every failure stays in the denominator
                        errors.append({'source_texts': texts, 'error': repr(error)})
                rows.append({'id': case['id'], 'outputs': outputs, 'errors': errors})
                print(condition, case['id'], ' | '.join(t for o in outputs for t in o['translations']), flush=True)
            report['conditions'][condition] = {'cases': rows, 'total_translation_seconds': round(total, 2),
                                               'error_count': sum(len(r['errors']) for r in rows)}
        report['status'] = 'complete'
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        report['server_released'] = proc.poll() is not None
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('wrote', out)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
