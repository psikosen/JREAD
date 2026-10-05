#!/usr/bin/env python3
"""Run frozen development streaming ASR with installed Whisper large-v3-turbo."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

ASR_SCRIPT = TESTS / 'evaluate_streaming_speech.py'
REPLAY_SCRIPT = TESTS / 'replay_streaming_boundaries.py'
MANIFEST = TESTS / 'fixtures/accuracy-corpus.json'
MODEL = Path.home() / '.local/share/JREAD/models/large-v3-turbo'
ASR_OUTPUT = ROOT / 'test-results/accuracy-streaming-turbo-speech-development.json'
REPLAY_OUTPUT = ROOT / 'test-results/accuracy-streaming-turbo-boundary-replay.json'
MODEL_NAME = 'faster-whisper large-v3-turbo'
WRAPPER_PATH = Path(__file__).resolve()
WRAPPER_SHA256 = None
REVISION = '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'
MODEL_BIN_SHA256 = 'e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da'
MODEL_CONFIG_SHA256 = None
MODEL_CONFIG_PROVENANCE = None
EXPECTED_CASES = 20
EXPECTED_UPDATES = 141
POLICY = 'sentence_terminal_eof'
CONFIG_ENV = 'JREAD_STREAMING_ASR_CONFIG'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load bounded evaluation helper: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def configure(config: dict | None = None) -> dict:
    """Apply an explicit model/report config; omitted config preserves Turbo defaults."""
    global MODEL, ASR_OUTPUT, REPLAY_OUTPUT, MODEL_NAME, REVISION, MODEL_BIN_SHA256
    global MODEL_CONFIG_SHA256
    global MODEL_CONFIG_PROVENANCE
    global WRAPPER_PATH, WRAPPER_SHA256
    config = dict(config or {})
    MODEL = Path(config.get('model_directory', MODEL))
    ASR_OUTPUT = Path(config.get('asr_output', ASR_OUTPUT))
    REPLAY_OUTPUT = Path(config.get('replay_output', REPLAY_OUTPUT))
    MODEL_NAME = config.get('model_name', MODEL_NAME)
    REVISION = config.get('revision', REVISION)
    MODEL_BIN_SHA256 = config.get('model_bin_sha256', MODEL_BIN_SHA256)
    MODEL_CONFIG_SHA256 = config.get('model_config_sha256', MODEL_CONFIG_SHA256)
    MODEL_CONFIG_PROVENANCE = config.get('model_config_provenance', MODEL_CONFIG_PROVENANCE)
    WRAPPER_PATH = Path(config.get('wrapper_path', WRAPPER_PATH))
    WRAPPER_SHA256 = config.get('wrapper_sha256', WRAPPER_SHA256)
    required = config.get('required_files', {})
    if not isinstance(required, dict):
        raise ValueError('required_files must be a filename-to-byte-size mapping.')
    validate_frozen_inputs.required_files = required
    return config


def validate_frozen_inputs() -> tuple[dict, list[dict]]:
    if not MODEL.is_dir() or (MODEL / '.verified').read_text().strip() != REVISION:
        raise ValueError(f'The locally installed {MODEL_NAME} revision does not match its pinned revision.')
    if sha256(MODEL / 'model.bin') != MODEL_BIN_SHA256:
        raise ValueError(f'The locally installed {MODEL_NAME} model.bin does not match its frozen SHA256.')
    if MODEL_CONFIG_SHA256 and sha256(MODEL / 'config.json') != MODEL_CONFIG_SHA256:
        raise ValueError(f'The locally installed {MODEL_NAME} config.json does not match its frozen SHA256.')
    for name, expected_size in getattr(validate_frozen_inputs, 'required_files', {}).items():
        if (MODEL / name).stat().st_size != expected_size:
            raise ValueError(f'Model side file has unexpected size: {name}')
    manifest = json.loads(MANIFEST.read_text())
    cases = [case for case in manifest['cases'] if case.get('partition') == 'development']
    if len(cases) != EXPECTED_CASES or len({str(case['id']) for case in cases}) != EXPECTED_CASES:
        raise ValueError('Frozen accuracy corpus must contain exactly20 unique development cases.')
    for case in cases:
        audio = (ROOT / case['audio_path']).resolve()
        if not audio.is_relative_to(ROOT / 'test-results/accuracy-audio') or sha256(audio) != case['audio_sha256']:
            raise ValueError(f"Frozen audio path or SHA256 mismatch for case {case['id']}.")
        if float(case['duration_seconds']) > 30:
            raise ValueError(f"Case {case['id']} exceeds the existing30-second retained-audio limit.")
    return manifest, cases


def gpu_snapshot():
    from asr_cuda_pilot import gpu_snapshot as snapshot
    return snapshot()


def self_test(check_model_files: bool = True) -> None:
    asr = load_script(ASR_SCRIPT, 'jread_streaming_asr_helpers')
    replay = load_script(REPLAY_SCRIPT, 'jread_boundary_replay_helpers')
    asr.self_test()
    replay.self_test()
    if check_model_files:
        if (MODEL / '.verified').read_text().strip() != REVISION:
            raise ValueError('The locally installed model revision does not match its pinned revision.')
        if sha256(MODEL / 'model.bin') != MODEL_BIN_SHA256:
            raise ValueError('The locally installed model.bin does not match its frozen SHA256.')
        if MODEL_CONFIG_SHA256 and sha256(MODEL / 'config.json') != MODEL_CONFIG_SHA256:
            raise ValueError('The locally installed model config.json does not match its frozen SHA256.')
        for name, expected_size in getattr(validate_frozen_inputs, 'required_files', {}).items():
            if (MODEL / name).stat().st_size != expected_size:
                raise ValueError(f'Model side file has unexpected size: {name}')
    assert ASR_OUTPUT != ROOT / 'test-results/accuracy-streaming-speech-development.json'
    assert REPLAY_OUTPUT != ROOT / 'test-results/accuracy-streaming-boundary-replay.json'
    # Selected corrected replay emits a punctuation-complete stable unit once,
    # then an EOF tail once; the last event owns both and no extra event/cost is added.
    Replay = replay.Replay
    def tok(text, start, end):
        return dict(text=text, start_seconds=start, end_seconds=end)
    policy = Replay(POLICY)
    update = dict(audio_end_seconds=2, inference_seconds=.2,
                  raw_hypothesis_words=[tok('短い文。', .1, .8)], raw_segments=[])
    policy.feed(update, 0)
    policy.feed({**update, 'audio_end_seconds': 4}, 1)
    policy.finish()
    result = policy.result('短い文。')
    assert result['transcript_alignment_status'] == 'aligned'
    assert len(result['units']) == 1
    assert result['units'][0]['stability'] == 'la2_stable'


def _asr_module():
    module = load_script(ASR_SCRIPT, 'jread_streaming_asr_turbo_worker')
    # Reuse the existing causal capture implementation, changing only the
    # already-installed model identity and report path.
    module.MODEL = MODEL
    module.OUTPUT = ASR_OUTPUT
    module.REVISION = REVISION
    module.FLAGS = dict(language='ja', task='transcribe', vad_filter=True,
                        beam_size=5, condition_on_previous_text=False)
    return module


def _run_asr_worker() -> None:
    _asr_module().worker()


def run_asr() -> dict:
    if ASR_OUTPUT.exists():
        raise RuntimeError(f'Preserve existing Turbo ASR artifact; refusing to overwrite {ASR_OUTPUT.relative_to(ROOT)}.')
    if REPLAY_OUTPUT.exists():
        raise RuntimeError(f'Preserve existing Turbo replay artifact; refusing to overwrite {REPLAY_OUTPUT.relative_to(ROOT)}.')
    manifest, cases = validate_frozen_inputs()
    asr_source_sha = sha256(ASR_SCRIPT)
    wrapper_sha = WRAPPER_SHA256 or sha256(WRAPPER_PATH)
    before = gpu_snapshot()
    if not any(row['index'] == 0 and row['free_mib'] >= 4096 for row in before['gpus']):
        raise RuntimeError('At least4096MiB of free GPU memory is required; no model was started.')
    report = {
        'schema_version': 1,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'starting',
        'partition': 'development',
        'runner_sha256': asr_source_sha,
        'wrapper': {'path': str(WRAPPER_PATH.relative_to(ROOT)), 'sha256': wrapper_sha},
        'orchestration_helper': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                                 'sha256': sha256(Path(__file__).resolve())},
        'corpus': {key: value for key, value in manifest.items() if key != 'cases'} |
                  {'manifest': str(MANIFEST.relative_to(ROOT)), 'manifest_sha256': sha256(MANIFEST)},
        'development_ids': [case['id'] for case in cases],
        'completed_cases': 0,
        'cases': [],
        'model': {
            'name': MODEL_NAME,
            'revision': REVISION,
            'model_directory': str(MODEL),
            'model_bin_sha256': MODEL_BIN_SHA256,
            'model_config_sha256': MODEL_CONFIG_SHA256,
            'model_config_provenance': MODEL_CONFIG_PROVENANCE,
            'device': 'cuda', 'device_index': 0,
            'compute_type': 'int8_float16', 'cpu_threads': 4,
            'num_workers': 1, 'local_files_only': True,
        },
        'flags': {'language': 'ja', 'task': 'transcribe', 'vad_filter': True,
                  'beam_size': 5, 'condition_on_previous_text': False},
        'streaming': {
            'algorithm': 'Existing LocalAgreement-2 capture, retained for raw observation diagnostics only.',
            'step_seconds': 2, 'max_buffer_seconds': 30, 'word_timestamps': True,
            'repaired_publication_policy': POLICY,
            'causality': 'Each observation decodes samples[0:horizon]; observations are strictly increasing two-second horizons, final horizon clipped to recording duration.',
            'source_flags': 'Japanese transcription, VAD on, beam size5, condition_on_previous_text=False; word_timestamps=True.',
            'capture_vs_replay': 'Legacy raw LocalAgreement publication units/conflicts remain capture diagnostics. Translation-facing units in the separate replay artifact are generated from saved raw hypotheses with sentence_terminal_eof; they do not use the legacy captured unit list.',
        },
        'baseline': {'chunk_seconds': 8, 'overlap_seconds': 1,
                     'boundaries': '0-8,7-15,14-22,...; final chunk clipped.',
                     'cer': None, 'cer_note': 'No aggregate CER for overlapping chunks.'},
        'gpu_before': before,
        'limitations': [
            'Only the 20 frozen development recordings were used; no held-out audio was accessed.',
            f'This repeats the medium streaming experiment with {MODEL_NAME} and preserves the raw LocalAgreement capture, but repairs the selected translation-facing boundary policy only in a separate replay artifact.',
            'Transcript CER is a source diagnostic, not semantic or translation accuracy.',
            'Repeated full-buffer decoding measurements do not guarantee live browser latency.',
            'No Gemma translation was run in this experiment.',
        ],
    }
    write_json(ASR_OUTPUT, report)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               **{CONFIG_ENV: json.dumps({
                   'model_directory': str(MODEL), 'asr_output': str(ASR_OUTPUT),
                   'replay_output': str(REPLAY_OUTPUT), 'model_name': MODEL_NAME,
                   'revision': REVISION, 'model_bin_sha256': MODEL_BIN_SHA256,
                   'model_config_sha256': MODEL_CONFIG_SHA256,
                   'model_config_provenance': MODEL_CONFIG_PROVENANCE,
                   'wrapper_path': str(WRAPPER_PATH), 'wrapper_sha256': wrapper_sha,
                   'required_files': getattr(validate_frozen_inputs, 'required_files', {}),
               })})
    cuda_lib = '/usr/local/cuda-12.8/lib64'
    env['LD_LIBRARY_PATH'] = cuda_lib + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    report['worker_environment'] = {key: env.get(key, '') for key in
                                   ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'LD_LIBRARY_PATH')}
    write_json(ASR_OUTPUT, report)

    child = None
    try:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                 cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = child.communicate(timeout=1500)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            report['status'] = 'timed_out'
            report['error'] = f'{MODEL_NAME} streaming ASR worker exceeded1500 seconds.'
        report = json.loads(ASR_OUTPUT.read_text())
        report.update(worker_pid=child.pid, worker_exit_code=child.returncode,
                      worker_stdout_tail=stdout[-6000:], worker_stderr_tail=stderr[-6000:])
        if child.returncode != 0:
            report['status'] = 'failed'
        elif report.get('status') != 'complete':
            report['status'] = 'failed'
            report['error'] = report.get('error') or 'Worker exited successfully without a complete report.'
        report['wrapper'] = {'path': str(WRAPPER_PATH.relative_to(ROOT)), 'sha256': wrapper_sha}
        report['model'].update(revision=REVISION, model_bin_sha256=MODEL_BIN_SHA256,
                               model_config_sha256=MODEL_CONFIG_SHA256,
                               model_config_provenance=MODEL_CONFIG_PROVENANCE,
                               model_directory=str(MODEL), name=MODEL_NAME)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        after = gpu_snapshot()
        report['gpu_after'] = after
        report['owned_gpu_process_released'] = child is None or all(
            row['pid'] != child.pid for row in after['processes'])
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(ASR_OUTPUT, report)
    if report.get('status') != 'complete' or not report.get('owned_gpu_process_released'):
        return report
    replay_turbo(report, manifest, cases)
    return report


def replay_turbo(asr: dict, manifest: dict, frozen_cases: list[dict]) -> dict:
    replay_helpers = load_script(REPLAY_SCRIPT, 'jread_turbo_replay_helpers')
    replay_helpers.self_test()
    if asr.get('status') != 'complete' or asr.get('owned_model_unloaded') is not True or asr.get('owned_gpu_process_released') is not True or asr.get('worker_exit_code') != 0:
        raise ValueError(f'{MODEL_NAME} source ASR artifact is incomplete or lacks owned cleanup verification.')
    if asr.get('model', {}).get('revision') != REVISION or asr.get('model', {}).get('model_bin_sha256') != MODEL_BIN_SHA256:
        raise ValueError(f'{MODEL_NAME} source ASR artifact model provenance is inconsistent.')
    if MODEL_CONFIG_SHA256 and asr.get('model', {}).get('model_config_sha256') != MODEL_CONFIG_SHA256:
        raise ValueError(f'{MODEL_NAME} source ASR artifact config provenance is inconsistent.')
    if asr.get('corpus', {}).get('manifest_sha256') != sha256(MANIFEST):
        raise ValueError(f'{MODEL_NAME} ASR report manifest hash differs from frozen manifest.')
    cases = [row for row in asr['cases'] if row.get('partition') == 'development']
    expected = {str(row['id']): row for row in frozen_cases}
    if len(cases) != EXPECTED_CASES or {str(row['id']) for row in cases} != set(expected):
        raise ValueError(f'{MODEL_NAME} source ASR must contain the 20 development IDs exactly.')
    if sum(len(row['streaming']['updates']) for row in cases) != EXPECTED_UPDATES:
        raise ValueError(f'{MODEL_NAME} source ASR must preserve all141 two-second causal observations.')
    for case in cases:
        frozen = expected[str(case['id'])]
        if case.get('audio_sha256') != frozen.get('audio_sha256') or case.get('japanese_reference') != frozen.get('japanese_reference'):
            raise ValueError(f"Turbo ASR audio/reference provenance mismatch for {case['id']}.")

    policy_cases = replay_helpers.replay_policy(POLICY, cases)
    for row in policy_cases:
        if row['update_count'] != len(row['events']):
            raise ValueError(f"Replay changed the causal observation count for {row['id']}.")
        if row['transcript_alignment_status'] == 'aligned' and not row['final_raw_text_equals_fused_text']:
            raise ValueError(f"Replay lost or reordered final source text for aligned case {row['id']}.")
        for index, event in enumerate(row['events']):
            for unit in event['emitted_units']:
                if unit['emitted_update_index'] != index or unit['emitted_at_audio_seconds'] > event['audio_end_seconds'] + 1e-4:
                    raise ValueError(f"Noncausal replay unit in case {row['id']} update {index}.")
                if unit['source_end_seconds'] > unit['emitted_at_audio_seconds'] + 1e-4:
                    raise ValueError(f"Replay unit uses future source time in case {row['id']}.")
    stream_seconds = sum(float(update['inference_seconds']) for case in cases for update in case['streaming']['updates'])
    baseline_seconds = sum(float(case['overlap_baseline']['inference_seconds']) for case in cases)
    summary = replay_helpers.summary(POLICY, policy_cases, EXPECTED_UPDATES, stream_seconds, baseline_seconds)
    replay_path_sha = sha256(REPLAY_SCRIPT)
    replay_report = {
        'schema_version': 1,
        'status': 'complete',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': f"Pure replay of saved {MODEL_NAME} development-only Japanese hypotheses under the corrected sentence_terminal_eof publication rule. No model inference or held-out data was used in replay.",
        'provenance': {
            'asr_report': str(ASR_OUTPUT.relative_to(ROOT)), 'asr_report_sha256': sha256(ASR_OUTPUT),
            'asr_runner': str(ASR_SCRIPT.relative_to(ROOT)), 'asr_runner_sha256': sha256(ASR_SCRIPT),
            'asr_wrapper': str(WRAPPER_PATH.relative_to(ROOT)),
            'asr_wrapper_sha256': WRAPPER_SHA256 or sha256(WRAPPER_PATH),
            'orchestration_helper': str(Path(__file__).resolve().relative_to(ROOT)),
            'orchestration_helper_sha256': sha256(Path(__file__).resolve()),
            'replay_runner': str(REPLAY_SCRIPT.relative_to(ROOT)), 'replay_runner_sha256': replay_path_sha,
            'manifest': str(MANIFEST.relative_to(ROOT)), 'manifest_sha256': sha256(MANIFEST),
            'partition': 'development', 'case_ids': [case['id'] for case in cases],
            'saved_hypothesis_count': EXPECTED_UPDATES,
            'original_asr_decode_seconds': round(stream_seconds, 4),
            'original_8s_overlap_decode_seconds': round(baseline_seconds, 4),
            'original_artifacts_preserved': True,
            'model': asr['model'],
            'selected_policy': POLICY,
        },
        'replay_controls': {
            'timestamp_jitter_seconds': replay_helpers.TIMESTAMP_JITTER_SECONDS,
            'agreement': 'Exact Japanese character prefix at shared complete timestamp-token boundaries with overlapping character times across adjacent saved horizons.',
            'immutable_text': 'Only source already emitted as a translation unit is immutable.',
            'eof': 'Use final saved hypothesis once; flush residual once at EOF.',
            'future_horizons': 'Replay inspects only the current saved update and immediately preceding saved hypothesis.',
            'legacy_capture': 'Legacy LocalAgreement publication conflicts/units are retained as raw capture diagnostics in the ASR report. Replay ignores those publication units and derives a separate translation-facing sequence directly from the saved raw hypotheses.',
            'policy': POLICY,
            'policy_description': 'Stable Japanese sentence terminal plus one EOF flush.',
        },
        'metrics': {POLICY: summary},
        'policies': {POLICY: {'cases': policy_cases, 'summary': summary}},
        'interpretation_limit': 'CER is a transcript diagnostic only. No semantic accuracy, translation accuracy, Gemma quality, or live extension latency is measured.',
    }
    write_json(REPLAY_OUTPUT, replay_report)
    return replay_report


def main() -> int:
    if '--self-test' in sys.argv:
        self_test()
        print('Turbo wrapper, frozen model identity, capture helper, and corrected boundary replay self-tests passed.')
        return 0
    if '--worker' in sys.argv:
        configure(json.loads(os.environ.get(CONFIG_ENV, '{}')))
        _run_asr_worker()
        return 0
    self_test()
    result = run_asr()
    print(json.dumps({
        'status': result.get('status'),
        'completed_cases': result.get('completed_cases'),
        'source_diagnostic_cer': result.get('aggregate', {}).get('streaming_diagnostic_cer'),
        'report': str(ASR_OUTPUT.relative_to(ROOT)),
        'replay_report': str(REPLAY_OUTPUT.relative_to(ROOT)) if REPLAY_OUTPUT.exists() else None,
        'owned_gpu_process_released': result.get('owned_gpu_process_released'),
    }, ensure_ascii=False))
    return 0 if result.get('status') == 'complete' else 1


def run_configured(config: dict) -> dict:
    """Python entry point for a pinned compatible CTranslate2 Whisper model."""
    configure(config)
    self_test(check_model_files=False)
    return run_asr()


if __name__ == '__main__':
    raise SystemExit(main())
