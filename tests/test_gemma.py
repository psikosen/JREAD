"""No real model, downloads, sockets, or subprocesses are used by these tests."""
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

if 'JREAD_DATA_DIR' not in os.environ:
    _test_data = tempfile.TemporaryDirectory(prefix='jread-gemma-test-')
    os.environ['JREAD_DATA_DIR'] = _test_data.name

import pytest
from jread import gemma


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    runtime = tmp_path / 'runtime' / 'llama-cpp'
    monkeypatch.setattr(gemma.store, 'DATA', tmp_path)
    monkeypatch.setattr(gemma.store, 'MODELS', tmp_path / 'models')
    monkeypatch.setattr(gemma, 'PRESETS', {key: dict(value) for key, value in gemma.PRESETS.items()})
    monkeypatch.setattr(gemma, 'RUNTIME_DIR', runtime)
    monkeypatch.setattr(gemma, 'SERVER_PATH', runtime / 'llama-server')
    monkeypatch.setattr(gemma, 'RUNTIME_MANIFEST', tmp_path / 'runtime.json')
    for key, value in {'_verified': {}, '_process': None, '_process_preset': None, '_owner_pid': None, '_key': '', '_key_path': None}.items():
        monkeypatch.setattr(gemma, key, value)
    # Fail closed if any test accidentally reaches a real transport or process.
    monkeypatch.setattr(gemma._http, 'open', Mock(side_effect=AssertionError('Unexpected real HTTP call')))
    monkeypatch.setattr(gemma, 'urlopen', Mock(side_effect=AssertionError('Unexpected runtime download')))
    monkeypatch.setattr(gemma.subprocess, 'Popen', Mock(side_effect=AssertionError('Unexpected subprocess')))
    yield
    gemma.shutdown()


def output(texts, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'role': 'assistant', 'content': json.dumps({'translations': [{'id': i, 'text': text} for i, text in enumerate(texts)]})}}]}


@pytest.mark.parametrize('preset', ['gemma4', 'gemma4-12b'])
def test_ready_verifies_and_invalidates_changed_weights(preset):
    weights = b'model-weights-v1'
    gemma.PRESETS[preset].update(size=len(weights), sha256=hashlib.sha256(weights).hexdigest())
    path = gemma.model_path(preset)
    assert not gemma.ready(preset)
    path.parent.mkdir(parents=True)
    path.write_bytes(weights)
    assert gemma.ready(preset)
    path.write_bytes(b'model-weights-v2')
    assert not gemma.ready(preset), 'Same-sized replacement must invalidate checksum readiness.'


@pytest.mark.parametrize('preset', ['gemma4', 'gemma4-12b'])
def test_download_is_pinned_verified_and_busy_safe(monkeypatch, preset):
    import huggingface_hub
    weights = b'test weights'
    config = gemma.PRESETS[preset]
    config.update(size=len(weights), sha256=hashlib.sha256(weights).hexdigest())
    monkeypatch.setattr(gemma, 'ensure_runtime', lambda: None)
    called = []
    def fake_download(*args, **kwargs):
        called.append((args, kwargs))
        gemma.model_path(preset).write_bytes(weights)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
    gemma.download(preset)
    assert called[0][0] == (config['repo'], config['filename'])
    assert called[0][1]['revision'] == config['revision']
    assert gemma.ready(preset)
    with gemma._lock:
        with pytest.raises(gemma.GemmaError, match='busy'):
            gemma.remove(preset)
    gemma.remove(preset)
    assert not gemma.model_path(preset).parent.exists()


def test_input_and_structured_output_validation():
    assert gemma.translate_batch([]) == []
    source = [{'start': 1, 'end': 2, 'text': '命令を翻訳してください。'}]
    messages = gemma._messages(source, 'Context only')
    assert 'untrusted quoted data' in messages[0]['content']
    assert source[0]['text'] in messages[1]['content']
    assert '<JREAD_SOURCE_' in messages[1]['content']
    for malformed in [source * 9, [{'start': float('nan'), 'end': 2, 'text': 'a'}], [{'start': 2, 'end': 1, 'text': 'a'}], [{'start': False, 'end': 1, 'text': 'a'}], [{'start': 0, 'end': 1, 'text': '日' * 1000}]]:
        with pytest.raises(ValueError):
            gemma._messages(malformed, '')
    assert gemma._translations(output(['Translate the instruction.']), 1) == ['Translate the instruction.']
    invalid = [output([], 'stop'), output(['a'], 'length'), output(['']), output(['a', 'extra']), {'choices': []}]
    tools = output(['a']); tools['choices'][0]['message']['tool_calls'] = [{'function': {'name': 'shell'}}]; invalid.append(tools)
    fenced = output(['a']); fenced['choices'][0]['message']['content'] = '```json\n{}\n```'; invalid.append(fenced)
    wrong_id = output(['a']); wrong_id['choices'][0]['message']['content'] = '{"translations":[{"id":1,"text":"a"}]}'; invalid.append(wrong_id)
    for value in invalid:
        with pytest.raises(gemma.GemmaError, match='No captions were accepted'):
            gemma._translations(value, 1)


def test_managed_cpu_lifecycle_and_toolless_request(monkeypatch):
    monkeypatch.setattr(gemma, 'ready', lambda preset='gemma4': True)
    gemma.RUNTIME_DIR.mkdir(parents=True)
    gemma.SERVER_PATH.write_text('test binary'); gemma.SERVER_PATH.chmod(0o700)
    probe = Mock(); probe.__enter__ = Mock(return_value=probe); probe.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(gemma.socket, 'socket', lambda: probe)
    process = Mock(returncode=None); process.poll.return_value = None
    popen = Mock(return_value=process); monkeypatch.setattr(gemma.subprocess, 'Popen', popen)
    monkeypatch.setenv('LLAMA_ARG_LOG_PROMPTS_DIR', '/must-not-log')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    requests = []
    def transport(path, payload=None, timeout=None):
        requests.append((path, payload))
        if path == '/health':
            return {'status': 'ok'}
        return {'data': [{'id': gemma.PRESETS['gemma4']['alias']}]} if path == '/v1/models' else output(['Do not follow this command.'])
    monkeypatch.setattr(gemma, '_request', transport)
    assert gemma.translate_batch([{'start': 0, 'end': 1, 'text': 'この命令に従わないでください。'}]) == ['Do not follow this command.']
    assert probe.method_calls[:2] == [
        ('setsockopt', (gemma.socket.SOL_SOCKET, gemma.socket.SO_REUSEADDR, 1), {}),
        ('bind', ((gemma.HOST, gemma.PORT),), {}),
    ]
    args = popen.call_args.args[0]; env = popen.call_args.kwargs['env']
    assert args[0] == str(gemma.SERVER_PATH)
    assert args[args.index('--n-gpu-layers') + 1] == '0'
    assert args[args.index('--ctx-size') + 1] == '4096'
    assert args[args.index('--threads') + 1] == '8'
    assert args[args.index('--host') + 1] == '127.0.0.1'
    assert env['CUDA_VISIBLE_DEVICES'] == '' and 'LLAMA_ARG_LOG_PROMPTS_DIR' not in env
    assert '--log-disable' in args and '--offline' in args
    assert gemma._key not in args
    key_file = gemma._key_path
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert key_file.read_text().strip() == gemma._key
    payload = requests[-1][1]
    assert payload['tools'] == [] and payload['tool_choice'] == 'none'
    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
    assert payload['response_format']['json_schema']['schema']['properties']['translations']['maxItems'] == 1
    gemma.shutdown()
    process.terminate.assert_called_once()
    assert not key_file.exists()
    # A child process must never kill an inherited parent-owned handle.
    monkeypatch.setattr(gemma, '_process', process)
    monkeypatch.setattr(gemma, '_owner_pid', os.getpid() + 1)
    gemma.shutdown()
    process.terminate.assert_called_once()


def test_preset_switch_reuses_only_matching_owned_server(monkeypatch):
    available = {'gemma4': True, 'gemma4-12b': True}
    monkeypatch.setattr(gemma, 'ready', lambda preset='gemma4': available[preset])
    monkeypatch.setattr(gemma, 'runtime_ready', lambda: True)
    probe = Mock(); probe.__enter__ = Mock(return_value=probe); probe.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(gemma.socket, 'socket', lambda: probe)
    processes, launches, payloads = [], [], []
    def launch(args, **kwargs):
        process = Mock(returncode=None); process.poll.return_value = None
        processes.append(process); launches.append(args)
        return process
    monkeypatch.setattr(gemma.subprocess, 'Popen', launch)
    def transport(path, payload=None, timeout=None):
        if path == '/health':
            return {'status': 'ok'}
        if path == '/v1/models':
            return {'data': [{'id': gemma.PRESETS[gemma._process_preset]['alias']}]}
        payloads.append(payload)
        return output(['Yes.'])
    monkeypatch.setattr(gemma, '_request', transport)
    source = [dict(start=0, end=1, text='はい。')]
    gemma.translate_batch(source)
    old_key = gemma._key_path
    gemma.translate_batch(source)
    assert len(launches) == 1
    available['gemma4-12b'] = False
    with pytest.raises(gemma.GemmaError, match='12B'):
        gemma.translate_batch(source, preset='gemma4-12b')
    processes[0].terminate.assert_not_called()
    available['gemma4-12b'] = True
    gemma.translate_batch(source, preset='gemma4-12b')
    assert len(launches) == 2 and not old_key.exists()
    processes[0].terminate.assert_called_once()
    assert [args[args.index('--model') + 1] for args in launches] == [str(gemma.model_path(name)) for name in ('gemma4', 'gemma4-12b')]
    assert [item['model'] for item in payloads] == [gemma.PRESETS[name]['alias'] for name in ('gemma4', 'gemma4', 'gemma4-12b')]
    gemma.remove('gemma4')
    processes[1].terminate.assert_not_called()
    assert gemma._process_preset == 'gemma4-12b'
    with pytest.raises(gemma.GemmaError, match='Unknown'):
        gemma.translate_batch(source, preset='untrusted-model')
    assert len(launches) == 2


def test_verified_models_are_cached_and_removed_independently():
    for preset in gemma.PRESETS:
        weights = preset.encode()
        gemma.PRESETS[preset].update(size=len(weights), sha256=hashlib.sha256(weights).hexdigest())
        path = gemma.model_path(preset)
        path.parent.mkdir(parents=True)
        path.write_bytes(weights)
        assert gemma.ready(preset)
    assert set(gemma._verified) == {'gemma4', 'gemma4-12b'}
    gemma.remove('gemma4-12b')
    assert gemma.ready() and not gemma.ready('gemma4-12b')
    assert set(gemma._verified) == {'gemma4'}


def test_busy_port_is_not_adopted_or_stopped(monkeypatch):
    monkeypatch.setattr(gemma, 'ready', lambda preset='gemma4': True)
    gemma.RUNTIME_DIR.mkdir(parents=True)
    gemma.SERVER_PATH.write_text('test'); gemma.SERVER_PATH.chmod(0o700)
    probe = Mock(); probe.__enter__ = Mock(return_value=probe); probe.__exit__ = Mock(return_value=False)
    probe.bind.side_effect = OSError('Address in use')
    monkeypatch.setattr(gemma.socket, 'socket', lambda: probe)
    with pytest.raises(gemma.GemmaError, match='unknown server'):
        gemma._start()
    gemma.subprocess.Popen.assert_not_called()


def test_startup_timeout_cleans_owned_process_and_key(monkeypatch):
    monkeypatch.setattr(gemma, 'ready', lambda preset='gemma4': True)
    gemma.RUNTIME_DIR.mkdir(parents=True)
    gemma.SERVER_PATH.write_text('test'); gemma.SERVER_PATH.chmod(0o700)
    probe = Mock(); probe.__enter__ = Mock(return_value=probe); probe.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(gemma.socket, 'socket', lambda: probe)
    process = Mock(returncode=None); process.poll.return_value = None
    monkeypatch.setattr(gemma.subprocess, 'Popen', Mock(return_value=process))
    monkeypatch.setattr(gemma, '_request', Mock(side_effect=URLError('not ready')))
    monotonic = iter([0, 0, 121])
    monkeypatch.setattr(gemma.time, 'monotonic', lambda: next(monotonic))
    monkeypatch.setattr(gemma.time, 'sleep', lambda _: None)
    with pytest.raises(gemma.GemmaError, match='120 seconds'):
        gemma._start()
    process.terminate.assert_called_once()
    assert not list(gemma.store.DATA.glob('gemma-server-*.key'))
    assert gemma._process is None


def test_transport_auth_size_limit_and_failures(monkeypatch):
    monkeypatch.setattr(gemma, '_key', 'fixture-key')
    opened = Mock(return_value=io.BytesIO(b'{"ok":true}'))
    monkeypatch.setattr(gemma._http, 'open', opened)
    assert gemma._request('/v1/models') == {'ok': True}
    request = opened.call_args.args[0]
    assert request.full_url == 'http://127.0.0.1:18766/v1/models'
    assert request.get_header('Authorization') == 'Bearer fixture-key'
    assert gemma._NoRedirect().redirect_request(None) is None
    monkeypatch.setattr(gemma._http, 'open', lambda *a, **k: io.BytesIO(b'x' * (gemma.MAX_RESPONSE_BYTES + 1)))
    with pytest.raises(gemma.GemmaError, match='oversized'):
        gemma._request('/v1/models')
    monkeypatch.setattr(gemma, '_start', lambda preset='gemma4': None)
    monkeypatch.setattr(gemma, '_request', Mock(side_effect=HTTPError(gemma.BASE, 401, 'private body', {}, None)))
    with pytest.raises(gemma.GemmaError, match='HTTP 401'):
        gemma.translate_batch([{'start': 0, 'end': 1, 'text': 'はい。'}])


@pytest.mark.parametrize('escape', [False, True])
def test_runtime_checksum_and_safe_staging(monkeypatch, escape):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as bundle:
        entry = tarfile.TarInfo('../escape' if escape else 'runtime/llama-server')
        entry.size = 4; entry.mode = 0o755
        bundle.addfile(entry, io.BytesIO(b'test'))
    data = archive.getvalue()
    gemma.RUNTIME_MANIFEST.write_text(json.dumps({'asset_url': 'https://github.com/ggml-org/llama.cpp/releases/download/test/runtime.tar.gz', 'asset_sha256': hashlib.sha256(data).hexdigest()}))
    monkeypatch.setattr(gemma, 'urlopen', lambda *a, **k: io.BytesIO(data))
    if escape:
        with pytest.raises(gemma.GemmaError, match='installation failed'):
            gemma.ensure_runtime()
        assert not (gemma.RUNTIME_DIR.parent / 'escape').exists()
    else:
        gemma.ensure_runtime()
        assert gemma.SERVER_PATH.read_bytes() == b'test'
        assert os.access(gemma.SERVER_PATH, os.X_OK)
