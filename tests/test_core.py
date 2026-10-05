import os
import tempfile
from pathlib import Path

os.environ['JREAD_DATA_DIR']=tempfile.mkdtemp(prefix='jread-test-')
import pytest
from fastapi.testclient import TestClient
from jread import engine,store,subtitles
from jread.server import app,EXTENSION,PAIR

client=TestClient(app,base_url='http://127.0.0.1:8765')
auth={'Authorization':'Bearer '+store.TOKEN}
SRT='1\n00:00:01,000 --> 00:00:03,250\nHello & welcome.\n\n2\n00:01:01,005 --> 00:01:02,500\nA second line.\n'


def test_subtitle_roundtrip_and_invalid_timing():
    cues=subtitles.parse(SRT)
    assert cues[1]['start']==61.005
    for kind in ('srt','vtt'):
        roundtrip=subtitles.parse(subtitles.export(cues,kind))
        assert [(c['start'],c['end'],c['text']) for c in roundtrip]==[(c['start'],c['end'],c['text']) for c in cues]
    for data in ['1\n00:00:03 --> 00:00:01\nBad','1\n00:60:00 --> 01:01:00\nBad','1\nnan:00 --> 00:01\nBad']:
        with pytest.raises(ValueError):subtitles.parse(data)


def test_long_generated_captions_preserve_words_and_flag_estimated_timing():
    text='If you search for the appropriate environmental course on the Internet, you will probably find the address of the local company.'
    cues=subtitles.assemble(3,13,text)
    assert len(cues)==2
    assert ' '.join(' '.join(c['text'].split()) for c in cues)==text
    assert cues[0]['start']==3 and cues[-1]['end']==13
    assert all(0<c['end']-c['start'] for c in cues)
    assert cues[0]['end']==cues[1]['start']
    assert all(len(c['text'].splitlines())<=2 and c['status']=='needs_review' for c in cues)
    assert subtitles.assemble(0,2,'Hello, world.')[0]['status']=='final'
    assert subtitles.assemble(0,.2,'Too quick to read.')[0]['status']=='needs_review'


def test_auth_origin_and_host_boundaries():
    assert client.get('/api/jobs').status_code==401
    assert client.get('/api/jobs',headers=auth|{'Origin':'https://evil.invalid'}).status_code==403
    assert client.get('/api/jobs',headers=auth|{'Host':'evil.invalid'}).status_code==403
    assert client.post('/api/import',headers=auth|{'content-length':'6000000'}).status_code==413
    assert client.patch('/api/settings',headers=auth,json={'device':'shell'}).status_code==422


def test_import_edit_undo_export_delete():
    response=client.post('/api/import',headers=auth,files={'file':('demo.srt',SRT.encode())})
    assert response.status_code==201,response.text
    job=response.json();cue=job['cues'][0]
    url=f"/api/jobs/{job['id']}/cues/{cue['id']}"
    changed=client.patch(url,headers=auth,json={'text':'Edited <b>text</b>','start':1.2,'end':3.3})
    assert changed.json()['status']=='user_edited'
    assert client.patch(url,headers=auth,json={'text':'bad','start':9,'end':2}).status_code==400
    assert client.post(url+'/undo',headers=auth).json()['text']==cue['text']
    exported=client.get(f"/api/jobs/{job['id']}/export?format=vtt",headers=auth)
    assert exported.status_code==200 and 'WEBVTT' in exported.text
    assert len(subtitles.parse(exported.text))==2
    assert client.delete(f"/api/jobs/{job['id']}",headers=auth).status_code==200
    assert client.get(f"/api/jobs/{job['id']}",headers=auth).status_code==404


def test_pairing_requires_approved_single_use_code():
    PAIR['attempts']=[]
    code=client.post('/api/pair/new',headers=auth).json()['code']
    assert client.post('/api/pair',json={'code':code}).status_code==403
    headers={'Origin':EXTENSION}
    result=client.post('/api/pair',headers=headers,json={'code':code})
    assert result.status_code==200,result.text
    paired={'Authorization':'Bearer '+result.json()['token'],'Origin':EXTENSION}
    assert client.get('/api/jobs',headers=paired).status_code==200
    assert client.post('/api/pair',headers=headers,json={'code':code}).status_code==403
    assert client.options('/api/jobs',headers={'Origin':EXTENSION,'Access-Control-Request-Method':'GET'}).headers['access-control-allow-origin']==EXTENSION


def test_companion_reuses_only_a_current_unused_pairing_code():
    assert client.post('/api/pair/new?reuse=true').status_code==401
    first=client.post('/api/pair/new',headers=auth).json()
    expires=PAIR['expires']
    reused=client.post('/api/pair/new?reuse=true',headers=auth).json()
    assert reused['code']==first['code'] and PAIR['expires']==expires
    assert 0<reused['expires_in']<=300
    PAIR['expires']=0
    renewed=client.post('/api/pair/new?reuse=true',headers=auth).json()
    assert renewed['code']!=first['code'] and renewed['expires_in']==300
    PAIR['used']=True
    after_use=client.post('/api/pair/new?reuse=true',headers=auth).json()
    assert after_use['code']!=renewed['code'] and not PAIR['used']
    refreshed=client.post('/api/pair/new',headers=auth).json()
    assert refreshed['code']!=after_use['code']


def test_media_boundary_rejects_playlists_before_probe():
    with pytest.raises(ValueError):engine.media_info(Path('/tmp/untrusted.m3u8'),'.m3u8')
    # Force a harmless WAV demuxer even if the uploaded bytes describe a network URL.
    path=Path(os.environ['JREAD_DATA_DIR'])/'not-a-wave.wav'
    path.write_text('#EXTM3U\nhttps://example.invalid/private\n')
    with pytest.raises(Exception):engine.media_info(path,'.wav')


def test_live_requires_authentication_and_valid_origin():
    with client.websocket_connect('ws://127.0.0.1:8765/api/live',headers={'Origin':EXTENSION}) as ws:
        ws.send_json({'type':'hello','token':'incorrect','epoch':0})
        assert ws.receive_json()['type']=='error'
    with pytest.raises(Exception):
        with client.websocket_connect('ws://127.0.0.1:8765/api/live',headers={'Origin':'https://evil.invalid'}):pass


def test_failed_model_switch_can_recover(monkeypatch):
    import faster_whisper
    monkeypatch.setattr(engine,'model_ready',lambda name:True)
    monkeypatch.setattr(engine,'_model',object())
    monkeypatch.setattr(engine,'_model_key',('small','cpu'))
    expected=object()
    def make(*args,**kwargs):
        if kwargs['device']=='cuda':raise RuntimeError('no GPU')
        return expected
    monkeypatch.setattr(faster_whisper,'WhisperModel',make)
    with pytest.raises(RuntimeError):engine.load('small','cuda')
    assert engine.load('small','cpu') is expected


def test_interrupted_upload_is_not_resumable():
    job=store.create('Interrupted upload',status='uploading',has_media=False)
    folder=store.MEDIA/job['id'];folder.mkdir();(folder/'source.part').write_bytes(b'partial')
    with TestClient(app,base_url='http://127.0.0.1:8765'):
        found=store.get(job['id'])
        assert found['status']=='failed' and not found['has_media']
        assert not folder.exists()


def test_restart_releases_pending_cancellation():
    job=store.create('Interrupted cancellation',status='cancelled',cancellation_pending=True)
    with TestClient(app,base_url='http://127.0.0.1:8765'):
        found=store.get(job['id'])
        assert found['status']=='cancelled' and not found['cancellation_pending']


@pytest.mark.parametrize('exceptional',[False,True])
def test_app_lifespan_stops_owned_gemma_runtime(monkeypatch,tmp_path,exceptional):
    import asyncio
    from unittest.mock import Mock
    from jread.server import lifespan
    runtime=Mock();runtime.poll.return_value=None
    key=tmp_path/'owned-runtime.key';key.write_text('fixture')
    for name,value in {'_process':runtime,'_process_preset':'gemma4','_owner_pid':os.getpid(),'_key':'fixture','_key_path':key}.items():
        monkeypatch.setattr(engine.gemma,name,value)
    async def run():
        async with lifespan(app):
            if exceptional:
                raise RuntimeError('Lifespan fixture failed')
    if exceptional:
        with pytest.raises(RuntimeError,match='Lifespan fixture failed'):
            asyncio.run(run())
    else:
        asyncio.run(run())
    runtime.terminate.assert_called_once()
    runtime.wait.assert_called_once_with(timeout=5)
    assert not key.exists() and engine.gemma._process is None


def test_gemma_translation_preserves_japanese_times_and_text(monkeypatch):
    sources=[dict(start=1.25,end=3.5,text='こんにちは。'),dict(start=4,end=6,text='元気ですか？')]
    seen=[]
    def translate(batch,context='',preset='gemma4'):
        seen.extend(batch)
        return ['Hello.','How are you?']
    monkeypatch.setattr(engine.gemma,'translate_batch',translate)
    results=list(engine.gemma_segments(sources))
    assert seen==sources
    assert [(s.start,s.end,s.source_text) for s in results]==[(s['start'],s['end'],s['text']) for s in sources]
    assert [s.text for s in results]==['Hello.','How are you?']


def test_gemma_long_segments_are_bounded_without_changing_source_clock(monkeypatch):
    original='日本語です。' * 240 + '\\"\n' * 500
    source=dict(start=2,end=20,text=original)
    seen=[]
    def translate(batch,context='',preset='gemma4'):
        engine.gemma._messages(batch,context)  # Exercise the real 2 KB boundary.
        seen.extend(batch)
        return ['English.'] * len(batch)
    monkeypatch.setattr(engine.gemma,'translate_batch',translate)
    result=list(engine.gemma_segments([source]))
    assert len(seen)>1 and ''.join(s['text'] for s in seen)==original
    assert len(result)==1 and (result[0].start,result[0].end)==(2,20)
    assert result[0].source_text==original and result[0].needs_review


def test_gemma_cancel_stops_before_next_batch_and_resume_skips_completed(monkeypatch):
    sources=[dict(start=i,end=i+1,text='はい。') for i in range(7)]
    calls=[]
    def translate(batch,context='',preset='gemma4'):
        calls.append(batch)
        return ['Yes.'] * len(batch)
    monkeypatch.setattr(engine.gemma,'translate_batch',translate)
    stopped=False
    segments=engine.gemma_segments(sources,start_at=1,should_stop=lambda:stopped)
    assert [next(segments).start for _ in range(4)]==[1,2,3,4]
    stopped=True
    assert list(segments)==[] and len(calls)==1


def test_gemma_readiness_requires_runtime_but_retains_model_removal(monkeypatch):
    monkeypatch.setattr(store,'settings',lambda:dict(model='small',device='cpu',translator='gemma4'))
    monkeypatch.setattr(engine,'model_ready',lambda _:True)
    monkeypatch.setattr(engine.gemma,'ready',lambda preset='gemma4':True)
    monkeypatch.setattr(engine.gemma,'runtime_ready',lambda:False)
    status=engine.status()
    assert status['gemma_model_ready'] and not status['gemma_ready'] and not status['ready']


@pytest.mark.parametrize('translator',['whisper','gemma4','gemma4-12b'])
def test_live_cues_are_clamped_to_captured_audio(monkeypatch,translator):
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setattr(store,'settings',lambda:dict(model='small',device='cpu',translator=translator))
    raw=[SimpleNamespace(start=-.5,end=2,text='はい。'),SimpleNamespace(start=3,end=4,text='Outside packet')]
    monkeypatch.setattr(engine,'load',lambda *a:SimpleNamespace(transcribe=lambda *a,**k:(iter(raw),None)))
    def translate(batch,context='',preset='gemma4'):
        assert preset==translator
        assert [(s['start'],s['end']) for s in batch]==[(0,1)]
        return ['Yes.']
    monkeypatch.setattr(engine.gemma,'translate_batch',translate)
    cues=engine.live_infer(np.zeros(16000), 'ja', 50, 2)
    assert len(cues)==1 and (cues[0]['start'],cues[0]['end'])==(50,52)


def test_translation_choice_is_fixed_for_saved_job():
    previous=store.settings()['translator']
    try:
        assert client.patch('/api/settings',headers=auth,json={'translator':'gemma4'}).status_code==200
        job=store.create('Gemma selection fixture')
        assert client.patch('/api/settings',headers=auth,json={'translator':'gemma4-12b'}).status_code==200
        newer=store.create('Gemma 12B selection fixture')
        assert store.get(job['id'])['translator']=='gemma4'
        assert store.get(newer['id'])['translator']=='gemma4-12b'
        assert client.patch('/api/settings',headers=auth,json={'translator':'remote-service'}).status_code==422
    finally:store.settings({'translator':previous})


def test_turbo_settings_validate_partial_updates_and_preserve_job_choices():
    previous=store.settings()
    try:
        store.settings({'model':'medium','translator':'whisper'})
        baseline=store.create('Saved medium baseline')
        response=client.patch('/api/settings',headers=auth,json={'model':'large-v3-turbo'})
        assert response.status_code==400 and 'Select Gemma' in response.json()['detail']
        assert store.settings()['model']=='medium'
        response=client.patch('/api/settings',headers=auth,json={'model':'large-v3-turbo','translator':'gemma4'})
        assert response.status_code==200,response.text
        turbo=store.create('Saved Turbo with E4B')
        assert client.patch('/api/settings',headers=auth,json={'translator':'whisper'}).status_code==400
        assert store.settings()['translator']=='gemma4'
        assert client.patch('/api/settings',headers=auth,json={'translator':'gemma4-12b'}).status_code==200
        assert client.patch('/api/settings',headers=auth,json={'model':'medium','translator':'whisper'}).status_code==200
        assert (store.get(baseline['id'])['model'],store.get(baseline['id'])['translator'])==('medium','whisper')
        assert (store.get(turbo['id'])['model'],store.get(turbo['id'])['translator'])==('large-v3-turbo','gemma4')
    finally:
        store.settings(previous)


@pytest.mark.parametrize('name,repository',[
    ('small','Systran/faster-whisper-small'),
    ('medium','Systran/faster-whisper-medium'),
    ('large-v3-turbo','dropbox-dash/faster-whisper-large-v3-turbo'),
])
def test_asr_download_uses_pinned_repo_and_complete_assets(monkeypatch,tmp_path,name,repository):
    import hashlib
    import huggingface_hub
    from types import SimpleNamespace
    revision=engine.PRESETS[name][0]
    monkeypatch.setattr(store,'MODELS',tmp_path)
    monkeypatch.setitem(engine.PRESETS,name,(revision,hashlib.sha256(b'test weights').hexdigest()))
    monkeypatch.setattr(engine,'STATE',dict(active_job=None,downloading=False,error='',live=False))
    def download(repo,**kwargs):
        assert repo==repository and kwargs['revision']==revision
        assert set(kwargs['allow_patterns'])=={'model.bin','config.json','preprocessor_config.json','tokenizer.json','vocabulary.json','vocabulary.txt'}
        folder=kwargs['local_dir'];folder.mkdir()
        (folder/'model.bin').write_bytes(b'test weights')
        for filename in ('config.json','tokenizer.json','vocabulary.json'):
            (folder/filename).write_text('{}')
        (folder/'preprocessor_config.json').write_text('{"feature_size":128}')
        return str(folder)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    monkeypatch.setattr(engine.threading,'Thread',lambda target,**kwargs:SimpleNamespace(start=target))
    engine.download(name)
    assert not engine.STATE['downloading'] and not engine.STATE['error']
    assert engine.model_ready(name) and (tmp_path/name/'.verified').read_text()==revision
    if name=='large-v3-turbo':
        (tmp_path/name/'preprocessor_config.json').write_text('{"feature_size":80}')
        assert not engine.model_ready(name)
        (tmp_path/name/'preprocessor_config.json').unlink()
        assert not engine.model_ready(name)


def test_turbo_status_and_model_endpoints(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(store,'settings',lambda:dict(model='large-v3-turbo',device='cuda',translator='gemma4'))
    monkeypatch.setattr(engine,'model_ready',lambda _:True)
    monkeypatch.setattr(engine.gemma,'ready',lambda preset:True)
    monkeypatch.setattr(engine.gemma,'runtime_ready',lambda:True)
    monkeypatch.setattr(engine.shutil,'disk_usage',lambda _:SimpleNamespace(free=4_000_000_000))
    actions=[]
    monkeypatch.setattr(engine,'download',lambda name:actions.append(('download',name)))
    monkeypatch.setattr(engine,'remove_model',lambda name:actions.append(('remove',name)))
    for action in ('download','remove'):
        response=client.post(f'/api/models/{action}',headers=auth,json={'model':'large-v3-turbo'})
        assert response.status_code==200,response.text
        assert response.json()['ready'] and response.json()['model_label']=='Large v3 Turbo'
    assert actions==[('download','large-v3-turbo'),('remove','large-v3-turbo')]


def test_invalid_stored_turbo_choice_fails_before_loading_or_inference(monkeypatch):
    import numpy as np
    def unexpected_load(*args,**kwargs):
        pytest.fail('Invalid Turbo translation must fail before loading the model')
    monkeypatch.setattr(engine,'load',unexpected_load)
    monkeypatch.setattr(engine,'model_ready',lambda _:True)
    monkeypatch.setattr(store,'settings',lambda:dict(model='large-v3-turbo',device='cpu',translator='whisper'))
    assert not engine.status()['ready'] and 'Select Gemma' in engine.status()['error']
    with pytest.raises(ValueError,match='Select Gemma'):
        engine.live_infer(np.zeros(16000),'ja',0,1)
    job=store.create('Invalid stored Turbo choice')
    engine.process(job['id'])
    found=store.get(job['id'])
    assert found['status']=='failed' and 'Select Gemma' in found['error']


def test_selected_12b_status_and_model_actions(monkeypatch):
    from types import SimpleNamespace
    selected=[]
    monkeypatch.setattr(engine.shutil,'disk_usage',lambda _:SimpleNamespace(free=10_000_000_000))
    monkeypatch.setattr(store,'settings',lambda:dict(model='small',device='cpu',translator='gemma4-12b'))
    monkeypatch.setattr(engine,'model_ready',lambda _:True)
    monkeypatch.setattr(engine.gemma,'runtime_ready',lambda:True)
    def ready(preset='gemma4'):
        selected.append(preset)
        return preset=='gemma4-12b'
    monkeypatch.setattr(engine.gemma,'ready',ready)
    status=engine.status()
    assert status['ready'] and status['gemma_ready'] and selected==['gemma4-12b']
    assert status['translator_label']=='Gemma 4 12B'
    assert status['translator_download_bytes']==6975879296
    actions=[]
    monkeypatch.setattr(engine,'download',lambda name:actions.append(('download',name)))
    monkeypatch.setattr(engine,'remove_model',lambda name:actions.append(('remove',name)))
    for action in ('download','remove'):
        response=client.post(f'/api/models/{action}',headers=auth,json={'model':'gemma4-12b'})
        assert response.status_code==200,response.text
    assert actions==[('download','gemma4-12b'),('remove','gemma4-12b')]


def test_resumed_e4b_job_uses_saved_preset_after_selecting_12b(monkeypatch):
    import json
    import numpy as np
    from types import SimpleNamespace
    job=store.create('Saved E4B retry fixture',translator='gemma4',duration=1,source_done=True,has_media=True)
    folder=store.MEDIA/job['id'];folder.mkdir()
    (folder/'audio.wav').touch()
    (folder/'source-cues.json').write_text(json.dumps([dict(start=0,end=1,text='はい。')]))
    monkeypatch.setattr(store,'settings',lambda:dict(model='small',device='cpu',translator='gemma4-12b'))
    monkeypatch.setattr(engine,'load',lambda **kwargs:SimpleNamespace())
    monkeypatch.setattr(engine,'read_audio',lambda _:np.zeros(16000))
    presets=[]
    def translate(batch,context='',preset='gemma4'):
        presets.append(preset)
        return ['Yes.']
    monkeypatch.setattr(engine.gemma,'translate_batch',translate)
    engine.process(job['id'])
    result=store.get(job['id'])
    assert result['status']=='complete' and result['translator']=='gemma4'
    assert result['cue_count']==1 and presets==['gemma4']


def test_cancel_queued_job_does_not_leave_pending():
    from concurrent.futures import Future
    job=store.create('Queued fixture')
    engine._futures[job['id']]=Future()
    result=engine.cancel(job['id'])
    assert result['status']=='cancelled' and not result['cancellation_pending']
    assert engine._futures.pop(job['id']).cancelled()


def test_live_stop_drains_and_epoch_discards_stale_audio(monkeypatch):
    import time
    import numpy as np
    monkeypatch.setattr(engine,'model_ready',lambda name:True)
    def infer(samples,language,offset,rate):
        time.sleep(.03)
        return [dict(id=f'cue-{offset}',start=offset,end=offset+1,text='Hello',source_text='',status='final')]
    monkeypatch.setattr(engine,'live_infer',infer)
    with client.websocket_connect('ws://127.0.0.1:8765/api/live',headers={'Origin':EXTENSION}) as ws:
        ws.send_json({'type':'hello','token':store.TOKEN,'epoch':0})
        job=ws.receive_json()['job_id']
        data=np.full(16000,10000,dtype='<i2').tobytes()
        ws.send_json({'type':'audio','epoch':0,'seq':0,'media_time':0,'sample_rate':16000});ws.send_bytes(data)
        ws.send_json({'type':'reset','epoch':1})
        ws.send_json({'type':'audio','epoch':1,'seq':1,'media_time':20,'sample_rate':16000});ws.send_bytes(data)
        ws.send_json({'type':'stop'})
        messages=[]
        while True:
            msg=ws.receive_json();messages.append(msg)
            if msg['type']=='stopped':break
        assert [m['cue']['start'] for m in messages if m['type']=='cue']==[20]
        assert any(m['type']=='ack' and m['seq']==1 for m in messages)
    result=store.get(job)
    assert result['status']=='complete' and result['cue_count']==1 and not result['error']


@pytest.mark.parametrize('safe',[False,True])
def test_live_inference_preserves_safe_gemma_error_in_socket_and_job(monkeypatch,safe):
    import numpy as np
    detail='Gemma port 18766 is already in use. JREAD will not reuse or stop an unknown server.'
    expected=detail if safe else 'Live inference failed. Check the selected device/model in the companion.'
    monkeypatch.setattr(engine,'status',lambda:{'ready':True})
    def infer(*args):
        raise engine.gemma.GemmaError(detail) if safe else RuntimeError('private implementation detail')
    monkeypatch.setattr(engine,'live_infer',infer)
    with client.websocket_connect('ws://127.0.0.1:8765/api/live',headers={'Origin':EXTENSION}) as ws:
        ws.send_json({'type':'hello','token':store.TOKEN,'epoch':0})
        job=ws.receive_json()['job_id']
        ws.send_json({'type':'audio','epoch':0,'seq':0,'media_time':0,'sample_rate':16000})
        ws.send_bytes(np.full(16000,10000,dtype='<i2').tobytes())
        ws.send_json({'type':'stop'})
        assert ws.receive_json()=={'type':'error','detail':expected}
    result=store.get(job)
    assert result['status']=='failed' and result['error']==expected


def test_live_failed_worker_with_pending_packets_finishes_without_drain_timeout(monkeypatch):
    import asyncio
    import threading
    import numpy as np
    started,release,finished=threading.Event(),threading.Event(),threading.Event()
    monkeypatch.setattr(engine,'status',lambda:{'ready':True})
    def infer(*args):
        started.set()
        assert release.wait(2),'The stop command never began draining.'
        raise engine.gemma.GemmaError('Fixture inference failed.')
    monkeypatch.setattr(engine,'live_infer',infer)
    original_join=asyncio.Queue.join
    async def join(queue):
        assert queue.qsize()==2,'Regression needs pending packets behind the failed worker.'
        release.set()
        await original_join(queue)
    monkeypatch.setattr(asyncio.Queue,'join',join)
    original_update=store.update
    def update(job_id,**fields):
        result=original_update(job_id,**fields)
        if fields.get('status')=='failed':finished.set()
        return result
    monkeypatch.setattr(store,'update',update)
    with client.websocket_connect('ws://127.0.0.1:8765/api/live',headers={'Origin':EXTENSION}) as ws:
        ws.send_json({'type':'hello','token':store.TOKEN,'epoch':0})
        job=ws.receive_json()['job_id']
        data=np.full(16000,10000,dtype='<i2').tobytes()
        for seq in range(3):
            ws.send_json({'type':'audio','epoch':0,'seq':seq,'media_time':seq,'sample_rate':16000})
            ws.send_bytes(data)
            if seq==0:assert started.wait(2)
        ws.send_json({'type':'stop'})
        assert ws.receive_json()=={'type':'error','detail':'Fixture inference failed.'}
        assert finished.wait(2),'A failed worker must not wait120 seconds for abandoned queued packets.'
        assert store.get(job)['status']=='failed' and not engine.STATE['live']
