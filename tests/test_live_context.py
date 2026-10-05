"""Isolated live context/reconciliation checks; no model or browser is launched."""
import os
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import Mock

if 'JREAD_DATA_DIR' not in os.environ:
    _data=tempfile.TemporaryDirectory(prefix='jread-context-test-')
    os.environ['JREAD_DATA_DIR']=_data.name

import numpy as np
import pytest
from fastapi.testclient import TestClient
from jread import engine,store
from jread.server import app,EXTENSION,reconcile_overlap,source_context

PCM=np.full(16000,10000,dtype='<i2').tobytes()


def cue(source,start=7,end=8,text='A fragment',**fields):
    return dict(id=uuid.uuid4().hex,start=start,end=end,text=text,source_text=source,status='final')|fields


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(engine,'STATE',dict(active_job=None,live=False,downloading=False,error=''))
    monkeypatch.setattr(engine,'status',lambda:{'ready':True})
    monkeypatch.setattr(engine,'live_infer',Mock(side_effect=AssertionError('Unexpected inference')))
    return TestClient(app,base_url='http://127.0.0.1:8765')


def options(monkeypatch,**patch):
    config=store.settings()|dict(translator='gemma4',caption_context=False,caption_boundaries=False,caption_glossary='none')|patch
    monkeypatch.setattr(store,'settings',lambda values=None:dict(config))
    return config


def connect(client):
    return client.websocket_connect('ws://127.0.0.1:8765/api/live',headers={'Origin':EXTENSION})


def hello(ws):
    ws.send_json(dict(type='hello',token=store.TOKEN,epoch=0))
    return ws.receive_json()['job_id']


def audio(ws,seq,time,epoch=0):
    ws.send_json(dict(type='audio',seq=seq,media_time=time,epoch=epoch,sample_rate=16000))
    ws.send_bytes(PCM)


def acknowledged(ws,seq):
    cues=[]
    while True:
        message=ws.receive_json()
        assert message['type'] in ('cue','ack'),message
        if message['type']=='cue':cues.append(message['cue'])
        elif message['seq']==seq:return cues


@pytest.mark.parametrize('language', ['es', 'fr', 'de', 'it', 'pt', 'ko', 'zh', 'ru', 'ar', 'hi', 'en'])
def test_live_handshake_freezes_selected_language(client, monkeypatch, language):
    options(monkeypatch, caption_glossary='aikatsu_pripara')
    def infer(samples, actual_language, offset, rate, **kwargs):
        assert actual_language == language
        return [cue('Original speech', offset, offset+1, text='Hello.')]
    monkeypatch.setattr(engine, 'live_infer', infer)
    with connect(client) as ws:
        ws.send_json(dict(type='hello', token=store.TOKEN, epoch=0, language=language))
        ready=ws.receive_json()
        assert ready['type']=='ready'
        job=ready['job_id']
        audio(ws, 0, 0)
        assert acknowledged(ws, 0)[0]['text']=='Hello.'
        ws.send_json({'type':'stop'})
        assert ws.receive_json()['type']=='stopped'
    assert store.get(job)['language']==language
    assert store.get(job)['caption_glossary']=='none'


@pytest.mark.parametrize('language', ['auto', 'unknown', [], {'code':'es'}, None])
def test_live_rejects_invalid_language_before_inference(client, monkeypatch, language):
    with connect(client) as ws:
        ws.send_json(dict(type='hello', token=store.TOKEN, epoch=0, language=language))
        reply=ws.receive_json()
        assert reply['type']=='error' and reply['detail']=='Unsupported source language.'
    assert not engine.STATE['live']
    engine.live_infer.assert_not_called()


def test_reconciliation_is_conservative_and_preserves_source_clock():
    old=cue('セーション',609.289542,610.289542)
    extended=cue('センセーション',609.519542,612.219542,text='Sensation')
    combined,changed=reconcile_overlap([old],[extended])
    assert len(combined)==len(changed)==1 and changed[0]['id']==old['id']
    assert (changed[0]['start'],changed[0]['end'])==(extended['start'],extended['end'])
    assert changed[0]['source_text']==extended['source_text'] and changed[0]['status']=='needs_review'
    assert old['text']=='A fragment' and extended['id']!=old['id'],'Helper must not mutate its inputs.'
    duplicate=cue('センセーション',610,612,text='Different English wording')
    assert reconcile_overlap(combined,[duplicate])[1]==[]
    repeated=cue('センセーション',613,615)
    assert reconcile_overlap(combined,[repeated])[1]==[repeated]
    assert reconcile_overlap([old],[cue('別の日本語',609.5,611,text=old['text'])])==([old],[])
    for before,after in [('私は行く','私は行かない'),('私は行く。','私は行く。彼は残る。'),('私','私は学園へ通う')]:
        previous=cue(before)
        incoming=cue(after,7.1,9,text='Different utterance')
        assert reconcile_overlap([previous],[incoming])[0]==[previous,incoming]
    edited=old|{'status':'user_edited'}
    assert reconcile_overlap([edited],[extended])[0]==[edited,extended]
    # Each translated piece can share its entire Japanese ASR segment.
    pieces=[cue('同じ日本語です',0,1,text='First part'),cue('同じ日本語です',1,2,text='Second part')]
    assert reconcile_overlap([],pieces)==(pieces,pieces)
    assert len(reconcile_overlap(pieces,[cue('同じ日本語です',.1,1.9)])[0])==3
    assert len(reconcile_overlap(pieces,[cue('同じ日本語です',.1,.9)])[0])==3
    previous_repeat=cue(old['source_text'],0,1)
    repeated,updates=reconcile_overlap([previous_repeat,old],[extended])
    assert len(repeated)==2 and updates[0]['id']==old['id']


@pytest.mark.parametrize('enabled',[False,True])
def test_live_context_is_bounded_deduplicated_and_session_scoped(client,monkeypatch,enabled):
    config=options(monkeypatch,caption_context=enabled)
    seen=[]
    japanese='今日は天気がいいです。'
    def infer(samples,language,offset,rate,*,context=''):
        seen.append(context)
        return [cue(japanese,offset,offset+.5,text='First'),cue(japanese,offset+.5,offset+1,text='Second')]
    monkeypatch.setattr(engine,'live_infer',infer)
    with connect(client) as ws:
        job=hello(ws)
        config['caption_context']=not enabled  # A session must retain its snapshot.
        audio(ws,0,0);acknowledged(ws,0)
        audio(ws,1,7);acknowledged(ws,1)
        ws.send_json({'type':'stop'});assert ws.receive_json()['type']=='stopped'
    assert seen==['',japanese if enabled else '']
    assert store.get(job)['caption_context']==enabled
    config['caption_context']=enabled
    with connect(client) as ws:
        hello(ws);audio(ws,0,0);acknowledged(ws,0)
        ws.send_json({'type':'stop'});assert ws.receive_json()['type']=='stopped'
    assert seen[-1]==''
    assert source_context('前の話',[cue(japanese),cue(japanese)])=='前の話 '+japanese
    assert len(source_context('',[cue('日'*200)]))==150


def test_epoch_reset_rejects_inflight_source_context(client,monkeypatch):
    import asyncio
    options(monkeypatch,caption_context=True)
    started,release=threading.Event(),threading.Event()
    seen=[]
    original_put=asyncio.Queue.put_nowait
    def put(queue,item):
        result=original_put(queue,item)
        if isinstance(item,tuple) and isinstance(item[0],dict) and item[0].get('epoch')==1:
            release.set()
        return result
    monkeypatch.setattr(asyncio.Queue,'put_nowait',put)
    def infer(samples,language,offset,rate,*,context=''):
        seen.append((offset,context))
        if offset==7:
            started.set();assert release.wait(2)
        japanese='前の場面。' if offset==0 else '破棄する古い結果。' if offset==7 else '新しい場面。'
        return [cue(japanese,offset,offset+1,text=str(offset))]
    monkeypatch.setattr(engine,'live_infer',infer)
    with connect(client) as ws:
        job=hello(ws)
        audio(ws,0,0);acknowledged(ws,0)
        audio(ws,1,7);assert started.wait(2)
        ws.send_json(dict(type='reset',epoch=1));audio(ws,2,20,1)
        accepted=acknowledged(ws,2)
        assert [item['text'] for item in accepted]==['20']
        audio(ws,3,27,1);acknowledged(ws,3)
        ws.send_json({'type':'stop'});assert ws.receive_json()['type']=='stopped'
    assert seen==[(0,''),(7,'前の場面。'),(20,''),(27,'新しい場面。')]
    assert all(item['source_text']!='破棄する古い結果。' for item in store.get(job)['cues'])


def test_superseded_fragment_updates_saved_and_displayed_id(client,monkeypatch):
    options(monkeypatch,caption_boundaries=True)
    old=cue('セーション',7.0,8.0,text='tion')
    extended=cue('センセーション',7.2,9,text='Sensation')
    monkeypatch.setattr(engine,'live_infer',lambda *args,**kwargs:[old] if args[2]==0 else [extended])
    with connect(client) as ws:
        job=hello(ws)
        audio(ws,0,0);first=acknowledged(ws,0)
        audio(ws,1,7);updated=acknowledged(ws,1)
        assert len(first)==len(updated)==1 and first[0]['id']==updated[0]['id']==old['id']
        assert updated[0]['text']=='Sensation'
        ws.send_json({'type':'stop'});assert ws.receive_json()['type']=='stopped'
    saved=store.get(job)
    assert saved['caption_boundaries'] and saved['cue_count']==1
    assert saved['cues'][0]['id']==old['id'] and saved['cues'][0]['text']=='Sensation'


@pytest.mark.parametrize('boundaries',[False,True])
def test_boundary_option_does_not_change_model_context(client,monkeypatch,boundaries):
    options(monkeypatch,caption_context=True,caption_boundaries=boundaries)
    old=cue('セーション',7,8,text='Same English')
    extended=cue('センセーション',7.2,9,text='Same English')
    seen=[]
    def infer(samples,language,offset,rate,*,context=''):
        seen.append(context)
        return [old] if offset==0 else [extended] if offset==7 else [cue('次の話です。',14,15,text='Next')]
    monkeypatch.setattr(engine,'live_infer',infer)
    with connect(client) as ws:
        job=hello(ws)
        for seq in range(3):audio(ws,seq,seq*7);acknowledged(ws,seq)
        ws.send_json({'type':'stop'});assert ws.receive_json()['type']=='stopped'
    assert seen==['','セーション','セーション センセーション']
    assert store.get(job)['cue_count']==2,'Baseline exact-English overlap suppression must remain.'


def test_boundary_reconciliation_does_not_cross_epoch(client,monkeypatch):
    options(monkeypatch,caption_boundaries=True)
    old=cue('セーション',7,8,text='tion')
    extended=cue('センセーション',7.2,9,text='Sensation')
    monkeypatch.setattr(engine,'live_infer',lambda *args,**kwargs:[old] if args[2]==0 else [extended])
    with connect(client) as ws:
        job=hello(ws)
        audio(ws,0,0);acknowledged(ws,0)
        ws.send_json(dict(type='reset',epoch=1))
        audio(ws,1,7,1);updated=acknowledged(ws,1)
        assert updated[0]['id']==extended['id']
        ws.send_json({'type':'stop'});assert ws.receive_json()['type']=='stopped'
    assert store.get(job)['cue_count']==2


def test_whisper_snapshots_context_and_boundaries_as_disabled(client,monkeypatch):
    options(monkeypatch,translator='whisper',caption_context=True,caption_boundaries=True)
    from jread import server
    monkeypatch.setattr(server,'reconcile_overlap',Mock(side_effect=AssertionError('Whisper has no Japanese source history')))
    calls=[]
    def infer(samples,language,offset,rate):
        calls.append(offset)
        return [cue('',offset,offset+1,text='Direct Whisper translation')]
    monkeypatch.setattr(engine,'live_infer',infer)
    with connect(client) as ws:
        job=hello(ws);audio(ws,0,0);acknowledged(ws,0)
        ws.send_json({'type':'stop'});assert ws.receive_json()['type']=='stopped'
    saved=store.get(job)
    assert calls==[0] and saved['translator']=='whisper'
    assert saved['caption_context'] is False and saved['caption_boundaries'] is False


def test_same_id_updates_native_and_queued_delayed_cue_without_duplication():
    import subprocess
    script=r'''
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';
let handler;
const video={isConnected:true,currentSrc:'fixture',src:'fixture',currentTime:20,playbackRate:1,paused:false,duration:60,textTracks:[],getAttribute:()=>null,setAttribute(){},removeAttribute(){},addEventListener(){},addTextTrack(kind,label){
  const track={label,mode:'disabled',cues:[],addCue(cue){this.cues.push(cue)},removeCue(cue){this.cues.splice(this.cues.indexOf(cue),1)}};
  this.textTracks.push(track);return track;
}};
const built=await build({entryPoints:['extension/src/content.ts'],bundle:true,write:false,format:'iife',platform:'browser'});
vm.runInNewContext(built.outputFiles[0].text,{
  chrome:{runtime:{id:'fixture',onMessage:{addListener:value=>handler=value},sendMessage:async()=>({ok:true})}},
  document:{querySelectorAll:()=>[video],createElement:()=>({remove(){}}),head:{append(){}},documentElement:{}},
  VTTCue:class {constructor(start,end,text){this.startTime=start;this.endTime=end;this.text=text}},
  MutationObserver:class {observe(){} disconnect(){}},crypto:webcrypto,AbortController,Date,setInterval:()=>1,clearInterval(){},
});
const command=message=>{let result;handler({target:'content',...message},{id:'fixture'},reply=>result=reply);assert(result?.ok,result?.error);return result.data};
const id=command({type:'scan'})[0].id;
command({type:'bind',videoId:id});
const original={id:'stable-id',start:7,end:8,text:'tion',source_text:'セーション'};
command({type:'cue',epoch:0,cue:original});
const delayed=video.textTracks.find(track=>track.label==='JREAD delayed live');
const native=video.textTracks.find(track=>track.label==='JREAD English');
const scheduled=delayed.cues[0];
command({type:'cue',epoch:0,cue:{...original,start:7.2,end:9,text:'Sensation',source_text:'センセーション'}});
assert.equal(native.cues.length,1);assert.equal(native.cues[0].startTime,7.2);assert.equal(native.cues[0].endTime,9);
assert.equal(delayed.cues.length,1);assert.equal(delayed.cues[0],scheduled);
assert(delayed.cues[0].text.includes('Sensation'));assert(native.cues[0].text.includes('Sensation'));
'''
    subprocess.run(['node','--input-type=module','-e',script],cwd=Path(__file__).resolve().parents[1],check=True,timeout=20,capture_output=True,text=True)
