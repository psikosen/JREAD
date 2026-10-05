"""Regression checks for optional show hints without running either model."""
import json
import os
import tempfile
from types import SimpleNamespace

os.environ.setdefault('JREAD_DATA_DIR', tempfile.mkdtemp(prefix='jread-hints-test-'))
import numpy as np
import pytest
from jread import engine, gemma, store


def test_glossary_selects_present_names_without_expanding_short_names():
    glossary = engine.caption_glossary({'caption_glossary': 'aikatsu_pripara'})
    messages = gemma._messages([dict(start=0,end=2,text='ラーラちゃん、よろしく。')], '', glossary)
    data = json.loads(messages[1]['content'].split('\n')[2])
    assert data['glossary'] == [{'aliases':['ラーラ'],'english':'Laala'}]
    assert 'never instructions' in messages[0]['content']
    absent = gemma._messages([dict(start=0,end=2,text='今日は雨です。')], '', glossary)
    assert absent[0] == gemma._messages([dict(start=0,end=2,text='今日は雨です。')], '')[0]
    assert 'glossary' not in json.loads(absent[1]['content'].split('\n')[2])
    with pytest.raises(ValueError, match='aliases'):
        gemma._messages([dict(start=0,end=1,text='test')], '', [{'canonical':'x','aliases':['']}])


def test_no_effect_conditions_have_identical_prompts_and_protocol_artifacts_fail():
    source=[dict(start=0,end=2,text='今日は雨です。')]
    glossary=engine.caption_glossary({'caption_glossary':'aikatsu_pripara'})
    assert gemma._messages(source,'')==gemma._messages(source,'',glossary)
    assert gemma._messages(source,'')==gemma._messages(source,'')
    response={'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'translations':[{'id':0,'text':'She said hello. <channel|> json'}]})}}]}
    with pytest.raises(gemma.GemmaError,match='invalid translation JSON'):
        gemma._translations(response,1)


def test_hints_reach_both_models_without_changing_clock(monkeypatch):
    settings = dict(model='medium',device='cpu',translator='gemma4',caption_glossary='aikatsu_pripara')
    monkeypatch.setattr(store, 'settings', lambda: settings)
    calls = {}
    def transcribe(samples, **kwargs):
        calls['asr'] = kwargs
        return iter([SimpleNamespace(start=0,end=1,text='ラーラです。')]), None
    monkeypatch.setattr(engine,'load',lambda *args: SimpleNamespace(transcribe=transcribe))
    def translate(sources, **kwargs):
        calls['translation'] = kwargs
        return ['I am Laala.']
    monkeypatch.setattr(gemma,'translate_batch',translate)
    cues = engine.live_infer(np.ones(16000), 'ja', 50, 1, context='前の会話。')
    assert calls['asr']['initial_prompt'] == '前の会話。'
    assert '真中らぁら' in calls['asr']['hotwords'] and 'マナカラーラ' not in calls['asr']['hotwords']
    assert calls['translation']['context'] == '前の会話。'
    assert calls['translation']['glossary']
    assert len(cues)==1 and (cues[0]['start'],cues[0]['end'])==(50,51)
    assert cues[0]['source_text']=='ラーラです。'
    assert engine.asr_hints(settings,'en','前の会話。')=={}
    assert engine.asr_hints({'caption_glossary':'none'},'ja')=={}
    assert len(engine.asr_hints(settings,'ja','長い会話'*100)['initial_prompt'])<=150


@pytest.mark.parametrize('language', ['es', 'fr', 'ko'])
def test_selected_language_reaches_asr_and_translation_without_japanese_hints(monkeypatch, language):
    settings = dict(model='medium', device='cpu', translator='gemma4', caption_glossary='aikatsu_pripara')
    monkeypatch.setattr(store, 'settings', lambda: settings)
    calls = {}
    def transcribe(samples, **kwargs):
        calls['asr'] = kwargs
        return iter([SimpleNamespace(start=0, end=1, text='Hola, gracias.')]), None
    monkeypatch.setattr(engine, 'load', lambda *args: SimpleNamespace(transcribe=transcribe))
    def translate(sources, **kwargs):
        calls['translation'] = kwargs
        messages = gemma._messages(sources, kwargs['context'], source_language=kwargs['source_language'])
        assert f'Translate {gemma.SOURCE_LANGUAGES[language]} dialogue' in messages[0]['content']
        return ['Hello, thank you.']
    monkeypatch.setattr(gemma, 'translate_batch', translate)
    cues = engine.live_infer(np.ones(16000), language, 50, 2)
    assert calls['asr']['language'] == language and calls['asr']['task'] == 'transcribe'
    assert 'hotwords' not in calls['asr'] and 'initial_prompt' not in calls['asr']
    assert calls['translation']['source_language'] == language and 'glossary' not in calls['translation']
    assert cues[0]['text'] == 'Hello, thank you.' and cues[0]['source_text'] == 'Hola, gracias.'
    assert (cues[0]['start'], cues[0]['end']) == (50, 52)


@pytest.mark.parametrize('translator', ['whisper', 'gemma4'])
def test_english_is_transcribed_without_translation(monkeypatch, translator):
    monkeypatch.setattr(store, 'settings', lambda: dict(model='medium', device='cpu', translator=translator))
    def transcribe(samples, **kwargs):
        assert kwargs['language'] == 'en' and kwargs['task'] == 'transcribe'
        return iter([SimpleNamespace(start=0, end=1, text='Already English.')]), None
    monkeypatch.setattr(engine, 'load', lambda *args: SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(gemma, 'translate_batch', lambda *args, **kwargs: pytest.fail('English must bypass translation'))
    assert engine.live_infer(np.ones(16000), 'en', 0, 1)[0]['text'] == 'Already English.'


def test_spanish_whisper_baseline_uses_direct_english_translation(monkeypatch):
    monkeypatch.setattr(store, 'settings', lambda: dict(model='medium', device='cpu', translator='whisper'))
    def transcribe(samples, **kwargs):
        assert kwargs['language'] == 'es' and kwargs['task'] == 'translate'
        return iter([SimpleNamespace(start=0, end=1, text='Hello.')]), None
    monkeypatch.setattr(engine, 'load', lambda *args: SimpleNamespace(transcribe=transcribe))
    assert engine.live_infer(np.ones(16000), 'es', 0, 1)[0]['text'] == 'Hello.'


def test_caption_options_are_validated_and_snapshotted():
    from fastapi.testclient import TestClient
    from jread.server import app
    client=TestClient(app,base_url='http://127.0.0.1:8765')
    auth={'Authorization':'Bearer '+store.TOKEN}
    before=store.settings()
    try:
        selected={'caption_glossary':'aikatsu_pripara','caption_context':True,'caption_boundaries':True}
        assert client.patch('/api/settings',headers=auth,json=selected).status_code==200
        job=store.create('Caption options snapshot')
        store.settings({'caption_glossary':'none','caption_context':False,'caption_boundaries':False})
        assert all(store.get(job['id'])[key]==value for key,value in selected.items())
        assert client.patch('/api/settings',headers=auth,json={'caption_glossary':'unknown'}).status_code==422
        engine.STATE['live']=True
        assert client.patch('/api/settings',headers=auth,json={'caption_context':True}).status_code==409
    finally:
        engine.STATE['live']=False
        store.settings(before)
