"""Development-only CUDA ASR evaluation. Never opens held-out audio."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MANIFEST=ROOT/'tests/fixtures/accuracy-corpus.json'
OUTPUT=ROOT/'test-results/accuracy-speech-development.json'
TURBO='--turbo' in sys.argv
TURBO_REVISION='0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'
TURBO_SHA256='e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da'
if TURBO:OUTPUT=ROOT/'test-results/accuracy-speech-turbo-development.json'
FLAGS=dict(language='ja',task='transcribe',vad_filter=True,beam_size=5,condition_on_previous_text=False)


def normalize(text):
    return ''.join(char for char in unicodedata.normalize('NFKC',text) if not char.isspace() and not unicodedata.category(char).startswith('P'))


def character_error(reference,hypothesis):
    reference,hypothesis=normalize(reference),normalize(hypothesis)
    if not reference:raise ValueError('CER requires a nonempty normalized reference.')
    previous=list(range(len(hypothesis)+1))
    for index,left in enumerate(reference,1):
        current=[index]
        for column,right in enumerate(hypothesis,1):
            current.append(min(previous[column]+1,current[column-1]+1,previous[column-1]+(left!=right)))
        previous=current
    return dict(errors=previous[-1],reference_characters=len(reference),cer=previous[-1]/len(reference),normalized_reference=reference,normalized_hypothesis=hypothesis)


def self_test():
    assert normalize('ＡＢ、「日 本」。')=='AB日本'
    for changed in ('あえう','あいいう','あう'):
        result=character_error('あいう',changed)
        assert result['errors']==1 and result['reference_characters']==3 and result['cer']==1/3
    assert character_error('日本語','日本語')['errors']==0
    assert character_error('日本語','')['errors']==3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(report):
    OUTPUT.parent.mkdir(exist_ok=True)
    temporary=OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(OUTPUT)


def development_cases(manifest):
    cases=[case for case in manifest['cases'] if case['partition']=='development']
    if len(cases)!=20 or len({case['id'] for case in cases})!=20:
        raise ValueError('The frozen manifest must contain20 distinct development cases.')
    return cases


def worker():
    import resource
    import numpy as np
    from jread import engine
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    report=json.loads(OUTPUT.read_text())
    if digest(MANIFEST)!=report['corpus']['manifest_sha256']:
        raise ValueError('Corpus manifest changed after preflight.')
    cases=development_cases(json.loads(MANIFEST.read_text()))
    model=None
    try:
        start=time.perf_counter()
        if TURBO:
            from faster_whisper import WhisperModel
            from jread import store
            path=store.MODELS/'large-v3-turbo'
            with (path/'model.bin').open('rb') as file:
                if hashlib.file_digest(file,'sha256').hexdigest()!=TURBO_SHA256:
                    raise ValueError('Turbo candidate weights failed the pinned checksum.')
            model=WhisperModel(str(path),device='cuda',compute_type='int8_float16',cpu_threads=4,num_workers=1,local_files_only=True)
        else:
            model=engine.load('medium','cuda')
        report['model']['revision']=TURBO_REVISION if TURBO else engine.PRESETS['medium'][0]
        report['model']['load_seconds']=time.perf_counter()-start
        def transcribe(samples,offset=0,context=''):
            started=time.perf_counter()
            segments,_=model.transcribe(samples,**FLAGS,**engine.asr_hints({'caption_glossary':'none'},'ja',context))
            rows=list(engine.bounded_segments(segments,len(samples)/16000))
            for row in rows:
                row['start']+=offset;row['end']+=offset
            return dict(start=offset,end=offset+len(samples)/16000,segments=rows,context_used=context,inference_seconds=time.perf_counter()-started)
        warmup_path=ROOT/'tests/fixtures/japanese-speech.wav'
        warmed=transcribe(engine.read_audio(warmup_path))
        report['warmup']=dict(excluded=True,path=str(warmup_path.relative_to(ROOT)),audio_sha256=digest(warmup_path),inference_seconds=warmed['inference_seconds'])
        report['status']='running';write(report)
        for case in cases:
            audio_path=(ROOT/case['audio_path']).resolve()
            if not audio_path.is_relative_to(ROOT/'test-results/accuracy-audio') or digest(audio_path)!=case['audio_sha256']:
                raise ValueError(f"Development audio path/hash failed for{case['id']}.")
            samples=engine.read_audio(audio_path)
            duration=len(samples)/16000
            if abs(duration-case['duration_seconds'])>1/16000:
                raise ValueError('Audio duration differs from frozen manifest.')
            record=dict(case)
            full=transcribe(samples)
            full['transcript']=' '.join(row['text'] for row in full['segments'])
            full['cer']=character_error(case['japanese_reference'],full['transcript'])
            record['full_clip']=full
            for enabled,name in ((False,'chunk_baseline'),(True,'chunk_context')):
                chunks=[];history='';left=0
                while left<len(samples):
                    right=min(len(samples),left+8*16000)
                    chunk=transcribe(samples[left:right],left/16000,history if enabled else '')
                    chunks.append(chunk)
                    if enabled:
                        sources=dict.fromkeys(row['text'] for row in chunk['segments'])
                        history=' '.join([history,*sources]).strip()[-150:]
                    if right==len(samples):break
                    left=right-16000
                record[name]=dict(chunks=chunks,inference_seconds=sum(chunk['inference_seconds'] for chunk in chunks),cer=None,cer_note='Overlap alignment is not certified; raw chunks are not scored as one transcript.')
            report['cases'].append(record)
            report['completed_cases']=len(report['cases'])
            write(report)
            print(json.dumps(dict(completed_cases=report['completed_cases'],development_id=case['id'])),flush=True)
        silence=transcribe(np.zeros(8*16000,dtype=np.float32))
        report['silence_control']=dict(duration_seconds=8,expected_empty=True,passed=not silence['segments'],**silence)
        errors=sum(case['full_clip']['cer']['errors'] for case in report['cases'])
        reference_chars=sum(case['full_clip']['cer']['reference_characters'] for case in report['cases'])
        report['aggregate']=dict(errors=errors,reference_characters=reference_chars,cer=errors/reference_chars,audio_seconds=sum(case['duration_seconds'] for case in report['cases']),inference_seconds={name:sum(case[name]['inference_seconds'] for case in report['cases']) for name in ('full_clip','chunk_baseline','chunk_context')})
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if model is not None:
            model.model.unload_model()
            report['owned_model_unloaded']=True
        engine._model=engine._model_key=None
        model=None;gc.collect()
        write(report)


def main():
    self_test()
    if '--self-test' in sys.argv:
        print('CER checks passed: Unicode normalization, substitution, insertion, deletion.');return
    if '--worker' in sys.argv:
        worker();return
    from asr_cuda_pilot import gpu_snapshot
    if OUTPUT.exists():raise SystemExit('Preserve the existing evaluation; move its artifact explicitly before a new run.')
    manifest=json.loads(MANIFEST.read_text())
    selected=development_cases(manifest)
    report=dict(schema_version=1,started_at_utc=datetime.now(timezone.utc).isoformat(),status='starting',partition='development',
                corpus={key:value for key,value in manifest.items() if key!='cases'}|dict(manifest=str(MANIFEST.relative_to(ROOT)),manifest_sha256=digest(MANIFEST)),
                development_ids=[case['id'] for case in selected],completed_cases=0,cases=[],
                model=dict(name='faster-whisper large-v3-turbo' if TURBO else 'faster-whisper medium',device='cuda',compute_type='int8_float16',cpu_threads=4,num_workers=1,local_files_only=True),
                flags=FLAGS,chunking=dict(chunk_seconds=8,overlap_seconds=1,context_characters=150,glossary='none'),
                metric=dict(name='Whole-clip normalized Japanese CER',normalization='Unicode NFKC, then remove Unicode punctuation and whitespace',numerator='Levenshtein substitutions+insertions+deletions',denominator='normalized reference characters',meaning_accuracy=False),
                versions={key:importlib.metadata.version(key) for key in ('faster-whisper','ctranslate2','numpy')},
                gpu_before=gpu_snapshot(),limitations=['Only20 development recordings; held-out audio and model outputs were not opened.','FLEURS read speech is not anime dialogue or a meaning-accuracy score.','Chunk outputs are not given aggregate CER because overlap is not aligned.','No live browser latency or Gemma translation is measured.'])
    child=None
    try:
        if not any(row['index']==0 and row['free_mib']>=4096 for row in report['gpu_before']['gpus']):
            report['status']='skipped_insufficient_gpu_memory';return
        env=os.environ|dict(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        report['worker_environment']={key:env.get(key,'') for key in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','LD_LIBRARY_PATH')}
        write(report)
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',*(['--turbo'] if TURBO else [])],cwd=ROOT,env=env,stderr=subprocess.PIPE,text=True)
        try:
            _,stderr=child.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            child.kill();_,stderr=child.communicate()
        report=json.loads(OUTPUT.read_text())
        report.update(worker_pid=child.pid,worker_exit_code=child.returncode,stderr=stderr[-4000:])
        if child.returncode!=0:report['status']='failed'
    finally:
        if child is not None and child.poll() is None:child.kill();child.wait(timeout=10)
        report['gpu_after']=gpu_snapshot()
        report['owned_gpu_process_released']=child is None or all(row['pid']!=child.pid for row in report['gpu_after']['processes'])
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        write(report)
        print(json.dumps(dict(status=report['status'],completed_cases=report['completed_cases'],report=str(OUTPUT.relative_to(ROOT)),owned_gpu_process_released=report['owned_gpu_process_released'])),flush=True)


if __name__=='__main__':main()
