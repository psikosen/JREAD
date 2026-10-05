"""Bounded, isolated CUDA pilot using only the already installed Whisper medium."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
AUDIO=ROOT/'tests/fixtures/japanese-speech.wav'
MODEL=Path.home()/'.local/share/JREAD/models/medium'
REVISION='08e178d48790749d25932bbc082711ddcfdfbc4f'
OUTPUT=ROOT/'test-results/asr-cuda-pilot.json'
FLAGS=dict(language='ja',task='transcribe',vad_filter=True,beam_size=5,condition_on_previous_text=False)


def gpu_snapshot():
    def query(kind,fields):
        result=subprocess.run(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        return list(csv.reader(io.StringIO(result.stdout),skipinitialspace=True))
    return dict(gpus=[dict(index=int(row[0]),name=row[1],total_mib=int(row[2]),used_mib=int(row[3]),free_mib=int(row[4])) for row in query('gpu','index,name,memory.total,memory.used,memory.free')],
                processes=[dict(pid=int(row[0]),name=row[1],used_mib=row[2]) for row in query('compute-apps','pid,process_name,used_gpu_memory')])


def worker():
    import resource
    import wave
    import numpy as np
    from faster_whisper import WhisperModel
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def emit(stage,**fields):
        print(json.dumps(dict(stage=stage,**fields),ensure_ascii=False),flush=True)
    if (MODEL/'.verified').read_text().strip()!=REVISION:
        raise ValueError('Existing Whisper medium verification marker does not match the pinned revision.')
    with wave.open(str(AUDIO),'rb') as wav:
        assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())==(1,2,16000)
        samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
    emit('loading')
    start=time.perf_counter()
    model=WhisperModel(str(MODEL),device='cuda',device_index=0,compute_type='int8_float16',cpu_threads=4,num_workers=1,local_files_only=True)
    emit('loaded',load_seconds=time.perf_counter()-start,gpu=gpu_snapshot())
    def transcribe(left,right):
        started=time.perf_counter()
        segments,_=model.transcribe(samples[round(left*16000):round(right*16000)],**FLAGS)
        rows=[]
        for segment in segments:
            begin,end=max(0,float(segment.start)),min(right-left,float(segment.end))
            if begin<end and segment.text.strip():
                rows.append(dict(start=left+begin,end=left+end,text=segment.text.strip()))
        return dict(start=left,end=right,inference_seconds=time.perf_counter()-started,segments=rows)
    duration=len(samples)/16000
    emit('warmup')
    warmup=transcribe(0,duration)
    emit('warmed',excluded_warmup_seconds=warmup['inference_seconds'])
    full=transcribe(0,duration)
    chunks=[]
    left=0.0
    while left<duration:
        right=min(duration,left+8)
        chunks.append(transcribe(left,right))
        if right==duration:break
        left=right-1
    emit('results',audio_seconds=duration,full=full,chunks=chunks,gpu=gpu_snapshot())
    # Process exit is the ownership boundary; CUDA allocations cannot outlive it.
    del model


def main():
    if '--worker' in sys.argv:
        worker();return
    import importlib.metadata
    prior_path=ROOT/'test-results/caption-asr-combos.json'
    prior=json.loads(prior_path.read_text())
    report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),status='starting',
                model=dict(name='faster-whisper medium',revision=REVISION,device='cuda',device_index=0,compute_type='int8_float16',cpu_threads=4,num_workers=1,local_files_only=True),
                flags=FLAGS,chunking=dict(chunk_seconds=8,overlap_seconds=1),
                versions={name:importlib.metadata.version(name) for name in ('faster-whisper','ctranslate2','numpy')},
                fixture=dict(path=str(AUDIO.relative_to(ROOT)),sha256=hashlib.sha256(AUDIO.read_bytes()).hexdigest(),reference=json.loads((AUDIO.with_suffix('.json')).read_text())['reference']),
                prior_cpu=dict(path=str(prior_path.relative_to(ROOT)),model=prior['model'],baseline=prior['results']['baseline']),
                limitations=['One short public-speech fixture is not a meaning-accuracy or anime benchmark.','Prior CPU timings were recorded at a different time; this is not a simultaneous controlled speed comparison.','This pilot measures ASR only; Gemma E4B translation remains a separate CPU stage.'])
    child=None
    try:
        report['gpu_before']=gpu_snapshot()
        if not any(gpu['index']==0 and gpu['free_mib']>=4096 for gpu in report['gpu_before']['gpus']):
            report['status']='skipped_insufficient_free_gpu_memory'
            return
        env=os.environ|dict(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        report['worker_environment']={key:env.get(key,'') for key in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','LD_LIBRARY_PATH')}
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        report['owned_child_pid']=child.pid
        try:
            stdout,stderr=child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill();stdout,stderr=child.communicate()
            report['status']='timed_out'
        report['worker_exit_code']=child.returncode
        report['stages']=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
        report['stderr']=stderr[-4000:]
        result=next((item for item in report['stages'] if item['stage']=='results'),None)
        if result:
            report['status']='complete';report['results']=result
            report['cuda_chunk_inference_seconds']=sum(chunk['inference_seconds'] for chunk in result['chunks'])
            report['prior_cpu_to_cuda_chunk_time_ratio']=prior['results']['baseline']['elapsed_seconds']/report['cuda_chunk_inference_seconds']
        elif report['status']=='starting':
            report['status']='failed_runtime'
    except Exception as error:
        report.update(status='failed_preflight',error=f'{type(error).__name__}: {error}')
    finally:
        if child is not None and child.poll() is None:
            child.kill();child.wait(timeout=10)
        report['gpu_after']=gpu_snapshot()
        report['owned_gpu_process_released']=child is None or all(row['pid']!=child.pid for row in report['gpu_after']['processes'])
        OUTPUT.parent.mkdir(exist_ok=True)
        OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(status=report['status'],report=str(OUTPUT.relative_to(ROOT)),owned_gpu_process_released=report['owned_gpu_process_released'])))


if __name__=='__main__':main()
