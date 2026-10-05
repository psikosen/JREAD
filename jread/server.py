import asyncio
import hashlib
import json
import math
import mimetypes
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, engine, gemma, store, subtitles

ROOT = Path(__file__).resolve().parent.parent
ORIGIN = 'http://127.0.0.1:8765'
EXTENSION = 'chrome-extension://' + (ROOT/'scripts/extension-id.txt').read_text().strip()
ORIGINS = {ORIGIN, EXTENSION}
MAX_UPLOAD = 2 * 1024**3
PAIR = {'code': '', 'expires': 0, 'attempts': [], 'used': False}


@asynccontextmanager
async def lifespan(app):
    for job in store.all_jobs():
        if job['status'] == 'cancelled' and job.get('cancellation_pending'):
            store.update(job['id'],cancellation_pending=False,error='Cancelled. Resume to continue saved work.')
        if job['status'] == 'uploading':
            shutil.rmtree(store.MEDIA/job['id'], ignore_errors=True)
            store.update(job['id'],status='failed',has_media=False,error='Upload was interrupted. Import the file again.')
            continue
        if job['status'] in ('queued','preparing','recognizing','translating','live'):
            store.update(job['id'],status='failed',error='Processing was interrupted. Resume to continue saved work.' if job.get('has_media') else 'Live capture ended when the companion stopped. Saved cues remain available.')
    for path in store.MEDIA.glob('*/audio.part.wav'):
        path.unlink(missing_ok=True)
    try:
        yield
    finally:
        gemma.shutdown()


app = FastAPI(title='JREAD', version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def token_valid(token):
    if not token:
        return False
    if secrets.compare_digest(token,store.TOKEN):
        return True
    with store.connect() as db:
        row = db.execute('SELECT expires FROM sessions WHERE hash=?',(hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        return bool(row and row['expires'] > time.time())


def request_token(request):
    authorization = request.headers.get('authorization','')
    if authorization.startswith('Bearer '):
        return authorization[7:]
    # A cookie is usable only on the companion origin, not extension requests.
    if request.headers.get('origin') in (None,ORIGIN):
        return request.cookies.get('jread_session','')
    return ''


@app.middleware('http')
async def guard(request: Request, call_next):
    host, origin = request.headers.get('host',''), request.headers.get('origin')
    if host != '127.0.0.1:8765' or (origin is not None and origin not in ORIGINS):
        return JSONResponse({'detail':'This origin is not allowed.'},status_code=403)
    if request.method == 'OPTIONS':
        response = Response(status_code=204)
    else:
        if request.url.path.startswith('/api/') and request.url.path != '/api/pair' and not token_valid(request_token(request)):
            return JSONResponse({'detail':'Open the companion or pair the extension to continue.'},status_code=401,headers={'Access-Control-Allow-Origin':origin} if origin in ORIGINS else {})
        if request.method in ('POST','PUT','PATCH'):
            try:
                length = int(request.headers.get('content-length','0'))
            except ValueError:
                return JSONResponse({'detail':'Invalid content length.'},status_code=400)
            limit = MAX_UPLOAD + 1024*1024 if request.url.path == '/api/jobs' else (5*1024*1024 if request.url.path == '/api/import' else 65536)
            if length > limit:
                return JSONResponse({'detail':'Request is too large.'},status_code=413)
            if 'multipart/form-data' in request.headers.get('content-type','') and not length:
                return JSONResponse({'detail':'Upload requires a known file size.'},status_code=411)
        response = await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'"
    if origin in ORIGINS:
        response.headers.update({'Access-Control-Allow-Origin':origin,'Access-Control-Allow-Methods':'GET,POST,PATCH,DELETE,OPTIONS','Access-Control-Allow-Headers':'Authorization,Content-Type','Access-Control-Allow-Credentials':'true','Vary':'Origin'})
    return response


@app.exception_handler(KeyError)
async def missing(request, exc):
    return JSONResponse({'detail':'This item no longer exists.'},status_code=404)


@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse({'detail':str(exc)},status_code=400)


@app.get('/health')
def health():
    return {'ok':True,'version':__version__}


@app.get('/api/status')
def status():
    return {'version':__version__,'engine':engine.status(),'active_job':engine.STATE['active_job'],'live':engine.STATE['live'],'disk_free_bytes':shutil.disk_usage(store.DATA).free}


@app.post('/api/session')
def session(request:Request):
    response=JSONResponse({'ok':True})
    response.set_cookie('jread_session',request_token(request),max_age=86400,httponly=True,samesite='strict',path='/')
    return response


class Settings(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model: Literal['small','medium','large-v3-turbo']|None=None
    device: Literal['cpu','cuda','auto']|None=None
    translator: Literal['whisper','gemma4','gemma4-12b']|None=None
    caption_context: bool|None=None
    caption_boundaries: bool|None=None
    caption_glossary: Literal['none','aikatsu_pripara']|None=None
    font_size: int|None=Field(None,ge=14,le=64)
    show_japanese: bool|None=None
    caption_background: bool|None=None
    caption_position: int|None=Field(None,ge=0,le=40)


@app.get('/api/settings')
def get_settings():
    return store.settings()


@app.patch('/api/settings')
def set_settings(values:Settings):
    fields=values.model_dump(exclude_none=True)
    if any(k in fields for k in ('model','device','translator','caption_context','caption_boundaries','caption_glossary')) and (engine.STATE['active_job'] or engine.STATE['live']):
        raise HTTPException(409,'Stop processing before changing the inference preset.')
    engine.validate_translation(store.settings() | fields)
    return store.settings(fields)


class ModelRequest(BaseModel):
    model: Literal['small','medium','large-v3-turbo','gemma4','gemma4-12b']


@app.post('/api/models/download')
def model_download(values:ModelRequest):
    size = gemma.PRESETS[values.model]['size'] if values.model in gemma.PRESETS else {'small':600_000_000,'medium':1_700_000_000,'large-v3-turbo':1_630_000_000}[values.model]
    if shutil.disk_usage(store.DATA).free < size+2_000_000_000:
        raise HTTPException(409,'Free more disk space before downloading this model.')
    engine.download(values.model)
    return engine.status()


@app.post('/api/models/remove')
def model_remove(values:ModelRequest):
    engine.remove_model(values.model)
    return engine.status()


@app.get('/api/jobs')
def jobs():
    return store.all_jobs()


@app.post('/api/jobs',status_code=201)
async def new_job(file:UploadFile=File(...),language:Literal['ja','en']=Form('ja'),title:str=Form('')):
    settings=store.settings()
    if not engine.status()['ready']:
        raise HTTPException(409,'Download the selected speech recognition and translation models in Setup first.')
    if engine.STATE['live']:
        raise HTTPException(409,'Stop live capture before importing a file.')
    if len([j for j in store.all_jobs() if j['status'] in ('queued','preparing','recognizing','translating')])>=5:
        raise HTTPException(409,'The queue is full. Wait for a job to finish.')
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in engine.FORMATS:
        raise ValueError('Unsupported media format. Use MP4, MKV, WebM, MOV, M4A, WAV, MP3, FLAC, OGG, or AAC.')
    job=store.create(title or file.filename or 'Untitled clip',language=language,status='uploading',has_media=False,suffix=suffix)
    folder=store.MEDIA/job['id'];folder.mkdir(mode=0o700)
    try:
        total=0
        with (folder/'source.part').open('wb') as target:
            while chunk:=await file.read(1024*1024):
                total+=len(chunk)
                if total>MAX_UPLOAD or shutil.disk_usage(store.DATA).free<1_000_000_000:
                    raise HTTPException(413,'The file exceeds the 2 GB limit or there is insufficient disk space.')
                target.write(chunk)
        duration,stream=await asyncio.to_thread(engine.media_info,folder/'source.part',suffix)
        (folder/'source.part').replace(folder/'source')
        job=store.update(job['id'],duration=duration,audio_stream=stream,bytes=total,has_media=True)
        engine.submit(job['id'])
        return store.get(job['id'])
    except Exception:
        shutil.rmtree(folder,ignore_errors=True)
        with store.connect() as db: db.execute('DELETE FROM jobs WHERE id=?',(job['id'],))
        raise
    finally:
        await file.close()


@app.post('/api/import',status_code=201)
async def import_track(file:UploadFile=File(...),title:str=Form('')):
    try:
        data=await file.read(5*1024*1024+1)
        if len(data)>5*1024*1024:
            raise HTTPException(413,'Subtitle file exceeds 5 MB.')
        try: cues=subtitles.parse(data.decode('utf-8-sig'))
        except UnicodeDecodeError: raise ValueError('Save subtitles as UTF-8 SRT or WebVTT.')
        job=store.create(title or file.filename or 'Imported subtitles',status='complete',progress=1,duration=max(c['end'] for c in cues),model='imported',device='none')
        for cue in cues:store.add_cue(job['id'],cue)
        return store.get(job['id'])
    finally:
        await file.close()


@app.get('/api/jobs/{job_id}')
def get_job(job_id:str):
    return store.get(job_id)


@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id:str):
    job=store.get(job_id)
    if job['status'] in ('queued','preparing','recognizing','translating'):
        return engine.cancel(job_id)
    return job


@app.post('/api/jobs/{job_id}/retry')
def retry(job_id:str):
    job=store.get(job_id)
    if not job['has_media']:
        raise ValueError('This job has no saved media to resume.')
    if job['status'] not in ('failed','cancelled'):
        raise HTTPException(409,'Only interrupted or cancelled jobs can be resumed.')
    if job.get('cancellation_pending'):
        raise HTTPException(409,'Cancellation is finishing. Resume when the current inference step has stopped.')
    engine.submit(job_id)
    return store.get(job_id)


@app.delete('/api/jobs/{job_id}')
def delete(job_id:str):
    job=store.get(job_id)
    if job['status'] in ('queued','preparing','recognizing','translating','live'):
        raise HTTPException(409,'Stop this job before deleting it.')
    future=engine._futures.get(job_id)
    if future and not future.done():
        raise HTTPException(409,'Cancellation is finishing. Try deleting again shortly.')
    with store.connect() as db:
        db.execute('DELETE FROM cues WHERE job_id=?',(job_id,));db.execute('DELETE FROM jobs WHERE id=?',(job_id,))
    shutil.rmtree(store.MEDIA/job_id,ignore_errors=True)
    return {'ok':True}


@app.get('/api/jobs/{job_id}/media')
def media(job_id:str):
    job=store.get(job_id,False)
    if not job['has_media']:
        raise HTTPException(404,'This track has no local video.')
    return FileResponse(store.MEDIA/job_id/'source',media_type=mimetypes.guess_type('file'+job['suffix'])[0] or 'application/octet-stream',content_disposition_type='inline')


@app.get('/api/jobs/{job_id}/export')
def export(job_id:str,format:Literal['srt','vtt']='vtt'):
    job=store.get(job_id)
    return Response(subtitles.export(job['cues'],format),media_type='text/vtt' if format=='vtt' else 'application/x-subrip',headers={'Content-Disposition':f'attachment; filename="jread-{job_id}.{format}"'})


class CueEdit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str=Field(min_length=1,max_length=4000)
    start:float=Field(ge=0,le=86400,allow_inf_nan=False)
    end:float=Field(gt=0,le=86400,allow_inf_nan=False)


@app.patch('/api/jobs/{job_id}/cues/{cue_id}')
def edit(job_id:str,cue_id:str,values:CueEdit):
    return store.edit_cue(job_id,cue_id,values.model_dump())


@app.post('/api/jobs/{job_id}/cues/{cue_id}/undo')
def undo(job_id:str,cue_id:str):
    return store.edit_cue(job_id,cue_id,undo=True)


@app.post('/api/pair/new')
def new_pair(reuse:bool=False):
    now=time.time()
    if not (reuse and PAIR['code'] and not PAIR['used'] and PAIR['expires']>now):
        PAIR.update(code=secrets.token_hex(4).upper(),expires=now+300,used=False)
    return {'code':PAIR['code'],'expires_in':math.ceil(PAIR['expires']-now)}


class PairRequest(BaseModel):
    code:str=Field(min_length=1,max_length=32)


@app.post('/api/pair')
def pair(request:Request,values:PairRequest):
    if request.headers.get('origin')!=EXTENSION:
        raise HTTPException(403,'Pairing is available only to the installed JREAD extension.')
    now=time.time();PAIR['attempts']=[t for t in PAIR['attempts'] if now-t<60]
    if len(PAIR['attempts'])>=5:raise HTTPException(429,'Wait a minute before trying another pairing code.')
    PAIR['attempts'].append(now)
    if PAIR['used'] or now>PAIR['expires'] or not secrets.compare_digest(values.code.strip().upper(),PAIR['code']):
        raise HTTPException(403,'Pairing code is incorrect or expired. Create a new code in the companion.')
    PAIR['used']=True;token=secrets.token_urlsafe(32)
    with store.connect() as db:
        db.execute('DELETE FROM sessions WHERE expires<?',(now,))
        db.execute('INSERT INTO sessions VALUES (?,?)',(hashlib.sha256(token.encode()).hexdigest(),now+86400))
    return {'token':token,'expires_in':86400}


def reconcile_overlap(existing, incoming):
    """Pure conservative reconciliation; return all cues and added/replaced cues.

    Only unambiguous, single-cue sources are reconciled. Multi-cue translations
    have no translated word alignment, so their pieces must not be discarded.
    """
    def source(cue):
        return ''.join(cue.get('source_text', '').split())
    combined=[dict(cue) for cue in existing]
    new_counts=Counter(source(cue) for cue in incoming)
    changed=[]
    for cue in incoming:
        text=source(cue)
        match=None
        if text and new_counts[text]==1:
            candidates=[]
            for index,old in enumerate(combined):
                original=source(old)
                if not original or old.get('status')=='user_edited':
                    continue
                overlap=min(old['end'],cue['end'])-max(old['start'],cue['start'])
                shorter=min(old['end']-old['start'],cue['end']-cue['start'])
                # Allow modest ASR timestamp jitter at the one-second overlap.
                if shorter<=0 or overlap/shorter<.75:
                    continue
                # Adjacent pieces sharing one ASR source are ambiguous; distant
                # repetitions of the same dialogue do not disable a valid match.
                if sum(source(item)==original and item['end']>=old['start'] and item['start']<=old['end'] for item in combined)!=1:
                    continue
                equal=text==original
                extended=len(original)>=4 and not original.endswith(('。','！','？','.','!','?')) and len(text)>len(original) and (text.startswith(original) or text.endswith(original))
                if equal or extended:
                    candidates.append((index,equal))
            if len(candidates)==1:
                match=candidates[0]
        if match:
            index,equal=match
            if equal:
                continue
            old=combined[index]
            cue=dict(cue,id=old['id'],status='needs_review')
        else:
            cue=dict(cue)
        # Preserve the baseline's exact-English overlap suppression in offline
        # replay of this helper as well as in the live persistence path.
        if any(old['id']!=cue['id'] and old['text']==cue['text'] and old['start']<cue['end'] and old['end']>cue['start'] for old in existing):
            continue
        if match:
            combined[index]=cue
        else:
            combined.append(cue)
        changed.append(cue)
    return combined,changed


def source_context(previous, cues):
    sources=dict.fromkeys(cue.get('source_text','').strip() for cue in cues if cue.get('source_text','').strip())
    return ' '.join([previous,*sources]).strip()[-150:]


@app.websocket('/api/live')
async def live(ws:WebSocket):
    if ws.headers.get('host')!='127.0.0.1:8765' or ws.headers.get('origin')!=EXTENSION:
        await ws.close(code=1008);return
    await ws.accept()
    job=None;worker=None;owned=False;graceful=False
    context={'epoch':0,'stopped':False,'source_history':'','cue_ids':set()};queue=asyncio.Queue(maxsize=2)
    try:
        raw=await asyncio.wait_for(ws.receive_text(),5)
        if len(raw)>4096:raise ValueError('Handshake is too large.')
        hello=json.loads(raw)
        if hello.get('type')!='hello' or not token_valid(hello.get('token','')):raise ValueError('Pair the extension before capture.')
        if engine.STATE['live'] or engine.STATE['active_job']:raise ValueError('Another job is using the model. Stop it before live capture.')
        if not engine.status()['ready']:raise ValueError('Download the selected model in the companion first.')
        language=hello.get('language','ja')
        if not isinstance(language,str) or language not in gemma.SOURCE_LANGUAGES:raise ValueError('Unsupported source language.')
        epoch=hello.get('epoch',0)
        if type(epoch) is not int or epoch<0:raise ValueError('Invalid capture epoch.')
        engine.STATE['live']=True;owned=True;context['epoch']=epoch
        options=store.settings()
        use_gemma=language!='en' and options.get('translator') in gemma.PRESETS
        use_context=use_gemma and bool(options.get('caption_context',False))
        use_boundaries=use_gemma and bool(options.get('caption_boundaries',False))
        job=store.create(str(hello.get('title','Live captions'))[:200],status='live',language=language,caption_context=use_context,caption_boundaries=use_boundaries,caption_glossary=options.get('caption_glossary','none') if language=='ja' else 'none')
        await ws.send_json({'type':'ready','job_id':job['id']})
        async def process_audio():
            while True:
                meta,data=await queue.get()
                try:
                    if meta['epoch']!=context['epoch']:continue
                    samples=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768
                    cues=[]
                    if samples.size and float(np.sqrt(np.mean(samples*samples)))>.0005:
                        prompt={'context':context['source_history'][-150:]} if use_context else {}
                        cues=await asyncio.to_thread(engine.live_infer,samples,job['language'],meta['media_time'],meta.get('rate',1),**prompt)
                    if meta['epoch']==context['epoch'] and not context['stopped']:
                        existing=store.get(job['id'])['cues']
                        if use_boundaries:
                            _,changed=reconcile_overlap([cue for cue in existing if cue['id'] in context['cue_ids']],cues)
                        else:
                            changed=cues
                        changed=[cue for cue in changed if not any(c['id']!=cue['id'] and c['text']==cue['text'] and c['start']<cue['end'] and c['end']>cue['start'] for c in existing)]
                        existing_ids={cue['id'] for cue in existing}
                        accepted=[]
                        for cue in changed:
                            if cue['id'] in existing_ids:
                                with store.connect() as db:
                                    updated=db.execute("UPDATE cues SET start=?,end=?,text=?,source_text=?,status=? WHERE id=? AND job_id=? AND status!='user_edited'",(cue['start'],cue['end'],cue['text'],cue.get('source_text',''),cue.get('status','final'),cue['id'],job['id'])).rowcount
                                if not updated:continue
                            else:
                                store.add_cue(job['id'],cue)
                            accepted.append(cue)
                        if use_context:
                            # Context depends on successful recognition, not the
                            # independent presentation/reconciliation option.
                            context['source_history']=source_context(context['source_history'],cues)
                        context['cue_ids'].update(cue['id'] for cue in accepted)
                        for cue in accepted:
                            if meta['epoch']!=context['epoch'] or context['stopped']:break
                            await ws.send_json({'type':'cue','epoch':meta['epoch'],'cue':cue})
                        await ws.send_json({'type':'ack','seq':meta['seq']})
                except Exception as exc:
                    context['stopped']=True
                    detail=str(exc) if isinstance(exc,gemma.GemmaError) else 'Live inference failed. Check the selected device/model in the companion.'
                    context['error']=detail
                    store.update(job['id'],error=detail)
                    await ws.send_json({'type':'error','detail':detail})
                    await ws.close(code=1011)
                    raise
                finally:queue.task_done()
        worker=asyncio.create_task(process_audio())
        last_seq=-1
        while True:
            if worker.done():
                worker.result()
            # A paused video may remain idle indefinitely. WebSocket transport
            # keepalives detect disconnects without treating a pause as failure.
            raw=await ws.receive_text()
            if len(raw)>4096:raise ValueError('Capture metadata is too large.')
            meta=json.loads(raw)
            kind=meta.get('type')
            if kind=='stop':
                drained=asyncio.create_task(queue.join())
                try:
                    done,_=await asyncio.wait((drained,worker),timeout=120,return_when=asyncio.FIRST_COMPLETED)
                    if worker in done:
                        worker.result()
                    if drained not in done:
                        raise asyncio.TimeoutError()
                finally:
                    drained.cancel()
                    await asyncio.gather(drained,return_exceptions=True)
                graceful=True
                await ws.send_json({'type':'stopped','job_id':job['id']})
                break
            if kind=='reset':
                epoch=meta.get('epoch')
                if type(epoch) is not int or epoch<=context['epoch']:raise ValueError('Capture epoch must increase.')
                context['epoch']=epoch
                context['source_history']=''
                context['cue_ids'].clear()
                continue
            if kind!='audio':raise ValueError('Unexpected capture message.')
            if type(meta.get('seq')) is not int or meta['seq']<=last_seq:raise ValueError('Audio sequence must increase.')
            last_seq=meta['seq']
            if type(meta.get('epoch')) is not int or meta['epoch']!=context['epoch']:raise ValueError('Stale capture epoch.')
            if meta.get('sample_rate')!=16000:raise ValueError('Capture must use mono 16 kHz PCM.')
            for field in ('media_time','rate'):
                value=meta.get(field,1 if field=='rate' else -1)
                if not isinstance(value,(float,int)) or not math.isfinite(value):raise ValueError('Invalid media clock.')
            if not 0<=meta['media_time']<=86400 or not .25<=meta.get('rate',1)<=4:raise ValueError('Unsupported media clock or playback rate.')
            data=await asyncio.wait_for(ws.receive_bytes(),10)
            if not 0<len(data)<=16000*2*30 or len(data)%2:raise ValueError('Invalid audio packet size.')
            try:queue.put_nowait((meta,data))
            except asyncio.QueueFull:raise ValueError('Inference cannot keep up. Stop playback and select a faster preset; no audio was silently dropped.')
    except (WebSocketDisconnect,asyncio.TimeoutError):
        pass
    except Exception as exc:
        detail=context.get('error') or (str(exc) if isinstance(exc,ValueError) else 'Live inference failed. Check the model/device in the companion.')
        if job:store.update(job['id'],error=detail)
        try:await ws.send_json({'type':'error','detail':detail})
        except Exception:pass
    finally:
        context['stopped']=True
        if worker:
            worker.cancel()
            try:await worker
            except (asyncio.CancelledError,Exception):pass
        if owned:engine.STATE['live']=False
        if job:
            current=store.get(job['id'])
            if not graceful and not current['error']:
                current=store.update(job['id'],error='Live capture disconnected before finishing. Saved captions may be incomplete.')
            store.update(job['id'],status='complete' if not current['error'] else 'failed',progress=1,duration=max((c['end'] for c in current['cues']),default=0))
        try:await ws.close()
        except Exception:pass



class BodyLimit:
    def __init__(self, app):
        self.app=app

    async def __call__(self, scope, receive, send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        path=scope.get('path','')
        limit=MAX_UPLOAD+1024*1024 if path=='/api/jobs' else 5*1024*1024 if path=='/api/import' else 65536
        size=0
        async def bounded_receive():
            nonlocal size
            message=await receive()
            size+=len(message.get('body',b''))
            if size>limit:
                raise HTTPException(413,'Request is too large.')
            return message
        await self.app(scope,bounded_receive,send)


app.add_middleware(BodyLimit)

app.mount('/',StaticFiles(directory=ROOT/'companion',html=True),name='companion')
