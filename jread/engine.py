"""One local inference worker, explicit model downloads, durable subtitle stages."""
import hashlib
import json
import math
import shutil
import subprocess
import threading
import uuid
import wave
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from . import gemma, store, subtitles

PRESETS = {
    'small': ('536b0662742c02347bc0e980a01041f333bce120', '3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671'),
    'medium': ('08e178d48790749d25932bbc082711ddcfdfbc4f', '9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae'),
    'large-v3-turbo': ('0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf', 'e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da'),
}
MODEL_LABELS = {'small':'Whisper small', 'medium':'Whisper medium', 'large-v3-turbo':'Large v3 Turbo'}
MODEL_FILES = ['model.bin','config.json','preprocessor_config.json','tokenizer.json','vocabulary.json','vocabulary.txt']
FORMATS = {'.mp4':'mov', '.mov':'mov', '.m4a':'mov', '.mkv':'matroska', '.webm':'matroska', '.wav':'wav', '.mp3':'mp3', '.flac':'flac', '.ogg':'ogg', '.aac':'aac'}
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jread')
LOCK = threading.Lock()
STATE = {'active_job': None, 'downloading': False, 'error': '', 'live': False}
_model = None
_model_key = None
_futures = {}
_processes = {}


def model_ready(name):
    marker = store.MODELS / name / '.verified'
    return name in PRESETS and marker.exists() and marker.read_text().strip() == PRESETS[name][0] and (marker.parent / 'model.bin').is_file() and (name != 'large-v3-turbo' or turbo_assets_ready(marker.parent))


def turbo_assets_ready(path):
    # Without this config faster-whisper defaults to 80 mel bins; Turbo needs 128.
    try:
        return all((path / name).is_file() for name in MODEL_FILES if name != 'vocabulary.txt') and json.loads((path / 'preprocessor_config.json').read_text()).get('feature_size') == 128
    except (OSError, ValueError, AttributeError):
        return False


def validate_translation(options):
    if options.get('model') == 'large-v3-turbo' and options.get('translator','whisper') not in gemma.PRESETS:
        raise ValueError('Large v3 Turbo recognizes speech only. Select Gemma for English translation, or choose another speech model.')


def status():
    settings = store.settings()
    configuration_error = ''
    try:
        validate_translation(settings)
    except ValueError as exc:
        configuration_error = str(exc)
    asr_ready = model_ready(settings['model'])
    translator = settings['translator']
    preset = translator if translator in gemma.PRESETS else 'gemma4'
    gemma_model_ready = gemma.ready(preset)
    gemma_ready = gemma_model_ready and gemma.runtime_ready()
    return {'model': settings['model'], 'model_label': MODEL_LABELS.get(settings['model'],settings['model']), 'device': settings['device'], 'translator': settings['translator'],
            'translator_label': gemma.PRESETS[translator]['label'] if translator in gemma.PRESETS else 'Whisper direct translation',
            'translator_download_bytes': gemma.PRESETS[translator]['size'] if translator in gemma.PRESETS else 0,
            'asr_ready': asr_ready, 'gemma_ready': gemma_ready, 'gemma_model_ready': gemma_model_ready,
            'ready': not configuration_error and asr_ready and (settings['translator'] == 'whisper' or gemma_ready),
            'downloading': STATE['downloading'], 'error': configuration_error or STATE['error']}


def download(name):
    if name not in PRESETS and name not in gemma.PRESETS:
        raise ValueError('Unknown model preset.')
    if STATE['downloading'] or STATE['active_job'] or STATE['live']:
        raise ValueError('Wait for the current operation to finish.')
    STATE.update(downloading=True, error='')
    def run():
        try:
            if name in gemma.PRESETS:
                gemma.download(name)
                return
            from huggingface_hub import snapshot_download
            revision, expected = PRESETS[name]
            repository = 'dropbox-dash/faster-whisper-large-v3-turbo' if name == 'large-v3-turbo' else f'Systran/faster-whisper-{name}'
            path = Path(snapshot_download(repository, revision=revision, local_dir=store.MODELS/name, allow_patterns=MODEL_FILES))
            with (path / 'model.bin').open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            if actual != expected:
                raise RuntimeError('Model checksum failed; retry the download.')
            if name == 'large-v3-turbo' and not turbo_assets_ready(path):
                raise ValueError('Large v3 Turbo preprocessing files are missing or invalid. Retry the model download.')
            (path / '.verified').write_text(revision)
        except Exception as exc:
            STATE['error'] = str(exc) if isinstance(exc,ValueError) else f'Model download failed: {type(exc).__name__}. Check disk space and network, then retry.'
        finally:
            STATE['downloading'] = False
    threading.Thread(target=run, daemon=True, name='jread-download').start()


def remove_model(name):
    global _model, _model_key
    if name not in PRESETS and name not in gemma.PRESETS:
        raise ValueError('Unknown model.')
    if STATE['active_job'] or STATE['live'] or STATE['downloading']:
        raise ValueError('Stop processing before removing models.')
    with LOCK:
        if name in gemma.PRESETS:
            gemma.remove(name)
            return
        _model = _model_key = None
        shutil.rmtree(store.MODELS / name, ignore_errors=True)


def load(name, device):
    global _model, _model_key
    if not model_ready(name):
        raise ValueError('Download this model in Setup before generating subtitles.')
    # CPU is predictable alongside other GPU applications; explicit CUDA remains selectable.
    resolved = 'cpu' if device == 'auto' else device
    key = (name, resolved)
    if key != _model_key:
        from faster_whisper import WhisperModel
        _model = None
        _model_key = None
        _model = WhisperModel(str(store.MODELS/name), device=resolved, compute_type='int8' if resolved == 'cpu' else 'int8_float16', cpu_threads=4, num_workers=1, local_files_only=True)
        _model_key = key
    return _model


def read_audio(path):
    with wave.open(str(path),'rb') as audio:
        if (audio.getnchannels(),audio.getsampwidth(),audio.getframerate()) != (1,2,16000):
            raise ValueError('Expected normalized mono 16 kHz PCM audio.')
        return np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(np.float32)/32768


def media_info(path, suffix):
    if suffix not in FORMATS:
        raise ValueError('Use MP4, MKV, WebM, MOV, M4A, WAV, MP3, FLAC, OGG, or AAC.')
    result = subprocess.run(['ffprobe','-v','error','-protocol_whitelist','file,pipe','-f',FORMATS[suffix],'-show_format','-show_streams','-of','json',str(path)], capture_output=True, timeout=30, check=True)
    info = json.loads(result.stdout)
    streams = [s for s in info.get('streams',[]) if s.get('codec_type') == 'audio']
    if not streams:
        raise ValueError('No audio track was found in this file.')
    stream = next((s for s in streams if s.get('tags',{}).get('language','').lower() in ('ja','jpn')), streams[0])
    duration = float(info.get('format',{}).get('duration') or stream.get('duration') or 0)
    if not 0 < duration <= 10800:
        raise ValueError('Use media with a known duration of up to three hours.')
    return duration, stream['index']


def submit(job_id):
    if job_id in _futures and not _futures[job_id].done():
        raise ValueError('This job is already running.')
    if STATE['live']:
        raise ValueError('Stop live capture before starting a file job.')
    store.update(job_id, status='queued', error='')
    _futures[job_id] = POOL.submit(process, job_id)


def cancelled(job_id):
    try:
        return store.get(job_id, False)['status'] == 'cancelled'
    except KeyError:
        return True


def cancel(job_id):
    job = store.update(job_id, status='cancelled', cancellation_pending=True,
                       error='Cancellation requested. Finishing the current inference step.')
    process = _processes.get(job_id)
    if process and process.poll() is None:
        process.terminate()
    future = _futures.get(job_id)
    if not future or future.cancel() or future.done():
        job = store.update(job_id, cancellation_pending=False, error='Cancelled. Completed cues are retained.')
    return job


def process(job_id):
    STATE['active_job'] = job_id
    folder = store.MEDIA / job_id
    try:
        with LOCK:
            if cancelled(job_id):
                return
            job = store.get(job_id)
            validate_translation(job)
            settings = {'model': job['model'], 'device': job['device']}
            model = load(**{'name':settings['model'], 'device':settings['device']})
            wav = folder / 'audio.wav'
            duration = job['duration']
            if not wav.exists():
                store.update(job_id, status='preparing', progress=.02)
                cmd = ['ffmpeg','-nostdin','-v','error','-threads','2','-protocol_whitelist','file,pipe','-f',FORMATS[job['suffix']],'-i',str(folder/'source'),'-map',f"0:{job['audio_stream']}",'-vn','-ac','1','-ar','16000','-t','10800','-y',str(folder/'audio.part.wav')]
                process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                _processes[job_id] = process
                if cancelled(job_id):
                    process.terminate()
                try:
                    _, stderr = process.communicate(timeout=600)
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, cmd)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    raise
                finally:
                    _processes.pop(job_id, None)
                (folder/'audio.part.wav').replace(wav)
            if cancelled(job_id):
                return
            samples = read_audio(wav)
            transcript_file = folder/'source-cues.json'
            sources = json.loads(transcript_file.read_text()) if transcript_file.exists() else []
            audio_duration = min(duration, len(samples) / 16000)
            sources = list(bounded_segments((SimpleNamespace(**s) for s in sources), audio_duration))
            if not store.get(job_id,False).get('source_done'):
                store.update(job_id, status='recognizing', progress=.1)
                start_at = sources[-1]['end'] if sources else 0
                segments, _ = model.transcribe(samples, language=job['language'], task='transcribe', vad_filter=True, beam_size=5, clip_timestamps=str(start_at), **asr_hints(job, job['language']))
                for segment in bounded_segments(segments, audio_duration):
                    if cancelled(job_id):
                        return
                    sources.append(segment)
                    temp = transcript_file.with_suffix('.tmp')
                    temp.write_text(json.dumps(sources,ensure_ascii=False)); temp.replace(transcript_file)
                    store.update(job_id, progress=min(.49,.1+.39*segment['end']/duration))
                store.update(job_id, source_done=True)
            if cancelled(job_id):
                return
            store.update(job_id, status='translating', progress=.5)
            start_at = float(store.get(job_id,False).get('translated_until',0))
            if job.get('translator','whisper') in gemma.PRESETS:
                segments = gemma_segments(sources, start_at, should_stop=lambda: cancelled(job_id), preset=job['translator'], glossary=caption_glossary(job) if job['language'] == 'ja' else None)
            else:
                segments, _ = model.transcribe(samples, language=job['language'], task='translate', vad_filter=True, beam_size=5, clip_timestamps=str(start_at))
            for segment in segments:
                if cancelled(job_id):
                    return
                start, end = max(0,start_at,segment.start), min(audio_duration,segment.end)
                if end <= start or not segment.text.strip():
                    continue
                source = ' '.join(s['text'] for s in sources if s['end']>start and s['start']<end)
                for index, cue in enumerate(subtitles.assemble(start,end,segment.text)):
                    if getattr(segment, 'needs_review', False):
                        cue['status'] = 'needs_review'
                    cue.update(id=hashlib.sha256(f'{job_id}:{start:.3f}:{end:.3f}:{index}'.encode()).hexdigest()[:24],source_text=source)
                    store.add_cue(job_id,cue)
                store.update(job_id,progress=min(.99,.5+.49*end/duration),translated_until=end)
            if not cancelled(job_id):
                count = store.get(job_id,False)['cue_count']
                store.update(job_id,status='complete',progress=1,error='' if count else 'No speech was detected. Try another clip or the medium model.')
                wav.unlink(missing_ok=True)
    except Exception as exc:
        if not cancelled(job_id):
            error = str(exc) if isinstance(exc,ValueError) else f'{type(exc).__name__}: local processing failed. Check the selected device and media, then retry.'
            store.update(job_id,status='failed',error=error)
    finally:
        STATE['active_job'] = None
        (folder/'audio.part.wav').unlink(missing_ok=True)
        try:
            if cancelled(job_id):
                store.update(job_id, cancellation_pending=False, error='Cancelled. Completed cues are retained; Resume continues processing.')
        except KeyError:
            pass


def bounded_segments(segments, duration):
    """ASR may extend beyond its input; captions must stay on the captured clock."""
    for segment in segments:
        if not all(math.isfinite(t) for t in (segment.start, segment.end)):
            raise ValueError('Speech recognition returned invalid timestamps.')
        start, end = max(0, segment.start), min(duration, segment.end)
        text = segment.text.strip()
        if start < end and text:
            yield dict(start=start, end=end, text=text)


def gemma_text_parts(text):
    """Bound JSON-escaped UTF-8 text, including unusual quotes/control characters."""
    part, size = [], 0
    for char in text:
        length = len(json.dumps(char, ensure_ascii=False).encode('utf-8')) - 2
        if size + length > 900:
            yield ''.join(part)
            part, size = [], 0
        part.append(char)
        size += length
    if part:
        yield ''.join(part)


def gemma_segments(sources, start_at=0, should_stop=lambda: False, preset='gemma4', context='', glossary=None, source_language='ja'):
    """Translate bounded groups while retaining the source ASR media clock."""
    remaining = [s for s in sources if s['end'] > start_at and s['text'].strip()]
    context = (context + ' ' + ' '.join(s['text'] for s in sources if s['end'] <= start_at)).strip()[-150:]
    hints = {'glossary': glossary} if glossary else {}
    if source_language != 'ja':
        hints['source_language'] = source_language
    index = 0
    while index < len(remaining):
        if should_stop():
            return
        source = remaining[index]
        parts = list(gemma_text_parts(source['text']))
        if len(parts) > 1:
            translated = []
            for part in parts:
                if should_stop():
                    return
                translated.extend(gemma.translate_batch([source | {'text': part}], context=context, preset=preset, **hints))
                context = (context + ' ' + part)[-150:]
            if should_stop():
                return
            # Text-only splitting has no word alignment; keep the original clock.
            yield SimpleNamespace(start=source['start'], end=source['end'], text=' '.join(translated), source_text=source['text'], needs_review=True)
            index += 1
            continue
        batch = []
        size = 0
        while index < len(remaining) and len(batch) < 4:
            source = remaining[index]
            length = len(json.dumps(source['text'], ensure_ascii=False).encode('utf-8')) - 2
            if batch and size + length > 900:
                break
            batch.append(source);size += length;index += 1
        translated = gemma.translate_batch(batch, context=context, preset=preset, **hints)
        for source, text in zip(batch, translated, strict=True):
            if should_stop():
                return
            yield SimpleNamespace(start=source['start'],end=source['end'],text=text,source_text=source['text'])
        context = (context + ' ' + ' '.join(s['text'] for s in batch))[-150:]


def caption_glossary(settings):
    if settings.get('caption_glossary', 'none') == 'none':
        return []
    # Explicit show profile; never infer names from unrelated browser page content.
    if settings.get('caption_glossary') != 'aikatsu_pripara':
        raise ValueError('Unknown caption glossary.')
    return json.loads((Path(__file__).parent / 'caption-glossary.json').read_text())['entries']


def asr_hints(settings, language, context=''):
    if language != 'ja':
        return {}
    glossary = caption_glossary(settings)
    hints = {}
    if glossary:
        hints['hotwords'] = '、'.join(dict.fromkeys(term for entry in glossary for term in entry.get('asr_terms', entry['aliases'])))
    if context:
        hints['initial_prompt'] = context[-150:]
    return hints


def live_infer(samples, language, offset, rate, *, context=''):
    if not isinstance(language, str) or language not in gemma.SOURCE_LANGUAGES:
        raise ValueError('Unsupported source language.')
    settings = store.settings()
    validate_translation(settings)
    with LOCK:
        model = load(settings['model'], settings['device'])
        use_gemma = language != 'en' and settings['translator'] in gemma.PRESETS
        segments, _ = model.transcribe(samples,language=language,task='transcribe' if use_gemma or language == 'en' else 'translate',vad_filter=True,beam_size=5 if use_gemma else 1,condition_on_previous_text=False,**asr_hints(settings, language, context))
        duration = len(samples) / 16000
        sources = list(bounded_segments(segments, duration))
        if use_gemma:
            segments = gemma_segments(sources, preset=settings['translator'], context=context, glossary=caption_glossary(settings) if language == 'ja' else None, source_language=language)
        else:
            segments = (SimpleNamespace(**s) for s in sources)
        cues=[]
        for s in segments:
            start,end = offset+s.start*rate,min(86400,offset+s.end*rate)
            if end<=start or not s.text.strip():
                continue
            for cue in subtitles.assemble(start,end,s.text):
                if getattr(s, 'needs_review', False):
                    cue['status'] = 'needs_review'
                cue.update(id=uuid.uuid4().hex,source_text=getattr(s,'source_text',''))
                cues.append(cue)
        return cues
