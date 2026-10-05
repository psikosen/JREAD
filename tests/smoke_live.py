"""Opt-in real PCM/WebSocket inference check against a running local companion."""
import asyncio
import json
import time
import wave
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

ROOT=Path(__file__).resolve().parent.parent
from jread import store


async def main():
    origin='chrome-extension://'+(ROOT/'scripts/extension-id.txt').read_text().strip()
    started=time.monotonic()
    with wave.open(str(ROOT/'tests/fixtures/japanese-speech.wav'),'rb') as audio:
        assert (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())==(16000,1,2)
        data=audio.readframes(audio.getnframes())
    cues=[]
    async with connect('ws://127.0.0.1:8765/api/live',origin=origin) as socket:
        await socket.send(json.dumps(dict(type='hello',token=store.TOKEN,epoch=0,language='ja',title='Live integration fixture')))
        ready=json.loads(await asyncio.wait_for(socket.recv(),10))
        assert ready['type']=='ready',ready
        await socket.send(json.dumps(dict(type='audio',epoch=0,seq=0,media_time=5,sample_rate=16000,rate=1)))
        await socket.send(data)
        await socket.send(json.dumps(dict(type='stop')))
        acknowledged=False
        while True:
            result=json.loads(await asyncio.wait_for(socket.recv(),120))
            assert result['type']!='error',result
            if result['type']=='cue':cues.append(result['cue'])
            if result['type']=='ack':acknowledged=True
            if result['type']=='stopped':break
        # The server closes only after persisting the terminal job state.
        await socket.wait_closed()
    with httpx.Client(base_url='http://127.0.0.1:8765',headers={'Authorization':'Bearer '+store.TOKEN}) as client:
        job=client.get('/api/jobs/'+ready['job_id']).json()
        assert job['status']=='complete' and not job['error'],job
        assert cues and acknowledged and job['cue_count']==len(cues),job
        assert all(5<=c['start']<c['end']<=5+len(data)/32000 for c in cues)
        assert all(len(c['text'].splitlines())<=2 for c in cues)
        report={'passed':True,'elapsed_seconds':round(time.monotonic()-started,2),'job':job,
                'scope':'Actual PCM -> WebSocket -> selected local speech/translation models -> durable captions, including stop drain. Browser tab capture is tested separately.'}
        (ROOT/'test-results').mkdir(exist_ok=True)
        (ROOT/'test-results/live-inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        deleted=client.delete('/api/jobs/'+ready['job_id'])
        assert deleted.status_code==200
    print(json.dumps({k:v for k,v in report.items() if k!='job'}))


if __name__=='__main__':asyncio.run(main())
