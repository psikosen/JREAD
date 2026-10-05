"""Opt-in Japanese -> English check using the installed, selected models."""
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
from jread.server import app
from jread import engine,store,subtitles

root=Path(__file__).resolve().parent.parent
client=TestClient(app,base_url='http://127.0.0.1:8765')
auth={'Authorization':'Bearer '+store.TOKEN}
started=time.monotonic()
with (root/'tests/fixtures/japanese-demo.mp4').open('rb') as f:
    settings=store.settings()
    translator={'gemma4':'Gemma 4 E4B','gemma4-12b':'Gemma 4 12B'}.get(settings['translator'],'Whisper translation')
    title='Japanese demo · '+settings['model']+' + '+translator
    result=client.post('/api/jobs',headers=auth,files={'file':('japanese-demo.mp4',f,'video/mp4')},data={'title':title,'language':'ja'})
assert result.status_code==201,result.text
job_id=result.json()['id']
engine._futures[job_id].result(timeout=180)
job=client.get(f'/api/jobs/{job_id}',headers=auth).json()
assert job['status']=='complete',job
assert job['cues'],job
assert all(0<=c['start']<c['end']<=job['duration'] for c in job['cues'])
assert any(any('\u3040'<=ch<='\u9fff' for ch in c['source_text']) for c in job['cues'])
assert all(any('a'<=ch.lower()<='z' for ch in c['text']) for c in job['cues'])
for kind in ('srt','vtt'):
    exported=client.get(f'/api/jobs/{job_id}/export?format={kind}',headers=auth)
    assert exported.status_code==200
    assert len(subtitles.parse(exported.text))==len(job['cues'])
report={'passed':True,'elapsed_seconds':round(time.monotonic()-started,2),'job':job,'fixture':'google/fleurs, ja_jp, test row 0 (CC BY 4.0)','quality_note':'Automated pipeline smoke check, not a bilingual quality benchmark.'}
(root/'test-results').mkdir(exist_ok=True)
(root/'test-results/inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
