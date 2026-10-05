"""Prepare a fixed, disjoint 20-case development / 50-case held-out speech set.

Opt-in network operation. FLEURS read speech is supplemental evidence, not an
anime benchmark. Fetches metadata and only selected Japanese audio (CC BY 4.0).
"""
import concurrent.futures
import csv
import hashlib
import io
import json
import subprocess
import urllib.parse
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
BASE = f'https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}'
MANIFEST = ROOT / 'tests/fixtures/accuracy-corpus.json'
CACHE = ROOT / 'test-results/accuracy-audio'


def fetch(url, limit=10_000_000):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Corpus download exceeded its per-file limit.')
    return data


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    if MANIFEST.exists():
        raise SystemExit('Manifest already exists; preserve the frozen set rather than overwriting it.')
    CACHE.mkdir(parents=True, exist_ok=True)
    urls = {language: f'{BASE}/data/{language}/dev.tsv' for language in ('ja_jp', 'en_us')}
    metadata = {language: fetch(url) for language, url in urls.items()}
    tables = {language: list(csv.reader(io.StringIO(data.decode()), delimiter='\t')) for language, data in metadata.items()}
    english = {int(row[0]): row[2] for row in tables['en_us']}
    japanese = {row[1]: row for row in tables['ja_jp']}
    rows = []
    for offset in range(0, len(japanese), 100):
        query = urllib.parse.urlencode(dict(dataset='google/fleurs', config='ja_jp', split='validation', offset=offset, length=100))
        page = json.loads(fetch('https://datasets-server.huggingface.co/rows?' + query))
        if page.get('partial') or page['num_rows_total'] != len(japanese):
            raise ValueError('Dataset Viewer and pinned metadata disagree.')
        rows.extend(page['rows'])
    unique = {}
    for item in rows:
        row = item['row']
        name = Path(row['path']).name
        source = japanese[name]
        if int(source[0]) != row['id'] or source[2] != row['raw_transcription']:
            raise ValueError('Audio row does not match pinned Japanese reference.')
        if row['id'] in english and row['id'] != 1828:
            unique.setdefault(row['id'], item)  # lowest source row per sentence ID
    selected = sorted(unique, key=lambda value: digest(f'jread-accuracy-v1:{value}'.encode()))[:70]
    if len(selected) != 70:
        raise ValueError('Need 70 distinct parallel sentence IDs.')

    def download(index_id):
        index, sentence_id = index_id
        item = unique[sentence_id]
        row = item['row']
        url = row['audio'][0]['src']
        if f'/--/{REVISION}/--/' not in url:
            raise ValueError('Dataset Viewer audio revision differs from frozen references.')
        raw = fetch(url)
        wav = CACHE / f'{sentence_id}.wav'
        temporary = wav.with_suffix('.download')
        temporary.write_bytes(raw)
        try:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(temporary), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-y', str(wav)], check=True, timeout=30)
        finally:
            temporary.unlink(missing_ok=True)
        with wave.open(str(wav)) as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
            seconds = audio.getnframes() / 16000
        return {'id': sentence_id, 'partition': 'development' if index < 20 else 'heldout',
                'row_index': item['row_idx'], 'source_filename': Path(row['path']).name,
                'japanese_reference': row['raw_transcription'], 'english_reference': english[sentence_id],
                'audio_path': str(wav.relative_to(ROOT)), 'duration_seconds': seconds,
                'download_sha256': digest(raw), 'audio_sha256': digest(wav.read_bytes())}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        cases = list(pool.map(download, enumerate(selected)))
    manifest = {'version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': 'google/fleurs', 'revision': REVISION, 'source_split': 'validation',
                'license': 'CC BY 4.0', 'attribution_url': 'https://huggingface.co/datasets/google/fleurs',
                'scope': 'General read speech. Not anime dialogue or a measure of live caption delay.',
                'selection': 'Lowest source row per shared Japanese/English sentence ID; exclude previously tested ID 1828; sort IDs by SHA256(jread-accuracy-v1:ID); first20 development, next50 heldout. No selection based on model output.',
                'metadata': {language: {'url': urls[language], 'sha256': digest(data)} for language, data in metadata.items()},
                'cases': cases}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'manifest': str(MANIFEST), 'development': 20, 'heldout': 50,
                      'seconds': round(sum(case['duration_seconds'] for case in cases), 2),
                      'sha256': digest(MANIFEST.read_bytes())}))


if __name__ == '__main__':
    prepare()
