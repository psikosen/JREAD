"""Opt-in same-transcript E4B/12B comparison using installed models; no downloads."""
import argparse
import json
import time
from pathlib import Path
from jread import gemma

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preset', choices=['gemma4', 'gemma4-12b'], required=True)
args = parser.parse_args()
# The installed companion owns 18766. This benchmark owns only its isolated port.
gemma.PORT = 18767
gemma.BASE = f'http://{gemma.HOST}:{gemma.PORT}'
comparison = json.loads((ROOT/'test-results/gemma-e4b-comparison.json').read_text())
cases = [{'name': 'fleurs', 'segments': [{'start': 0, 'end': 10.44,
          'text': comparison['fixture']['source']}]}]
for index, chunk in enumerate(comparison['page_chunks']):
    cases.append({'name': f'dialogue-{index+1:02d}', 'media_start': chunk['media_start'],
                  'segments': chunk['source']})
live = json.loads((ROOT/'test-results/actual-page-gemma-e4b.json').read_text())['job']['cues']
for index, (start, end) in enumerate([(0, 2), (2, 4), (4, 8), (8, 9)]):
    cases.append({'name': f'live-{index+1:02d}', 'segments': [
        {'start': cue['start'], 'end': cue['end'], 'text': cue['source_text']}
        for cue in live[start:end]]})
report = {'preset': args.preset, 'model': gemma.PRESETS[args.preset], 'complete': False,
          'runtime': 'b11146', 'device': 'cpu', 'threads': 8,
          'thinking': False, 'temperature': 0,
          'system_prompt': gemma._messages(cases[0]['segments'], '')[0]['content'],
          'scope': 'Same saved Japanese transcripts and current prompt; no recognition or browser latency included. Small qualitative sample, not an accuracy benchmark.',
          'cases': []}
output = ROOT/'test-results'/f'{args.preset}-matched-comparison.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
started = time.monotonic()
try:
    for case in cases:
        before = time.monotonic()
        translations = gemma.translate_batch(case['segments'], preset=args.preset)
        row = case | {'translations': translations, 'seconds': round(time.monotonic()-before, 3)}
        report['cases'].append(row)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({'case': case['name'], 'seconds': row['seconds'], 'translations': translations}, ensure_ascii=False), flush=True)
finally:
    gemma.shutdown()
report['elapsed_seconds'] = round(time.monotonic()-started, 3)
report['complete'] = len(report['cases']) == len(cases)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f'Saved {len(cases)} cases to {output}', flush=True)
