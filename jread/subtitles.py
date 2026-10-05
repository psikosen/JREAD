"""Plain-text subtitles with a single media clock (seconds)."""
import html
import math
import re
import textwrap
import uuid


def validate(start, end, text):
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (start, end)):
        raise ValueError('Cue times must be finite numbers.')
    if not 0 <= start < end <= 86400:
        raise ValueError('Cue end must follow its start (maximum 24 hours).')
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError('Cue text must contain 1–4000 characters.')
    return text.strip()


def parse_time(value):
    parts = value.replace(',', '.').split(':')
    if len(parts) not in (2, 3):
        raise ValueError('Invalid subtitle timestamp.')
    values = [float(p) for p in parts]
    if any(not math.isfinite(v) or v < 0 for v in values) or values[-1] >= 60 or values[-2] >= 60:
        raise ValueError('Invalid subtitle timestamp.')
    return values[-1] + values[-2] * 60 + (values[0] * 3600 if len(values) == 3 else 0)


def parse(data):
    data = data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    cues = []
    for block in re.split(r'\n\s*\n', data.strip()):
        lines = block.splitlines()
        if not lines or lines[0].startswith(('WEBVTT', 'NOTE', 'STYLE', 'REGION')):
            continue
        index = next((i for i, line in enumerate(lines[:2]) if '-->' in line), None)
        if index is None:
            raise ValueError('Expected SRT or WebVTT cue timestamps.')
        times = lines[index].split('-->')
        if len(times) != 2:
            raise ValueError('Invalid cue timing line.')
        start, end = parse_time(times[0].strip()), parse_time(times[1].strip().split()[0])
        text = html.unescape(re.sub(r'<[^>]*>', '', '\n'.join(lines[index + 1:])))
        text = validate(start, end, text)
        cues.append(dict(id=uuid.uuid4().hex, start=start, end=end, text=text, source_text='', status='final'))
        if len(cues) > 20000:
            raise ValueError('Subtitle file contains too many cues.')
    if not cues:
        raise ValueError('No subtitle cues found.')
    return sorted(cues, key=lambda cue: (cue['start'], cue['end']))


def timestamp(seconds, separator='.'):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02}:{m:02}:{s:02}{separator}{ms:03}'


def export(cues, format='vtt'):
    if format not in ('vtt', 'srt'):
        raise ValueError('Choose SRT or WebVTT.')
    blocks = ['WEBVTT\n'] if format == 'vtt' else []
    for i, cue in enumerate(cues, 1):
        validate(cue['start'], cue['end'], cue['text'])
        sep = ',' if format == 'srt' else '.'
        # Escape cue markup so subtitle text cannot become WebVTT tags.
        text = html.escape(cue['text'], quote=False)
        blocks.append(f"{i}\n{timestamp(cue['start'], sep)} --> {timestamp(cue['end'], sep)}\n{text}\n")
    return '\n'.join(blocks)


def assemble(start, end, text):
    """Keep generated English captions to two lines without dropping words.

    Whisper supplies segment timing, not translated word alignment. Splitting a
    long segment therefore estimates its internal boundaries and needs review.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    validate(start, end, text)
    lines = textwrap.wrap(text, width=42, break_long_words=False, break_on_hyphens=False)
    chunks = ['\n'.join(lines[i:i+2]) for i in range(0, len(lines), 2)]
    weight = sum(len(chunk) for chunk in chunks)
    elapsed = 0
    cues = []
    for i, chunk in enumerate(chunks):
        cue_start = start + (end-start) * elapsed/weight
        elapsed += len(chunk)
        cue_end = end if i == len(chunks)-1 else start + (end-start) * elapsed/weight
        review = (len(chunks) > 1 or len(chunk)/(cue_end-cue_start) > 20
                  or any(len(line) > 42 for line in chunk.splitlines())
                  or any('\u3040' <= char <= '\u9fff' for char in chunk))
        cues.append(dict(start=cue_start, end=cue_end, text=chunk,
                         status='needs_review' if review else 'final'))
    return cues
