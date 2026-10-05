import json
import os
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

DATA = Path(os.environ.get('JREAD_DATA_DIR', Path.home() / '.local/share/JREAD'))
DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
MEDIA = DATA / 'media'
MEDIA.mkdir(exist_ok=True, mode=0o700)
MODELS = DATA / 'models'
MODELS.mkdir(exist_ok=True, mode=0o700)
TOKEN_FILE = DATA / 'token'
if not TOKEN_FILE.exists():
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(secrets.token_urlsafe(32))
os.chmod(TOKEN_FILE, 0o600)
TOKEN = TOKEN_FILE.read_text().strip()
DEFAULT_SETTINGS = {'model': 'small', 'device': 'cpu', 'translator': 'whisper', 'caption_glossary': 'none', 'caption_context': False, 'caption_boundaries': False, 'font_size': 28, 'show_japanese': False, 'caption_background': True, 'caption_position': 10}


def connect():
    conn = sqlite3.connect(DATA / 'library.sqlite3', timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


with connect() as db:
    db.executescript('''
      PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, data TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS cues(id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
        start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL,
        source_text TEXT NOT NULL, status TEXT NOT NULL, revisions TEXT NOT NULL DEFAULT '[]');
      CREATE INDEX IF NOT EXISTS cues_job ON cues(job_id,start);
      CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS sessions(hash TEXT PRIMARY KEY, expires REAL NOT NULL);
    ''')


def settings(values=None):
    with connect() as db:
        if values:
            db.executemany('INSERT OR REPLACE INTO settings VALUES (?,?)', [(k, json.dumps(v)) for k, v in values.items()])
        return DEFAULT_SETTINGS | {row['key']: json.loads(row['value']) for row in db.execute('SELECT * FROM settings')}


def create(title, **kwargs):
    config = settings()
    job = dict(id=uuid.uuid4().hex, title=title[:200], status='queued', progress=0, created_at=time.time(), language='ja', duration=0, error='', has_media=False, model=config['model'], device=config['device'], translator=config['translator'], caption_glossary=config.get('caption_glossary','none'), caption_context=config.get('caption_context',False), caption_boundaries=config.get('caption_boundaries',False)) | kwargs
    with connect() as db:
        db.execute('INSERT INTO jobs VALUES (?,?)', (job['id'], json.dumps(job)))
    return job | {'cue_count': 0}


def get(job_id, with_cues=True):
    with connect() as db:
        row = db.execute('SELECT data FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        job = json.loads(row['data'])
        job['cue_count'] = db.execute('SELECT COUNT(*) FROM cues WHERE job_id=?', (job_id,)).fetchone()[0]
        if with_cues:
            job['cues'] = [dict(c) for c in db.execute('SELECT id,start,end,text,source_text,status FROM cues WHERE job_id=? ORDER BY start,end', (job_id,))]
    return job


def all_jobs():
    with connect() as db:
        return [json.loads(row['data']) | {'cue_count': row['n']} for row in db.execute('SELECT data,(SELECT COUNT(*) FROM cues WHERE job_id=jobs.id) n FROM jobs ORDER BY rowid DESC')]


def update(job_id, **fields):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT data FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        value = json.loads(row['data']) | fields
        db.execute('UPDATE jobs SET data=? WHERE id=?', (json.dumps(value), job_id))
    return get(job_id)


def add_cue(job_id, cue):
    with connect() as db:
        db.execute('INSERT OR IGNORE INTO cues(id,job_id,start,end,text,source_text,status) VALUES (?,?,?,?,?,?,?)', (cue['id'], job_id, cue['start'], cue['end'], cue['text'], cue.get('source_text', ''), cue.get('status', 'final')))


def edit_cue(job_id, cue_id, values=None, undo=False):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM cues WHERE id=? AND job_id=?', (cue_id, job_id)).fetchone()
        if not row:
            raise KeyError(cue_id)
        cue = dict(row)
        revisions = json.loads(cue.pop('revisions'))
        if undo:
            if not revisions:
                raise ValueError('No earlier revision to restore.')
            cue.update(revisions.pop())
        else:
            revisions.append({key: cue[key] for key in ('text', 'start', 'end', 'status')})
            cue.update(values or {})
            cue['status'] = 'user_edited'
        from .subtitles import validate
        cue['text'] = validate(cue['start'], cue['end'], cue['text'])
        db.execute('UPDATE cues SET text=?,start=?,end=?,status=?,revisions=? WHERE id=?', (cue['text'], cue['start'], cue['end'], cue['status'], json.dumps(revisions[-50:]), cue_id))
    cue.pop('job_id')
    return cue
