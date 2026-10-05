#!/usr/bin/env python3
"""Resume the unscored Laya screen after restoring its pinned tokenizer bytes."""
import argparse
import fcntl
import json
from pathlib import Path
import evaluate_caption_verifiers as frozen


if __name__ == '__main__':
    original_base = frozen.BASE
    manifest_path = original_base / 'laya-tokenizer-reset-v2/recovery.json'
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['bindings']:
        if frozen.sha(item['path']) != item['sha256']:
            raise ValueError(f'Recovery binding changed: {item["path"]}')
    with (original_base / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen.BASE = original_base / 'laya-tokenizer-reset-v2'
        prior = frozen.BASE / 'runs/laya-smoke-v1/report.json'
        prior.parent.mkdir(parents=True, exist_ok=True)
        # Read the exact successful smoke report through a symlink. No smoke
        # generation is repeated and no previous report is rewritten.
        prior.symlink_to(original_base / 'runs/laya-smoke-v1/report.json')
        args = argparse.Namespace(model='laya', phase='screen', protocol=frozen.PROTOCOL)
        raise SystemExit(frozen.run(args))
