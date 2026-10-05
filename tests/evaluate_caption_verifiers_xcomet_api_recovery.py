#!/usr/bin/env python3
"""Reuse the frozen verifier runner in a separate, serialized API-repair namespace."""
import argparse
import fcntl
from pathlib import Path
import evaluate_caption_verifiers as frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['validate', 'smoke', 'screen'])
    args = parser.parse_args()
    original_base = frozen.BASE
    args.model = 'xcomet'
    args.protocol = Path(__file__).parent / 'fixtures/accuracy-caption-verifiers-xcomet-api-v3.json'
    # Preserve the v1 runner, reports, inputs and failed load. Hold the original
    # global lock as well as the runner's namespace lock, so model jobs cannot overlap.
    with (original_base / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen.BASE = original_base / 'xcomet-api-fix-v3'
        frozen.BASE.mkdir(exist_ok=True)
        raise SystemExit(frozen.run(args))
