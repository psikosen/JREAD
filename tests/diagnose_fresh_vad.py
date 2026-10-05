"""CPU-only development diagnostic; stop at the first new-voice rejection risk."""
import hashlib
import importlib.metadata
import json
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tests/fixtures/accuracy-corpus.json'
OUTPUT = ROOT / 'test-results/accuracy-fresh-vad-controls.json'
RATE = 16000
FRESH = RATE


def digest(data):
    return hashlib.sha256(data).hexdigest()


def energy(samples):
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0
    return dict(rms=rms, peak=float(np.max(np.abs(samples))) if len(samples) else 0,
                rms_dbfs=20 * float(np.log10(rms)) if rms else None)


def write(report):
    OUTPUT.parent.mkdir(exist_ok=True)
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(OUTPUT)


def main():
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    cases = [case for case in manifest['cases'] if case['partition'] == 'development']
    assert len(cases) == 20
    vad = get_vad_model()
    assert vad.session.get_providers() == ['CPUExecutionProvider']
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), status='running',
                  scope='Development audio only; CPU Silero, no ASR/translation/GPU. Known source speech is rescaled, not synthesized. This tests speech preservation risk, not word fidelity or human audibility.',
                  corpus=dict(dataset=manifest['dataset'], revision=manifest['revision'], manifest_sha256=digest(raw)),
                  runtime=dict(faster_whisper=importlib.metadata.version('faster-whisper'), onnxruntime=importlib.metadata.version('onnxruntime'), providers=vad.session.get_providers()),
                  vad_options=dict(threshold=.5, neg_threshold=.35, min_speech_duration_ms=0, min_silence_duration_ms=2000, baseline_speech_pad_ms=400, candidate_speech_pad_ms=0),
                  design='Use the first >=1.312s run whose original 32ms speech probabilities are all >=0.9. Place exact source crops around the 1s overlap boundary; use fixed 0/-12/-24dB gains and 4s packets. Stop on the first fresh voiced crop rejected by the candidate but retained by padded VAD. No threshold tuning.',
                  source_files_opened=[], rows=[])
    for case in cases:
        path = (ROOT / case['audio_path']).resolve()
        assert path.is_relative_to(ROOT / 'test-results/accuracy-audio')
        content = path.read_bytes()
        assert digest(content) == case['audio_sha256']
        report['source_files_opened'].append(dict(id=case['id'], partition=case['partition'], audio_sha256=case['audio_sha256']))
        with wave.open(str(path), 'rb') as file:
            assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, RATE)
            audio = np.frombuffer(file.readframes(file.getnframes()), dtype='<i2').astype(np.float32) / 32768
        probabilities = np.asarray(vad(np.pad(audio, (0, (-len(audio)) % 512)))).reshape(-1)
        run = 0
        for index, probability in enumerate(probabilities):
            run = run + 1 if probability >= .9 else 0
            if run >= 41:
                left = (index - run + 1) * 512
                break
        else:
            continue
        source = audio[left:left + 41 * 512]
        provenance = dict(case_id=case['id'], audio_path=case['audio_path'], audio_sha256=case['audio_sha256'],
                          source_filename=case['source_filename'], row_index=case['row_index'],
                          continuous_source_start_sample=left, continuous_source_end_sample=left + len(source),
                          original_speech_probability_min=float(min(probabilities[left // 512:left // 512 + 41])),
                          source_energy=energy(source))
        specifications = [('retained_voice_fresh_zeros', 0)]
        specifications += [('continuous_boundary_crossing', db) for db in (0, -12, -24)]
        specifications += [('fresh_short_voiced_excerpt', db) for db in (0, -12, -24)]
        for kind, db in specifications:
            packet = np.zeros(4 * RATE, dtype=np.float32)
            gain = 10 ** (db / 20)
            pieces = []
            if kind == 'continuous_boundary_crossing':
                # A continuous 1.2s speech excerpt leaves 200ms of known speech fresh.
                packet[:19200] = source[:19200] * gain
                pieces.append(dict(packet_start=0, packet_end=19200, source_start=left, source_end=left + 19200, gain_db=db))
                fresh_voice = (FRESH, 19200)
            else:
                # Retained voice ends at .8s; a new 160ms crop begins at 1.04s.
                packet[1600:12800] = source[:11200]
                pieces.append(dict(packet_start=1600, packet_end=12800, source_start=left, source_end=left + 11200, gain_db=0))
                fresh_voice = None
                if kind == 'fresh_short_voiced_excerpt':
                    packet[16640:19200] = source[16000:18560] * gain
                    pieces.append(dict(packet_start=16640, packet_end=19200, source_start=left + 16000, source_end=left + 18560, gain_db=db))
                    fresh_voice = (16640, 19200)
            unpadded = get_speech_timestamps(packet, VadOptions(speech_pad_ms=0))
            padded = get_speech_timestamps(packet, VadOptions())
            admitted = any(interval['end'] > FRESH for interval in unpadded)
            covered = sum(max(0, min(interval['end'], fresh_voice[1]) - max(interval['start'], fresh_voice[0])) for interval in padded) if fresh_voice else 0
            row = dict(kind=kind, gain_db=db, provenance=provenance, source_crops=pieces,
                       packet_sha256=digest(packet.astype('<f4').tobytes()), packet_samples=len(packet), fresh_start_samples=FRESH,
                       fresh_voice_interval=list(fresh_voice) if fresh_voice else None, fresh_region_energy=energy(packet[FRESH:]),
                       fresh_voiced_crop_energy=energy(packet[slice(*fresh_voice)]) if fresh_voice else None,
                       unpadded_intervals=unpadded, padded_intervals=padded,
                       baseline_admits_packet=bool(padded), candidate_admits_packet=admitted,
                       padded_coverage_of_known_fresh_voice_samples=covered,
                       added_rejection_risk=bool(fresh_voice and padded and not admitted),
                       known_fresh_voice_only_in_padding=bool(fresh_voice and covered and not admitted))
            report['rows'].append(row)
            if row['added_rejection_risk']:
                report.update(status='counterexample_found', recommendation='Keep experimental; do not enable by default. The unpadded fresh-region gate rejects a packet containing known source speech that the existing padded VAD admits. No word fidelity or audibility claim is made.',
                              stopped_early=True, completed_at=datetime.now(timezone.utc).isoformat())
                write(report)
                print(json.dumps(dict(status=report['status'], case_id=case['id'], kind=kind, gain_db=db, in_padding=row['known_fresh_voice_only_in_padding'], report=str(OUTPUT.relative_to(ROOT)))))
                return
            write(report)
    report.update(status='bounded_controls_complete', stopped_early=False,
                  recommendation='No counterexample in these bounded development controls; this does not establish safety for quiet or short unseen speech. Keep experimental pending broader preservation tests.',
                  completed_at=datetime.now(timezone.utc).isoformat())
    write(report)
    print(json.dumps(dict(status=report['status'], rows=len(report['rows']), report=str(OUTPUT.relative_to(ROOT)))))


if __name__ == '__main__':
    main()
