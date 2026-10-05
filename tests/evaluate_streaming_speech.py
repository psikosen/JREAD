"""Bounded development-only Whisper LocalAgreement-2 feasibility experiment."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'tests/fixtures/accuracy-corpus.json'
MODEL = Path.home() / '.local/share/JREAD/models/medium'
OUTPUT = ROOT / 'test-results/accuracy-streaming-speech-development.json'
REVISION = '08e178d48790749d25932bbc082711ddcfdfbc4f'
FLAGS = dict(language='ja', task='transcribe', vad_filter=True, beam_size=5,
             condition_on_previous_text=False)
SAMPLE_RATE = 16000
STEP_SECONDS = 2
MAX_SECONDS = 30
TIMESTAMP_JITTER_SECONDS = 0.75
SENTENCE_END = re.compile(r'[。！？!?][」』】）》”’"\']*$')
CLOSERS = set('」』】）》”’"\']')


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def character_error(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        raise ValueError('CER requires a nonempty normalized reference.')
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1,
                               previous[j-1] + (left != right)))
        previous = current
    return dict(errors=previous[-1], reference_characters=len(reference),
                cer=previous[-1] / len(reference), normalized_reference=reference,
                normalized_hypothesis=hypothesis)


def development_cases(manifest):
    cases = [case for case in manifest['cases'] if case['partition'] == 'development']
    if len(cases) != 20 or len({case['id'] for case in cases}) != 20:
        raise ValueError('Frozen manifest must contain20 distinct development cases.')
    return cases


def word(text, start, end, **extra):
    return dict(text=text, start_seconds=float(start), end_seconds=float(end), **extra)


def token_text(tokens):
    return ''.join(token['text'] for token in tokens)


def _character_spans(tokens, length):
    spans = []
    for token in tokens:
        spans.extend([(token['start_seconds'], token['end_seconds'])] * len(token['text']))
        if len(spans) >= length:
            return spans[:length]
    return spans


def _timed_prefix_compatible(previous, current, length):
    left, right = _character_spans(previous, length), _character_spans(current, length)
    if len(left) != length or len(right) != length:
        return False
    for (a_start, a_end), (b_start, b_end) in zip(left, right):
        gap = max(a_start - b_end, b_start - a_end, 0.0)
        if gap > TIMESTAMP_JITTER_SECONDS:
            return False
    return True


def common_complete_prefix(previous, current):
    """Exact Japanese LCP at shared token boundaries and overlapping times."""
    left, right = token_text(previous), token_text(current)
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    left_boundaries = {0}
    right_boundaries = {0}
    for tokens, bounds in ((previous, left_boundaries), (current, right_boundaries)):
        position = 0
        for token in tokens:
            position += len(token['text'])
            bounds.add(position)
    eligible = sorted((boundary for boundary in left_boundaries & right_boundaries
                       if boundary <= common), reverse=True)
    for boundary in eligible:
        if _timed_prefix_compatible(previous, current, boundary):
            return boundary
    return 0


class LocalAgreement:
    """LA-2 over complete timestamp-token boundaries, preserving exact text."""
    def __init__(self):
        self.committed = []
        self.pending = []
        self.units_pending = []
        self.units = []
        self.boundary_conflicts = []
        self.finished = False
        self.alignment_valid = True
        self.sentence_terminal_waiting_for_closer = False

    @property
    def committed_text(self):
        return token_text(self.committed)

    def _translation_units(self, new_tokens, emitted_at):
        units = []
        for token in new_tokens:
            if self.sentence_terminal_waiting_for_closer:
                if token['text'] and all(char in CLOSERS for char in token['text']):
                    self.units_pending.append(token)
                    continue
                group, self.units_pending = self.units_pending, []
                unit = self._unit(group, emitted_at, 'stable_sentence_terminal')
                self.units.append(unit)
                units.append(unit)
                self.sentence_terminal_waiting_for_closer = False
            self.units_pending.append(token)
            if SENTENCE_END.search(token['text']):
                # Wait one stable token so a separate closing quote stays attached.
                self.sentence_terminal_waiting_for_closer = not any(c in CLOSERS for c in token['text'][-1:])
                if self.sentence_terminal_waiting_for_closer:
                    continue
                group, self.units_pending = self.units_pending, []
                unit = self._unit(group, emitted_at, 'stable_sentence_terminal')
                self.units.append(unit)
                units.append(unit)
        return units

    @staticmethod
    def _unit(tokens, emitted_at, stability):
        return dict(text=token_text(tokens).strip(),
                    word_timestamps=[dict(token) for token in tokens],
                    source_start_seconds=min(t['start_seconds'] for t in tokens),
                    source_end_seconds=max(t['end_seconds'] for t in tokens),
                    emitted_at_audio_seconds=emitted_at, stability=stability)

    def update(self, hypothesis, emitted_at):
        if self.finished:
            raise RuntimeError('Cannot update after EOF flush.')
        before = [dict(t) for t in self.pending]
        full_text = token_text(hypothesis)
        stable_text = self.committed_text
        self.last_hypothesis_text = full_text
        if not full_text.startswith(stable_text):
            conflict = dict(audio_end_seconds=emitted_at,
                            committed_text=stable_text,
                            hypothesis_prefix=full_text[:len(stable_text)],
                            hypothesis_text=full_text,
                            reason='fresh hypothesis rewrote committed prefix')
            self.boundary_conflicts.append(conflict)
            self.alignment_valid = False
            # No stale-character slicing: the next independent audio update must
            # re-establish the immutable prefix before any new text can commit.
            self.pending = []
            return dict(pending_before=before, stable_words=[], pending_after=[],
                        boundary_conflict=conflict, emitted_units=[])
        if stable_text and not _timed_prefix_compatible(self.committed, hypothesis, len(stable_text)):
            conflict = dict(audio_end_seconds=emitted_at,
                            committed_text=stable_text,
                            hypothesis_prefix=full_text[:len(stable_text)],
                            hypothesis_text=full_text,
                            reason='fresh hypothesis text matches but committed timestamp spans do not overlap')
            self.boundary_conflicts.append(conflict)
            self.alignment_valid = False
            self.pending = []
            return dict(pending_before=before, stable_words=[], pending_after=[],
                        boundary_conflict=conflict, emitted_units=[])
        self.alignment_valid = True

        # The whole-buffer decode repeats its prior prefix. Align only by exact
        # character content; if a new token spans that boundary, preserve the
        # suffix and timestamp explicitly instead of dropping it.
        current = []
        offset = 0
        for token in hypothesis:
            next_offset = offset + len(token['text'])
            if next_offset <= len(stable_text):
                offset = next_offset
                continue
            if offset < len(stable_text):
                cut = len(stable_text) - offset
                current.append(word(token['text'][cut:], token['start_seconds'],
                                    token['end_seconds'],
                                    split_at_committed_text_boundary=True))
            else:
                current.append(dict(token))
            offset = next_offset

        count = common_complete_prefix(self.pending, current)
        stable, self.pending = [], []
        stable_chars = 0
        for token in current:
            if stable_chars + len(token['text']) > count:
                self.pending.extend(current[len(stable):])
                break
            stable.append(token)
            stable_chars += len(token['text'])
        else:
            self.pending = []
        self.committed.extend(stable)
        emitted = self._translation_units(stable, emitted_at)
        return dict(pending_before=before, stable_words=[dict(t) for t in stable],
                    pending_after=[dict(t) for t in self.pending],
                    boundary_conflict=None, emitted_units=emitted)

    def finish(self, emitted_at):
        if self.finished:
            return []
        self.finished = True
        tail = [dict(t) for t in self.pending]
        self.pending = []
        tokens = self.units_pending + tail
        self.units_pending = []
        if not tokens:
            return []
        stability = ('stable_sentence_terminal_eof' if self.sentence_terminal_waiting_for_closer and not tail
                     else 'end_of_recording_unconfirmed')
        unit = self._unit(tokens, emitted_at, stability)
        unit['contains_unstable_tail'] = bool(tail)
        unit['stable_word_count'] = len(tokens) - len(tail)
        unit['unstable_tail_word_count'] = len(tail)
        self.units.append(unit)
        self.committed.extend(tail)
        return [unit]


def self_test():
    from types import SimpleNamespace
    # Exact Japanese punctuation is preserved and sentence units wait for LA-2.
    la = LocalAgreement()
    sentence = [word('私は', 0, .4), word('日本語を', .4, .8), word('話します。', .8, 1.4)]
    assert not la.update(sentence, 2)['stable_words']
    second = la.update(sentence, 4)
    assert token_text(second['stable_words']) == '私は日本語を話します。'
    assert not second['emitted_units']  # Hold one horizon for a separate closing quote.
    assert la.finish(4)[0]['stability'] == 'stable_sentence_terminal_eof'

    # A real repeated phrase at a different time stays duplicated.
    repeated = LocalAgreement()
    two = [word('はい', 0, .2), word('はい', .3, .5)]
    repeated.update(two, 2)
    assert len(repeated.update(two, 4)['stable_words']) == 2

    shifted = LocalAgreement()
    shifted.update([word('はい', 0, .2)], 2)
    assert not shifted.update([word('はい', 2.2, 2.4)], 4)['stable_words']

    committed_repeat = LocalAgreement()
    committed_repeat.update([word('はい', 0, .2)], 2)
    committed_repeat.update([word('はい', 0, .2)], 4)
    phrase = [word('はい', 0, .2), word('はい', 2.2, 2.4)]
    assert not committed_repeat.update(phrase, 6)['stable_words']
    assert token_text(committed_repeat.update(phrase, 8)['stable_words']) == 'はい'

    quoted = LocalAgreement()
    quoted_tokens = [word('「『はい', 0, .2), word('。', .2, .3),
                     word('』', .3, .35), word('」', .35, .4),
                     word('次です。', .4, .8)]
    quoted.update(quoted_tokens, 2)
    quoted_units = quoted.update(quoted_tokens, 4)['emitted_units']
    assert quoted_units[0]['text'] == '「『はい。』」'

    # Revisions commit only the shared complete token and then re-agree.
    revise = LocalAgreement()
    revise.update([word('行き', 0, .2), word('ます。', .2, .5)], 2)
    revised = revise.update([word('行き', 0, .2), word('ました。', .2, .6)], 4)
    assert token_text(revised['stable_words']) == '行き'
    assert token_text(revised['pending_after']) == 'ました。'
    assert token_text(revise.update([word('行き', 0, .2), word('ました。', .2, .6)], 6)['stable_words']) == 'ました。'

    # Different tokenizer groupings agree only at a boundary shared by both.
    grouped = common_complete_prefix([word('東京', 0, .2), word('大学', .2, .5)],
                                     [word('東京大学', 0, .5)])
    assert grouped == len('東京大学')
    assert common_complete_prefix([word('東京大学', 0, .5)],
                                  [word('東京', 0, .2), word('大学', .2, .5)]) == len('東京大学')

    # A rewritten committed prefix is surfaced; no stale offset is spliced in.
    conflict = LocalAgreement()
    conflict.update([word('これは', 0, .3)], 2)
    conflict.update([word('これは', 0, .3)], 4)
    changed = conflict.update([word('それは', 0, .3), word('大事。', .3, .7)], 6)
    assert changed['boundary_conflict'] and not changed['stable_words']

    # One EOF flush retains an unconfirmed final token and is idempotent.
    tail = LocalAgreement()
    tail.update([word('最後の語', 0, .5)], 2)
    flushed = tail.finish(2)
    assert len(flushed) == 1 and flushed[0]['contains_unstable_tail']
    assert not tail.finish(2)
    assert normalize('ＡＢ、「日 本」。') == 'AB日本'
    assert common_complete_prefix([word('はい', 0, .2)],
                                  [word('はい', 2.2, 2.4)]) == 0
    clamped_segments, clamped_words = words_from_segments(
        [SimpleNamespace(start=-1, end=9, text='。', words=[
            SimpleNamespace(start=0, end=0, word='。', probability=1.0),
            SimpleNamespace(start=-1, end=9, word='語', probability=1.0)])], 4)
    assert clamped_segments[0]['start_seconds'] == 0
    assert clamped_segments[0]['end_seconds'] == 4
    assert clamped_words[0]['start_seconds'] == clamped_words[0]['end_seconds'] == 0
    assert clamped_words[1]['end_seconds'] == 4


def write(report):
    OUTPUT.parent.mkdir(exist_ok=True)
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUTPUT)


def words_from_segments(segments, horizon_seconds, offset=0):
    rows, tokens = [], []
    for segment in segments:
        seg_start, seg_end = float(segment.start), float(segment.end)
        if not (float('-inf') < seg_start < float('inf') and float('-inf') < seg_end < float('inf')) or seg_end < seg_start:
            raise ValueError('Whisper returned nonfinite or reversed segment timestamps.')
        seg_start = min(max(seg_start, 0.0), horizon_seconds)
        seg_end = min(max(seg_end, 0.0), horizon_seconds)
        segment_words = []
        for item in (getattr(segment, 'words', None) or []):
            raw_start, raw_end = float(item.start), float(item.end)
            if not (float('-inf') < raw_start < float('inf') and float('-inf') < raw_end < float('inf')) or raw_end < raw_start:
                raise ValueError('Whisper returned nonfinite or reversed word timestamps.')
            start = offset + min(max(raw_start, 0.0), horizon_seconds)
            end = offset + min(max(raw_end, 0.0), horizon_seconds)
            token = word(str(item.word), start, end,
                         probability=float(item.probability) if item.probability is not None else None)
            tokens.append(token)
            segment_words.append(dict(token))
        rows.append(dict(start_seconds=offset + seg_start,
                         end_seconds=offset + seg_end,
                         text=str(segment.text), words=segment_words))
    return rows, tokens


def process_gpu_memory(snapshot, pid):
    for process in snapshot.get('processes', []):
        if process.get('pid') == pid:
            try:
                return int(process['used_mib'])
            except (KeyError, ValueError, TypeError):
                return None
    return None


def worker():
    import resource
    import numpy as np
    from faster_whisper import WhisperModel
    from jread import engine
    from asr_cuda_pilot import gpu_snapshot

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = json.loads(OUTPUT.read_text())
    if digest(MANIFEST) != report['corpus']['manifest_sha256']:
        raise ValueError('Corpus manifest changed after preflight.')
    if digest(Path(__file__).resolve()) != report['runner_sha256']:
        raise ValueError('Runner source changed after preflight.')
    if (MODEL / '.verified').read_text().strip() != REVISION:
        raise ValueError('Whisper medium verification marker differs from pinned revision.')
    cases = development_cases(json.loads(MANIFEST.read_text()))
    model = None
    try:
        started = time.perf_counter()
        model = WhisperModel(str(MODEL), device='cuda', device_index=0,
                             compute_type='int8_float16', cpu_threads=4,
                             num_workers=1, local_files_only=True)
        report['model']['load_seconds'] = time.perf_counter() - started
        after_load = gpu_snapshot()
        report['gpu_after_model_load'] = after_load

        def transcribe(samples, timestamps):
            kwargs = dict(FLAGS)
            if timestamps:
                kwargs['word_timestamps'] = True
            start = time.perf_counter()
            segments, _ = model.transcribe(samples, **kwargs)
            segment_list = list(segments)
            elapsed = time.perf_counter() - start
            rows, tokens = words_from_segments(segment_list, len(samples) / SAMPLE_RATE)
            return rows, tokens, elapsed

        warmup_path = ROOT / 'tests/fixtures/japanese-speech.wav'
        warmup_samples = engine.read_audio(warmup_path)
        warmup_rows, _, warmup_time = transcribe(warmup_samples, True)
        report['warmup'] = dict(excluded=True, path=str(warmup_path.relative_to(ROOT)),
                                audio_sha256=digest(warmup_path),
                                inference_seconds=warmup_time,
                                segment_count=len(warmup_rows))
        silence_rows, _, silence_time = transcribe(np.zeros(8 * SAMPLE_RATE, dtype=np.float32), True)
        report['silence_control'] = dict(duration_seconds=8, expected_empty=True,
                                         passed=not any(s['text'].strip() for s in silence_rows),
                                         inference_seconds=silence_time, segments=silence_rows)
        report['status'] = 'running'
        write(report)

        for case in cases:
            audio_path = (ROOT / case['audio_path']).resolve()
            if not audio_path.is_relative_to(ROOT / 'test-results/accuracy-audio') or digest(audio_path) != case['audio_sha256']:
                raise ValueError(f"Development audio path/hash failed for {case['id']}.")
            samples = engine.read_audio(audio_path)
            duration = len(samples) / SAMPLE_RATE
            if abs(duration - case['duration_seconds']) > 1 / SAMPLE_RATE:
                raise ValueError(f"Duration differs from frozen manifest for {case['id']}.")
            if duration > MAX_SECONDS:
                raise ValueError(f"{case['id']} exceeds the explicitly bounded30second buffer.")

            la = LocalAgreement()
            updates, memory_observations = [], []
            update_total = 0.0
            for end_sample in range(min(STEP_SECONDS * SAMPLE_RATE, len(samples)),
                                    len(samples) + STEP_SECONDS * SAMPLE_RATE,
                                    STEP_SECONDS * SAMPLE_RATE):
                end_sample = min(end_sample, len(samples))
                if end_sample == 0 or (updates and updates[-1]['audio_end_seconds'] >= duration):
                    break
                horizon = end_sample / SAMPLE_RATE
                segments, hypothesis, elapsed = transcribe(samples[:end_sample], True)
                update_total += elapsed
                agreement = la.update(hypothesis, horizon)
                snap = gpu_snapshot()
                memory = process_gpu_memory(snap, os.getpid())
                if memory is not None:
                    memory_observations.append(memory)
                updates.append(dict(audio_end_seconds=horizon,
                                    audio_buffer_start_seconds=0.0,
                                    audio_buffer_seconds=horizon,
                                    inference_seconds=elapsed,
                                    raw_segments=segments,
                                    raw_hypothesis_words=hypothesis,
                                    raw_hypothesis_text=token_text(hypothesis),
                                    committed_text_after=la.committed_text,
                                    **agreement,
                                    owned_process_gpu_memory_mib_after=memory))
                if end_sample == len(samples):
                    break

            final_units = la.finish(duration)
            if not updates:
                raise ValueError(f"No distinct audio observations were made for {case['id']}.")
            updates[-1]['emitted_units'].extend(final_units)
            updates[-1]['eof_flush_unit_count'] = len(final_units)
            final_tokens = list(la.committed)
            final_text = token_text(final_tokens).strip()
            aligned = la.alignment_valid

            # Production chunk baseline: same ASR flags without word timestamps.
            chunks, chunk_time, left = [], 0.0, 0
            while left < len(samples):
                right = min(len(samples), left + 8 * SAMPLE_RATE)
                rows, _, elapsed = transcribe(samples[left:right], False)
                adjusted = []
                for row in rows:
                    if row['start_seconds'] < row['end_seconds'] and row['text'].strip():
                        adjusted.append(dict(start_seconds=row['start_seconds'] + left / SAMPLE_RATE,
                                             end_seconds=row['end_seconds'] + left / SAMPLE_RATE,
                                             text=row['text'].strip()))
                chunks.append(dict(start_seconds=left / SAMPLE_RATE,
                                   end_seconds=right / SAMPLE_RATE,
                                   overlap_with_previous_seconds=1 if left else 0,
                                   inference_seconds=elapsed, segments=adjusted,
                                   raw_text=''.join(row['text'] for row in adjusted).strip()))
                chunk_time += elapsed
                if right == len(samples):
                    break
                left = right - SAMPLE_RATE

            record = dict(case)
            record['streaming'] = dict(step_seconds=STEP_SECONDS,
                                       max_buffer_seconds=MAX_SECONDS,
                                       audio_buffer_policy='Retain all audio from time0; reject longer clips.',
                                       transcript=final_text if aligned else None,
                                       transcript_alignment_status='aligned' if aligned else 'unresolved_committed_prefix_conflict',
                                       cer=character_error(case['japanese_reference'], final_text) if aligned else None,
                                       stable_text=token_text(la.committed[:len(la.committed)-sum(u.get('unstable_tail_word_count', 0) for u in final_units)]).strip() if final_units else token_text(la.committed).strip(),
                                       final_tail_units=final_units,
                                       unresolved_final_hypothesis=(None if aligned or not updates else
                                                                    dict(text=updates[-1]['raw_hypothesis_text'],
                                                                         words=updates[-1]['raw_hypothesis_words'])),
                                       updates=updates,
                                       translation_units=la.units,
                                       boundary_conflicts=la.boundary_conflicts,
                                       total_inference_seconds=update_total,
                                       word_timestamped_update_count=len(updates),
                                       peak_owned_process_gpu_memory_mib=max(memory_observations) if memory_observations else None)
            record['overlap_baseline'] = dict(chunk_seconds=8, overlap_seconds=1,
                                              flags=FLAGS, word_timestamps=False,
                                              chunks=chunks, inference_seconds=chunk_time,
                                              cer=None,
                                              cer_note='No aggregate CER: the raw8s chunks overlap by1s and were not aligned/reconciled.')
            report['cases'].append(record)
            report['completed_cases'] = len(report['cases'])
            write(report)
            print(json.dumps(dict(completed_cases=report['completed_cases'], development_id=case['id'])), flush=True)

        valid_cer = [case['streaming']['cer'] for case in report['cases'] if case['streaming']['cer']]
        errors = sum(item['errors'] for item in valid_cer)
        chars = sum(item['reference_characters'] for item in valid_cer)
        report['aggregate'] = dict(streaming_diagnostic_cer=dict(errors=errors,
                                                                  reference_characters=chars,
                                                                  cer=errors / chars if chars else None,
                                                                  meaning_accuracy=False),
                                   streaming_inference_seconds=sum(c['streaming']['total_inference_seconds'] for c in report['cases']),
                                   baseline_chunk_inference_seconds=sum(c['overlap_baseline']['inference_seconds'] for c in report['cases']),
                                   unresolved_alignment_cases=sum(c['streaming']['transcript_alignment_status'] != 'aligned' for c in report['cases']))
        report['gpu_peak_note'] = 'Per-update nvidia-smi process memory samples; null if the driver did not report this worker.'
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if model is not None:
            model.model.unload_model()
            report['owned_model_unloaded'] = True
        model = None
        gc.collect()
        write(report)


def main():
    self_test()
    if '--self-test' in sys.argv:
        print('LocalAgreement-2, repeated Japanese phrase, revisions, boundaries, conflicts and EOF checks passed.')
        return
    if '--worker' in sys.argv:
        worker()
        return
    from asr_cuda_pilot import gpu_snapshot
    if OUTPUT.exists():
        raise SystemExit('Preserve existing evaluation; move its artifact explicitly before a new run.')
    manifest = json.loads(MANIFEST.read_text())
    selected = development_cases(manifest)
    for case in selected:
        path = (ROOT / case['audio_path']).resolve()
        if not path.is_relative_to(ROOT / 'test-results/accuracy-audio') or digest(path) != case['audio_sha256']:
            raise SystemExit(f"Frozen development audio failed path/hash preflight for {case['id']}.")
        if case['duration_seconds'] > MAX_SECONDS:
            raise SystemExit(f"{case['id']} exceeds the explicit30second experiment limit.")
    before = gpu_snapshot()
    report = dict(schema_version=1, started_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='starting', partition='development',
                  runner_sha256=digest(Path(__file__).resolve()),
                  corpus={key: value for key, value in manifest.items() if key != 'cases'} |
                         dict(manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=digest(MANIFEST)),
                  development_ids=[c['id'] for c in selected], completed_cases=0, cases=[],
                  model=dict(name='faster-whisper medium', revision=REVISION, device='cuda',
                             device_index=0, compute_type='int8_float16', cpu_threads=4,
                             num_workers=1, local_files_only=True),
                  flags=dict(FLAGS), streaming=dict(algorithm='LocalAgreement-2', step_seconds=STEP_SECONDS,
                             max_buffer_seconds=MAX_SECONDS, word_timestamps=True,
                             exact_japanese_text=True,
                             timestamp_jitter_seconds=TIMESTAMP_JITTER_SECONDS,
                             agreement='Longest common concatenated-text prefix at complete timestamp-token boundaries across consecutive distinct 2s audio horizons.',
                             unit_rule='Accumulate stable text; emit at stable Japanese sentence terminal punctuation only (ASR segment boundaries are recorded but do not trigger unit emission); flush residual once at EOF as unconfirmed.'),
                  baseline=dict(chunk_seconds=8, overlap_seconds=1, flags=dict(FLAGS), word_timestamps=False,
                                boundaries='0-8,7-15,14-22,...; final chunk clipped to recording end.',
                                cer=None, cer_note='No aggregate CER: overlapping raw chunks are not aligned/reconciled.'),
                  metric=dict(name='Whole-clip normalized Japanese CER',
                              normalization='Unicode NFKC, then remove Unicode punctuation and whitespace',
                              meaning_accuracy=False),
                  versions={key: importlib.metadata.version(key) for key in ('faster-whisper', 'ctranslate2', 'numpy')},
                  gpu_before=before,
                  limitations=['Only20 frozen development recordings; no held-out audio or outputs were opened.',
                                'All current clips are <=29.52s; this feasibility run retains audio from time0 and explicitly rejects clips over30s.',
                                'LocalAgreement adds word_timestamps=True to the production ASR flags; baseline retains production flags.',
                                'Repeated full-buffer decoding cost is measured but not a real-time latency guarantee.',
                                'CER is a transcript diagnostic only; it does not score meaning.',
                                'Translation units wait for stable sentence punctuation or EOF; this measures ASR emissions, not translation quality.'])
    child = None
    try:
        if not any(row['index'] == 0 and row['free_mib'] >= 4096 for row in before['gpus']):
            report['status'] = 'skipped_insufficient_gpu_memory'
            return
        env = os.environ | dict(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        report['worker_environment'] = {key: env.get(key, '') for key in ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'LD_LIBRARY_PATH')}
        write(report)
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                 cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = child.communicate(timeout=1200)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            report['status'] = 'timed_out'
        report = json.loads(OUTPUT.read_text())
        report.update(worker_pid=child.pid, worker_exit_code=child.returncode,
                      worker_stdout_tail=stdout[-4000:], stderr=stderr[-4000:])
        if child.returncode != 0:
            report['status'] = 'failed'
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        report['gpu_after'] = gpu_snapshot()
        report['owned_gpu_process_released'] = child is None or all(row['pid'] != child.pid for row in report['gpu_after']['processes'])
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(report)
        print(json.dumps(dict(status=report['status'], completed_cases=report['completed_cases'],
                              report=str(OUTPUT.relative_to(ROOT)),
                              owned_gpu_process_released=report['owned_gpu_process_released'])), flush=True)


if __name__ == '__main__':
    main()
