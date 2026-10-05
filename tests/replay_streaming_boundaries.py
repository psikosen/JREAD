"""Replay saved Whisper hypotheses under three causal publication policies."""
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluate_speech import character_error  # saved references are used only for final CER

ASR_PATH = ROOT / 'test-results/accuracy-streaming-speech-development.json'
ASR_SCRIPT = ROOT / 'tests/evaluate_streaming_speech.py'
MANIFEST_PATH = ROOT / 'tests/fixtures/accuracy-corpus.json'
OUTPUT_PATH = ROOT / 'test-results/accuracy-streaming-boundary-replay.json'
ASR_SHA256 = '31bbf2675fb2d207a81d51f7ddf67e40d92df62f6c09686fd1abbf8341e8ca18'
ASR_SCRIPT_SHA256 = 'c2c341b632ec55eb580610e673b036757dd629ead9836d3d09007cf901edfa3f'
TIMESTAMP_JITTER_SECONDS = 0.75
MAX_PENDING_SECONDS = 8.0
SENTENCE_END = set('。！？!?')
CLOSERS = set('」』】）》”’"\'）)]}')

POLICIES = (
    ('sentence_terminal_eof', 'Stable sentence terminal plus one EOF flush.'),
    ('current_segment_or_sentence_eof', 'Current-horizon stable ASR segment boundary or sentence terminal plus one EOF flush.'),
    ('segment_sentence_max8_eof', 'Policy (b) plus a forced stable word-boundary fragment when pending source span reaches8s.'),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_of(tokens):
    return ''.join(token['text'] for token in tokens)


def char_spans(tokens, length):
    spans = []
    for token in tokens:
        spans.extend([(float(token['start_seconds']), float(token['end_seconds']))] * len(token['text']))
        if len(spans) >= length:
            return spans[:length]
    return spans


def timed_prefix_compatible(previous, current, length):
    left, right = char_spans(previous, length), char_spans(current, length)
    if len(left) != length or len(right) != length:
        return False
    return all(max(a0 - b1, b0 - a1, 0.0) <= TIMESTAMP_JITTER_SECONDS
               for (a0, a1), (b0, b1) in zip(left, right))


def common_complete_prefix(previous, current):
    left, right = text_of(previous), text_of(current)
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    left_bounds, right_bounds = {0}, {0}
    for tokens, bounds in ((previous, left_bounds), (current, right_bounds)):
        position = 0
        for token in tokens:
            position += len(token['text'])
            bounds.add(position)
    eligible = sorted((b for b in left_bounds & right_bounds if b <= common), reverse=True)
    return next((b for b in eligible if timed_prefix_compatible(previous, current, b)), 0)


def slice_tokens(tokens, start_char, end_char):
    """Slice by exact Japanese characters while retaining original word clocks."""
    result, offset = [], 0
    for token in tokens:
        next_offset = offset + len(token['text'])
        left, right = max(start_char, offset), min(end_char, next_offset)
        if left < right:
            row = dict(token)
            if left != offset or right != next_offset:
                row['text'] = token['text'][left-offset:right-offset]
                row['text_sliced_at_publication_boundary'] = True
            result.append(row)
        offset = next_offset
        if offset >= end_char:
            break
    return result


def strip_prefix(tokens, prefix):
    full = text_of(tokens)
    if not full.startswith(prefix):
        return None
    return slice_tokens(tokens, len(prefix), len(full))


def sentence_boundaries(text):
    result = set()
    for index, char in enumerate(text):
        if char not in SENTENCE_END:
            continue
        end = index + 1
        while end < len(text) and text[end] in CLOSERS:
            end += 1
        result.add(end)
    return result


def segment_boundaries(update):
    boundaries, position = set(), 0
    for segment in update.get('raw_segments', []):
        words = segment.get('words') or []
        if not words:
            continue
        position += sum(len(word['text']) for word in words)
        boundaries.add(position)
    return boundaries


def unit_from_slice(tokens, horizon, update_index, kind, forced=False):
    if not tokens:
        return None
    text = text_of(tokens)
    starts = [float(t['start_seconds']) for t in tokens]
    ends = [float(t['end_seconds']) for t in tokens]
    return dict(text=text, word_timestamps=[dict(t) for t in tokens],
                source_start_seconds=min(starts), source_end_seconds=max(ends),
                emitted_at_audio_seconds=float(horizon), emitted_update_index=update_index,
                boundary_kind=kind, stability='la2_stable_forced_fragment' if forced else 'la2_stable',
                forced_fragment=forced)


class Replay:
    """One frozen policy; only emitted translation source is immutable."""
    def __init__(self, policy):
        self.policy = policy
        self.published_tokens = []
        self.previous_hypothesis = None
        self.previous_anchored = False
        self.units = []
        self.events = []
        self.conflicts = []
        self.finished = False
        self.latest_hypothesis = []
        self.latest_horizon = 0.0

    @property
    def published_text(self):
        return text_of(self.published_tokens)

    def _publish(self, current, boundary, horizon, update_index, kind, forced=False):
        start = len(self.published_text)
        chunk = slice_tokens(current, start, boundary)
        unit = unit_from_slice(chunk, horizon, update_index, kind, forced)
        if unit is not None and unit['text']:
            self.units.append(unit)
            self.published_tokens.extend(chunk)
        return unit

    def _natural_boundaries(self, update, current_text, stable_end):
        published_len = len(self.published_text)
        sentence = sentence_boundaries(current_text)
        segment = segment_boundaries(update)
        if self.policy == 'sentence_terminal_eof':
            allowed = sentence
        else:
            allowed = sentence | segment
        return sorted(b for b in allowed if published_len < b <= stable_end)

    def _force_boundaries(self, current, stable_end):
        if self.policy != 'segment_sentence_max8_eof':
            return []
        published_len = len(self.published_text)
        stable_suffix = slice_tokens(current, published_len, stable_end)
        if not stable_suffix:
            return []
        pending_start = min(float(t['start_seconds']) for t in stable_suffix)
        token_ends, chars = [], published_len
        for token in stable_suffix:
            chars += len(token['text'])
            token_ends.append((chars, float(token['end_seconds'])))
        selected = []
        while token_ends:
            if max(end - pending_start for _, end in token_ends) < MAX_PENDING_SECONDS:
                break
            at_or_before = [(pos, end) for pos, end in token_ends
                            if end - pending_start <= MAX_PENDING_SECONDS]
            if at_or_before:
                boundary, _ = at_or_before[-1]
            else:
                boundary = token_ends[0][0]
            if boundary <= published_len:
                break
            selected.append(boundary)
            published_len = boundary
            remaining = slice_tokens(current, published_len, stable_end)
            if not remaining:
                break
            pending_start = min(float(t['start_seconds']) for t in remaining)
            token_ends = []
            chars = published_len
            for token in remaining:
                chars += len(token['text'])
                token_ends.append((chars, float(token['end_seconds'])))
        return selected

    def feed(self, update, update_index):
        if self.finished:
            raise RuntimeError('Cannot feed after EOF.')
        horizon = float(update['audio_end_seconds'])
        current = [dict(t) for t in update['raw_hypothesis_words']]
        current_text = text_of(current)
        base_published_text = self.published_text
        conflict = None
        if not current_text.startswith(base_published_text):
            conflict = dict(update_index=update_index, audio_end_seconds=horizon,
                            published_text=base_published_text,
                            hypothesis_prefix=current_text[:len(base_published_text)],
                            hypothesis_text=current_text,
                            reason='current hypothesis rewrote already-published source text')
        elif base_published_text and not timed_prefix_compatible(self.published_tokens, current, len(base_published_text)):
            conflict = dict(update_index=update_index, audio_end_seconds=horizon,
                            published_text=base_published_text,
                            hypothesis_prefix=current_text[:len(base_published_text)],
                            hypothesis_text=current_text,
                            reason='published source text matches but its word-time anchors do not overlap')

        emitted = []
        stable_count = 0
        stable_end = len(base_published_text)
        current_sentence_boundaries = sentence_boundaries(current_text)
        current_segment_boundaries = segment_boundaries(update)
        current_suffix = strip_prefix(current, base_published_text) if conflict is None else None
        previous_suffix = None
        if conflict is None and self.previous_hypothesis is not None and self.previous_anchored:
            previous_suffix = strip_prefix(self.previous_hypothesis, base_published_text)
        if conflict is None and current_suffix is not None and previous_suffix is not None:
            stable_count = common_complete_prefix(previous_suffix, current_suffix)
            stable_end = len(base_published_text) + stable_count
            boundaries = self._natural_boundaries(update, current_text, stable_end)
            if self.policy == 'segment_sentence_max8_eof':
                for boundary in boundaries:
                    for forced_boundary in self._force_boundaries(current, boundary):
                        if forced_boundary < boundary:
                            unit = self._publish(current, forced_boundary, horizon,
                                                 update_index, 'forced_max_pending_span', forced=True)
                            if unit:
                                emitted.append(unit)
                    kind = 'stable_sentence_terminal' if boundary in sentence_boundaries(current_text) else 'current_asr_segment_boundary'
                    unit = self._publish(current, boundary, horizon, update_index, kind)
                    if unit:
                        emitted.append(unit)
                for boundary in self._force_boundaries(current, stable_end):
                    unit = self._publish(current, boundary, horizon, update_index,
                                         'forced_max_pending_span', forced=True)
                    if unit:
                        emitted.append(unit)
            else:
                for boundary in boundaries:
                    kind = 'stable_sentence_terminal' if boundary in sentence_boundaries(current_text) else 'current_asr_segment_boundary'
                    unit = self._publish(current, boundary, horizon, update_index, kind)
                    if unit:
                        emitted.append(unit)

        if conflict is not None:
            self.conflicts.append(conflict)
        self.previous_hypothesis = current
        # A conflicting observation cannot seed agreement after the published
        # anchor changes; one later compatible horizon is required to re-seed.
        self.previous_anchored = conflict is None
        self.latest_hypothesis = current
        self.latest_horizon = horizon
        self.events.append(dict(update_index=update_index, audio_end_seconds=horizon,
                                inference_seconds=float(update['inference_seconds']),
                                hypothesis_text=current_text,
                                previous_hypothesis_text=(text_of(previous_suffix) if previous_suffix is not None else None),
                                stable_prefix_characters=stable_count,
                                stable_prefix_text=(text_of(current_suffix)[:stable_count] if current_suffix is not None else ''),
                                stable_prefix_end_char_offset=stable_end,
                                current_sentence_boundary_char_offsets=sorted(current_sentence_boundaries),
                                current_segment_boundary_char_offsets=sorted(current_segment_boundaries),
                                published_text_after=self.published_text,
                                boundary_conflict=conflict, emitted_units=emitted))

    def finish(self):
        if self.finished:
            return []
        self.finished = True
        if not self.latest_hypothesis:
            return []
        full_text = text_of(self.latest_hypothesis)
        if not full_text.startswith(self.published_text) or not self.previous_anchored:
            return []
        tail_tokens = strip_prefix(self.latest_hypothesis, self.published_text)
        if not tail_tokens:
            return []
        tail = unit_from_slice(tail_tokens, self.latest_horizon,
                               max(0, len(self.events)-1), 'end_of_recording_unconfirmed')
        if tail is None:
            return []
        tail['stability'] = 'end_of_recording_unconfirmed'
        tail['contains_unstable_tail'] = True
        self.units.append(tail)
        self.published_tokens.extend(tail_tokens)
        self.events[-1]['eof_flush_unit_count'] = 1
        self.events[-1]['emitted_units'].append(tail)
        return [tail]

    def result(self, reference):
        full_final = text_of(self.published_tokens)
        unresolved = bool(self.conflicts and not self.previous_anchored)
        strict_pre_eof = [u for u in self.units if u['emitted_at_audio_seconds'] < self.latest_horizon - 1e-4]
        return dict(transcript=None if unresolved else full_final,
                    transcript_alignment_status='unresolved_published_prefix_conflict' if unresolved else 'aligned',
                    protocol_status='fail_published_prefix_rewrite' if self.conflicts else 'pass',
                    cer=character_error(reference, full_final) if not unresolved else None,
                    boundary_conflicts=self.conflicts,
                    events=self.events, units=self.units,
                    non_eof_flush_unit_count=sum(u['boundary_kind'] != 'end_of_recording_unconfirmed' for u in self.units),
                    strict_before_eof_emission_unit_count=len(strict_pre_eof),
                    strict_before_eof_emission_case=bool(strict_pre_eof),
                    eof_flush_count=sum(u['boundary_kind'] == 'end_of_recording_unconfirmed' for u in self.units),
                    final_raw_hypothesis_text=text_of(self.latest_hypothesis),
                    final_raw_hypothesis_words=[dict(t) for t in self.latest_hypothesis],
                    final_raw_hypothesis_cer=character_error(reference, text_of(self.latest_hypothesis)),
                    final_raw_hypothesis_retained=True,
                    final_raw_text_equals_fused_text=(not unresolved and full_final == text_of(self.latest_hypothesis)))


def replay_policy(policy, cases):
    output_cases = []
    for case in cases:
        simulation = Replay(policy)
        updates = case['streaming']['updates']
        for index, update in enumerate(updates):
            simulation.feed(update, index)
        simulation.finish()
        row = simulation.result(case['japanese_reference'])
        row['id'] = case['id']
        row['duration_seconds'] = case['duration_seconds']
        row['update_count'] = len(updates)
        row['source_hypothesis_sha256s'] = [hashlib.sha256(u['raw_hypothesis_text'].encode()).hexdigest() for u in updates]
        output_cases.append(row)
    return output_cases


def self_test():
    def tok(text, start, end):
        return dict(text=text, start_seconds=start, end_seconds=end)

    def update(horizon, tokens, segments=None):
        segments = segments if segments is not None else [dict(words=tokens)]
        return dict(audio_end_seconds=horizon, inference_seconds=.1,
                    raw_hypothesis_words=tokens, raw_segments=segments)

    # A partial negation changes before publication; only the latest agreement is flushed.
    before = Replay('sentence_terminal_eof')
    before.feed(update(2, [tok('彼は', 0, .3), tok('行かない', .3, .8)]), 0)
    before.feed(update(4, [tok('彼は', 0, .3), tok('行かない', .3, .8)]), 1)
    before.feed(update(6, [tok('彼は', 0, .3), tok('行く。', .3, .8)]), 2)
    before.feed(update(8, [tok('彼は', 0, .3), tok('行く。', .3, .8)]), 3)
    before.finish()
    assert before.result('彼は行く。')['transcript'] == '彼は行く。'
    assert '行かない' not in before.published_text

    # Once an English-facing unit was published, a polarity rewrite is explicit.
    after = Replay('sentence_terminal_eof')
    no = [tok('彼は来ない。', 0, .8)]
    after.feed(update(2, no), 0)
    after.feed(update(4, no), 1)
    yes = [tok('彼は来る。', 0, .8)]
    conflict = after.feed(update(6, yes), 2)
    assert after.conflicts and after.conflicts[-1]['reason'].startswith('current hypothesis rewrote')
    assert after.published_text == '彼は来ない。'
    assert after.finish() == []
    assert after.result('彼は来ない。')['cer'] is None

    # Distinct repeated occurrences do not agree across disjoint word times.
    repeated = Replay('sentence_terminal_eof')
    repeated.feed(update(2, [tok('はい。', 0, .2)]), 0)
    repeated.feed(update(4, [tok('はい。', 0, .2), tok('はい。', 2.1, 2.3)]), 1)
    repeated.feed(update(6, [tok('はい。', 0, .2), tok('はい。', 2.1, 2.3)]), 2)
    repeated.finish()
    assert repeated.published_text == 'はい。はい。'

    # Current-horizon segment boundary is used; the next segment is not peeked.
    segment = Replay('current_segment_or_sentence_eof')
    one = [tok('先行', 0, .5), tok('継続中', .5, 1.2)]
    first = dict(words=one)
    second = [tok('先行', 0, .5), tok('継続中', .5, 1.2)]
    segment.feed(update(2, one, [first]), 0)
    segment.feed(update(4, second, [dict(words=[second[0]]), dict(words=[second[1]])]), 1)
    assert segment.events[-1]['emitted_units'][0]['text'] == '先行'
    assert segment.events[-1]['emitted_units'][0]['emitted_update_index'] == 1

    # Forced splitting occurs at word boundaries after8s, with zero-time punctuation retained.
    forced = Replay('segment_sentence_max8_eof')
    long = [tok('語', 0, 4.0), tok('、', 4.0, 4.0), tok('長い', 4.0, 8.2), tok('末尾', 8.2, 9.0)]
    forced.feed(update(10, long, []), 0)
    forced.feed(update(12, long, []), 1)
    assert forced.events[-1]['emitted_units'][0]['forced_fragment']
    assert '、' in forced.events[-1]['emitted_units'][0]['text']

    # EOF preserves every residual token and is idempotent.
    tail = Replay('sentence_terminal_eof')
    final_words = [tok('最後', 0, .4), tok('、', .4, .4)]
    tail.feed(update(2, final_words), 0)
    tail.feed(update(4, final_words), 1)
    emitted = tail.finish()
    assert len(emitted) == 1 and emitted[0]['text'] == '最後、'
    assert emitted[0]['stability'] == 'end_of_recording_unconfirmed'
    assert tail.finish() == []
    assert tail.result('最後、')['final_raw_text_equals_fused_text']


def summary(policy, cases, update_count, stream_seconds, baseline_seconds):
    covered = [case['cer'] for case in cases if case['cer'] is not None]
    errors = sum(item['errors'] for item in covered)
    chars = sum(item['reference_characters'] for item in covered)
    raw = [case['final_raw_hypothesis_cer'] for case in cases]
    raw_errors = sum(item['errors'] for item in raw)
    raw_chars = sum(item['reference_characters'] for item in raw)
    spans = [u['source_end_seconds'] - u['source_start_seconds'] for c in cases for u in c['units']]
    lags = [u['emitted_at_audio_seconds'] - u['source_end_seconds'] for c in cases for u in c['units']]
    stable_spans = [u['source_end_seconds'] - u['source_start_seconds']
                    for c in cases for u in c['units']
                    if u['boundary_kind'] != 'end_of_recording_unconfirmed']
    return dict(policy=policy, case_count=len(cases), update_count=update_count,
                cer_coverage_count=len(covered), cer_uncovered_count=len(cases)-len(covered),
                diagnostic_cer=dict(errors=errors, reference_characters=chars,
                                    cer=errors/chars if chars else None, meaning_accuracy=False),
                final_raw_hypothesis_cer=dict(coverage_count=len(raw), errors=raw_errors,
                                              reference_characters=raw_chars,
                                              cer=raw_errors/raw_chars if raw_chars else None,
                                              meaning_accuracy=False),
                failed_published_prefix_cases=[c['id'] for c in cases if c['transcript_alignment_status'] != 'aligned'],
                published_prefix_conflict_failures=[c['id'] for c in cases if c['boundary_conflicts']],
                non_eof_flush_unit_count=sum(c['non_eof_flush_unit_count'] for c in cases),
                strict_before_eof_emission_unit_count=sum(c['strict_before_eof_emission_unit_count'] for c in cases),
                strict_before_eof_emission_case_count=sum(c['strict_before_eof_emission_case'] for c in cases),
                eof_flush_unit_count=sum(c['eof_flush_count'] for c in cases),
                forced_fragment_count=sum(u['forced_fragment'] for c in cases for u in c['units']),
                emitted_unit_count=len(spans),
                mean_units_per_case=len(spans)/len(cases) if cases else 0,
                unit_source_span_seconds=dict(min=min(spans) if spans else None,
                                              median=statistics.median(spans) if spans else None,
                                              max=max(spans) if spans else None),
                non_eof_unit_source_span_max_seconds=max(stable_spans) if stable_spans else None,
                unit_emit_lag_seconds=dict(median=statistics.median(lags) if lags else None,
                                           max=max(lags) if lags else None),
                same_saved_asr_decode_count=update_count,
                same_saved_asr_inference_seconds=round(stream_seconds, 4),
                unchanged_8s_overlap_baseline_inference_seconds=round(baseline_seconds, 4))


def main():
    self_test()
    if '--self-test' in sys.argv:
        print('Streaming-boundary replay self-tests passed.')
        return
    if OUTPUT_PATH.exists():
        raise SystemExit(f'Preserve existing replay report; move explicitly first: {OUTPUT_PATH.relative_to(ROOT)}')
    asr_sha, script_sha = sha256(ASR_PATH), sha256(ASR_SCRIPT)
    if asr_sha != ASR_SHA256 or script_sha != ASR_SCRIPT_SHA256:
        raise SystemExit('Negative ASR report or runner bytes changed; refusing replay.')
    report = json.loads(ASR_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    cases = [row for row in report['cases'] if row.get('partition') == 'development']
    refs = [row for row in manifest['cases'] if row.get('partition') == 'development']
    if report.get('status') != 'complete' or report.get('owned_model_unloaded') is not True or report.get('owned_gpu_process_released') is not True:
        raise SystemExit('Frozen ASR source report is incomplete or lacks cleanup verification.')
    if len(cases) != 20 or len(refs) != 20 or {c['id'] for c in cases} != {c['id'] for c in refs}:
        raise SystemExit('Replay requires exactly the frozen20 development IDs.')
    if sha256(MANIFEST_PATH) != report.get('corpus', {}).get('manifest_sha256'):
        raise SystemExit('Frozen corpus manifest hash differs from the original ASR report.')
    if report.get('runner_sha256') != script_sha:
        raise SystemExit('Original ASR runner hash is not recorded in the negative ASR report.')
    ref_by_id = {c['id']: c for c in refs}
    for case in cases:
        frozen = ref_by_id[case['id']]
        if case['audio_sha256'] != frozen['audio_sha256'] or case['japanese_reference'] != frozen['japanese_reference']:
            raise SystemExit(f"Frozen case/hash/reference mismatch for {case['id']}.")
    update_count = sum(len(case['streaming']['updates']) for case in cases)
    if update_count != 141:
        raise SystemExit(f'Expected141 saved hypotheses, found{update_count}.')
    stream_seconds = sum(float(update['inference_seconds']) for case in cases for update in case['streaming']['updates'])
    baseline_seconds = sum(float(case['overlap_baseline']['inference_seconds']) for case in cases)
    policy_rows = {}
    for policy, _description in POLICIES:
        policy_cases = replay_policy(policy, cases)
        for case in policy_cases:
            if case['transcript_alignment_status'] == 'aligned' and not case['final_raw_text_equals_fused_text']:
                raise SystemExit(f"Replay lost/reordered final source text for {policy} case {case['id']}.")
            if len(case['events']) != case['update_count']:
                raise SystemExit(f"Replay changed observation count for {policy} case {case['id']}.")
            for index, event in enumerate(case['events']):
                for unit in event['emitted_units']:
                    if unit['emitted_update_index'] != index or unit['emitted_at_audio_seconds'] > event['audio_end_seconds'] + 1e-4:
                        raise SystemExit(f"Noncausal replay emission for {policy} case {case['id']} update {index}.")
                    if unit['source_end_seconds'] > unit['emitted_at_audio_seconds'] + 1e-4:
                        raise SystemExit(f"Future source timestamp in {policy} case {case['id']} unit.")
        policy_rows[policy] = dict(cases=policy_cases,
                                   summary=summary(policy, policy_cases, update_count,
                                                   stream_seconds, baseline_seconds))
    output = dict(schema_version=1, status='complete', created_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Pure replay of saved development-only Whisper hypotheses; no model inference or held-out data.',
                  provenance=dict(asr_report=str(ASR_PATH.relative_to(ROOT)), asr_report_sha256=asr_sha,
                                  asr_runner=str(ASR_SCRIPT.relative_to(ROOT)), asr_runner_sha256=script_sha,
                                  replay_runner=str(Path(__file__).resolve().relative_to(ROOT)),
                                  replay_runner_sha256=sha256(Path(__file__).resolve()),
                                  manifest=str(MANIFEST_PATH.relative_to(ROOT)),
                                  manifest_sha256=sha256(MANIFEST_PATH),
                                  partition='development', case_ids=[c['id'] for c in cases],
                                  saved_hypothesis_count=update_count,
                                  original_asr_decode_seconds=round(stream_seconds, 4),
                                  original_8s_overlap_decode_seconds=round(baseline_seconds, 4),
                                  original_artifacts_preserved=True),
                  replay_controls=dict(timestamp_jitter_seconds=TIMESTAMP_JITTER_SECONDS,
                                       agreement='Exact concatenated Japanese character prefix at shared complete timestamp-token boundaries with overlapping character times, across only adjacent saved audio horizons.',
                                       immutable_text='Only source text already emitted as a translation unit is immutable. LA-stable but unpublished text may revise against the next saved hypothesis.',
                                       published_prefix='A mismatch against emitted source text or its timestamp anchors is recorded as explicit conflict; no stale character offset is spliced.',
                                       eof='Use the last saved hypothesis once; no duplicate decode or extra agreement confirmation. EOF flush is idempotent and marked unconfirmed.',
                                       future_horizons='Policies inspect only the current saved update and its immediately previous observation.',
                                       policy_descriptions=dict(POLICIES),
                                       max_pending_source_span_seconds=MAX_PENDING_SECONDS),
                  metrics={key: row['summary'] for key, row in policy_rows.items()},
                  policies=policy_rows,
                  interpretation_limit='CER is a transcript diagnostic only. This replay makes no semantic or translation-accuracy claim.')
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    temp = OUTPUT_PATH.with_suffix('.tmp')
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    temp.replace(OUTPUT_PATH)
    print(json.dumps(dict(status='complete', report=str(OUTPUT_PATH.relative_to(ROOT)),
                          saved_hypotheses=update_count,
                          policies={key: dict(cer_coverage=val['summary']['cer_coverage_count'],
                                              conflict_failures=len(val['summary']['failed_published_prefix_cases']),
                                              units=val['summary']['emitted_unit_count'])
                                    for key, val in policy_rows.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
