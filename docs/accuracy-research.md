# Research directions for caption accuracy

Research checked October 4, 2026. This is an engineering record, not a claim that
90% accuracy is met. The best reviewed repaired-pipeline result remains 13/20.
Using the same frozen Turbo Japanese sources, the best single translation candidate
(E4B, original prompt, sampled seed 17) passed 14/20; the corrected ID-0 E4B
selector passed 13/20, with 14/20 oracle coverage. None reaches the 18/20 target.
ASR reconciliation and Qwen0.6B/1.7B trials did not improve the repaired pipeline.
The small-model candidate-pool comparison is complete; further selection is only
useful if new generation recovers cases beyond the current 14/20 coverage ceiling.
The separate local E4B QLoRA pilot is also complete: it did not improve its new
validation set and did not qualify for an original-development follow-up.

## Small bilingual checkers: Laya multilingual and XCOMET-lite

### Bidirectional source-support diagnostic

The [Horizon hallucination-guard-base model](https://huggingface.co/Horizon-Labs/hallucination-guard-base)
was tested as a different way to check source support. Its model card reports
Japanese hallucination tests, which do not establish Japanese-to-English
translation performance. The local experiment uses the pinned native 308M
checkpoint, CPU FP32, four threads, and no remote model code.

This diagnostic reuses the exact 32-group quantity fixture and its already
accepted labels. It is development data, not fresh validation. Every English
candidate is scored independently against the Japanese source in both directions.
The frozen primary score is the smaller of the two support probabilities, intended
to penalize additions and omissions. Forward-only support was predefined as a
secondary diagnostic and cannot replace the primary after seeing results.

The primary passed **93/128 (72.7%)**, versus the saved XCOMET baseline's 97/128
and quantity combination's 98/128. Against the latter it fixed 13 preferences and
regressed 18. Forward-only passed **99/128 (77.3%)**, fixing 14 and regressing 13
against the quantity combination. Both new conditions fully passed 12/32 groups,
versus 14/32 for each saved baseline. These are correlated preference comparisons,
not independent caption accuracy cases.

Primary stratum scores were dates/times 17/32, money/measurement 18/32,
equivalent/unsupported expressions 30/32, and other meaning contrasts 28/32.
Forward-only scored 20/32, 20/32, 30/32, and 29/32 respectively. Primary faithful
variant scores were 56/64 for the first wording and 37/64 for paraphrases;
forward-only scored 60/64 and 39/64. The reverse direction hurt overall ordering,
and faithful reformulations remain a weakness. The frozen gate required primary
90% overall, 75% per stratum, and 90% on paraphrase comparisons; it failed.

All 304 requests passed model-free token validation. The separate six-group smoke
completed all 48 directional requests and scored 23/24 primary preferences and
21/24 forward-only preferences. Its gate was technical, not perfect semantic
performance. The main worker completed 256 measured requests plus one excluded
control warmup, with finite untruncated outputs, a complete native checkpoint
load, verified CPU FP32 parameters/logits, and no CUDA initialization. Scoring
took 10.712 seconds across both directions; model loading took 0.139 seconds.
These timings exclude process startup, imports and artifact hashing, and do not
measure live translation latency. Workers released their process groups and all
frozen model/runtime/data hashes still matched after execution.

Independent aggregation audit confirms the counts, coverage, and timing. No model was retrained, labels changed,
or successful generation repeated. No production use or expanded inference is
selected, and the original 50 held-out cases were not used. The original
translation development best remains 14/20.

Evidence: [frozen grounding protocol](../tests/fixtures/accuracy-grounding-verifier.json),
[runner](../tests/evaluate_grounding_verifier.py), and
[outcome and bound raw reports](../test-results/accuracy-grounding-verifier-outcome.json).

### Fresh quantity-guard follow-up

A separately authored 32-group fixture tested XCOMET alone against the same raw
scores combined with a bounded, source-aware quantity check. The parser and unit
tests were frozen before root opened the fresh cases; all 128 English candidates
were reviewed before scoring. Three Japanese subjects were made explicit and one
English case was made neutral about agency before the final data freeze. Numbers,
intended errors, case membership and comparison rules were unchanged.

| Stratum | XCOMET alone | XCOMET + quantity guard |
| --- | ---: | ---: |
| Dates and times | 24/32 | 25/32 |
| Money and measurements | 17/32 | 17/32 |
| Equivalent or unsupported expressions | 29/32 | 29/32 |
| Other meaning contrasts | 27/32 | 27/32 |
| Total | **97/128 (75.8%)** | **98/128 (76.6%)** |

The rule acts only when one candidate has comparable preserved quantities and
the other has a clear quantity mismatch. All other pairs retain XCOMET's strict
score ordering. It applied to three pairs and changed one wrong preference to a
correct preference: a 09:20 start over an incorrect 09:40 start. There were no
regressions. Both conditions fully passed 14/32 source groups, so the extra pair
win did not make another entire group pass.

No faithful candidate was marked quantity-mismatched (0/64), but 58/64 faithful
and 55/64 incorrect candidates made the guard abstain. Compound amounts, grouping,
uncertain values and unsupported grammar are deliberately excluded. This avoids
known false alarms such as treating “1 hour 30 minutes” versus “90 minutes” as
inconsistent, but yields low coverage. It does not prove zero false-positive risk.
XCOMET's first faithful variants passed 57/64 comparisons while the paraphrases
passed 40/64; after the guard these were 58/64 and 40/64. Valid reformulation
remains a substantial weakness of this quality scorer.

The new result fails the frozen gate: at least 90% overall, 75% per stratum,
positive gain, no paired regressions, and no faithful-candidate mismatch. No
natural-caption follow-up or production use is selected. These are new focused
synthetic cases; 97/128 cannot be directly compared with the previous 95/112,
and neither count changes the historical 14/20 translation result.

One CPU FP32 four-thread worker produced 128 fresh scores plus one excluded
warmup. The same raw scores serve both conditions. Scoring took 10.770 seconds,
the guard 0.054 seconds, and cold loading 2.383 seconds. This is not browser or
end-to-end translation latency. Coverage, token limits and worker cleanup passed;
independent aggregation audit confirms both counts. No parser was tuned or case regraded
after scoring. The original 50 held-out cases and production remain untouched.

Evidence: [frozen paired protocol](../tests/fixtures/accuracy-quantity-verifier-v2.json),
[guard implementation](../tests/quantity_guard.py),
[paired outcome and bound raw artifacts](../test-results/accuracy-quantity-verifier-v2-outcome.json).

### Initial checker comparison

A fresh fixture was authored and bilingually reviewed before scoring: 28 Japanese
source groups covering seven error families, plus six separate smoke groups.
Each source has two faithful English variants and two fluent incorrect variants.
Models receive only the source and candidate text; answer labels, propositions,
references and error-family labels remain outside inference. Four good/bad
comparisons per source yield 112 correlated comparisons, not 112 independent cases.
The original development/held-out text was not used to author or score this fixture.

| Checker | Correct comparisons | Groups passing all four | Measured scoring work |
| --- | ---: | ---: | --- |
| Laya multilingual | 40/112 (35.7%) | 2/28 | 14.38 s for 224 pair requests |
| XCOMET-lite, reference-free | 95/112 (84.8%) | 19/28 | 8.84 s for 112 candidate scores |

Laya must prefer the faithful translation in both A/B orders; rounded ties fail.
Its individual directional choices were 123/224 (54.9%), with consistent choices
in 69/112 reversed pairs. XCOMET scores each candidate independently. Both use
four CPU threads and FP32 compute. Cold loads were 2.19 s and 2.41 s respectively;
scoring totals exclude one warmup, startup and orchestration/file overhead. The
workloads differ, so these are not matched end-to-end caption or browser latencies.
Laya's stored checkpoint values include F16 tensors copied into FP32 parameters.

| Error family | Laya | XCOMET |
| --- | ---: | ---: |
| Negation/modality | 7/16 | 15/16 |
| Agent/recipient roles | 9/16 | 15/16 |
| Numbers/units/dates | 3/16 | 7/16 |
| Causality/temporal order | 4/16 | 14/16 |
| Omission | 1/16 | 16/16 |
| Named entities | 11/16 | 12/16 |
| Scope/exclusivity | 5/16 | 16/16 |

Neither passes the prospectively frozen requirement of at least 90% overall and
75% in every family. XCOMET's faithful first variant passes 51/56 comparisons,
while the faithful paraphrase passes 44/56. Nine of its 17 errors involve numbers
or dates, four involve names, and four involve other meaning contrasts. For example,
it can rank an incorrect October 21 date above the correct October 12 date. A high
quality score is not a calibrated probability of factual correctness.

The bounded follow-up hypothesis was XCOMET plus explicit number/date/unit
consistency checks, tested on new synthetic cases with legitimate equivalent
expressions. The follow-up above measured only one additional correct pair.
Simply rejecting unlike strings would damage valid paraphrases. No threshold
was fitted, and neither setup qualifies for the planned natural-caption follow-up.

Three setup failures are retained: the first XCOMET loader expected safetensors
instead of the published PyTorch archive; the next reached the weights but failed
on a DeBERTa return-type API mismatch; the initial Laya screen stopped before load
because upstream Laya rewrites tokenizer metadata after verification. Recovery
namespaces preserve all failed artifacts. XCOMET now safely loads verified weights
with strict state matching and normalizes only the layer's return interface.
Laya resumed with the exact original tokenizer bytes and reused its successful
smoke report without regenerating those cases. Actual forward execution is CPU
FP32 with CUDA uninitialized; all successful workers released. Independent result
audits confirm both counts. These synthetic scores do not improve the historical 14/20
translation result or establish anime accuracy; production remains unchanged.

Evidence: [frozen initial protocol](../tests/fixtures/accuracy-caption-verifiers.json),
[XCOMET recovery protocol](../tests/fixtures/accuracy-caption-verifiers-xcomet-api-v3.json),
[outcome and bound raw reports](../test-results/accuracy-caption-verifiers-outcome.json).

## ReazonSpeech K2 with the existing E4B translator

The fixed 20-case comparison completed with 13 meaning passes and seven failures,
equal to the deterministic Turbo → E4B baseline and below the sampled 14/20 best.
Root and independent blind reviews agree on all 20 case labels under the unchanged
rubric. Ten exact previous-output grades were retained and ten new outputs were
independently reviewed. These are model-assisted development judgments, not human
certification or evidence of anime performance.

Reazon corrected two baseline failures: being followed and grabbed by the arm
(1632), and travelers needing time to adjust (1592). Two earlier passes regressed:
the worthwhile stroll became a confection (1607), and the park's square-kilometre
area became square metres while its ecological zones became prefectures (1643).
Swapping recognizers therefore produced no net whole-case improvement.

The seven failures identify problems in both stages. Recognition introduced the
wrong village-stroll wording and park units. Translation still omitted the Iranian
Revolution's causal relation (1604), broadened the Hindustan Plain to the Indian
subcontinent (1641), and omitted the emperor's host role (1649), despite those
details surviving recognition. Case 1569 continues to fail the frozen, previously
documented ambiguous coordination check. Case 1539 corrupts “subculture” during
recognition and drops “slang” during translation. Better recognition alone does
not resolve these translation losses.

All audio frames were retained, with the upstream recipe's 0.9 seconds of padding
on each side. The pinned FP32 model ran with sherpa-onnx on four CPU threads;
Gemma retained its original prompt, eight CPU threads, temperature zero, and
thinking disabled. All 20 ASR and translation outputs completed without technical
errors. Fifteen translations were newly generated; five exact source/prompt
matches reused the frozen baseline. No reference text entered model requests.
Owned processes released and both private ports closed.

Recognition cost 3.035 seconds. Translation cost 80.973 seconds, including 18.044
seconds of historical cache costs, giving 84.008 seconds combined / 4.200 per case.
Startup, warmups, controls and process/file overhead are excluded. The old 81.945
seconds / 4.097 per case is translation-only. CPU whole-clip Reazon and historical
CUDA Whisper also differ in hardware and execution mode; this is not a matched
end-to-end or browser-latency comparison.

Normalized ASR character errors were 86/1,028, versus 24/1,028 for saved full-clip
Turbo. Kanji versus Arabic numeral spellings inflate this metric without always
changing meaning; no number normalizer was introduced after seeing results.

The original readiness gate remains failed: “course” became “concept/plan,” and
eight seconds of zero PCM produced “はい.” A separate prospectively frozen offline
diagnostic authorized the 20-case comparison with the failure explicitly retained.
No settings or cases were adjusted to turn readiness into a pass. This candidate
is not selected for expanded validation or production, and the original 50 held-out
recordings remain untouched. No successful generation was rerun.

Evidence: [outcome and case reconciliation](../test-results/accuracy-reazon-outcome.json),
[ASR audit](../test-results/accuracy-reazon-asr-audit.json),
[translation audit](../test-results/accuracy-reazon-translation-audit.json), and
[independent meaning review](../test-results/accuracy-reazon-independent-grades.json).

## What our experiments establish

The frozen development set contains 20 FLEURS recordings. Reviews are
model-assisted, not human-certified, and read speech does not establish anime
performance. The 50 held-out recordings remain unused. Counts use the
[whole-example meaning criteria](accuracy-target.md), including missing captions.

| Comparable input | Method | Passes |
| --- | --- | ---: |
| Correct Japanese reference text | Original E4B | 17/20 |
| Correct Japanese reference text | E4B, bounded reasoning | 17/20 |
| Correct Japanese reference text | General precision prompt | 16/20 |
| Correct Japanese reference text | Two-example prompt | 16/20 |
| Correct Japanese reference text | NLLB-600M | 11/20 |
| Recorded Japanese audio | Medium ASR → E4B | 12/20 |
| Recorded Japanese audio | Turbo ASR → E4B, segmented | 12/20 |
| Recorded Japanese audio | Turbo ASR → E4B, joined | 12/20 |
| Recorded Japanese audio | Medium direct translation | 7/20 |

Bounded reasoning retained exactly the same three failures while increasing mean
translation time from 4.049 to 13.335 seconds in separate CPU runs. Keep it off.
Evidence: [reasoning review](../test-results/accuracy-gemma-reasoning-review.json).
The precision review originally missed the Emperor's organizer role in case
1649. Its JSON now includes the correction, archived original, and audit trail.

A [reproducible oracle audit](../tests/audit_accuracy_oracle.py) measures whether
any existing candidate is correct for each case; it does not evaluate a selector.
The [report](../test-results/accuracy-oracle-ceilings.json) hashes every input review.

- On correct Japanese, the five-method pool contains a passing answer for 19/20.
  Few-shot fixes baseline case 1604; NLLB fixes 1641. No candidate passes 1569
  under the frozen strict reading, which has a Japanese coordination ambiguity.
- On actual audio, the four-method pool contains a passing answer for 16/20.
  Every candidate fails 1539, 1569, 1604, and 1632. Even a perfect selector cannot
  reach 18/20 with these whole-case candidates.
- These are hindsight upper bounds conditional on review labels. The joined
  candidate also has coarse timing: ten spans exceed eight seconds, two exceed
  twelve. No live ranker has achieved either upper bound. Generating every
  candidate would add substantial cost.

The existing audio comparisons use whole-clip recognition and its segments.
They are not a measured accuracy score for continuous live capture. The live
eight-second transport introduces an additional boundary problem.

### Completed boundary repair and remaining failures

The [streaming meaning review](../test-results/accuracy-streaming-meaning-review.json)
finds **12/20** passes for repaired sentence-terminal publication versus **6/20**
for independently translated eight-second packets. These are different baselines
from the whole-clip comparisons above. All 20 repaired transcripts retain their
final recognized text, but recognition itself is unchanged: CER is 58/1,028.
Measured compute reconstructed into a serial schedule gives median caption delay
of 6.69 seconds, versus 5.01 seconds for packets; this is not browser latency.
The first English caption completes after clip EOF in all 20 repaired cases.
Thus the prototype improves meaning while retaining a serious latency problem.

The following identifies where a consequential error first appears. Categories
do not prove that later stages could never correct an earlier error.

| First visible problem | Cases | Examples |
| --- | --- | --- |
| Corrupted Japanese recognition | 1632, 1569, 1514, 1643, 1539, 1562 | Following becomes pushing; pleading becomes `単元`; prisoners become death-row prisoners; ecological zones become prefectures; slang becomes clothing; horse-drawn wagons become Japanese cars. |
| Correct Japanese, wrong translation/context | 1592 | People know travelers need time to adjust becomes the speaker needing to get used to travelers. |
| Mixed/uncertain recognition and interpretation | 1641 | A damaged phonetic rendering of Hindustan Plain becomes Hindusthan Highlands. |

Repairing only 1592 would yield 13/20 if nothing regressed; including 1641 would
yield 14/20. Reaching 18/20 therefore requires recovering several corrupted-source
cases too. The previous clean-reference 17/20 used different input grouping from
this Medium replay and cannot be added to these results. Only case 1643 contains its exact corrected
focal token (`各圏`) in the saved evolving ASR hypotheses. Selecting among those
old hypotheses alone cannot recover most of the missing words.

## Where CLM fits

[CLM's model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) describes a
candidate scorer with frozen Qwen3-8B last-token embeddings and learned
projection heads. It generates no translations. The heads require that encoder,
so they cannot simply be attached to Gemma E4B. Probabilities are relative to the
candidate set. Published verifier results use task-specific fine-tuning and do
not establish Japanese translation quality. The card is English-tagged and cites
a project blog; this search did not identify a corresponding CLM arXiv paper.

The [inspected engine](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/engine.py)
encodes context plus instructions separately from each candidate and scores
projected vectors. Its
[confidence calculation](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/schema.py)
is the largest softmax probability minus the mean of the others. A confident
winner among bad translations can still be wrong; this is not a calibrated
probability of preserving meaning.

Possible uses, all untested in JREAD:

1. Rank two English alternatives against the same stable Japanese source before
   `gemma_segments` emits cues. Supply the source and approved context, never
   development reference translations. Evaluate regressions as well as fixes.
2. Choose a fixed action such as continue buffering, retry recognition, or emit
   a candidate. Fixed action embeddings can be cached, but buffering correctness
   should first be tested with ordinary timing/stability rules.
3. Eventually train a caption-specific contrastive head using correct translations
   and plausible errors involving negation, numbers, roles, and omissions. Split
   by source sentence and episode; near-duplicate variants must share a split.
   Twenty development examples cannot support reliable training and calibration.

A text-only scorer cannot establish whether ASR heard the audio correctly. A
faithful translation of a wrong Japanese transcript remains an end-to-end failure.
Fresh caption candidates also offer less action-cache reuse than fixed tool lists.

Resource feasibility was checked without downloading weights. The CLM head is
75,557,149 bytes at model revision
`e939398d4556fcd9400c76fa8c5a513202f42b0a`. The five standard Qwen3-8B weight
shards total 16,381,516,776 bytes at revision
`b968826d9c46dd6066d109eabc6255188de91218`
([encoder files](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)).
The RTX 5090 snapshot had 7,125 MiB free. Standard encoder weights alone exceed
that free memory, before runtime overhead. Quantization/offload is a separate
embedding-fidelity and latency experiment, not an established solution. No CLM
runtime was installed and other GPU workloads were left running.

### Audit of the proposed Whisper/Gemma → CLM phrase cache → TTS pipeline

Only the Whisper → text-only Gemma part has been exercised in JREAD. The complete
three-model phrase-routing pipeline, direct Gemma audio, and its proposed TTS
output have not been benchmarked here.

CLM can choose an existing phrase or a fallback action; the application must
provide the vetted bilingual catalog, fallback execution, and rejection policy.
It does not create that catalog or translate new sentences without a generator.
An intent such as travel assistance is broader than a meaning-equivalent sentence;
names, destinations, numbers, negation and speaker roles must survive a match.
We have no measurement showing that most anime dialogue fits a fixed catalog.

The [official CLM README](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/README.md)
reports up to ninefold gains on action-selection tasks, not Japanese speech
translation. Its architecture still performs a Qwen3-8B forward pass for each
fresh input string; cached candidate dot products are only part of the cost.
The pinned [game benchmark setup](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/examples/t_rex/README.md)
and [CLM result](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/examples/t_rex/results/clm_realtime.json)
report 16.5 ms median client latency on a local RTX 4090 versus 149.8 ms for hosted
Jev, with six requests in flight. This chooses among three game actions and uses
a physics safety shield; equal game survival does not establish equal standalone
accuracy. Neither that result nor the proposed 120–180 ms complete speech pipeline
is a JREAD measurement. Audio-frame duration also does not measure model execution
time. Our current seconds-long CPU translation and buffering costs cannot be
replaced by those quoted millisecond figures without a measured implementation.

For a future cache experiment, first measure exact repeats with all relevant
inputs in the key: source, prior context, languages, glossary, prompt and model
versions. An ordinary cache suffices for identical inputs. Semantic phrase reuse
is a separate accuracy experiment requiring out-of-catalog and near-match tests;
high relative CLM confidence cannot certify equivalence or repair wrong ASR.

## Papers with practical application

| Source | Evidence from the source | Proposed JREAD use and limits |
| --- | --- | --- |
| [Whisper-Streaming, 2307.14743](https://arxiv.org/abs/2307.14743), [implementation](https://github.com/ufal/whisper_streaming) | LocalAgreement commits a prefix shared by successive decodes of growing audio; buffers can trim at segment boundaries. The implementation supports faster-whisper. | Reuse current ASR weights, retain the unstable tail, and avoid treating transport packet edges as completed speech. More re-decoding costs compute. The reported 3.3-second latency is not a Japanese or JREAD measurement. |
| [CUNI simultaneous translation, IWSLT 2025](https://aclanthology.org/2025.iwslt-1.41.pdf) | The English→Japanese cascade encountered hallucinations with sentence-paired buffer trimming but not segment-paired trimming in its development comparison. | Avoid assuming source and target sentences correspond one-to-one. Borrow the buffer policy, not the full Whisper-large/EuroLLM-9B stack. Reverse-direction Japanese→English behavior remains untested. |
| [Non-speech Whisper hallucinations, 2501.11378](https://arxiv.org/abs/2501.11378) | English large-v3 experiments found effective VAD beneficial, but none of the tested safeguards eliminated hallucinations. | Test endings, music, silence, and quiet short replies together. Do not deploy a phrase blacklist or copy thresholds based only on English results. |
| [xCOMET-lite, 2406.14553](https://arxiv.org/html/2406.14553v2#A6), [model](https://huggingface.co/myyycroft/XCOMET-lite) | A 278M scorer supports source-plus-translation evaluation. Reported reference-free memory is 1.4 GB at batch eight. Reference-free evaluation covers English→Spanish/German/Chinese, not Japanese→English. | Most attractive small offline scorer trial; validate Japanese discrimination before any live use. Its 92.1% retained metric quality is not 92.1% caption accuracy, and batched throughput is not caption latency. |
| [COMETKiwi, WMT 2022](https://aclanthology.org/2022.wmt-1.60/), [released scorer](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) | Source-conditioned quality estimation needs no human reference at inference. The multilingual encoder covers Japanese; the paper's Japanese task is English→Japanese. | Compare against Lite on the same fixed candidate pool. Neither language coverage nor a high numerical score validates our direction and domain. |
| [XCOMET](https://aclanthology.org/2024.tacl-1.54.pdf) | Predicts error spans and severity as well as quality scores; tested targeted entity, number, and negation corruptions. | Useful design for debugging and offline review. Full XL/XXL models add substantial memory; reported corruption experiments do not validate Japanese role/omission detection. |
| [Quality-Aware Decoding, 2205.00978](https://arxiv.org/abs/2205.00978) | Directly maximizing a fixed COMET-QE score can worsen human-rated quality; tuned reranking and MBR performed better in tested European-language directions. | Evaluate selection against independent meaning labels. Start with two source-conditioned candidates; full pairwise MBR adds scoring work and cannot recover an answer absent from the pool. |
| [ACES, 2401.16313](https://arxiv.org/html/2401.16313v1) | Contrastive error tests expose weaknesses hidden by aggregate metric correlation; no tested metric dominates all phenomena. | Add separate Japanese challenge pairs for polarity, numbers, roles, omissions, proper names, and correct paraphrases before trusting a ranker. Keep synthetic diagnostics separate from natural-speech accuracy. |

## Follow-up Documents and online research

The Documents scan triaged the first pages of 91 top-level PDFs and found four
nested PDFs; no directory named `articles` was found under Documents. No directly
matching Japanese ASR or contextual translation paper was identified there.
The relevant local file is
[Reducing Hallucinated Transcripts in Whisper via Hallucination Space Projection](../../Documents/2609.04561v1.pdf),
especially PDF pp. 1–7. It suppresses invented text on non-speech using decoder
activation projection and a no-speech gate. Its speech controls are English
LibriSpeech, and suppression trades off against false rejection of genuine
speech. It does not demonstrate repair of the wrong Japanese words above, and
would require decoder hooks outside our ordinary faster-whisper path. Keep it
for future silence/music testing, not as the next accuracy fix. The locally
stored general reasoning and inference-speed papers do not supply evidence for
these Japanese errors; bounded reasoning already failed to improve our score.

More directly relevant primary research found online:

- [Ko et al., Japanese multi-pass ASR correction](https://arxiv.org/abs/2408.16180):
  diverse recognizer hypotheses plus trained Japanese correction models can help,
  but some single-model N-best correction and voting combinations worsen results.
  This supports measuring genuinely different hypotheses and preserving an
  unchanged option, rather than blindly applying another text rewrite.
- [Li et al., phonetic ASR correction](https://www.isca-archive.org/interspeech_2024/li24h_interspeech.pdf):
  Japanese CER improves from 11.72 to 7.87 in their supervised Qwen1.5-7B/IPA
  experiment. This validates a research direction, not zero-shot E4B correction
  or anime accuracy. Exact homophones such as `各県`/`各圏` need context too.
- [Shimazu et al., Japanese zero-pronoun evaluation](https://aclanthology.org/2020.lrec-1.447/):
  a 724-example benchmark isolates context-dependent antecedents that aggregate
  translation scores can miss. It motivates a bounded previous-source-sentence
  experiment and explicit role checks for 1592, without promising E4B improvement.

### Small-model candidates: capability and observed evidence

| Candidate | Why evaluate it | Limits and integration cost |
| --- | --- | --- |
| Whisper Turbo + repaired publication policy | The 20-case trial reached 13/20 meaning passes and 24/1,028 CER, versus 12/20 and 58/1,028 for medium. | Streaming-like units still yield English only after EOF in all 20 recordings. The 2.33% CER is not anime accuracy or a live-latency result. |
| Direct Gemma 4 E4B audio | [Google documents](https://ai.google.dev/gemma/docs/core/model_card_4) speech recognition and speech-to-translated-text, with audio up to 30 seconds. The local runtime now passes an audio smoke request. | One development speech control changed prisoners/courtyard; silence invented speech. This was a two-control feasibility test, not a 20-case accuracy run. |
| Kotoba-Whisper v2.2 | [Official model](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2): Japanese specialist, Apache-2.0, roughly 0.8B; an existing CT2 conversion is close to our recognition interface. | The derived alignment fallback completed the 20-case trial at 12/20, 36/1,028 CER. Word-timestamp accuracy is unvalidated; every English output arrived after EOF. Community conversion and TV-derived evaluation do not establish anime accuracy. |
| Qwen3-ASR-0.6B / 1.7B | [Official model](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [report](https://arxiv.org/abs/2601.21337) explicitly cover Japanese. Both sizes were tested with the original E4B translator on 20 full-clip recordings. | The 0.6B run scored 8/20 and 79/1,028 CER; 1.7B scored 11/20 and 49/1,028. Both hallucinated speech on silence. Whole-clip results do not measure streaming. |
| ReazonSpeech K2 v2 | [Official model](https://huggingface.co/reazon-research/reazonspeech-k2-v2): Japanese 159M RNN-T/Zipformer, Apache-2.0, roughly 775 MB repository with ONNX models; independent recognizer family. | Adds sherpa-onnx rather than CT2. No comparable accuracy or local latency result established here. An alternative to Qwen when runtime size matters. |
| TranslateGemma 4B | [Google's translation-specific model](https://huggingface.co/google/translategemma-4b-it), based on Gemma 3, is distinct from Gemma 4 E4B. Its stock template has a 2K input limit, one source-text item, language codes, and no system role. | The completed BF16 stock-template readiness test passed negation but failed explicit gender and hearsay constraints: 1/3 synthetic controls, both reviews agreeing. The planned 20-case comparison did not advance; this is not a general accuracy score. [Outcome](../test-results/accuracy-translategemma-readiness-outcome.json). |

For direct E4B audio, the matching official repository at the already-used revision
`4b4a2c1d584be7264f87aac328a1bc739ce81b6c` lists
`gemma-4-E4B-it-mmproj.gguf`, 991,552,256 bytes, SHA-256
`7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578`. The
projector was downloaded and verified for the isolated smoke test. The installed llama.cpp
revision `7fe450e19305b828c199d602c23a8337aaa1f03b` contains both
[Gemma audio processing](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/mtmd/models/gemma4a.cpp)
and [`input_audio` request parsing](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/tools/server/server-common.cpp).
This establishes runtime feasibility only; the speech and silence quality controls
failed. See [`gemma4-audio-smoke.json`](../test-results/gemma4-audio-smoke.json).

## Earlier experiment order (boundary steps now completed)

1. **Replay stable-prefix recognition offline.** Compare the current eight-second
   chunks with LocalAgreement-2 on a bounded rolling buffer. Prototype in an
   evaluation runner before changing `engine.live_infer`. Keep timestamps and
   provisional text separate; tokenize Japanese without assuming whitespace.
   Do not translate every tentative hypothesis. Measure whole-example meaning,
   dropped/duplicated text, tail loss, ASR compute, peak memory, and backlog.
   Compare buffer policies with the same ASR model before combining with Turbo.

2. **Preserve translation units without arbitrary whole-clip joining.** Use stable
   speech plus bounded boundary lookahead; preserve original source spans and
   mark uncertain alignment. LocalAgreement establishes textual stability, not
   semantic completeness. Test final negation and clause roles explicitly. The
   failed joined-Turbo experiment shows that longer text alone is insufficient.

3. **Test small source-conditioned scoring on saved candidates.** Lite then Kiwi
   can be evaluated without generating more E4B output. Separate clean-text and
   audio pools, randomize candidate order, and report pairwise decisions, overall
   selected accuracy, fixes, regressions, and score calibration. Keep the evaluator
   independent of the ranker. New generations become justified only when existing
   selection shows a benefit and candidate diversity addresses the residual errors.

4. **Add bounded retries only after measuring them.** Preserve ASR confidence
   metadata now discarded by `bounded_segments` in the experimental runner.
   Test uncertainty-based recognition retries with more acoustic context and
   score all missed or late captions as failures. Confidence is a routing feature,
   not proof of correctness. Keep cheap number/entity checks as flags; a list
   of names must not silently rewrite uncertain speech.

5. **Validate the fixed combination once.** Freeze settings and scoring before
   running the 50 held-out examples; then separately test representative anime
   audio and sustained browser playback. Report accuracy, coverage, and latency
   together. A 45/50 observed result still has sampling uncertainty and cannot
   certify anime quality. Research findings do not change this requirement.

Existing VAD remains enabled. A previous attempt to skip packets lacking fresh
unpadded VAD speech rejected a real 160 ms quiet utterance; the
[control report](../test-results/accuracy-fresh-vad-controls.json) rules out that
shortcut. Endpoint experiments must retain this counterexample and must not
discard a fresh tail merely because it is short or quiet.

## Proposed controlled comparison sequence

This sequence was drafted before the later experiments summarized below; treat it
as research history rather than a statement of the current queue.

1. **Turbo plus repaired boundaries**, keeping the translator, prompt and empty
   context fixed. Run all 20 development cases, including every existing pass.
   Keep the medium-policy result as the comparison; record new raw observations,
   source completeness, meaning, memory, compute and reconstructed delay.
2. **One preceding completed Japanese sentence as translation context**, using
   the existing 22 medium-policy units to isolate context from recognition.
   Translate only the current unit. Never use future speech, reference answers,
   or prior generated English. Include separate speaker-change, context-leakage,
   polarity and reset controls; old broad-context trials sometimes invented names.
   Combine it with the ASR winner only after checking fixes and regressions.
3. **New audio evidence:** first smoke-test direct E4B audio with its matching
   projector, then evaluate the frozen 20 clips if it works. Compare Japanese
   transcription and English meaning separately. Kotoba is the next low-churn ASR
   challenger; Qwen is the independent-model alternative if more runtime work is
   justified. Whole-clip success is a screening result, not a streaming result.
4. **Bounded second-pass selection only where useful.** Compare two genuinely
   different audio hypotheses on uncertain spans, retaining originals and available
   acoustic scores. Measure the trigger's missed errors and unnecessary retries.
   Do not use correct reference words as a live glossary. A small translator or
   scorer trial follows if translation errors remain; CLM is not a prerequisite.
5. **Freeze a winning configuration and evaluate the 50 untouched cases once.**
   Then test representative anime audio and sustained browser playback, including
   speech beyond 30 seconds, quiet replies, music, pause, seek and final tails.
   Measure latency from source time through displayed caption with a stated budget;
   offline meaning accuracy alone cannot establish a usable live extension.

That research pass itself installed no model and ran no inference. Subsequent
authorized experiments now supersede the proposed first steps:

- Turbo plus repaired boundaries reached 13/20, gaining three cases and losing
  two compared with medium's 12/20. Recognition CER improved to 24/1,028, but the
  reconstructed median caption delay remained about 6.7 seconds.
- Medium plus one preceding source sentence also reached 13/20, fixing 1592
  without changing the other 19 outputs. Two of six separate context diagnostics
  still invented a first-person speaker. Context is not a universal repair.
- Native E4B audio ran successfully with its official projector, but the first
  speech control mistranslated prisoners/courtyard and the silence control
  invented speech. This is a failed feasibility-quality check, not an overall
  native-audio accuracy score.

See [development status](development-status.md) and the linked raw/review artifacts
for the completed comparisons. The proposed Kotoba CT2 experiment, Qwen trials,
translation correction studies, and ensemble study described below have since
run. Production remains medium CUDA plus text-only E4B, thinking off, original
prompts and segmentation. No configuration has demonstrated 90% meaning accuracy.

The first Kotoba run exposed invalid word-alignment metadata before any benchmark
case completed. Its [pinned model configuration](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2/blob/9d33482a0eb9b57f1ad80708e8ac5538246d8355/config.json)
has two decoder layers and 20 heads, while its
[generation configuration](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2/blob/9d33482a0eb9b57f1ad80708e8ac5538246d8355/generation_config.json)
references layers 7–25 for word alignment. Locally, the verified weights load and
transcribe the warmup audio without word timestamps; enabling them crashes inside
the alignment call. The original download is retained intact. A separate derived
configuration tests all 20 heads in decoder layer 1, the standard last-half-layer
fallback in the [installed CTranslate2 version's converter](https://github.com/OpenNMT/CTranslate2/blob/v4.8.2/python/ctranslate2/converters/transformers.py#L992-L1001).
This repairs invalid indices, not calibrated timestamp quality; caption timing
remains an experimental limitation. No reference-specific correction is involved.

The derived-config Kotoba experiment subsequently completed all 20 cases and
passed 12/20 under independent meaning review and root audit. Against Turbo it
gains 1592 but loses 1643 and 1517; all-case source CER is 36/1,028 versus Turbo's
24/1,028. Recognition compute is lower (12.69 versus 15.06 seconds), but every
Kotoba source unit is published at EOF and no first English output precedes EOF.
The original recognizer choice remains unchanged. These results motivate a
source-repair diagnostic using saved hypotheses rather than promoting Kotoba.
See the [complete review](../test-results/accuracy-streaming-kotoba-meaning-review.json).

The subsequent source-reconciliation trial also passed 13/20, with no gain over
Turbo: 19 English outputs were identical, and the only edited transcript worsened
source fidelity without repairing the English chronology error. Six separate
synthetic controls passed, but the extra 115.16 seconds of correction compute
produced no development benefit. The [review](../test-results/accuracy-asr-reconciliation-meaning-review.json)
and original outputs are preserved. Qwen3-ASR-0.6B and 1.7B full-clip trials are
now complete and remain below the best repaired-pipeline result; their accuracy
and silence-control failures are summarized below and in the linked reviews.

The completed Qwen0.6B screening subsequently produced 79/1,028 transcript errors
and only 8/20 clearly faithful English recordings with the original E4B translator
(11 fail, one unclear). Its initial semantic review was corrected because case
1607 omitted the required worthwhile judgment and case 1649 scoped “solely” to
the host without clearly preserving exclusive beneficiaries. The
[audited review](../test-results/accuracy-qwen-meaning-review.json) retains those
corrections and the failed silence control. Qwen1.7B subsequently reached only
11/20 passes (eight fail, the same one unclear) with 49/1,028 character edits;
its [review](../test-results/accuracy-qwen-1.7b-meaning-review.json) also records
the failed silence control. Neither recognizer improves on the best 13/20 result.

An independent English-editing pass with E4B also remained at 13/20, despite
changing ten drafts and adding 102.67 seconds. The same model did not correct its
own important omissions and reversed relation. Its four simple separate controls
passed; they do not establish effectiveness on the natural-speech cases. See the
[post-edit review](../test-results/accuracy-translation-review-meaning-review.json).
Further repetitions of these same correction prompts are not supported by the
results. Recognition errors and translation errors must remain separate targets.

### Small-model candidate generation and selection

The frozen 20-case development ensemble comparison used eight translation
conditions: five E4B variants and three E2B variants. E4B original and literal
prompts each passed 13/20; natural passed 12/20; sampled seed 17 passed 14/20;
sampled seed 43 passed 12/20. Each E2B prompt condition passed 12/20 with one
unclear result. The candidate pools had oracle coverage up to 14/20, but the best
initial first-order selection passed 12/20 and the best order-stable selection
passed 13/20. The separate ID-0 p03 rerun passed 13/20 in both orders, matching
the original E4B baseline pass set. No selection exceeds that 13/20 baseline. The
full ten-pool comparison and sequential offline timing are in
[development status](development-status.md) and
[`accuracy-ensemble-selection-scores.json`](../test-results/accuracy-ensemble-selection-scores.json).

The blinded review marked 51/91 candidates pass, 39 fail and one unclear; the
root's independent audit agreed on 90/91. The disagreement is whether singular
“Comoro” should pass the rubric's “Comoros” entity requirement. We keep the
blinded unclear label in the published score; accepting it would raise each E2B
single condition from 12/20 to 13/20, without meeting the 18/20 target. The
[reconciliation note](../test-results/accuracy-ensemble-root-reconciliation.json)
records the score sensitivity and preserves original artifact hashes.

The original p03 result is a completed 11/20 primary and stable measurement with
protocol-invalid responses counted as failures; it was not provisional or gated
on an earlier control. A separate schema-only ID-0 rerun passed all 12/12 control
choices and strict protocol checks, then passed 13/20 primary and stable on the
same candidates and order. It recovers 1514 and 1647 versus the original p03
selection without regressions, and exactly matches the baseline's pass set. It
still trails the seed-17 single candidate at 14/20. The new score is in
[`accuracy-ensemble-id0-p03-scores.json`](../test-results/accuracy-ensemble-id0-p03-scores.json);
the original pool score remains preserved separately. The 12/12 control pass
applies only to the corrected ID-0 run. In the initial control run, the 4B raw
candidate texts were semantically acceptable, but six responses failed solely on
the outer ID schema; control meaning was not order-dependent. Order sensitivity
reported for development pools refers to natural-case candidate choices. In the
corrected p03 run, 27/30 valid calls chose the first displayed candidate and
12/15 two-order pairs changed choices; the stable fallback was used 12 times.
All measured pool timings are sequential offline model costs, not live caption
latency. No held-out examples were opened, no production settings changed, and
the 90% meaning target remains unmet.

### Local E4B adapter pilot

One rank-8 QLoRA epoch trained 2,269,184 LM query/value adapter parameters on
128 independently sourced clean Japanese–English pairs plus 30 reviewed real
Whisper variants. Training used 20 optimizer steps; the final partial step used
six examples. A disposable one-example smoke test verified finite gradients,
adapter updates and save/reload, and exact unchanged adapter-off logits. A root
audit confirmed every training row was used once and no validation group entered
training. The [frozen protocol](../tests/fixtures/accuracy-adapter-protocol.json)
records pinned model/runtime/data hashes, masking, decoding and the progression rule.

The source was [KU-NLP CoVoST Native Japanese](https://github.com/ku-nlp/covost2NativeJa/tree/52f01d398c24017d104021897e022676117bbe7b),
whose repository specifies CC BY 4.0. All selected pairs received model-assisted
bilingual review before inference: 28 English targets were corrected and two
ambiguous training pairs replaced in the original hash order. All 20 selected
official-development groups remained in validation. Exact/near-text and audio-hash
guards excluded the original 70 FLEURS cases without opening held-out audio.
Only two retained training variants contain substantial lexical ASR corruption;
the other 28 mainly vary spelling or punctuation. This is a small feasibility
pilot, not broad noisy-speech specialization or native-speaker certification.

The first ASR preparation attempt failed all 52 MP3s before recognition because
of a PyAV decoder API mismatch; its original report is preserved with an explicit
invalidation record. A corrected FFmpeg-to-PCM run succeeded on all 52 unchanged
selected clips. Error cases now prevent a successful preparation status.

The GPU comparison used the official unquantized E4B QAT checkpoint, loaded as
NF4, with the adapter enabled or disabled in the same runtime. It differs from
the historical CPU Q4_0 GGUF condition. All 160 validation outputs passed JSON
validation; 50 distinct captions were graded blind, before revealing condition
labels. Root and independent reviews agreed on 48 captions; two bottle→“it”
renderings remain unclear in the conservative result below.

| New validation input | Decoding | Adapter off | Adapter on |
| --- | --- | --- | --- |
| Clean Japanese | Deterministic | 17 pass, 2 fail, 1 unclear | 17 pass, 2 fail, 1 unclear |
| Whisper Japanese | Deterministic | 16 pass, 3 fail, 1 unclear | 16 pass, 3 fail, 1 unclear |
| Clean Japanese | Temperature 0.6, seed 17 | 18 pass, 1 fail, 1 unclear | 17 pass, 3 fail |
| Whisper Japanese | Temperature 0.6, seed 17 | 17 pass, 2 fail, 1 unclear | 16 pass, 4 fail |

Each denominator is 20 source groups. Clean/Whisper variants are paired, not
40 independent recordings. Accepting both uncertain captions under the already
frozen independent review adds one pass wherever an unclear appears; it does not
change the conclusion. The sampled adapter loses the beheading specificity and
tentative secretary-employment proposal. Deterministic grades do not change.
The adapter changed wording in 31 of 80 matched pairs, mostly through paraphrase
and shortening. The corrupted obstetrician/care Whisper input still produced an
incorrect translation. These observations do not establish which training
examples caused the changes. A separate numerical audit verified all 160
condition mappings and recomputed both score sets from the frozen grades.
Neither configuration satisfies the frozen requirement of more noisy-input
passes without fewer clean-input passes, so no original-development rerun or
production promotion followed. The original best remains 14/20, not a new score.

Mean GPU generation-plus-parsing time was 0.55–0.61 seconds across conditions,
with separate warmup excluded. Fixed evaluation order and different generated
lengths prevent claiming a reliable adaptation speedup; these are not live
browser or CPU-history comparisons. All owned workers exited. Full counts,
uncertainty sensitivity, training audit and artifact hashes are linked from the
[pilot outcome](../test-results/accuracy-adapter-pilot-outcome.json).

### E4B adapter data expansion (v2)

The fixed one-epoch followup used 128 clean groups plus 119 observed Whisper
variants (247 rows, 31 optimizer steps). A single ASR pass over the remaining
96 previously reviewed training clips yielded 89 accepted variants: 83 mostly
equivalent spellings/formatting and six plausible contextual repairs. Seven
ambiguous variants were excluded; original reviews and the supplemental
repairability assessment remain preserved. Together with the original two
repairs, only eight training rows contain substantial lexical ASR corruption.

Fresh validation used official-dev hash ranks 21–40: 20 text/source groups,
40 paired clean/Whisper rows and 160 outputs. Targets and full grading rubric
were reviewed and frozen before translation, with six target corrections or
clarifications. All overlap guards passed. Four clips share a training speaker;
these are fresh sentences rather than wholly unseen speakers or anime speech.
All 20 Whisper variants were retained, including two material corruptions.

| New v2 validation input | Decoding | Adapter off | Adapter on |
| --- | --- | --- | --- |
| Clean Japanese | Deterministic | 18 pass, 2 fail | 19 pass, 1 fail |
| Whisper Japanese | Deterministic | 16 pass, 3 fail, 1 unclear | 16 pass, 3 fail, 1 unclear |
| Clean Japanese | Temperature 0.6, seed 17 | 18 pass, 2 fail | 18 pass, 2 fail |
| Whisper Japanese | Temperature 0.6, seed 17 | 17 pass, 2 fail, 1 unclear | 17 pass, 3 fail |

Two blind reviews graded 60 distinct captions and agreed on 58. Conservative
reconciliation retained the root’s stricter handling of an invented train-exit
event and ambiguous “defeated by love”; fixed independent-grade sensitivity is
reported separately. The deterministic clean-text gain fixes an omitted train
boarding detail. The sampled noisy condition clarifies heartbreak but loses
reported/hearsay status on another case; no net noisy-input gain results. Under
the independent sensitivity grades, sampled noisy passes instead decrease
18/20 → 17/20. Neither condition is eligible under the frozen rule. No original
20-case generation or original 50-case held-out evaluation followed.

GPU generation-plus-parsing means were 0.575–0.632 seconds with warmup excluded.
These timings are not browser latency or a reliable adaptation speedup. More
training rows also mean more optimizer updates, so this run tests the data and
exposure together. Full provenance, sensitivity and selection are in the
[v2 outcome](../test-results/accuracy-adapter-pilot-v2-outcome.json).
A separate numerical audit reproduced both score sets exactly and verified all
160 output snapshots, condition mappings, timing records and data bindings.

The separate HF direct-audio readiness check completed using the staged E4B
checkpoint and Google’s documented
[speech translation prompt](https://ai.google.dev/gemma/docs/capabilities/audio),
with the frozen two TRAIN speech controls plus one second of generated silence.
Root and independent model-assisted reviews agreed on all three controls:
`train0666` changed “was a little drunk” into “stopped by” in both its Japanese
transcript and English translation; `train0041` transcribed Japanese correctly
but reversed “I think it is defective” into “I don't think it's defective.”
Silence emitted only `English:`, failing the exact `<NO_SPEECH>` and format
requirements without inventing substantive speech. Neither English speech
control passed. This is a failed readiness gate, not a general accuracy estimate.

Technical checks passed: all three predetermined controls were covered, every
generation ended with EOS, actual acoustic modules remained BF16 with zero
four-bit modules, language projections used 343 NF4 Linear4bit modules, and the
output head remained unquantized. The owned worker and GPU process were released.
Peak allocated GPU memory was 9.18 GiB; model load took 2.59 seconds. Sequential
control times were 1.05, 0.52 and 0.10 seconds, including processor, transfer,
generation and parsing but excluding model load and file-to-PCM decoding. The
first decode was cold; these are not browser latency or matched CPU speedup
measurements. Both text reviews used saved outputs and frozen references,
without audio listening or human certification.

The [final readiness review](../test-results/gemma4-audio-hf-readiness-review.json)
binds the protocol, runner, raw reports and both reviews by hash. No successful
generation was rerun, no larger experiment was started, and production and the
original 50 held-out cases remain unchanged. The older CPU GGUF smoke and this
HF/BF16-acoustic/NF4-language configuration are distinct failed readiness checks.

### TranslateGemma stock-template readiness and narration hypothesis

The user selected the TranslateGemma 4B challenger. The official BF16 snapshot
`10042cb0e6e7fdce748996a71dc3dc432a4e0c89` was staged and all 15 files verified.
The initial verifier incorrectly compared full LFS payload Git hashes with the
small pointer-file hashes; both weight content SHA256 values were already correct.
The repaired verifier checks content SHA256/size and canonical pointer Git SHA1
for LFS files, and content Git SHA1 for regular files, with corruption tests and
the original error retained. See the [repair audit](../test-results/translategemma-verifier-repair.json).

The stock Japanese-to-English template passed CPU validation for all 3 readiness
and 20 planned development inputs, without CUDA initialization or reference
leakage. Only the three synthetic readiness controls then ran on GPU. All outputs
were nonempty and EOS-complete, model parameters were BF16 without quantization,
and the owned worker/GPU process released. Root and independent reviews agreed:
explicit negation passed; giver/recipient roles were correct but both names
acquired unsupported male titles; the meeting's day/hour survived but reported
status was omitted. This fails the unchanged three-control progression rule.
The [outcome](../test-results/accuracy-translategemma-readiness-outcome.json) records
one pass and two failures as diagnostics, not a general accuracy estimate. No
20-case or anime TranslateGemma output was generated and no prompt was tuned.

The user's subsequent suggestions distinguish three hypotheses. Whole-recording
translation and multi-recognizer reconciliation have already been tried without
a development gain; this does not exclude better candidate diversity or learned
phonetic correction. An isolated “Japanese dialogue” to “Japanese text” change
in E4B's opening prompt tested the narration hypothesis. TranslateGemma
made unsupported additions/omissions without using that production prompt, so the
word “dialogue” cannot by itself explain all observed failures. The completed test
kept all original 20 Turbo sources and remaining prompt/settings unchanged.

Neutral wording scored **12/20 versus 13/20** for the original deterministic
wording, with no fixes and one regression. Seventeen outputs were identical.
Case 1627 weakened impossibility to “unlikely”; case 1604 changed date prepositions
but retained unsupported “I” and the reversed Revolution/hostage-crisis relation;
case 1517 remained a correct paraphrase. Root and independent reviews both counted
12 passes, with 18/20 label agreement. They differed on two unchanged outputs
(1559 and 1649); frozen original grades remain primary for identical outputs,
and both raw reviews are retained. All three changed-output grades agree. This
is model-assisted review, not human certification. The sampled E4B development
best remains 14/20; neither production nor the original 50 held-out cases changed.

All 20 requests completed without errors and the private process released.
Mean request time was 3.933 seconds versus the archived baseline's 4.097 seconds,
excluding warmup/startup; separate historical runs do not establish a reliable
speedup. See the [wording outcome](../test-results/accuracy-e4b-text-wording-outcome.json).
The captured anime minute supports a separate segment-versus-whole-input
diagnostic, but its ASR and English site subtitles are not independent Japanese
ground truth. Its 18 segments are one correlated group, not 18 accuracy cases.

The frozen anime 2×2 diagnostic completed all 38 calls without output errors,
using the 18 raw ASR segments and their exact newline join under both wording
conditions. The private process released. Thirteen segment output pairs were
identical. Root qualitative review found that whole-input context connected the
academy/system clause and replaced the isolated glasses interpretation with a
name-like output, but this is not a verified acoustic repair. Both whole outputs
incorrectly attached “You are” to the following self-introduction, omitted the
separate “As expected” confirmation, and left uncertain recognized terms
unresolved. Independent qualitative review also identified the self-introduction
role error and added detector agent. Root checked its observations against the
packet: a segment/whole-output mixup and the interpretation of the shared
“Actually” passage are documented separately; both original reviews are retained.
Whole-input English has no subtitle alignment; one clip cannot support an
accuracy percentage. A future
sentence-grouping test would need to preserve speaker turns, not merely join an
entire minute. The [anime outcome](../test-results/accuracy-anime-grouping-outcome.json)
binds the raw outputs and reviews; no production change followed.

The subsequent raw-versus-verified-Japanese diagnostic is still a draft. After
repairing the Brave connection, the active player's caption menu exposed only
English and Off. English scene captions mention changing outfits, while the
saved recognizer text contains stage show; this is a useful discrepancy to
investigate, not a verified Japanese correction or acoustic-error grade.
The saved capture provenance contains no measured recorder/playback start
offset. Therefore the derived `episode_time = 677 + WAV_offset` mapping and
episode labels in the cropped review filename are unverified. The crop itself
is exactly WAV seconds 15.86–31.86, and prior source-text comparisons retain
their original inputs and grades. No new recognition/translation run or claimed
accuracy gain follows this check. See the
[repair and reference audit](/media/psikosen/newvolume/jread-translategemma/data/anime-reference-v1/browser-repair-v1/completion-and-caption-check.json).

### Matched source-condition audit

A closer audit of the saved **Turbo** comparison confirms that both reference
and recognized Japanese used one segment and one translation call per case,
the same E4B model/runtime, original prompt, empty context, no glossary, and
deterministic decoding. All 40 prompt hashes reconstruct successfully; source
timestamps never enter the prompt. The existing frozen scores are 17/20 versus
13/20, with reference-text passes but recognized-text failures on 1632, 1592,
1539 and 1649. Lexical content, orthography and punctuation vary together, so
this is not an acoustic-only attribution. Separate historical timing remains
unmatched, and there is no sampled seed-17 clean counterpart.

In case 1649, the input difference is only Japanese commas/full stops, yet the
recognized-input translation omits the Emperor's organizer role. Nine cases
differ only in those two punctuation characters and three inputs match exactly.
This motivates an automatic punctuation-restoration trial with an exact
character-preservation guard, not reference-derived corrections. The audit
uses existing grades and performs no inference. Evidence:
[source-condition audit](../test-results/accuracy-source-condition-audit.json).

### Punctuation-only restoration outcome

The frozen follow-up completed all 20 cases using the original E4B translator.
Its restoration stage could change only `、` and `。`; comparison of all remaining
characters rejected a lexical edit in 1639 and reused the baseline. Four separate
integrity controls passed. Nineteen restoration results preserved the invariant;
five of those left the source unchanged. The other fourteen used nine new
translation calls and five exact saved-request matches. No ASR was rerun, and
the owned model process released.

The primary result is **14/20**, retaining thirteen exact prior-output grades
and using agreed root/independent judgments for seven new outputs. It has one
nominal fix (1649), no nominal regressions, and only ties the existing sampled
best. That nominal fix depends on an old pristine pass grade: the output now
says the event was “hosted solely by the Emperor for himself and the nobles.”
Independent review fails the exclusive-beneficiary constraint, while root marks
the scope unclear. Independent review also marks the hospitalization/fighting
order in reused output 1514 unclear; root accepts the narrative order.

Fresh-review sensitivity is therefore 12 pass / 7 fail / 1 unclear independently
and 13 pass / 6 fail / 1 unclear for root, with 18/20 exact label agreement. These
reviews are retained separately and do not rewrite frozen labels. The chronology
in 1604 improves, but its agent error remains; word-recognition errors also remain.
There is no robust whole-case gain from punctuation alone.

Restoration took 86.01 seconds. New translation calls took 39.50 seconds, and
the saved translation costs on reused paths add 40.34 seconds: 165.84 total,
8.29 mean, versus 81.95 total / 4.10 mean for the historical baseline. This
roughly 2.02× serial processing cost excludes controls, warmup, startup and ASR;
it is not browser latency. Despite meeting the frozen nominal numeric rule for
considering fresh validation, the disputed gain and added cost do not justify
expansion or a production change. The experiment is complete; original held-out
cases remain unused. See the [outcome](../test-results/accuracy-e4b-punctuation-outcome.json)
and [independent review](../test-results/accuracy-e4b-punctuation-independent-grades.json).
