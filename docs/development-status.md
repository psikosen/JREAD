# Development status

**Snapshot: Linux development build, October 2026.** JREAD implements a local subtitle workflow and an early browser companion. It is not the full proposed AI Subs release plan in [original-brief.txt](original-brief.txt), and no Mac build or signed desktop package is available.

The frozen Hy-MT2-1.8B Q8 comparison and separate glossary/context tests are
complete. On the same 20 clean Japanese development texts, Hy-MT2 passed **13/20**
against the saved Gemma E4B **17/20** same-source/same-prompt reference. It fixed
no baseline failures and regressed four passes. On eight fresh synthetic controls,
Hy-MT2 passed **4/8** versus Gemma **6/8**; both invented first-person speakers in
the two omitted-subject controls. On nine previously exposed publisher captions,
both had six passes: Hy-MT2 fixed one previously unclear newcomer wording but
regressed a drinking line by inventing “I.” These sets are reported separately.

Source-matched glossary hints fixed both fresh Hy-MT2 animation-terminology
controls (**0/2 to 2/2**); Gemma stayed **2/2**. The difficult anime correction
line still failed for both models because name, gender, or agent errors remained.
The separate following-line probe produced **no agreed context-fit passes out
of two** for either model. Hy-MT2 appended information from the following line
on one case; Gemma's two context-fit results remained unclear. Plausible
source-only readings were graded separately from context fit.

All 55 new translations completed without execution errors, plus two exact
readiness reuses. Fresh eight-case CPU8 translation totals were **7.23 seconds
for Hy-MT2 and 26.37 seconds for Gemma**, a 3.65x ratio in this one run, excluding
ASR, startup and browser overhead. Independent model-assisted review agreed on
all 54 new target-meaning labels and six of seven additional context-fit labels;
the remaining disagreement is retained as unclear. Technical audit confirmed
coverage, serial execution, response integrity and owned-process cleanup.

Hy-MT2 failed the frozen advancement rule, so no recognized-input followup was
run and no production change was made. The original 50 held-out cases remain
unused, and the historical best full-pipeline development result remains **14/20**.
These clean-text and synthetic diagnostics do not establish speech or general
anime accuracy. See the [Hy-MT2 comparison outcome](../test-results/accuracy-hymt2-outcome.json).

The new matched-input anime-caption diagnostic completed all nine ASR clips and
36 translation slots using the same E4B settings. Five Nakano passages were used
for calibration; four passages from a separate Saga film were reserved until one
participant-preservation instruction was frozen. On calibration, clean-caption
meaning stayed **4/5**, while recognized-input caption agreement fell **5/5 to
4/5**. The instruction replaced an invented “we” with an incorrect “story flows”
reading and regressed the previously correct recognized-input version.

On the separate validation film, both input conditions went from **two passes,
one unclear and one failure out of four** to **three passes and one failure**.
The changed result was “new hires/employees” becoming “a newcomer”: a disputed
narrowing became an agreed pass. No clear failure was fixed. One line still
failed even with clean Japanese because Gemma changed names, gender and revision
responsibility; Whisper additionally confused animation-supervision/full-correction
terms with author/collected-works terms. The instruction is not adopted.

All requests completed without output errors, with serial model ownership and
cleanup verified. Added-instruction translation totals were roughly 5–7% slower
in this single run; startup and ASR are separate, and these are not live-browser
latencies. Root and an independent reviewer graded 24 unique case/output pairs,
agreeing on 22; the two disagreements remain unclear. Caption text is visually
verified, but caption/audio correspondence could not be independently heard.
These are caption-reference diagnostics, not verified speech accuracy or a new
20-case score. Production, the historical **14/20** best and original 50 held-out
cases remain unchanged. See the [matched-input outcome](../test-results/accuracy-anime-matched-outcome.json).

Index-Translate 2B Q8 scored **11/20** on the unchanged recognized development
inputs, versus **13/20** for the exact saved Gemma E4B baseline. Both reviewers
agreed on all 19 new unique English outputs; 21 comparison slots retained exact
historical grades. Index fixed no baseline failures and introduced two regressions:
it dropped “about” from “about 40 prisoners” and weakened “could not” to “unlikely.”

Index took 29.57 seconds for the 20 sequential CPU8 translation requests, compared
with the baseline's saved 81.95 seconds, about 2.77 times faster. These figures
exclude the shared ASR work and model startup; the baseline timing is historical,
not a new contemporaneous speed benchmark. All 20 outputs completed normally.

The fixed Index-draft-to-Gemma revision was also tested on three fresh controls.
Index and the original Gemma prompt each passed 3/3; revision had two passes and
one unclear result after changing “three documents” to “three materials.” Under
the frozen rule, revision did not advance to the 20-case comparison. These small,
repeatedly studied development results do not establish anime accuracy. Production
and the original 50 held-out cases remain unchanged. See the
[Index comparison outcome](../test-results/accuracy-index-translate-outcome.json).

The source-span semantic-ledger control experiment is complete and failed its
frozen gate. The original E4B prompt passed **9/12** fresh synthetic controls;
ledger extraction followed by English rendering passed **5/12**, with six failures
and one unclear. Both reviewers agreed on all 17 unique English grades. Four
ledgers violated the exact-source-span contract and were retained as failures
with rendering skipped. The eight mechanically valid ledgers also failed to
preserve all required roles or scope unambiguously. There were no new passes.

The ledger pipeline used 192.56 seconds of sequential CPU8 model inference versus
39.03 seconds for the baseline, about 4.93 times as much, despite skipping four
render calls. Startup and warmup are excluded. Technical audit verified all 32
completed calls and cleanup. The planned 20-case followup was not run; production,
the original 50 held-out cases, and prior development scores remain unchanged.
See the [completed ledger control outcome](../test-results/accuracy-semantic-ledger-outcome.json).

The native Whisper three-candidate experiment is complete. With Gemma E4B fixed,
the first and second candidates each score **13/20**, and the third scores **9/20**.
The first candidate fixes 1592 but regresses 1607, matching the deterministic
baseline overall. Even a reference-guided choice from all three English outputs
would cover only **15/20**; that is hypothetical coverage, not achieved accuracy.
Five cases have no passing English candidate. Frozen prior grades were retained;
the two reviewers agreed on 19 of 20 new unique outputs, with the remaining
geographic-scope disagreement kept unclear and in the denominator.

All 60 slots completed: 42 new translations and 18 exact-request cache reuses.
Independent audit passed all 123 technical checks and confirmed process cleanup.
Sequential accounting including historical cached costs is 82.32 seconds for
ASR plus the first translation, or 244.69 seconds for all three translations.
Startup and warmup are separate; these are not browser latency measurements or
a direct speed comparison with the previous streaming procedure. Raw decoding
without VAD hallucinated on digital silence, so production remains unchanged.
The original 50 held-out recordings remain unused. See the
[completed ASR recovery outcome](../test-results/accuracy-asr-recovery-outcome.json).

The matched diagnostic uses two official Japanese-captioned animated shorts.
Its nine passages have independently reviewed native-frame caption text and
visibility brackets in the external `candidate-manifest-v4.json`. Earlier boundary
annotations incorrectly counted blank frames or following captions and are
superseded. Caption/audio correspondence, spoken boundaries, and speaker
disjointness remain unverified; no general anime accuracy result is claimed.

The bidirectional source-support experiment completed all technical stages. Its
frozen primary rule passed **93/128** comparisons on the reused 32-group synthetic
set, below XCOMET alone (97/128) and XCOMET plus the quantity guard (98/128).
Forward-only support, a predefined secondary diagnostic, passed 99/128 but added
13 regressions against the previous 98/128 result while fixing 14 errors. It does
not replace the failed primary after results. The gate failed; no larger test or
production promotion is selected. Independent aggregation audit confirms the counts, coverage, and timing.
The historical translation best remains **14/20**. See the
[grounding outcome](../test-results/accuracy-grounding-verifier-outcome.json).

The fresh quantity-check follow-up completed scoring: XCOMET alone passed **97/128**
pair comparisons, and the same scores with the conservative quantity guard passed
**98/128**. There was one fix, no regressions, and no quantity-mismatch verdict on
the 64 faithful candidates. The guard abstained on 58 of those 64 candidates.
Both conditions fully passed 14/32 synthetic groups. The result misses the frozen
gate and does not improve the historical 14/20 translation score. Independent
aggregation audit confirms both counts. See the [paired outcome](../test-results/accuracy-quantity-verifier-v2-outcome.json).

The fresh synthetic bilingual-checker experiment has completed scoring. XCOMET-lite
ranked the faithful translation above the incorrect one in **95/112 comparisons
(84.8%)**; Laya multilingual passed **40/112 (35.7%)** under the stricter requirement
to choose correctly in both option orders. Neither meets the frozen 90% overall /
75% per-family gate. XCOMET's weakest family was numbers, units and dates (7/16).
These are correlated comparisons across 28 synthetic sources, not caption accuracy;
the historical development best remains 14/20. Independent result audits confirm both counts.
Production and the original 50 held-out cases remain unchanged. See the
[checker outcome](../test-results/accuracy-caption-verifiers-outcome.json).

The ReazonSpeech K2 → unchanged Gemma E4B comparison is complete: **13/20**
meaning passes, matching the deterministic Turbo baseline and below the sampled
14/20 best. Root and independent reviews agree on all 20 labels. It fixes being
followed (1632) and travelers needing time to adjust (1592), but regresses a
worthwhile village stroll into a confection (1607) and square kilometres into
square metres, with ecological zones becoming prefectures (1643). Ten exact
previous-output grades were retained; ten new outputs received agreeing reviews.

All 20 cases completed without output errors. Fifteen new translations and five
exact-request cache reuses cost 80.97 seconds including historical cache costs;
Reazon recognition added 3.04 seconds, for 84.01 seconds total / 4.20 per case.
The historical 81.95-second figure covers translation only, so this is not an
end-to-end speed comparison. Recognition processed whole clips after EOF on CPU;
the result does not establish live or anime performance.

Readiness remains failed: a speech control changed “course” to “concept/plan,”
and eight seconds of silence produced “はい” (“yes”). The all-20 run was a
separately frozen offline diagnostic, not a readiness pass. It is not selected
for expanded validation or production. Production and the original 50 held-out
recordings remain unchanged. See the [reconciled outcome](../test-results/accuracy-reazon-outcome.json).

The local E4B QLoRA pilot is complete and was **not selected**. On a separate
20-group CoVoST validation set, adapter-off/on meaning passes were 16/20 → 16/20
for deterministic decoding of Whisper transcripts and 17/20 → 16/20 for sampled
decoding. Conservative counts retain one uncertain bottle/pronoun case in the
denominator where present; the independently frozen permissive grades also show
no gain. These new validation counts are not comparable to the original FLEURS
score: the historical best remains 14/20 on those unchanged development inputs.
The app, original 50 held-out cases, and production model were left unchanged.
See the [pilot outcome](../test-results/accuracy-adapter-pilot-outcome.json) and
[training/evaluation details](accuracy-research.md#local-e4b-adapter-pilot).

The second data-expansion pilot also completed without qualifying for promotion.
On its separate 20-group CoVoST set, deterministic clean-text passes improved
18/20 → 19/20, while Whisper-input passes stayed 16/20. Sampled Whisper-input
passes stayed 17/20 under conservative grading; the unchanged independent grades
also showed no noisy-input gain. All 160 outputs passed format checks, and both
blind reviews were frozen before comparison. Four validation clips share a
training speaker. The original benchmark best remains 14/20; these are different
datasets. See the [v2 outcome](../test-results/accuracy-adapter-pilot-v2-outcome.json).

The E4B HF direct-audio readiness test completed but failed its frozen quality
gates. Both reviews found that the two TRAIN speech controls mistranslated
“was a little drunk” as “stopped by” and reversed the suspicion that a product
is defective into a denial. Silence returned `English:` instead of the required
`<NO_SPEECH>` marker, without inventing speech content. Acoustic BF16 / language
NF4 precision, coverage and cleanup passed. These three controls are not an
accuracy benchmark; no larger experiment or production change follows this
failure. See the [readiness review](../test-results/gemma4-audio-hf-readiness-review.json).

TranslateGemma 4B also completed a separate stock-template text readiness test.
All 15 checkpoint files verified after fixing a verifier that confused Git LFS
pointer hashes with payload hashes; no redownload was needed. BF16 inference and
cleanup passed. Both meaning reviews agreed on one pass and two failures among
three synthetic controls: negation survived, but gender-neutral names acquired
“Mr.” and a reported meeting time became an asserted schedule. Under the frozen
all-three-pass rule, the planned 20-case and anime translation comparisons did
not run. These controls do not measure general accuracy. See the
[TranslateGemma outcome](../test-results/accuracy-translategemma-readiness-outcome.json).

The user's anime window at 11:17–12:17 was captured once and restored to its
paused starting position. It is exploratory audio, without an independent
Japanese reference. The completed E4B wording experiment changed only “Japanese
dialogue” to “Japanese text” on the same 20 saved Turbo inputs. Meaning passes
fell from 13/20 to 12/20: no fixes, one regression from “impossible” to “unlikely.”
Seventeen outputs were identical; the narration still invented “I.” Root and
independent reviews each scored 12/20, agreeing on 18 labels; unchanged outputs
retain the original frozen grades. Production wording stays unchanged. See the
[wording outcome](../test-results/accuracy-e4b-text-wording-outcome.json).
The separate anime diagnostic completed all 38 calls: 18 saved ASR segments and
their whole-clip join, each under both wording conditions. No output errors;
the owned private process released. Root review found that whole-input context
connected fragments but also misattached a self-introduction to the preceding
“You are,” crossing a likely speaker turn. Independent qualitative review also
identified that role error. The single clip has no verified Japanese reference
or accuracy score; whole output is not timestamp-aligned. Sentence grouping that
preserves speaker turns is a future hypothesis, not a validated improvement.
See the [anime outcome](../test-results/accuracy-anime-grouping-outcome.json).

The Brave connection was repaired and the existing anime player's active HD-1
caption menu was inspected: it exposes only Off and English. The paired raw-ASR
versus audio-verified-Japanese diagnostic remains pending a Japanese reference
and checked speaker boundaries. No new inference ran. A timing audit also found
that adding the player's paused start time (677 seconds) to WAV-relative ASR
timestamps had never been synchronized with playback. Treat those derived
episode labels, including the 16-second review crop's filename, as unverified;
the audio frames and previous text-only results are unchanged. Playback was
restored paused at 677.179031 seconds. See the
[repair and caption check](/media/psikosen/newvolume/jread-translategemma/data/anime-reference-v1/browser-repair-v1/completion-and-caption-check.json).

The punctuation-only diagnostic and independent review are complete. All 20
cases finished, all four integrity controls passed, and the guard rejected one
attempted word change (1639), retaining the saved baseline. Fourteen accepted
changed sources required nine new translations and five exact-request cache
reuses; six cases retained their baseline output. The model process released.

Preserving exact prior-output grades gives 14/20 versus the deterministic
baseline's 13/20, tying the historical sampled best. Independent review finds
12 passes, seven failures and one unclear case; root's fresh audit finds 13
passes, six failures and one unclear. The sole primary gain, case 1649, restores
the organizer but puts “solely” on the host rather than clearly on the intended
beneficiaries. Its old pristine pass grade remains frozen, with the new scope
disagreement disclosed. Independent review also questions the chronology in
the reused pristine output for 1514. The seven outputs without prior grades
have complete root/independent agreement.

Sequential processing cost is 165.84 seconds total, 8.29 seconds per case,
versus 81.95 total / 4.10 mean for the saved baseline. This includes historical
translation costs for reused outputs, excludes controls/startup/warmup, and is
not live browser latency. The frozen numeric follow-up rule is met, but the
gain is not robust to independent review and cost is about twice the baseline;
this method is not selected for expansion or production. See the
[completed outcome](../test-results/accuracy-e4b-punctuation-outcome.json).

During preparation, an earlier self-test accidentally issued one restoration
request on an invented sentence because a fake callback was passed into the
wrong argument. It did not use development or held-out cases. Optional arguments
are now keyword-only, the CLI self-test forbids model startup, and the owned
orphan server and its temporary key were released before the real trial. The
corrected CPU checks passed. See the
[preparation erratum](../test-results/accuracy-e4b-punctuation-preparation-erratum.json).

## Implemented

- Tauri 2 desktop window, local Python/FastAPI service, loopback authentication, and a Linux user install script with native-messaging registration for Chrome, Brave, and Chromium.
- Explicitly downloaded, checksum-verified faster-whisper `small` and `medium` speech models, with a separate English translator selection: Whisper baseline, Gemma 4 E4B, or Gemma 4 12B. Fresh settings keep Whisper as the default while broader Gemma evaluation continues. Gemma uses pinned Google GGUF files and an isolated, checksum-verified llama.cpp CPU runtime; both run locally after explicit installation. Runtime and E4B provenance are in [runtime.json](runtime.json), with the larger model in [gemma-12b-model.json](gemma-12b-model.json).
- Local media upload, Japanese transcription followed by the selected translator, saved jobs/cues, progress, cancellation/resume, SRT/WebVTT import and export, cue editing and undo, native local playback captions, and library deletion.
- MV3 extension workflows for pairing, accessible player selection, applying saved subtitles, and explicitly started live tab-audio capture. Live captions arrive after speech has been processed; they cannot caption audio that has not played yet.
- Companion settings for caption size, dark background, vertical position, and optional Japanese text.
- Two-line generated English cues, with estimated internal timing flagged for review. Source Japanese segments remain available for comparison.

## Known limits

- The initial Gemma comparison covered 13 accepted requests: one FLEURS phrase and 12 dialogue chunks. CPU translation took 3.64–10.24 seconds per chunk; cold FLEURS translation took 9.08 seconds. These timings use the initial prompt configuration; a more compact JSON prompt is being evaluated. In this small sample, Gemma improved two dialogue readings over Whisper: “we ended up singing together” instead of “let’s sing together,” and “that’s so Akari-like” instead of “I think it’s Akari.” Remaining errors include ASR’s `マナカラーラ` becoming “Manacala,” the possible catchphrase `普通のアルプリ` becoming “normal Alpri,” an awkward first live-show phrase, and “Adversarial Environment Course” for “hostile-environment course.” These examples are preliminary and do not establish an accuracy rate or broad translation quality.
- No approved glossary, selected-span retranslation, broad human-reviewed evaluation corpus, or production performance benchmark is included. Do not treat the small sample or proposed one-to-three-minute episode targets as representative accuracy or measured episode throughput.
- Live-capture delay varies with playback and device speed and has not been benchmarked. Capture is from currently playing tab audio, not a way to read ahead through an episode.
- Browser and player compatibility is limited to tested, accessible paths. Unsupported frames, inaccessible tracks, and DRM-protected media receive no bypass or extraction workaround. Universal site support and mobile browsers are outside this build.
- Player discovery covers ordinary HTML video elements and permitted frames. Shadow-root players and picture-in-picture have not been validated. Media-clock sampling, resampling quality, and the proposed 100 ms timing target remain unmeasured.
- Model downloads need network access. Media processing and inference run locally after installation; Gemma uses CPU only, with one local server process owned by the backend and no fallback to Whisper if Gemma translation fails.
- Linux is the only implemented target. The Tauri bundle is disabled; `scripts/install-linux.sh` installs development binaries and a user-level launcher/native host, not a signed distributable package. macOS, including Apple Silicon, has not passed a backend validation gate.

## Current checks

The repository contains:

- `tests/test_core.py` and `tests/test_gemma.py`: API auth/origin limits, subtitle parsing/export, import/edit/undo/delete, pairing, model recovery, Gemma weight/runtime verification, local server lifecycle, translation response validation, cancellation, and live WebSocket lifecycle checks.
- `extension/tests/core.test.ts` and `extension/tests/lifecycle.test.mjs`, run by `node extension/tests/run.mjs`.
- `tests/ui-smoke.mjs`: optional real Brave/Playwright companion check. It creates and removes a temporary subtitle job; it requires a running service, a usable local token, and a ready model.
- `tests/smoke_inference.py`: optional real-model run on the short fixture. It requires an installed model and writes a report to `test-results/inference.json`.
- `tests/smoke_live.py`: real PCM over WebSocket through local inference, checking saved cues, clock offsets, and graceful stop/drain; this is separate from browser tab capture.
- `tests/extension-smoke.mjs`: compiled extension in an isolated Brave profile, real pairing/import, native captions, same-origin iframe application, pause/seek/rate/fullscreen, credential isolation, sequential delayed captions, and cleanup.
- A native-host framing test in `src-tauri/src/bin/native_host.rs`, run with `cargo test --manifest-path src-tauri/Cargo.toml`.

The installed Linux desktop has been launched and visually checked, including WebKit video decoding and visible native captions. The installed native host rejects unapproved origins and unsupported actions with correctly framed responses. Backend regression tests, extension TypeScript/core/lifecycle checks, companion UI checks, and extension renderer checks pass. Real medium-model file processing and PCM/WebSocket inference also pass on the fixture. The Gemma comparison completed 13 accepted requests and produced preliminary examples and CPU timings, but no broad human-reviewed accuracy result or production throughput benchmark. Passing these checks is not bilingual review or a release sign-off.

Real tab capture also passed in disposable Brave 154: toolbar activation, playing Japanese audio, offscreen AudioWorklet conversion, local inference, three saved captions, automatic completion when playback ended, and release of the browser capture. Evidence: `test-results/extension-smoke.json`. Reproduce with `JREAD_TEST_LIVE=1 node tests/extension-smoke.mjs`; the isolated test enables Chromium's extension-debugging interface to trigger the toolbar action and removes Playwright's default audio mute. The user's browser profile and extension permissions are not changed by the test.

The initial installed-machine checks used Whisper medium for Japanese recognition and Gemma 4 E4B for English translation. With the compact JSON prompt, the complete 10.44-second file fixture passed in 11.15 seconds, and the PCM/WebSocket fixture passed in 9.02 seconds. Real Brave tab capture also passed with E4B selected. These are short functional checks, not sustained live-throughput or translation-accuracy guarantees. The initial same-dialogue comparison is saved in `test-results/gemma-e4b-comparison.json`; both translation tracks are retained in the companion library.

The installed extension also captured the user's existing Aikatsu page with E4B selected: nine cues were saved, a delayed caption was visibly displayed about 10.8 seconds after its source time, and explicit stop drained the final audio and completed without errors. The page was left paused and capture stopped. Evidence: `test-results/actual-page-gemma-e4b.json` and `test-results/installed-gemma-popup.png`. Temporary captured audio/video used for the comparison was deleted after review.

## Data and test fixture

The 12B trial adds model selection without changing speech segmentation or the translation prompt. `tests/compare_gemma.py --preset gemma4` and `--preset gemma4-12b` compare 17 identical sets of saved Japanese text using the current compact JSON prompt. These opt-in checks require the saved reports from this test session and installed weights. Both matched runs are complete, and the 12B file passed its pinned SHA256 check. Excluding each first cold request, median CPU translation time was 4.308 seconds for E4B versus 8.678 seconds for 12B (16 requests each). 12B fixed the hostile-environment-course term, but the academy-affiliation sentence, two-person address and character name remain wrong. The larger model also worsened a fragmented PriPara name. These observations do not establish a broad accuracy gain. The installed machine returned to E4B at the user’s request after the 12B trial; existing jobs retain their original translator. Comparison evidence is in `test-results/gemma-size-comparison.json`, with trial progress in `test-results/gemma-12b-trial-status.json`. The 12B file pipeline passed in 18.27 seconds on the 10.44-second fixture.

By default, JREAD keeps its database, local token, imported media, and downloaded models under `~/.local/share/JREAD/`; `JREAD_DATA_DIR` can redirect the service data directory. The companion clears saved episodes separately from model files.

The 10.44-second FLEURS Japanese speech fixture is Google FLEURS test row 0 (`ja_jp`, row ID 1828), licensed CC BY 4.0. Its exact source, reference transcript, and generated-video provenance are recorded in [fixtures/README.md](../tests/fixtures/README.md). The fixture does not represent anime dialogue or conversational translation quality.

The 12B PCM/WebSocket fixture passed in 18.20 seconds, and real tab capture in disposable Brave passed with three saved cues and automatic capture release. A restart defect was also fixed: FastAPI lifespan shutdown now explicitly stops the owned model process and removes its temporary key; this passed a real shutdown/restart with 12B loaded. Live model errors now retain their safe diagnostic message. The installed extension was paired and visibly reported Gemma 4 12B during that trial.

The longer installed-browser 12B check failed to sustain live dialogue on CPU. During roughly 54 seconds of page playback, a visible delayed cue reached 22.6 seconds of lag and the extension stopped with three audio chunks waiting. The video was paused around11:15. This distinguishes successful short pipeline checks from continuous live usability: the machine has returned to E4B, and this CPU setup was not fast enough for continuous 12B live captions. Evidence: `test-results/actual-page-gemma-12b.json` and `test-results/installed-12b-live-throughput-failure.png`.

## E4B caption experiments

Optional show glossary, per-capture rolling Japanese context, and conservative overlap reconciliation are implemented. Model source delimiters are now content-derived so identical effective inputs yield identical prompts; leaked model protocol markers are rejected instead of shown in captions. The initial random-delimiter comparison is retained separately and is not the controlled result. `test-results/caption-combos.json` records the controlled 24-case comparison and both overlap policies; `test-results/caption-asr-combos.json` records a separate short speech-recognition replay. Corpus and checks were written before new inference. No human-certified accuracy percentage is claimed.

The controlled comparison's per-case synthetic reset bug was repaired using ten
targeted context/context-plus-glossary requests. The pre-repair report is retained;
its synthetic context conclusions are invalid. After repair, four independent
synthetic meaning checks pass under all conditions; the quoted-instruction case
fails output validation under all four, with no captions accepted. Context helps
the academy-affiliation example but can invent a name in a clipped phrase. The
show glossary helps preferred names but biases recognition on an unrelated
FLEURS clip. These findings do not support enabling the show glossary universally.

An isolated CUDA pilot using the existing Whisper medium model succeeded on the
installed RTX5090. Its warmed 8-second/1-second-overlap replay took 0.615 seconds
of ASR inference, versus 4.282 seconds in the earlier CPU report. Those timings
were recorded separately and exclude Gemma translation and browser delay. Chunk
boundary errors remained. The owned worker exited and released its GPU allocation;
unrelated GPU processes were unchanged. Reproduce with
`tests/asr_cuda_pilot.py`; evidence is in `test-results/asr-cuda-pilot.json`.

The next target is 90% meaning-preserving captions under the explicit criteria in
[accuracy target](accuracy-target.md). A frozen 70-recording supplemental FLEURS
corpus separates 20 development cases from 50 held-out cases. Its read speech is
not representative of anime; no held-out accuracy result has been established.

The installed app now uses medium recognition on CUDA with E4B translation. The
real PCM/WebSocket pipeline passed in 6.95 seconds. Disposable Brave tab capture
also passed with that combination: a 10.44-second recording produced three cues,
completed playback in 10.605 seconds and drained in another 4.841 seconds. These
are fixture timings, not sustained live delay. The extension now avoids flushing
already-sent overlap when no fresh samples remain; exact 8/15-second and very
short fresh-tail regressions pass. The installed extension was reloaded and paired.

On 20 development recordings, medium recognition made 58 normalized character
edits across 1,028 reference characters (5.64% CER); Large v3 Turbo made 24
(2.33%). Both passed the zero-PCM silence check. Strict model-assisted semantic
review found 17/20 passes when E4B received pristine reference text, 12/20 with
medium ASR segments, and 12/20 with Turbo ASR segments. These small development
counts are not human-certified or representative anime accuracy. One ambiguous
Japanese coordination is retained as a strict-rubric failure with its caveat.

Joining Turbo's transcript fragments into one translation unit also scored
12/20. It removed one duplicated sentence but introduced a negation reversal;
it is an evaluation-only experiment. The Turbo review originally reported 13/20;
an audit found a missed omission of the Emperor as organizer in case 1649 and
corrected both stages, retaining an audit note in the report. The general
precision-instruction prompt and two-example prompt each scored 16/20 on pristine
references, below the original 17/20. Those experiments remain separate artifacts.
Production keeps the original translation prompt and segment handling. Turbo is
selectable but the installed selection remains medium. Its real companion UI
selection, persistence and incompatible-choice handling passed in disposable
Brave; the test restored the exact original settings afterward.

Two additional baselines did not improve the result. A pinned community INT8
conversion of NLLB-200-distilled-600M passed11/20 pristine Japanese-reference
translations, versus E4B's17/20 on the same text. Whisper medium direct
speech-to-English translation passed7/20, versus12/20 for medium recognition
followed by E4B on the same audio. The pristine-text and audio comparisons are
separate; neither is held-out or human-certified evidence. NLLB is an isolated
experiment with CC-BY-NC/research-use limitations recorded in its provenance;
it is not a production preset. Installed E4B selection is unchanged.

The out-of-domain FLEURS control shows why a show glossary must stay optional: the anime glossary caused an academy term to appear in unrelated speech. The runtime uses only the correctly spelled Japanese hints for recognition; phonetic variants are translation aliases within the explicitly selected show profile.

The bounded-reasoning E4B trial also passed 17/20 pristine-reference cases, with
the same three failures as the original prompt. Mean translation time increased
from 4.049 to 13.335 seconds in separate CPU runs (3.29×); thinking stays off.
The precision-prompt review JSON has been audited to match its 16/20 result:
case 1649 omits the Emperor as organizer, despite preserving the beneficiaries.
The original review is archived and the correction is recorded in the artifact.

The [CLM and paper research](accuracy-research.md) recommends testing stable
speech boundaries before adding candidate ranking. A development-only oracle
audit finds that perfect selection among existing actual-audio candidates would
pass at most 16/20; the corresponding pristine-text pool reaches 19/20. Neither
is an achieved reranker score. CLM was inspected but not installed; its standard
Qwen3-8B encoder exceeded the approximately 7 GiB free GPU memory observed during
that feasibility check. Available memory has since increased; that earlier limit
does not establish that it cannot run now. The research identifies smaller
quality estimators for an offline comparison, with Japanese validation required.

The first bounded LocalAgreement-2 recognition replay completed all 20
development recordings, but is unsuitable for deployment. It froze internally
confirmed text before translation units were emitted; later recognition
corrections then caused six unresolved prefixes (1632, 1604, 1514, 1643, 1539,
1562). These cases must remain failures, not be excluded from meaning accuracy.
The reported 20/652 character edits cover only the other 14 recordings and are
not comparable to the prior full-20 CER. The final raw, whole-buffer hypotheses
across all 20 still have 58/1,028 edits, the same aggregate as medium's earlier
whole-clip result. This is a diagnostic after EOF, not streaming accuracy.

Only one of 22 translation units appeared strictly before EOF. Recognition
compute was 40.84 seconds versus 12.77 seconds for the same-run eight-second
packet baseline. The silence control passed, and the worker unloaded its model
and released its GPU allocation. No translation benchmark or production change
was justified for that broken state. Evidence is retained in
`test-results/accuracy-streaming-speech-development.json`.

A follow-up replay repaired the immutability boundary: source text may revise
until its translation unit has actually been emitted. The sentence-terminal
policy retains the complete final ASR transcript in all 20 recordings without
published-prefix conflicts. Two earlier-emission variants still fail one case
and were not selected. Recognition was not rerun; all policies use the same
141 saved observations. This fixes prototype truncation, not the remaining
recognition errors: all-20 CER stays 58/1,028.

The completed E4B comparison passes 12/20 whole examples with the repaired policy
versus 6/20 with independently translated eight-second packets. This streaming
baseline differs from the earlier whole-clip-recognition 12/20 baseline. Six
cases improve and no previously passing case fails, but case 1592 gains a new
wrong relation in its second sentence while its first sentence improves. The
independent model-assisted review and root audit retain the 20-case denominator;
no held-out or anime accuracy is established. Evidence:
`test-results/accuracy-streaming-meaning-review.json` and
`test-results/accuracy-streaming-repaired-translation-development.json`.

The repaired candidate produced 22 English units with no translation errors.
Measured translation time fell from 126.54 to 83.87 seconds, but waiting for
complete units delays captions: reconstructed median source-end-to-English
completion rises from 5.01 to 6.69 seconds, with a 15.63-second maximum. The first
English caption completes after the recording ends in all 20 repaired cases,
compared with 6/20 packet-baseline cases. These are simulated schedules using
offline measured compute, not browser latency. The isolated E4B worker exited
and port 18767 closed. Production remains unchanged: rolling audio beyond
30 seconds, pause/seek handling and bounded sentence buffers need validation
before this experimental policy can be integrated safely into live capture.

The follow-up Turbo plus repaired-boundary comparison is complete. All 20
development transcripts remain aligned, with no publication conflicts. Japanese
CER falls from 58/1,028 (5.64%) to 24/1,028 (2.33%); measured recognition compute
falls from 40.84 to 15.06 seconds across the same 141 causal observations.
Independent meaning review and root audit find 13/20 passes, versus 12/20 for
medium plus repaired boundaries. Cases 1514, 1643 and 1562 improve; 1604 reverses
the revolution/hostage-crisis chronology and 1649 loses the Emperor's organizer
role. These two regressions prevent a blanket quality improvement claim.
Evidence: `test-results/accuracy-streaming-turbo-meaning-review.json` and
`test-results/accuracy-streaming-turbo-none-translation-development.json`.

Turbo's reconstructed median caption delay is 6.66 seconds versus 6.69 seconds;
the maximum is 12.16 versus 15.63 seconds. First English output still completes
after clip EOF in every case. Faster recognition alone has not resolved live
caption timing. The run made 20 English calls with zero protocol failures and
released its owned server. A latent adapter issue could pass within-update
context into the nominal no-context arm; the actual 20 calls were independently
verified to have empty context, so no observed prompt was affected. Its executed
runner source is preserved beside the report, and the adapter was fixed and
self-checked before the context trial.

The separate medium plus one-previous-Japanese-sentence trial has also finished:
all outputs are identical to medium policy A except the second unit of 1592,
which now correctly says travelers need time to adjust. Root comparison therefore
finds 13/20 with no new development failures; independent review confirms this.
Only one of 21 calls uses nonempty context. Six separate synthetic controls remain
outside the accuracy denominator, and two still infer an unsupported first-person
speaker despite supplied context. Estimated median delay is 6.81 seconds. This
is not evidence to enable unrestricted context across anime scenes.
Evidence: `test-results/accuracy-streaming-medium-previous-source-translation-development.json`
and `test-results/accuracy-streaming-context-meaning-review.json`.

Direct E4B audio works with the verified official projector and native WAV input,
but its two smoke controls failed. On development case 1514, it changed prisoners
to casualties and the courtyard to inside; the Japanese line also lacked its
requested label. The call took 6.43 seconds including audio processing and decoding.
One second of silence produced invented speech: "I speak Japanese." Both isolated
servers exited, keys were removed and the port was released. These controls
establish runtime feasibility and concrete failures, not an overall accuracy
score. No broader native-audio run or production promotion followed this result.
Evidence: `test-results/gemma4-audio-smoke.json`. Production settings remain unchanged.

Kotoba's completed development trial passes 12/20 whole recordings, below Turbo's
13/20. Independent review and root audit agree. Compared with Turbo, it fixes
1592 (travelers need time to adjust) but regresses 1643 (ecological zones become
prefectures) and 1517 (making machines becomes machine policies). Relative to
medium's repaired policy it gains three and loses three. Evidence:
`test-results/accuracy-streaming-kotoba-meaning-review.json` and
`test-results/accuracy-streaming-kotoba-none-translation-development.json`.

The initial Kotoba conversion crashed in word alignment because its alignment
settings reference nonexistent decoder layers. The failed report and exact
runner sources were preserved. A separate derived configuration uses CTranslate2's
standard last-half-layer alignment fallback; the verified original download and
weights remain intact. The derived model completed all 141 causal observations,
with all 20 transcripts retained by the repaired publication rule and no
published-prefix conflicts. Final transcript CER is 36/1,028 (3.50%); the raw
capture's partial-coverage CER must not replace this all-case result.

Kotoba recognition compute totaled 12.69 seconds versus Turbo's 15.06. English
generation took 78.49 seconds, with no protocol errors or missing outputs. Every
source unit was emitted at EOF, and no first English caption completed before EOF.
The 6.58-second median reconstructed source-end delay also depends on the fallback
word timestamps, whose accuracy is unvalidated. This is not a demonstrated live
speed improvement. Owned CPU/GPU processes were released. Kotoba is not promoted
to production; the best reviewed repaired-pipeline development result remains
13/20, and the 50 held-out recordings remain unused.

The three-transcript reconciliation trial also finishes at 13/20. It supplies
only the final recognized Japanese from Turbo (primary), medium and Kotoba to
E4B, then translates with the original prompt. Independent review and root audit
find 19 identical English outputs. The only changed case, 1604, still reverses
the revolution/hostage-crisis chronology; its Japanese loses another character.
All six separate controls pass, including abstention on conflicting numbers and
negation, but they do not outweigh the lack of improvement on real recordings.
Correction adds 115.16 seconds across 20 clips, followed by 78.15 seconds of
translation. The additional correction stage is not enabled. Evidence:
`test-results/accuracy-asr-reconciliation-development.json` and
`test-results/accuracy-asr-reconciliation-meaning-review.json`.

A different recognizer, Qwen3-ASR-0.6B, is staged in an isolated Python runtime.
Pinned weights, dependencies, and a CUDA matrix-operation check are verified.
Its first smoke test runs successfully but fails silence quality: eight seconds
of zero PCM produces "ええ。". On development case 1632 it still mistakes 尾行 for
威光. The call takes 0.31 seconds after a separate warmup, ends with EOS, and
does not truncate. All three smoke calls complete and the owned GPU process exits.
Evidence: `test-results/qwen-asr-smoke.json`. The smoke tests no VAD, word alignment
or live streaming; its silence failure must remain visible in further comparisons.
An all-20 full-clip screening is authorized to measure complementary recognition
errors, not to promote the model based on speed. Production settings are unchanged.

Qwen0.6B's full-clip screening and original-prompt E4B translation are complete.
It makes 79/1,028 character edits (7.68%) versus Turbo's 24/1,028 (2.33%), although
the 20 whole-clip recognition calls take only 5.84 seconds. The recognition modes
differ: this timing is not a matched comparison with 141 streaming observations.
All 20 English calls complete in 81.62 seconds, but meaning review and root audit
find only 8/20 passing recordings, with 11 failures and one unclear case. The
initial review's 10/20 was corrected: an interesting walk is not equivalent to
saying a stroll is worthwhile (case 1607), and “organized solely by the Emperor”
does not clearly preserve exclusive beneficiaries (case 1649). Unclear cases
remain in the denominator. The original reviews and correction history are preserved. Silence still
hallucinates speech; this model is not enabled in production. Evidence:
`test-results/accuracy-qwen-meaning-review.json` and
`test-results/accuracy-qwen-translation-development.json`.

The separate E4B English post-edit trial changes ten drafts but leaves the same
13 passes and seven failures as Turbo's original outputs. Root and independent
review agree. It adds 102.67 seconds across 20 drafts without repairing the
chronology, organizer omission, traveler adjustment, or lexical errors. The
extra editing pass is not enabled. Evidence:
`test-results/accuracy-translation-review-development.json` and
`test-results/accuracy-translation-review-meaning-review.json`.

Qwen1.7B's full-clip screening and original E4B translation are also complete.
Recognition makes 49/1,028 character edits (4.77%) in 6.41 seconds; all 20 English
outputs finish without protocol errors in 79.12 seconds. Independent review and
root audit find 11/20 passes, eight failures and one unclear case. The initial
12/20 review is archived: case 1649 has the same host-versus-beneficiary scope
ambiguity as Qwen0.6B. Silence still produces “はい。”; the owned processes were
released. It does not improve on the best repaired pipeline's 13/20, and its
whole-clip timing does not establish live-caption speed. Evidence:
`test-results/accuracy-qwen-1.7b-meaning-review.json`.

The frozen small-model ensemble trial is complete on the 20 development cases.
It tested original/literal/natural prompts and two sampled E4B outputs, three E2B
outputs, and ten candidate pools. Independent blind grading covered 91 unique
English outputs; the root audit agrees on 90, with the sole difference discussed
in [the reconciliation note](../test-results/accuracy-ensemble-root-reconciliation.json).
The blinded grade is 51 pass, 39 fail and one conservative unclear; the root audit
counts 52 pass and 39 fail. The unclear output says “Comoro,” where the rubric
specifies “Comoros”; root accepts it as a recognizable minor variant. Scores below
retain the blinded unclear label. Accepting it would raise each E2B single condition
from 12/20 to 13/20, but would not establish the 18/20 target.

Single-candidate results (pass counts out of 20; measured generation time):

| Candidate | Passes | Total / mean seconds |
| --- | ---: | ---: |
| E4B original prompt | 13 | 81.9 / 4.10 |
| E4B literal prompt | 13 | 82.0 / 4.10 |
| E4B natural prompt | 12 | 81.4 / 4.07 |
| E4B sampled, seed 17 | 14 | 79.0 / 3.95 |
| E4B sampled, seed 43 | 12 | 78.3 / 3.91 |
| E2B original prompt | 12 pass, 1 unclear | 38.1 / 1.90 |
| E2B literal prompt | 12 pass, 1 unclear | 39.4 / 1.97 |
| E2B natural prompt | 12 pass, 1 unclear | 39.4 / 1.97 |

Pool scores are oracle candidate coverage / first-order primary selection /
order-stable selection. Times are sequential totals over 20 cases: candidate
generation / primary with one judge / stable with two judge orders. They include
reused-baseline generation where applicable and are offline costs, not live
caption latency.

| Pool | Candidates | Passes: oracle / primary / stable | Total seconds: generation / primary / stable |
| --- | --- | ---: | ---: |
| p01 | E4B original + literal | 13 / 12 / 12 | 163.9 / 200.9 / 237.8 |
| p02 | E4B original + literal + natural | 13 / 12 / 11 | 245.3 / 293.6 / 342.7 |
| p03 | E4B original + seeds 17 + 43 | 14 / 11 / 11 | 239.2 / 292.6 / 345.1 |
| p04 | E2B original + literal | 12 / 12 / 12 | 77.5 / 97.5 / 117.5 |
| p05 | E2B original + literal + natural | 12 / 12 / 12 | 116.9 / 137.5 / 158.0 |
| p06 | E4B original + E2B original; E4B judge | 13 / 12 / 12 | 120.0 / 186.2 / 251.8 |
| p07 | E4B original + literal + E2B original; E4B judge | 13 / 12 / 11 | 202.0 / 269.4 / 337.7 |
| p08 | p07 candidates; E2B judge | 13 / 12 / 13 | 202.0 / 236.3 / 270.6 |
| p09 | All five E4B candidates | 14 / 11 / 11 | 402.6 / 467.6 / 531.3 |
| p10 | All eight E4B/E2B candidates | 14 / 11 / 10 | 519.4 / 596.3 / 672.5 |

The best single candidate reached 14/20; the original pool run's best primary and
stable selections reached 12/20 and 13/20. That first p03 report is a completed
measurement: it scored 11/20 primary and stable, counting its protocol-invalid
responses as failures. It was not provisional and did not fail a prior control
gate. Its original artifacts remain unchanged.

A separate ID-0 schema correction reran only the p03 selector, with the same 20
sources, audio hashes, candidate texts and candidate orders. It constrained the
outer response ID to `[0]`; prompts, model, candidate generation and settings did
not change. The new run had 30/30 valid judge responses, all 20 primary and stable
outputs, and 13/20 passes for each selection. It recovers cases 1514 and 1647
versus the original p03 selections, with no regressions. Its passing-case set is
exactly the original E4B baseline's 13/20 set; it remains below the single sampled
E4B seed-17 result at 14/20. New measured costs were 102.96 seconds for 30 judge
calls, 290.93 seconds total sequential primary execution (14.55 seconds/case),
and 342.17 seconds for both judge orders (17.11 seconds/case). These are offline
sequential costs, not browser latency. Of 30 valid judge calls, 27 selected the
first displayed candidate; 12 of 15 two-order pairs changed choice, and the stable
fallback was used in 12 cases. Five cases needed no judge calls because duplicate
candidate texts collapsed to one. See
[`accuracy-ensemble-id0-p03-scores.json`](../test-results/accuracy-ensemble-id0-p03-scores.json)
and [`accuracy-ensemble-id0-p03-development.json`](../test-results/accuracy-ensemble-id0-p03-development.json).
The original ten-pool scores remain in
`test-results/accuracy-ensemble-selection-scores.json`; the Comoro label
sensitivity and root audit are in
[`accuracy-ensemble-root-reconciliation.json`](../test-results/accuracy-ensemble-root-reconciliation.json).

Synthetic controls are outside the 20-case denominator. In the initial control
run, all 12 E4B candidate texts selected across six controls were semantically
acceptable; six replies failed strict protocol solely because the outer response
ID was not literal `0`. The corrected ID-0 control gate then passed all 12/12 raw
choices and protocol checks. This is a schema-compliance correction, not a change
in control meaning. Do not describe E4B control meaning as order-dependent; order
sensitivity applies to natural-case selections. E2B passed both raw orders for all
six controls. Production prompts, model settings and selection behavior remain
unchanged. The 50 held-out recordings remain unused, and the best actual selection
still misses the 18/20 target.
