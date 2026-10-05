# Caption accuracy target

The target is at least 90% of evaluated speech units preserving the intended
meaning. No current result establishes that target for anime or live captions.

A unit passes when its major propositions, speaker/agent relationships, names,
numbers, polarity and intent survive translation without invented facts.
Natural paraphrases and harmless stylistic differences are allowed. Missing
captions, rejected output and untranslatable fragments count as failures, not
removed rows. Score each complete audio example first; do not inflate the
denominator by splitting easy clauses into extra cases.

Report separately:

- Japanese recognition error: whole-clip character error rate, with NFKC and
  punctuation/whitespace normalization. This is a diagnostic, not English
  meaning accuracy. Overlapping chunk outputs cannot simply be concatenated
  for this metric.
- Translation from the reference Japanese: tests the translator independently
  of recognition mistakes.
- Translation from recognized Japanese: measures the combined pipeline.
- Live usability: missing/repeated captions, timing and sustained backlog.
  Whole-file processing time does not establish live delay.

The supplemental FLEURS corpus has 20 development recordings and 50 held-out
recordings with distinct sentence IDs, selected before model inference. It uses
all aligned IDs in the pinned Japanese/English validation metadata, one audio
recording per ID, a fixed hash ordering, and excludes the previously tested
sentence 1828. The exact selection, source revision, text references and audio
hashes are in `tests/fixtures/accuracy-corpus.json`. Audio is cached under
`test-results/accuracy-audio/`; it is not sent to a remote inference service.

Tune using development cases only. Freeze the candidate and rubric before
opening held-out outputs. A 45/50 result would be an observed 90% on that set,
with substantial sampling uncertainty, not a guarantee on new material. If
held-out errors guide another change, that set becomes development evidence;
use fresh cases for the next independent result. Record evaluator identity and
label model-assisted review as such; it is not human certification.

FLEURS contains read speech. It supplements, but cannot replace, a representative
anime sample including fast dialogue, speaker changes, names, music, silence,
negation, and sentence boundaries. Saved uncertain ASR transcripts cannot serve
as ground truth for what was spoken. Unknown or ambiguous references must be
reported explicitly, with coverage alongside any score.

Source and license: [Google FLEURS](https://huggingface.co/datasets/google/fleurs),
CC BY 4.0. Japanese audio is converted to mono 16 kHz PCM for the existing app;
reference text is retained, and English parallel text is a scoring aid rather
than an exact-output requirement.
