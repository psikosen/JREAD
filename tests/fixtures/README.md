# Japanese speech fixture

`japanese-speech.wav` is test row 0 (ID 1828) from Google FLEURS, config `ja_jp`.

- Source: [Google FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Row source: [FLEURS `ja_jp` test parquet](https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/ja_jp/test/0000.parquet)

The embedded audio was decoded and rewritten as mono PCM WAV without changing its samples. Its Japanese reference text and source row metadata are in `japanese-speech.json`.

`japanese-demo.mp4` is a locally generated JREAD title card with an AAC-encoded copy of that FLEURS audio. It contains no third-party video footage; attribution for its audio remains the FLEURS attribution above.

These 10.44-second fixtures test pipeline execution. They do not establish anime translation quality, full-episode performance, alignment accuracy, or live latency.

`accuracy-corpus.json` freezes 70 additional, distinct Japanese/English FLEURS
sentence IDs from the validation split: 20 for development and 50 held out.
The manifest records the pinned dataset revision, reference text, source rows,
selection method and audio hashes. `tests/prepare_accuracy_corpus.py` downloads
only selected Japanese audio and converts it to mono 16 kHz PCM under the ignored
`test-results/accuracy-audio/` directory. The same CC BY 4.0 attribution applies.
See [accuracy criteria](../../docs/accuracy-target.md) before interpreting scores.
