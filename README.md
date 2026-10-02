# Error signals for language-model information extraction

Code, prompts, raw model outputs and analysis for the paper *Detecting erroneous records in information extraction by
small language models* (O. Kholodniak).

Three small local language models turn requests to a voice assistant into JSON records with an intent and typed
slots. Without the reference annotation, each record gets an error score, and the records with the highest scores
are sent to manual review. The study compares three kinds of score:

- **R**, rule-based validation: valid JSON, known intent and slot types, slot types seen with the intent in the
  training data, slot values present in the request, no duplicates (`validate.py`);
- **D**, disagreement: share of four sampled records (temperature 0.7) that differ from the greedy record;
- **L**, token probabilities of the greedy output: negative log-probability of the whole output (`L_seq`) and the
  negative of the lowest log-probability among the tokens of the intent, slot types and slot values (`L_field`)
  (`detectors.py`).

Outcomes are AUROC for detecting wrong records and the error left among accepted records when 10–30% of the records
go to review.

## Data and models

- [MASSIVE 1.1](https://github.com/alexa/massive), locale en-US, CC BY 4.0. The archive is downloaded and checked by
  SHA-256 on first use. 1000 test requests are drawn with seed 2026 (`data/sample.json`).
- `qwen2.5:3b-instruct` (digest 357c53fb659c), `gemma3:4b` (a2af6cc3eb7f) and `qwen2.5:7b-instruct` (845dbda0ea48)
  in Ollama 0.34.3, Q4_K_M, on an Apple M1 Pro with 16 GB; context 4096 tokens, at most 256 output tokens.
- Prompt: `data/system_prompt.txt` plus the eight most similar training requests (TF-IDF) in the user message.

## Layout

```
src/xreview/   data, validation, prompts, Ollama client, extraction runner, error scores, error kinds, statistics,
               analysis
scripts/       Fig. 1 (pipeline diagram)
results/       raw_test.jsonl (all model outputs with token log-probabilities), summary.json;
               pilot_*_dev.jsonl: 40 development requests, Qwen2.5-3B, with fixed and with retrieved examples
figures/       figures of the paper
tests/         unit tests
PROTOCOL.md    analysis plan written before the test run; protocol_freeze.sha256 holds the hashes at that time
CHANGELOG.md   changes after the freeze
```

## Usage

Python 3.10+ and [uv](https://docs.astral.sh/uv/). Install from the checkout (editable), because the commands read
`data/` and `results/` relative to it; `requirements.lock.txt` lists the exact package versions used.

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run xreview analyze             # results/summary.json and figures/ from the stored outputs
uv run python scripts/draw_scheme.py
```

Re-running the models needs Ollama with the three models pulled. `uv run xreview run test --out runs/test.jsonl`
writes a new run to its own file and resumes it if interrupted (resuming assumes unchanged prompts, models and
settings); without `--out` the command resumes `results/raw_test.jsonl`, which is already complete. A new run is
analysed with `uv run xreview analyze --raw runs/test.jsonl --figures runs/figures`. Sampling in Ollama is seeded,
but outputs can differ between Ollama versions and hardware.

## License

Code: MIT. The stored outputs contain MASSIVE requests (CC BY 4.0); see `THIRD_PARTY_NOTICES.md`.
