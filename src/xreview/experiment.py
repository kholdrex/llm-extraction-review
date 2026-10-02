"""Greedy extraction with token log-probabilities plus K sampled extractions per utterance, stored as JSONL."""

import json
from dataclasses import asdict
from pathlib import Path

from . import prompts
from .data import Utterance
from .llm import chat
from .retrieval import Retriever

SAMPLES = 4
SAMPLE_TEMPERATURE = 0.7


def run_item(model: str, u: Utterance, system: str, retriever: Retriever) -> dict:
    messages = prompts.messages(system, u.text, retriever.nearest(u.text))
    greedy = chat(model, messages, logprobs=True)
    samples = [chat(model, messages, temperature=SAMPLE_TEMPERATURE, seed=s) for s in range(1, SAMPLES + 1)]
    return {"model": model, "id": u.id, "greedy": asdict(greedy), "samples": [asdict(s) for s in samples]}


def load_rows(path: Path) -> list[dict]:
    """Stored records; a last line cut off by an interrupted write is ignored."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def run(models: list[str], items: list[Utterance], system: str, retriever: Retriever, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["model"], r["id"]) for r in load_rows(out)}
    with out.open("a") as f:
        for model in models:
            for u in items:
                if (model, u.id) not in done:
                    f.write(json.dumps(run_item(model, u, system, retriever)) + "\n")
                    f.flush()
                    done.add((model, u.id))
