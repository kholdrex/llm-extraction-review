"""Error scores for one extracted record that need no gold annotation. Higher score = more likely wrong."""

import re
from dataclasses import dataclass

from .data import Ontology
from .validate import check, normalize

FIELD_RE = re.compile(r'"(intent|type|value)"\s*:\s*"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Scores:
    rules: float
    disagreement: float
    seq_nll: float
    field_nll: float


def record_key(record: dict | None) -> tuple | None:
    if record is None:
        return None
    return record["intent"], tuple(sorted((s["type"], normalize(s["value"])) for s in record["slots"]))


def field_spans(text: str) -> list[tuple[int, int]]:
    return [m.span(2) for m in FIELD_RE.finditer(text)]


def min_field_logprob(text: str, tokens: list[tuple[str, float]]) -> float:
    """Lowest token log-probability among tokens that overlap the intent, slot-type or slot-value strings."""
    if "".join(t for t, _ in tokens) != text:
        raise ValueError("token log-probabilities do not match the output text")
    spans = field_spans(text)
    lowest, pos = 0.0, 0
    for token, lp in tokens:
        end = pos + len(token)
        if any(pos < b and end > a for a, b in spans):
            lowest = min(lowest, lp)
        pos = end
    return lowest


def score(greedy: dict, samples: list[dict], utterance: str, onto: Ontology) -> tuple[dict | None, Scores]:
    record, issues = check(greedy["text"], utterance, onto)
    key = record_key(record)
    sample_keys = [record_key(check(s["text"], utterance, onto)[0]) for s in samples]
    agree = sum(k is not None and k == key for k in sample_keys) / len(sample_keys)
    seq = -sum(lp for _, lp in greedy["tokens"])
    field = -min_field_logprob(greedy["text"], greedy["tokens"])
    return record, Scores(float(bool(issues)), 1.0 - agree, seq, field)
