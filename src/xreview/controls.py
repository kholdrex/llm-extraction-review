"""Output format, finer disagreement among samples and scores that use no token probabilities."""

import json
from collections import Counter
from fractions import Fraction

import numpy as np

from .data import Utterance
from .detectors import record_key
from .tokens import json_span
from .validate import normalize, parse


def outside_text(text: str) -> str:
    """Non-whitespace text around the JSON span; the whole reply when it has no span."""
    span = json_span(text)
    if span is None:
        return text.strip()
    return (text[: span[0]] + text[span[1] :]).strip()


def strict_parse(text: str) -> bool:
    """Whether the whole reply is a JSON object, as a strict JSON API would require."""
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def permissive_parse(text: str) -> bool:
    """Whether the JSON span parses, as in the October analysis."""
    try:
        parse(text)
    except json.JSONDecodeError:
        return False
    return True


def bag(record: dict | None) -> Counter | None:
    if record is None:
        return None
    return Counter([("intent", record["intent"])] + [(s["type"], normalize(s["value"])) for s in record["slots"]])


def jaccard_distance(a: dict | None, b: dict | None) -> Fraction:
    """1 − |A ∩ B| / |A ∪ B| for the multisets of intent and typed slots, exact; 1 when either record is missing."""
    x, y = bag(a), bag(b)
    if x is None or y is None:
        return Fraction(1)
    return 1 - Fraction(sum((x & y).values()), sum((x | y).values()))


def entropy(records: list[dict | None]) -> float:
    """Entropy (nats) of the distinct records; unparsable records form one class."""
    counts = np.array(list(Counter(record_key(r) for r in records).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def slot_types(record: dict) -> tuple[str, ...]:
    return tuple(sorted(s["type"] for s in record["slots"]))


def retrieved_disagreement(record: dict | None, neighbours: list[Utterance]) -> tuple[float, float]:
    """Share of the retrieved examples whose intent, and whose multiset of slot types, differs from the record."""
    if record is None:
        return 1.0, 1.0
    intent = np.mean([u.intent != record["intent"] for u in neighbours])
    types = np.mean([tuple(sorted(t for t, _ in u.slots)) != slot_types(record) for u in neighbours])
    return float(intent), float(types)
