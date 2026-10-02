"""Scoring of final records against the MASSIVE gold annotation."""

from collections import Counter
from dataclasses import dataclass

from .data import Utterance
from .validate import normalize


@dataclass(frozen=True)
class Score:
    record: bool
    intent: bool
    tp: int
    fp: int
    fn: int


def slot_bag(slots) -> Counter:
    return Counter((t, normalize(v)) for t, v in slots)


def score(record: dict | None, gold: Utterance) -> Score:
    truth = slot_bag(gold.slots)
    if record is None:
        return Score(False, False, 0, 0, sum(truth.values()))
    pred = slot_bag((s["type"], s["value"]) for s in record["slots"])
    tp = sum((pred & truth).values())
    intent = record["intent"] == gold.intent
    return Score(intent and pred == truth, intent, tp, sum(pred.values()) - tp, sum(truth.values()) - tp)
