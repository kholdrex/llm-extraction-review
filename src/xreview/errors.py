"""Kinds of error in an extracted record compared with the gold annotation."""

from collections import Counter

from .data import Utterance
from .validate import normalize

KINDS = ("unparsable", "intent", "missing slot", "extra slot", "slot type", "slot boundary")


def kinds(record: dict | None, gold: Utterance) -> set[str]:
    """All kinds of error present in a record; an empty set means the record is correct."""
    if record is None:
        return {"unparsable"}
    found = set()
    if record["intent"] != gold.intent:
        found.add("intent")
    pred = Counter((s["type"], normalize(s["value"])) for s in record["slots"])
    truth = Counter((t, normalize(v)) for t, v in gold.slots)
    spare_pred = list((pred - truth).elements())
    spare_truth = list((truth - pred).elements())
    for t, v in list(spare_truth):
        same_value = next((p for p in spare_pred if p[1] == v), None)
        overlap = next((p for p in spare_pred if p[0] == t and (p[1] in v or v in p[1])), None)
        if same_value is not None:
            found.add("slot type")
            spare_pred.remove(same_value)
            spare_truth.remove((t, v))
        elif overlap is not None:
            found.add("slot boundary")
            spare_pred.remove(overlap)
            spare_truth.remove((t, v))
    if spare_truth:
        found.add("missing slot")
    if spare_pred:
        found.add("extra slot")
    return found
