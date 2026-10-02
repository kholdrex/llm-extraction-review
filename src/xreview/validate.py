"""Checks that an information system can run on a model output without knowing the correct answer."""

import json
from dataclasses import dataclass

from .data import Ontology


@dataclass(frozen=True)
class Issue:
    path: str
    code: str
    message: str


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def parse(raw: str) -> object:
    """The JSON object between the first '{' and the last '}', so that labels or code fences around it are ignored."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object", raw, 0)
    return json.loads(raw[start : end + 1])


def check(raw: str, utterance: str, onto: Ontology) -> tuple[dict | None, list[Issue]]:
    """Parse a raw output and list every violated check. Returns the parsed record when its structure is valid."""
    try:
        obj = parse(raw)
    except json.JSONDecodeError as e:
        return None, [Issue("$", "parse", f"output is not valid JSON ({e.msg})")]

    issues = structure_issues(obj)
    if issues:
        return None, issues

    issues = []
    intent = obj["intent"]
    if intent not in onto.intents:
        issues.append(Issue("intent", "unknown_intent", f"'{intent}' is not one of the allowed intents"))
    text = normalize(utterance)
    seen = set()
    for i, slot in enumerate(obj["slots"]):
        path = f"slots[{i}]"
        kind, value = slot["type"], slot["value"]
        if kind not in onto.slot_types:
            issues.append(Issue(path, "unknown_slot_type", f"'{kind}' is not one of the allowed slot types"))
        elif intent in onto.allowed and kind not in onto.allowed[intent]:
            issues.append(Issue(path, "incompatible_slot", f"slot type '{kind}' is not used with intent '{intent}'"))
        if not normalize(value) or normalize(value) not in text:
            issues.append(Issue(path, "value_not_in_text", f"value '{value}' does not occur in the request"))
        key = (kind, normalize(value))
        if key in seen:
            issues.append(Issue(path, "duplicate_slot", f"slot ({kind}, '{value}') is repeated"))
        seen.add(key)
    return obj, issues


def structure_issues(obj: object) -> list[Issue]:
    if not isinstance(obj, dict):
        return [Issue("$", "schema", "output must be a JSON object")]
    issues = []
    extra = set(obj) - {"intent", "slots"}
    if extra:
        issues.append(Issue("$", "schema", f"unexpected keys: {', '.join(sorted(extra))}"))
    if not isinstance(obj.get("intent"), str):
        issues.append(Issue("intent", "schema", "'intent' must be a string"))
    slots = obj.get("slots")
    if not isinstance(slots, list):
        issues.append(Issue("slots", "schema", "'slots' must be a list"))
        return issues
    for i, slot in enumerate(slots):
        ok = isinstance(slot, dict) and set(slot) == {"type", "value"}
        ok = ok and isinstance(slot["type"], str) and isinstance(slot["value"], str)
        if not ok:
            issues.append(Issue(f"slots[{i}]", "schema", "each slot must be an object with string 'type' and 'value'"))
    return issues
