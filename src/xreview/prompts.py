"""Extraction prompt: the ontology, output rules and optional worked examples."""

import json

from .data import Ontology, Utterance

SYSTEM = """You convert requests to a voice assistant into a JSON record of the form
{{"intent": "<intent>", "slots": [{{"type": "<slot type>", "value": "<words from the request>"}}]}}

Rules:
- "intent" is exactly one of the allowed intents.
- Every slot "type" is exactly one of the allowed slot types.
- Every slot "value" is copied word for word from the request.
- "slots" is an empty list when the request mentions no slot.
- Output the JSON record only.

Allowed intents: {intents}

Allowed slot types: {slot_types}"""


def record(u: Utterance) -> dict:
    return {"intent": u.intent, "slots": [{"type": t, "value": v} for t, v in u.slots]}


def system_prompt(onto: Ontology, shots: list[Utterance]) -> str:
    text = SYSTEM.format(intents=", ".join(onto.intents), slot_types=", ".join(onto.slot_types))
    if not shots:
        return text
    return text + "\n\nExamples:\n" + example_block(shots)


def example_block(shots: list[Utterance]) -> str:
    return "\n".join(f"Request: {u.text}\nJSON: {json.dumps(record(u))}" for u in shots)


def messages(system: str, text: str, similar: list[Utterance]) -> list[dict]:
    user = f"Similar annotated requests:\n{example_block(similar)}\n\nRequest: {text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
