"""Sets of tokens in a greedy output and aggregations of their negative log-probabilities."""

import numpy as np

from .detectors import field_spans

MASKS = ("all", "json", "json-open", "field-span", "field")
AGGREGATES = ("worst", "mean", "sum")


def offsets(text: str, tokens: list[tuple[str, float]]) -> list[tuple[int, int]]:
    if "".join(t for t, _ in tokens) != text:
        raise ValueError("token log-probabilities do not match the output text")
    out, pos = [], 0
    for token, _ in tokens:
        out.append((pos, pos + len(token)))
        pos += len(token)
    return out


def overlapping(bounds: list[tuple[int, int]], spans: list[tuple[int, int]]) -> np.ndarray:
    return np.array([any(a < e and b > s for s, e in spans) for a, b in bounds], dtype=bool)


def json_span(text: str) -> tuple[int, int] | None:
    """Characters from the first '{' to the last '}' inclusive, the part the parser reads."""
    start, end = text.find("{"), text.rfind("}")
    return (start, end + 1) if 0 <= start < end else None


def masks(text: str, tokens: list[tuple[str, float]]) -> dict[str, np.ndarray]:
    """Token sets:
    all: every token of the reply;
    json: tokens overlapping the JSON span (every token when the reply has no span);
    json-open: json without the token that contains the opening brace;
    field-span: tokens from the first to the last intent, slot-type or slot-value string, markup between them included;
    field: tokens overlapping the intent, slot-type and slot-value strings (the October L-field).
    """
    bounds = offsets(text, tokens)
    n = len(bounds)
    span = json_span(text)
    inside = overlapping(bounds, [span]) if span else np.ones(n, dtype=bool)
    opening = overlapping(bounds, [(span[0], span[0] + 1)]) if span else np.zeros(n, dtype=bool)
    fields = field_spans(text)
    interval = overlapping(bounds, [(fields[0][0], fields[-1][1])]) if fields else np.zeros(n, dtype=bool)
    return {
        "all": np.ones(n, dtype=bool),
        "json": inside,
        "json-open": inside & ~opening,
        "field-span": interval,
        "field": overlapping(bounds, fields),
    }


def aggregate(nll: np.ndarray, how: str) -> float:
    """Worst (largest), mean or sum of negative log-probabilities; 0 for an empty set, as in L-field."""
    if len(nll) == 0:
        return 0.0
    return float({"worst": nll.max, "mean": nll.mean, "sum": nll.sum}[how]())


def token_scores(text: str, tokens: list[tuple[str, float]]) -> dict[str, float]:
    nll = -np.array([lp for _, lp in tokens])
    return {f"{m} {a}": aggregate(nll[mask], a) for m, mask in masks(text, tokens).items() for a in AGGREGATES}


def worst_position(text: str, tokens: list[tuple[str, float]]) -> str:
    """Where the least probable token of the reply lies: before the JSON span, in the token containing the opening
    brace, inside the rest of the span, or after it."""
    nll = -np.array([lp for _, lp in tokens])
    i = int(nll.argmax())
    m = masks(text, tokens)
    if m["json-open"][i]:
        return "inside"
    if m["json"][i]:
        return "opening"
    return "before" if offsets(text, tokens)[i][1] <= json_span(text)[0] else "after"


def worst_in_span_is_opening(text: str, tokens: list[tuple[str, float]]) -> bool:
    """Whether, among the tokens of the JSON span, the least probable one contains the opening brace."""
    nll = -np.array([lp for _, lp in tokens])
    m = masks(text, tokens)
    inside = np.flatnonzero(m["json"])
    return not m["json-open"][inside[nll[inside].argmax()]]
