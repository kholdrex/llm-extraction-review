from fractions import Fraction

import numpy as np
import pytest

from xreview.controls import (
    entropy,
    jaccard_distance,
    outside_text,
    permissive_parse,
    retrieved_disagreement,
    strict_parse,
)
from xreview.data import Utterance
from xreview.detectors import min_field_logprob
from xreview.reanalysis import budget_row, caught
from xreview.stats import auroc, cross_fitted, ecdf, out_of_fold
from xreview.tokens import aggregate, json_span, masks, token_scores, worst_in_span_is_opening, worst_position

TEXT = 'JSON: {"intent": "alarm_set", "slots": []}'
TOKENS = [
    ("JSON", -0.01),
    (":", -0.02),
    (' {"', -3.0),
    ("intent", 0.0),
    ('":', 0.0),
    (' "', -0.1),
    ("alarm", -0.5),
    ("_set", -0.2),
    ('",', -0.05),
    (' "', 0.0),
    ("slots", 0.0),
    ('":', 0.0),
    (" []", -0.4),
    ("}", -0.01),
]
RECORD = {"intent": "alarm_set", "slots": [{"type": "time", "value": "Five am"}, {"type": "date", "value": "today"}]}


def test_json_span_covers_first_to_last_brace():
    assert json_span(TEXT) == (6, len(TEXT))
    assert json_span("no record") is None
    assert json_span("} {") is None


def test_token_masks():
    m = masks(TEXT, TOKENS)
    assert m["all"].all()
    assert list(np.flatnonzero(~m["json"])) == [0, 1]
    assert list(np.flatnonzero(~m["json-open"])) == [0, 1, 2]
    assert list(np.flatnonzero(m["field"])) == [6, 7]
    assert list(np.flatnonzero(m["field-span"])) == [6, 7]


def test_field_span_keeps_markup_between_fields():
    text = '{"intent": "a", "slots": [{"type": "b", "value": "c"}]}'
    tokens = [(c, -0.1) for c in text]
    m = masks(text, tokens)
    assert m["field-span"].sum() == text.index('"c"') + 2 - text.index('"a"') - 1
    assert m["field"].sum() == 3


def test_reply_without_record_uses_every_token():
    m = masks("sorry", [("sorry", -1.0)])
    assert m["json"].all() and m["json-open"].all()
    assert not m["field"].any()


def test_field_worst_equals_published_score():
    scores = token_scores(TEXT, TOKENS)
    assert scores["field worst"] == -min_field_logprob(TEXT, TOKENS) == 0.5
    assert scores["field sum"] == pytest.approx(0.7)
    assert scores["json worst"] == 3.0
    assert scores["json-open worst"] == 0.5
    assert scores["all sum"] == pytest.approx(-sum(lp for _, lp in TOKENS))


def test_aggregate():
    nll = np.array([0.1, 0.5, 0.3])
    assert aggregate(nll, "worst") == 0.5
    assert aggregate(nll, "mean") == pytest.approx(0.3)
    assert aggregate(nll, "sum") == pytest.approx(0.9)
    assert aggregate(np.array([]), "sum") == 0.0


def test_misaligned_tokens_are_rejected():
    with pytest.raises(ValueError):
        masks(TEXT, TOKENS[:-1])


def test_worst_token_position():
    assert worst_position(TEXT, TOKENS) == "opening"
    assert worst_in_span_is_opening(TEXT, TOKENS)
    before = [("JSON", -4.0)] + TOKENS[1:]
    assert worst_position(TEXT, before) == "before"
    assert worst_in_span_is_opening(TEXT, before)
    inside = TOKENS[:6] + [("alarm", -5.0)] + TOKENS[7:]
    assert worst_position(TEXT, inside) == "inside"
    assert not worst_in_span_is_opening(TEXT, inside)
    after = TOKENS + [("\n```", -6.0)]
    assert worst_position(TEXT + "\n```", after) == "after"


def test_format_of_reply():
    assert outside_text(TEXT) == "JSON:"
    assert outside_text('```json\n{"a": 1}\n```') == "```json\n\n```"
    assert outside_text(' {"a": 1}\n') == ""
    assert outside_text("none") == "none"
    assert strict_parse(' {"a": 1}\n') and not strict_parse(TEXT) and not strict_parse("[1]")
    assert permissive_parse(TEXT) and not permissive_parse('{"a": }')


def test_jaccard_distance():
    other = {"intent": "alarm_set", "slots": [{"type": "time", "value": "five  am"}]}
    assert jaccard_distance(RECORD, RECORD) == 0.0
    assert jaccard_distance(RECORD, other) == Fraction(1, 3)
    assert jaccard_distance(RECORD, None) == 1.0
    assert jaccard_distance({"intent": "x", "slots": []}, {"intent": "y", "slots": []}) == 1.0


def test_entropy_of_records():
    other = {"intent": "weather_query", "slots": []}
    assert entropy([RECORD] * 4) == 0.0
    assert entropy([RECORD, RECORD, other, None]) == pytest.approx(-0.5 * np.log(0.5) - 2 * 0.25 * np.log(0.25))
    assert entropy([None, None]) == 0.0


def test_retrieved_disagreement():
    neighbours = [
        Utterance("1", "train", "alarm", "", "alarm_set", (("date", "today"), ("time", "six"))),
        Utterance("2", "train", "alarm", "", "alarm_set", (("time", "six"),)),
        Utterance("3", "train", "alarm", "", "alarm_query", (("time", "six"), ("date", "today"))),
        Utterance("4", "train", "weather", "", "weather_query", ()),
    ]
    assert retrieved_disagreement(RECORD, neighbours) == (0.5, 0.5)
    assert retrieved_disagreement(None, neighbours) == (1.0, 1.0)


def test_ecdf_uses_mid_ranks_of_reference():
    reference = np.array([[1.0], [2.0], [2.0], [3.0]])
    assert list(ecdf(reference, np.array([[0.0], [2.0], [5.0]]))[:, 0]) == [0.0, 0.5, 1.0]


def test_out_of_fold_fits_the_transform_on_training_folds_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = x[:, 0] + rng.normal(size=200) > 0
    held_out = np.flatnonzero(np.random.default_rng(1).permutation(200) % 5 == 0)
    changed = x.copy()
    changed[held_out[0]] = 50.0
    assert np.allclose(out_of_fold(x, y, 5, 1)[held_out[1:]], out_of_fold(changed, y, 5, 1)[held_out[1:]])
    assert not np.allclose(cross_fitted(x, y, 5, 1)[held_out[1:]], cross_fitted(changed, y, 5, 1)[held_out[1:]])
    assert auroc(out_of_fold(x, y, 5, 1), y) > 0.75


def test_caught_and_accepted_error():
    score = np.array([0.9, 0.8, 0.1, 0.5, 0.5])
    wrong = np.array([True, False, False, True, False])
    tie = np.array([0.0, 0.0, 0.0, 0.2, 0.1])
    assert caught(score, wrong, 0.4, tie) == 1
    assert caught(score, wrong, 0.6, tie) == 1
    assert caught(score, wrong, 0.6, np.array([0.0, 0.0, 0.0, 0.1, 0.2])) == 2
    row = budget_row(np.arange(10.0), np.arange(10) >= 7, np.zeros(10))
    assert row["0.1"]["caught_per_1000"] == 100.0
    assert row["0.3"]["accepted_error"] == 0.0
