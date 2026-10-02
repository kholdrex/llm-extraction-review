import numpy as np
import pytest

from xreview.analysis import rules_first
from xreview.data import Ontology, Utterance, parse_annotation
from xreview.detectors import min_field_logprob, score
from xreview.metrics import score as score_record
from xreview.stats import auroc
from xreview.validate import check

ONTO = Ontology(
    intents=("alarm_set", "weather_query"),
    slot_types=("date", "time"),
    allowed={"alarm_set": frozenset({"date", "time"}), "weather_query": frozenset({"date"})},
)
TEXT = "wake me up at five am tomorrow"
GOLD = Utterance("1", "test", "alarm", TEXT, "alarm_set", (("time", "five am"), ("date", "tomorrow")))
GOOD = '{"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}, {"type": "date", "value": "tomorrow"}]}'


def test_parse_annotation():
    assert parse_annotation("wake me up at [time : five am] [date : this week]") == (
        ("time", "five am"),
        ("date", "this week"),
    )


def test_valid_record_passes_all_checks():
    record, issues = check("JSON: " + GOOD, TEXT, ONTO)
    assert issues == []
    assert score_record(record, GOLD).record


def test_checks_detect_form_errors():
    slot = '{"type": "time", "value": "six pm"}'
    raw = f'{{"intent": "weather_query", "slots": [{slot}, {slot}]}}'
    _, issues = check(raw, TEXT, ONTO)
    assert {i.code for i in issues} == {"incompatible_slot", "value_not_in_text", "duplicate_slot"}
    assert check("no json here", TEXT, ONTO)[1][0].code == "parse"


def test_slot_order_and_case_do_not_matter():
    slots = '[{"type": "date", "value": "Tomorrow"}, {"type": "time", "value": "five  am"}]'
    raw = f'{{"intent": "alarm_set", "slots": {slots}}}'
    assert score_record(check(raw, TEXT, ONTO)[0], GOLD).record


def test_field_logprob_ignores_structural_tokens():
    text = '{"intent": "alarm_set"}'
    tokens = [('{"', -5.0), ("intent", 0.0), ('": "', 0.0), ("alarm", -0.2), ("_set", -0.7), ('"}', -4.0)]
    assert min_field_logprob(text, tokens) == -0.7


def test_disagreement_counts_unparsable_samples_as_different():
    greedy = {"text": GOOD, "tokens": [(GOOD, -0.1)]}
    samples = [{"text": GOOD}, {"text": GOOD}, {"text": "oops"}, {"text": GOOD.replace("five am", "five")}]
    _, s = score(greedy, samples, TEXT, ONTO)
    assert s.disagreement == 0.5
    assert s.rules == 0.0


def test_auroc_handles_ties():
    wrong = np.array([True, False, True, False])
    assert auroc(np.array([1.0, 0.0, 1.0, 0.0]), wrong) == 1.0
    assert auroc(np.array([1.0, 1.0, 1.0, 1.0]), wrong) == 0.5


def test_misaligned_log_probabilities_are_rejected():
    with pytest.raises(ValueError):
        min_field_logprob('{"intent": "x"}', [("{", -0.1)])


def test_rules_first_puts_flagged_records_on_top():
    rules = np.array([0.0, 1.0, 0.0, 1.0])
    field = np.array([5.0, 0.1, 3.0, 0.2])
    assert list(np.argsort(-rules_first(rules, field))) == [3, 1, 0, 2]
