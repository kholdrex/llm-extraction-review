import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from xreview import llm, snips
from xreview import replication as rep
from xreview import replication_analysis as ra
from xreview.data import Utterance, ontology
from xreview.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
HAS_MASSIVE = (ROOT / "data" / "massive").exists()
HAS_DATA = HAS_MASSIVE and (ROOT / "data" / "snips").exists()
RIGHT = '{"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]}'
WRONG = '{"intent": "alarm_query", "slots": []}'


def test_slots_from_bio():
    words = "play the new album by sia on spotify".split()
    tags = "O O B-sort B-music_item O B-artist O B-service".split()
    assert snips.slots_from_bio(words, tags) == (("sort", "new"), ("music_item", "album"), ("artist", "sia"),
                                                 ("service", "spotify"))
    assert snips.slots_from_bio(["a", "b", "c"], ["B-x", "I-x", "I-y"]) == (("x", "a b"), ("y", "c"))
    assert snips.slots_from_bio(["a", "b"], ["I-x", "I-x"]) == (("x", "a b"),)
    assert snips.slots_from_bio(["7", "day"], ["B-timeRange", "B-timeRange"]) == (("timeRange", "7"),
                                                                                ("timeRange", "day"))
    with pytest.raises(ValueError):
        snips.slots_from_bio(["a"], ["O", "O"])


@pytest.mark.skipif(not HAS_DATA, reason="SNIPS not downloaded")
def test_adjacent_begin_tags_in_snips():
    item = {u.id: u for u in rep.snips_rows()}["train-2080"]
    assert ("timeRange", "7") in item.slots and ("timeRange", "day") in item.slots


def write_snips(dest: Path) -> None:
    content = {
        "train": ("add sia to my playlist \nrate it 5\n", "O B-artist O O B-playlist \nO O B-rating_value\n",
                  "AddToPlaylist\nRateBook\n"),
        "valid": ("rate it 5\n", "O O B-rating_value\n", "RateBook\n"),
        "test": ("add sia now\n", "O B-artist O\n", "AddToPlaylist\n"),
    }
    for split, files in content.items():
        (dest / split).mkdir(parents=True)
        for name, text in zip(snips.FILES, files, strict=True):
            (dest / split / name).write_text(text)


def test_snips_load(tmp_path):
    write_snips(tmp_path)
    rows = snips.load(tmp_path)
    assert [(u.id, u.partition) for u in rows] == [("train-0", "train"), ("train-1", "train"), ("valid-0", "dev"),
                                                   ("test-0", "test")]
    assert rows[0].text == "add sia to my playlist"
    assert rows[0].slots == (("artist", "sia"), ("playlist", "playlist"))
    assert rows[0].scenario == rows[0].intent == "AddToPlaylist"
    assert snips.line("test-17") == 17


def test_snips_checksums_are_enforced(tmp_path):
    write_snips(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        snips.download(tmp_path)


def utterance(i: int, partition: str, text: str | None = None) -> Utterance:
    return Utterance(str(i), partition, "s", text or f"request number {i}", "x", (("t", "request"),))


def synthetic():
    massive = ([utterance(i, "train", f"train {i}") for i in range(30)]
               + [utterance(i, "test") for i in range(100, 2700)]
               + [utterance(i, "dev") for i in range(3000, 3400)]
               + [utterance(5000, "test", "train 3"), utterance(5001, "dev", "Train  4"),
                  utterance(5002, "test", "request number 100"), utterance(5003, "test", "request number 3150"),
                  utterance(5004, "test", "request number 3105")])
    snips_rows = ([Utterance(f"train-{i}", "train", "A", f"s train {i}", "A", (("t", "s"),)) for i in range(20)]
                  + [Utterance(f"valid-{i}", "dev", "A", f"s dev {i}", "A", ()) for i in range(150)]
                  + [Utterance("valid-150", "dev", "A", "s test 3", "A", ())]
                  + [Utterance(f"test-{i}", "test", "A", f"s test {i}", "A", ()) for i in range(40)]
                  + [Utterance("test-40", "test", "A", "s train 2", "A", ()),
                     Utterance("test-41", "test", "A", "s test 5", "A", ())])
    october = {"test": [str(i) for i in range(100, 1100)], "dev": [str(i) for i in range(3000, 3100)]}
    gate = tuple(str(i) for i in range(3100, 3120))
    return massive, snips_rows, october, gate


def test_manifest_is_stable_and_text_disjoint():
    massive, snips_rows, october, gate = synthetic()
    m = rep.manifest(massive, snips_rows, october, gate)
    assert m == rep.manifest(massive, snips_rows, october, gate)
    sets = m["sets"]
    assert {k: len(v) for k, v in sets.items()} == {"snips_test": 42, "massive_test": 500, "snips_dev": 100,
                                                    "massive_dev": 100, "format_arm": 200}
    assert m["pools"]["massive_test"] == 1601 and m["pools"]["snips_dev"] == 150
    assert "5000" not in sets["massive_test"] and "5002" not in sets["massive_test"]
    assert "5001" not in sets["massive_dev"] and not set(gate) & set(sets["massive_dev"])
    assert not set(sets["massive_dev"]) & set(october["dev"])
    texts = {u.id: u.text for u in massive}
    assert not {texts[i] for i in sets["massive_test"]} & {texts[i] for i in sets["massive_dev"]}
    assert "valid-150" not in sets["snips_dev"]
    assert m["snips_test_train_matches"] == ["test-40"] and "test-40" in sets["snips_test"]
    assert m["snips_test_text_duplicates"] == [["test-5", "test-41"]]
    assert sets["massive_test"] == sorted(sets["massive_test"], key=int)
    assert set(sets["format_arm"]) <= set(sets["massive_test"])
    assert m["gate"] == list(gate)


@pytest.mark.skipif(not HAS_DATA, reason="MASSIVE and SNIPS not downloaded")
def test_stored_manifest_regenerates_and_matches_the_protocol():
    october = json.loads((ROOT / "data" / "sample.json").read_text())
    stored = rep.load_manifest()
    assert rep.manifest(rep.massive_rows(), rep.snips_rows(), october, rep.GATE_IDS) == stored
    assert stored["pools"] == {"massive_test": 1958, "massive_dev": 1898, "snips_test": 700, "snips_dev": 684}
    assert len(stored["snips_test_train_matches"]) == 26
    gate = [json.loads(line)["id"] for line in (ROOT / "results/replication/gate.jsonl").open()]
    assert gate == list(rep.GATE_IDS)


def test_manifest_hash_is_enforced(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError):
        rep.load_manifest(path, "0" * 64)
    assert rep.load_manifest(path, rep.sha256(path)) == {}


def small_dataset(name: str = "massive") -> rep.Dataset:
    train = [Utterance(f"tr{i}", "train", "alarm", f"wake me at {i}", "alarm_set", (("time", str(i)),))
             for i in range(10)]
    items = {u.id: u for u in train}
    for i in range(60):
        slots = (("time", "five am"),) if i % 2 == 0 else (("time", "x"),)
        items[str(i)] = Utterance(str(i), "test", "alarm", f"wake me at five am {i}", "alarm_set", slots)
    return rep.Dataset(name, items, ontology(train), train[:2], Retriever(train, 3))


def test_bare_arm_differs_only_by_the_label():
    ds = small_dataset()
    labelled = rep.messages(ds, ds.items["0"], "labelled")
    bare = rep.messages(ds, ds.items["0"], "bare")
    assert "\nJSON: " in labelled[0]["content"] and "\nJSON: " in labelled[1]["content"]
    for a, b in zip(labelled, bare, strict=True):
        assert a["content"].replace("\nJSON: ", "\n") == b["content"]
    with pytest.raises(ValueError):
        rep.messages(ds, ds.items["0"], "other")


@pytest.mark.skipif(not HAS_MASSIVE, reason="MASSIVE not downloaded")
def test_labelled_arm_is_the_october_prompt():
    ds = rep.build("massive", rep.massive_rows())
    system = rep.messages(ds, next(iter(ds.items.values())), "labelled")[0]["content"]
    assert system + "\n" == (ROOT / "data" / "system_prompt.txt").read_text()


def test_request_is_the_october_request(monkeypatch):
    sent = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": RIGHT}, "logprobs": [{"token": RIGHT, "logprob": -0.1}],
                    "done_reason": "stop"}

    monkeypatch.setattr("requests.post", lambda url, json, timeout: sent.append(json) or Response())
    msgs = [{"role": "user", "content": "x"}]
    for kwargs in (rep.settings(0), rep.settings(3)):
        llm.chat("m", msgs, **kwargs)
        reply = rep.ollama_chat("m", msgs, **kwargs)
        assert sent[-1] == sent[-2]
    assert reply["done_reason"] == "stop"


def test_reply_problem():
    assert rep.reply_problem({"text": RIGHT, "tokens": [[RIGHT, -0.1]]}, 0) is None
    assert rep.reply_problem({"text": RIGHT, "tokens": []}, 1) is None
    for reply in ({"text": "", "tokens": []}, {"text": RIGHT, "tokens": [["{", -0.1]]},
                  {"text": RIGHT, "tokens": [[RIGHT, math.nan]]}, {"text": RIGHT, "tokens": [[RIGHT, -math.inf]]},
                  {"text": RIGHT, "tokens": [[None, -0.1]]}, {"text": RIGHT, "tokens": [[RIGHT, None]]},
                  {"text": RIGHT, "tokens": [[RIGHT, True]]}, {"text": None, "tokens": []}, [1]):
        assert rep.reply_problem(reply, 0)
    assert rep.reply_problem({"text": None, "tokens": []}, 2)


def header(set_name, arm="labelled", invocation="i", models=None, manifest="m", thresholds="t", commit="c"):
    phase = rep.SETS[set_name][1]
    models = list(models or rep.roster(set_name, arm))
    h = {"invocation": invocation, "phase": phase, "set": set_name, "arm": arm, "models": models,
         "manifest_sha256": manifest, "commit": commit, "ollama": rep.OLLAMA_VERSION,
         "digests": {m: rep.DIGESTS.get(m) for m in models}}
    if phase in ("test", "format"):
        h["thresholds_sha256"] = thresholds
    return h


def run(client, set_name, ds, ids, out, arm="labelled", invocation="i", **kwargs):
    h = header(set_name, arm, invocation, **kwargs)
    return rep.run(client, h, ds, ids, out, manifest_sha=h["manifest_sha256"],
                   thresholds_sha=h.get("thresholds_sha256"))


class FakeClient:
    def __init__(self, stop_after=None, failures=0, invalid=0, text=None):
        self.calls, self.stop_after, self.failures, self.invalid, self.text = [], stop_after, failures, invalid, text

    def __call__(self, model, messages, temperature, seed, logprobs):
        if self.stop_after is not None and len(self.calls) == self.stop_after:
            raise KeyboardInterrupt
        self.calls.append((model, logprobs, seed))
        if self.failures:
            self.failures -= 1
            raise ValueError("server error")
        if self.invalid:
            self.invalid -= 1
            return {"text": RIGHT, "tokens": [], "seconds": 0.1}
        text = self.text(model, messages, seed) if self.text else RIGHT
        return {"text": text, "tokens": [[text, -0.1 - seed / 10]] if logprobs else [], "seconds": 0.1}


P1, P2 = rep.PRIMARY


def test_run_resumes_inside_an_item_without_repeating_calls(tmp_path):
    ds = small_dataset()
    out = tmp_path / "test.jsonl"
    first = FakeClient(stop_after=7)
    with pytest.raises(KeyboardInterrupt):
        run(first, "massive_test", ds, ["0", "1"], out, invocation="i1")
    second = FakeClient()
    run(second, "massive_test", ds, ["0", "1"], out, invocation="i2")
    rows, tail = rep.read_rows(out)
    keys = [rep.key(r) for r in rows]
    assert len(keys) == len(set(keys)) == 20 and tail == ""
    assert len(first.calls) + len(second.calls) == 20
    assert second.calls[0] == (P1, False, 2)
    assert [h["invocation"] for h in rep.read_rows(rep.meta_path(out))[0]] == ["i1", "i2"]
    assert not rep.admission([out], "m", "t")[1]


def test_failed_attempts_are_kept_and_retried(tmp_path):
    ds = small_dataset()
    out = tmp_path / "format.jsonl"
    run(FakeClient(failures=1, invalid=1), "format_arm", ds, ["0"], out, arm="labelled")
    rows, _ = rep.read_rows(out)
    assert [(r["status"], r["attempt"]) for r in rows[:3]] == [("error", 1), ("invalid", 2), ("ok", 3)]
    assert rows[1]["text"] == RIGHT


def test_malformed_server_replies_are_recorded_attempts(tmp_path, monkeypatch):
    payloads = [[1], {"message": {"content": RIGHT}, "logprobs": [{"token": None, "logprob": -0.1}]},
                {"message": {"content": RIGHT}, "logprobs": [{"token": RIGHT, "logprob": -0.1}]},
                {"message": {"content": None}}, {"message": {"content": RIGHT}}]

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    monkeypatch.setattr("requests.post", lambda url, json, timeout: Response(payloads.pop(0)))
    out = tmp_path / "test.jsonl"
    fallback = FakeClient()

    def client(*args, **kwargs):
        return rep.ollama_chat(*args, **kwargs) if payloads else fallback(*args, **kwargs)

    run(client, "massive_test", small_dataset(), ["0"], out)
    rows, _ = rep.read_rows(out)
    assert [(r["call"], r["status"]) for r in rows[:5]] == [(0, "error"), (0, "invalid"), (0, "ok"), (1, "invalid"),
                                                            (1, "ok")]


def test_no_fourth_attempt_after_three_failures(tmp_path):
    ds = small_dataset()
    out = tmp_path / "format.jsonl"
    result = run(FakeClient(failures=3), "format_arm", ds, ["0", "1"], out, arm="labelled")
    first = rep.FORMAT_ONLY[0]
    assert result["exhausted"] == [["format", "massive", "labelled", first, "0", 0]]
    client = FakeClient()
    result = run(client, "format_arm", ds, ["0", "1"], out, arm="labelled", invocation="i2")
    assert client.calls == [] and not result["header_written"]
    assert len(rep.read_rows(rep.meta_path(out))[0]) == 1
    rows, _ = rep.read_rows(out)
    assert sum(r["model"] == first for r in rows) == 3


def test_success_after_three_failures_is_not_admitted(tmp_path):
    ds = small_dataset()
    out = tmp_path / "format.jsonl"
    run(FakeClient(failures=3), "format_arm", ds, ["0"], out, arm="labelled")
    rows, _ = rep.read_rows(out)
    late = dict(rows[0], status="ok", attempt=4, text=RIGHT, tokens=[[RIGHT, -0.1]])
    with out.open("a") as f:
        f.write(json.dumps(late) + "\n")
    rows, problems = rep.admission([out], "m", "t")
    assert problems["success after three failed attempts"] == 1
    assert ra.select(rows, [rep.key(late)])[1] == {"inconsistent": [list(rep.key(late))]}
    with pytest.raises(RuntimeError, match="not admitted"):
        run(FakeClient(), "format_arm", ds, ["0"], out, arm="labelled", invocation="i2")


def test_row_must_match_its_header(tmp_path):
    ds = small_dataset()
    out = tmp_path / "format.jsonl"
    run(FakeClient(), "format_arm", ds, ["0"], out, arm="bare")
    rows, _ = rep.read_rows(out)
    stray = dict(rows[0], phase="test", arm="labelled", model=P2)
    with out.open("a") as f:
        f.write(json.dumps(stray) + "\n")
    admitted, problems = rep.admission([out], "m", "t")
    assert stray not in admitted and problems["format.jsonl: row does not match its invocation"] == 1
    with pytest.raises(RuntimeError, match="not admitted"):
        run(FakeClient(), "format_arm", ds, ["1"], out, arm="bare", invocation="i2")


def test_bad_header_in_another_ledger_invalidates_the_invocation(tmp_path):
    ds = small_dataset()
    dev = tmp_path / "dev.jsonl"
    run(FakeClient(), "massive_dev", ds, ["0"], dev, invocation="shared")
    other = tmp_path / "other.jsonl"
    bad = header("massive_dev", invocation="shared")
    bad["digests"][P1] = "0" * 64
    rep.meta_path(other).write_text(json.dumps(bad) + "\n")
    assert len(rep.admission([dev], "m")[0]) == 2
    rows, problems = rep.admission([dev, other], "m")
    assert rows == [] and problems["dev.jsonl: row of an unknown or invalid invocation"] == 2


def test_headers_the_protocol_does_not_admit():
    good = header("format_arm", "bare")
    assert rep.header_problem(good, "m", "t") is None
    assert rep.header_problem({**good, "models": [*good["models"], "llama3.2:3b"]}, "m", "t")
    unknown = header("format_arm", "bare")
    unknown["models"][0] = "llama3.2:3b"
    unknown["digests"] = {"llama3.2:3b": None}
    assert rep.header_problem(unknown, "m", "t")
    assert rep.header_problem({**good, "phase": "test"}, "m", "t")
    assert rep.header_problem({**good, "arm": None}, "m", "t")
    assert rep.header_problem({**good, "commit": ""}, "m", "t")
    assert rep.header_problem(good, "other", "t")
    assert rep.header_problem(good, "m", "other")
    assert rep.header_problem({**good, "ollama": "0.35.0"}, "m", "t")


def test_lock_covers_both_files_and_a_run_without_calls_writes_nothing(tmp_path):
    ds = small_dataset()
    out = tmp_path / "dev.jsonl"
    run(FakeClient(), "massive_dev", ds, ["0"], out)
    before = rep.meta_path(out).read_bytes()
    with rep.single_writer(out):
        with pytest.raises(RuntimeError, match="another process"):
            run(FakeClient(), "massive_dev", ds, ["0", "1"], out, invocation="i2")
    assert rep.meta_path(out).read_bytes() == before
    result = run(FakeClient(), "massive_dev", ds, ["0"], out, invocation="i3")
    assert not result["header_written"] and rep.meta_path(out).read_bytes() == before


def test_torn_tails_of_both_files_are_repaired_under_the_lock(tmp_path):
    ds = small_dataset()
    out = tmp_path / "dev.jsonl"
    run(FakeClient(), "massive_dev", ds, ["0"], out)
    with out.open("a") as f:
        f.write('{"partial":')
    with rep.meta_path(out).open("a") as f:
        f.write('{"invocation": "half')
    assert rep.read_rows(out)[1] == '{"partial":'
    assert rep.admission([out], "m")[1]
    run(FakeClient(), "massive_dev", ds, ["0", "1"], out, invocation="i2")
    rows, tail = rep.read_rows(out)
    assert [r["id"] for r in rows] == ["0", "0", "1", "1"] and tail == ""
    assert rep.read_rows(rep.meta_path(out))[1] == "" and not rep.admission([out], "m")[1]
    assert json.loads((tmp_path / "dev.jsonl.torn").read_text())["fragment"] == '{"partial":'
    assert (tmp_path / "dev.jsonl.meta.jsonl.torn").exists()


def test_repair_cuts_at_a_byte_offset(tmp_path):
    out = tmp_path / "calls.jsonl"
    out.write_bytes('{"a": "éééé"}\n{"b":'.encode())
    rep.repair(out)
    assert rep.read_rows(out) == ([{"a": "éééé"}], "")
    assert json.loads((tmp_path / "calls.jsonl.torn").read_text())["offset"] == len('{"a": "éééé"}\n'.encode())


def test_complete_unterminated_row_is_kept(tmp_path):
    out = tmp_path / "calls.jsonl"
    out.write_text('{"a": 1}\n{"a": 2}')
    assert rep.read_rows(out) == ([{"a": 1}, {"a": 2}], "")
    rep.repair(out)
    assert out.read_text() == '{"a": 1}\n{"a": 2}\n'


def test_environment_must_match_the_protocol():
    good = {"ollama": rep.OLLAMA_VERSION, "digests": dict(rep.DIGESTS)}
    assert rep.environment_problem(good, rep.FORMAT_MODELS) is None
    assert rep.environment_problem({**good, "ollama": "0.35.0"}, rep.PRIMARY)
    assert rep.environment_problem({**good, "digests": {**rep.DIGESTS, "gemma3:4b": None}}, ["gemma3:4b"])
    assert rep.environment_problem({"ollama": rep.OLLAMA_VERSION, "digests": {"x": None}}, ["x"])


def test_rosters():
    assert rep.roster("massive_test", "labelled") == rep.PRIMARY
    assert rep.roster("format_arm", "bare") == rep.FORMAT_MODELS
    assert rep.roster("format_arm", "labelled") == rep.FORMAT_ONLY
    for set_name, arm in (("snips_dev", "bare"), ("format_arm", "other")):
        with pytest.raises(ValueError):
            rep.roster(set_name, arm)


def git(path: Path, *args) -> None:
    subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


def frozen_repo(path: Path) -> None:
    git(path, "init", "-q")
    for f in (*rep.FROZEN_FILES, *rep.FROZEN_CODE):
        (path / f).parent.mkdir(parents=True, exist_ok=True)
        (path / f).write_text(f)
    git(path, "add", ".")
    git(path, "commit", "-qm", "freeze")


def test_committed_files_are_compared_with_their_blobs(tmp_path):
    frozen = (*rep.FROZEN_FILES, *rep.FROZEN_CODE)
    assert "exited with" in rep.committed_problem(frozen, tmp_path)
    frozen_repo(tmp_path)
    assert rep.committed_problem(frozen, tmp_path) is None
    (tmp_path / "src/xreview/scratch.py").write_text("untracked")
    (tmp_path / "notes.txt").write_text("untracked")
    assert rep.committed_problem(frozen, tmp_path) is None
    (tmp_path / "results/replication/gate.meta.jsonl").write_text("changed")
    assert "gate.meta.jsonl" in rep.committed_problem(frozen, tmp_path)
    git(tmp_path, "add", ".")
    assert "gate.meta.jsonl" in rep.committed_problem(frozen, tmp_path)


def test_git_errors_are_refusals(tmp_path, monkeypatch):
    frozen_repo(tmp_path)

    class Failed:
        returncode, stdout = 128, b""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: Failed())
    assert "exited with 128" in rep.committed_problem(rep.FROZEN_FILES, tmp_path)


def test_preflight_refuses_before_any_call(tmp_path, monkeypatch):
    frozen_repo(tmp_path)
    monkeypatch.setattr(rep, "MANIFEST_SHA256", rep.sha256(tmp_path / "data/replication_manifest.json"))
    calls = []

    def info(models):
        calls.append(models)
        return {"ollama": rep.OLLAMA_VERSION, "digests": dict(rep.DIGESTS)}

    h = rep.preflight("massive_dev", "labelled", info, tmp_path)
    assert h["commit"] and h["models"] == list(rep.PRIMARY) and "thresholds_sha256" not in h
    with pytest.raises(RuntimeError, match="thresholds"):
        rep.preflight("massive_test", "labelled", info, tmp_path)
    (tmp_path / "REPLICATION_PROTOCOL.md").write_text("changed")
    with pytest.raises(RuntimeError, match="differs"):
        rep.preflight("massive_dev", "labelled", info, tmp_path)
    assert len(calls) == 1


def frame(wrong, field, d, wrapped=None, all_worst=None):
    wrong = np.array(wrong, dtype=bool)
    field = np.array(field, dtype=float)
    n = len(wrong)
    scores = {"L-field": field, "D": np.array(d, dtype=float), "field sum": field * 2, "json-open worst": field,
              "all worst": np.array(all_worst if all_worst is not None else field, dtype=float)}
    return {"ids": [str(i) for i in range(n)], "wrong": wrong, "scores": scores,
            "wrapped": np.array(wrapped if wrapped is not None else [False] * n, dtype=bool),
            "malformed": np.zeros(n, dtype=bool), "strict": np.ones(n, dtype=bool)}


def test_cell_passes_only_when_the_lower_bound_is_above_zero(monkeypatch):
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.4
    assert ra.primary(frame(y, y + rng.normal(0, 0.5, 400), rng.random(400)))["replicated"]
    cell = frame(y, rng.random(400), rng.random(400))
    for value, low, expected in ((0.05, -0.01, False), (0.1, 0.0, False), (0.1, 1e-12, True)):
        monkeypatch.setattr(ra, "paired_difference",
                            lambda a, b, y, v=value, lo=low: {"value": v, "ci": (lo, 0.2), "undefined_draws": 0})
        assert ra.primary(cell)["replicated"] is expected


def test_one_class_cell_is_not_estimable_and_blocks_the_verdict():
    one = ra.primary(frame(np.zeros(50), np.arange(50.0), np.zeros(50)))
    assert not one["estimable"] and not one["replicated"]
    good = {"status": "complete", "primary": {"estimable": True, "replicated": True}}
    failed = {"status": "complete", "primary": {"estimable": True, "replicated": False}}
    names = ra.cell_names()
    cells = dict.fromkeys(names, good)
    assert ra.overall(cells) == {"verdict": "replicated", "replicated_cells": 4}
    assert ra.overall({**cells, names[0]: failed})["verdict"] == "replicated"
    assert ra.overall({**cells, names[0]: failed, names[1]: failed})["verdict"] == "not replicated"
    assert ra.overall({**cells, names[0]: {"status": "complete", "primary": one}})["verdict"] == "incomplete"
    assert ra.overall({n: cells[n] for n in names[:3]})["verdict"] == "incomplete"
    assert ra.overall({**cells, "extra / model": good, names[3]: {"status": "incomplete"}})["verdict"] == "incomplete"
    assert json.loads(ra.dumps(one))["difference"]["value"] is None


def test_select_issues_and_extras():
    def row(k, status="ok"):
        return dict(zip(("phase", "dataset", "arm", "model", "id", "call"), k, strict=True), status=status)

    k1, k2, k3, k4, k5 = (("test", "massive", "labelled", "m", i, 0) for i in "12345")
    rows = [row(k1), row(k1), row(k3, "error"), row(k3, "invalid"), row(k3, "error"), row(k4),
            row(k5, "error"), row(k5, "error"), row(k5, "error"), row(k5),
            row(("dev", "massive", "labelled", "m", "2", 0))]
    chosen, issues = ra.select(rows, [k1, k2, k3, k5])
    assert chosen == {} and issues == {"missing": [list(k2)], "exhausted": [list(k3)], "conflicting": [list(k1)],
                                       "inconsistent": [list(k5)], "extra": [list(k4)]}
    chosen, issues = ra.select([row(k1), row(k4)], [k1])
    assert list(chosen) == [k1] and issues == {"extra": [list(k4)]} and ra.fatal(issues) == {}


def test_threshold_rule():
    t = 0.5
    field = np.array([-math.log(0.2), -math.log(t), -math.log(0.9), 0.0])
    assert np.exp(-field)[1] == t
    out = ra.threshold(t, frame([True, False, False, True], field, np.zeros(4)))
    assert out["review_share"] == 0.25
    assert out["accepted_error"] == pytest.approx(1 / 3)


def format_pair(rng, wrong_bare=None, labelled_wrapped=0.6):
    y = rng.random(200) < 0.4
    yb = y if wrong_bare is None else wrong_bare
    field = y + rng.normal(0, 0.5, 200)
    noise = rng.random(200) * 5
    labelled = frame(y, field, np.zeros(200), wrapped=rng.random(200) < labelled_wrapped,
                     all_worst=np.maximum(field, noise))
    bare = frame(yb, yb + rng.normal(0, 0.5, 200), np.zeros(200), wrapped=np.zeros(200))
    return labelled, bare


def test_format_contrast_uses_each_arm_outcomes_and_serialises():
    rng = np.random.default_rng(1)
    labelled, bare = format_pair(rng, wrong_bare=rng.random(200) < 0.3)
    labelled["malformed"][:3] = True
    c = ra.format_contrast(labelled, bare)
    assert c["arms"]["labelled"]["wrong"] != c["arms"]["bare"]["wrong"]
    assert c["arms"]["labelled"]["not_well_formed"] == 3 and c["arms"]["bare"]["not_well_formed"] == 0
    assert c["W_bare_minus_labelled"] < 0 and c["result"] == "supported"
    assert c["G_bare_minus_labelled"]["ci"][1] < 0
    json.loads(ra.dumps(c))
    bare_wrapped = dict(bare, wrapped=rng.random(200) < 0.9)
    assert ra.format_contrast(labelled, bare_wrapped)["result"] == "not supported"
    assert ra.format_contrast(bare, bare)["not_informative_because"] == ["no wrapped reply in the labelled arm"]
    with pytest.raises(ValueError):
        ra.format_contrast(labelled, frame(np.zeros(10), np.zeros(10), np.zeros(10)))


def test_text_outside_the_span():
    from xreview.controls import outside_text

    assert not outside_text('  {"a": 1}\n') and not outside_text("   ")
    assert outside_text('JSON: {"a": 1}') and outside_text("sorry")


@pytest.mark.parametrize("wrong", [19, 20])
def test_format_needs_twenty_wrong_records_in_each_arm(wrong):
    rng = np.random.default_rng(2)
    labelled, bare = format_pair(rng)
    y = np.zeros(200, dtype=bool)
    y[:wrong] = True
    c = ra.format_contrast(dict(labelled, wrong=y), bare)
    assert c["informative"] is (wrong == 20)
    assert ("labelled arm: 19 wrong records" in c["not_informative_because"]) is (wrong == 19)


def test_shared_finding_needs_both_historical_wrappers():
    supported = {"result": "supported"}
    assert ra.shared_finding({"qwen2.5:3b-instruct": supported, "gemma3:4b": supported})["shared"]
    one = {"qwen2.5:3b-instruct": supported, "gemma3:4b": {"result": "not informative"}}
    assert not ra.shared_finding(one)["shared"]
    assert not ra.shared_finding({"qwen2.5:3b-instruct": supported})["shared"]


def fake_text(correct_texts):
    def text(model, messages, seed):
        item = messages[1]["content"].rsplit("Request: ", 1)[1]
        return RIGHT if item in correct_texts and (seed == 0 or seed % 2) else WRONG
    return text


def study(root: Path, monkeypatch, wrapped_labelled=False):
    """A miniature study in a git repository: frozen files, development runs, the thresholds command, the second
    commitment, test and format runs, through the public entry points."""
    frozen_repo(root)
    massive, snips_ds = small_dataset(), small_dataset("snips")
    ids = [str(i) for i in range(60)]
    manifest = {"sets": {"massive_test": ids, "snips_test": ids, "massive_dev": ids[:40], "snips_dev": ids[:40],
                         "format_arm": ids}, "snips_test_train_matches": ["0"]}
    (root / "data/replication_manifest.json").write_text(json.dumps(manifest))
    git(root, "commit", "-qam", "manifest")
    sha = rep.sha256(root / "data/replication_manifest.json")
    monkeypatch.setattr(rep, "MANIFEST_SHA256", sha)
    monkeypatch.setattr(ra, "DEV_ITEMS", 40)
    monkeypatch.setattr(ra, "load_data_sets", lambda: {"massive": massive, "snips": snips_ds})
    right = fake_text({f"wake me at five am {i}" for i in range(0, 60, 2)})

    def wrapped(model, messages, seed):
        return "JSON: " + right(model, messages, seed)

    client = FakeClient(text=right)
    labelled = FakeClient(text=wrapped if wrapped_labelled else right)
    ledger = {p: root / f for p, f in rep.LEDGERS.items()}
    for name, ds in (("massive", massive), ("snips", snips_ds)):
        h = header(f"{name}_dev", invocation=f"dev-{name}", manifest=sha)
        rep.run(client, h, ds, ids[:40], ledger["dev"])
    ra.main(["thresholds"], root)
    git(root, "add", ".")
    git(root, "commit", "-qm", "thresholds")
    t_sha = rep.sha256(root / rep.THRESHOLDS)
    for name, ds in (("massive", massive), ("snips", snips_ds)):
        h = header(f"{name}_test", invocation=f"test-{name}", manifest=sha, thresholds=t_sha)
        rep.run(labelled, h, ds, ids, ledger["test"], thresholds_sha=t_sha)
    rep.run(client, header("format_arm", "bare", "bare", manifest=sha, thresholds=t_sha), massive, ids,
            ledger["format"], thresholds_sha=t_sha)
    rep.run(labelled, header("format_arm", "labelled", "lab", manifest=sha, thresholds=t_sha), massive, ids,
            ledger["format"], thresholds_sha=t_sha)
    return ledger


def test_command_line_analysis_with_the_committed_thresholds(tmp_path, monkeypatch):
    ledger = study(tmp_path, monkeypatch, wrapped_labelled=True)
    out = tmp_path / "analysis.json"
    ra.main(["analyze", "--out", str(out)], tmp_path)
    loaded = json.loads(out.read_text())
    assert "NaN" not in out.read_text() and loaded["not_admitted"] == {}
    assert loaded["overall"]["verdict"] in ("replicated", "not replicated")
    gemma = loaded["format_arm"]["gemma3:4b"]
    assert gemma["status"] == "complete" and gemma["arms"]["labelled"]["wrapped"] == 60
    assert isinstance(gemma["informative"], bool)
    assert loaded["format_arm"][P1]["arms"]["labelled"]["wrapped"] == 60
    sens = loaded["cells"][f"snips / {P2}"]["sensitivity"]
    assert set(sens) == {"without_malformed", "without_train_matches"} and sens["without_train_matches"]["n"] == 59
    assert loaded["cells"][f"massive / {P1}"]["secondary"]["strict_parse"] == {"count": 0, "of": 60}
    assert set(loaded["thresholds"]) == set(ra.cell_names())
    rows, _ = rep.read_rows(ledger["test"])
    extra = dict(rows[0], id="999")
    with ledger["test"].open("a") as f:
        f.write(json.dumps(extra) + "\n")
    ra.main(["analyze", "--out", str(out)], tmp_path)
    cell = json.loads(out.read_text())["cells"][f"{extra['dataset']} / {extra['model']}"]
    assert cell["status"] == "complete" and cell["issues"]["extra"]


def test_analysis_refuses_development_input_other_than_the_bound_one(tmp_path, monkeypatch):
    ledger = study(tmp_path, monkeypatch)
    out = tmp_path / "analysis.json"
    rows, _ = rep.read_rows(ledger["dev"])
    ledger["dev"].write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    with pytest.raises(SystemExit, match="differs from the last commit"):
        ra.main(["analyze", "--out", str(out)], tmp_path)
    git(tmp_path, "commit", "-qam", "changed development ledger")
    with pytest.raises(SystemExit, match="changed"):
        ra.main(["analyze", "--out", str(out)], tmp_path)
    assert not out.exists()


def test_thresholds_file_must_list_both_development_inputs(tmp_path, monkeypatch):
    study(tmp_path, monkeypatch)
    assert ra.threshold_problem(tmp_path) is None
    path = tmp_path / rep.THRESHOLDS
    committed = json.loads(path.read_text())
    del committed["dev_inputs"][rep.LEDGERS["dev"] + ".meta.jsonl"]
    path.write_text(ra.dumps(committed))
    git(tmp_path, "commit", "-qam", "drop metadata hash")
    assert "both be listed" in ra.threshold_problem(tmp_path)
    committed = json.loads(git_show(tmp_path, "HEAD~1", rep.THRESHOLDS))
    committed["thresholds"][f"massive / {P2}"] += 0.01
    path.write_text(ra.dumps(committed))
    git(tmp_path, "commit", "-qam", "edit a threshold")
    assert "reproduce" in ra.threshold_problem(tmp_path)


def git_show(path: Path, rev: str, name: str) -> str:
    return subprocess.run(["git", "-C", str(path), "show", f"{rev}:{name}"], capture_output=True, text=True,
                          check=True).stdout


def test_thresholds_are_refused_when_incomplete(tmp_path, monkeypatch):
    ledger = study(tmp_path, monkeypatch)
    rows, _ = rep.read_rows(ledger["dev"])
    ledger["dev"].write_text("".join(json.dumps(r) + "\n" for r in rows if not (r["dataset"] == "snips"
                                                                                 and r["id"] == "3")))
    with pytest.raises(RuntimeError, match="not complete"):
        ra.main(["thresholds"], tmp_path)


def test_wrong_environment_rows_are_not_used(tmp_path, monkeypatch):
    ledger = study(tmp_path, monkeypatch)
    bad = dict(rep.read_rows(rep.meta_path(ledger["test"]))[0][0], ollama="0.35.0")
    with rep.meta_path(ledger["test"]).open("a") as f:
        f.write(json.dumps(bad) + "\n")
    out = tmp_path / "analysis.json"
    ra.main(["analyze", "--out", str(out)], tmp_path)
    loaded = json.loads(out.read_text())
    assert loaded["overall"]["verdict"] == "incomplete"
    assert loaded["cells"][f"massive / {P1}"]["status"] == "incomplete"


@pytest.mark.skipif(not HAS_MASSIVE, reason="MASSIVE not downloaded")
def test_stored_gate_passes_and_copies_do_not(tmp_path):
    ds = rep.build("massive", rep.massive_rows())
    assert ra.gate_result(ROOT, ds) == {"requests": True, "environment": True, "aligned": 20, "well_formed": 20,
                                        "passed": True}
    copy = tmp_path / "results/replication"
    copy.mkdir(parents=True)
    first = (ROOT / rep.GATE_FILES[0]).read_text().splitlines()[0]
    (tmp_path / rep.GATE_FILES[0]).write_text((first + "\n") * 20)
    (tmp_path / rep.GATE_FILES[1]).write_text((ROOT / rep.GATE_FILES[1]).read_text())
    assert ra.gate_result(tmp_path, ds)["passed"] is False


@pytest.mark.skipif(not HAS_MASSIVE, reason="MASSIVE not downloaded")
@pytest.mark.parametrize("model", ["qwen2.5:3b-instruct", "gemma3:4b", "qwen2.5:7b-instruct"])
def test_analysis_reproduces_october(model):
    from xreview import cli
    from xreview.experiment import load_rows

    onto, _, _, split = cli.setup()
    items = {u.id: u for u in split["test"]}
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    reanalysis = json.loads((ROOT / "results" / "reanalysis.json").read_text())
    rows = load_rows(ROOT / "results" / "raw_test.jsonl")
    entries = [(items[r["id"]], r["greedy"], r["samples"]) for r in rows if r["model"] == model and r["id"] in items]
    f = ra.full_frame(entries, onto)
    p = json.loads(ra.dumps(ra.primary(f)))
    assert p["difference"]["value"] == summary[model]["difference"]["L-field - D"]["value"]
    assert p["difference"]["ci"] == summary[model]["difference"]["L-field - D"]["ci"]
    d = json.loads(ra.dumps(ra.descriptive(f)))
    assert d["review_yield"] == reanalysis[model]["budget_contrasts"]["L-field - D"]
    assert d["field_sum_minus_field"]["value"] == reanalysis[model]["contrasts"]["field sum - field worst"]["value"]
    assert d["unanimous"] == reanalysis[model]["unanimous"]
    assert d["strict_parse"]["count"] == reanalysis[model]["format"]["strict_parse"]
    assert d["malformed"]["count"] == 1000 - reanalysis[model]["format"]["well_formed"]


def test_any_exception_of_the_client_is_a_recorded_attempt(tmp_path):
    raised = []

    def client(*args, **kwargs):
        if not raised:
            raised.append(1)
            raise AttributeError("'list' object has no attribute 'get'")
        return FakeClient()(*args, **kwargs)

    out = tmp_path / "format.jsonl"
    run(client, "format_arm", small_dataset(), ["0"], out, arm="labelled")
    rows, _ = rep.read_rows(out)
    assert rows[0]["status"] == "error" and rows[0]["error"].startswith("AttributeError")
    assert rows[1]["status"] == "ok"


def test_unsolicited_token_metadata_on_a_sampled_call(tmp_path, monkeypatch):
    payload = {"message": {"content": RIGHT}, "logprobs": [None]}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr("requests.post", lambda url, json, timeout: Response())
    sampled = rep.ollama_chat("m", [], **rep.settings(1))
    assert sampled["tokens"] == [] and rep.reply_problem(sampled, 1) is None
    with pytest.raises(ValueError):
        rep.ollama_chat("m", [], **rep.settings(0))
    out = tmp_path / "test.jsonl"
    greedy = FakeClient()

    def client(model, messages, temperature, seed, logprobs):
        if logprobs:
            return greedy(model, messages, temperature, seed, logprobs)
        return rep.ollama_chat(model, messages, temperature, seed, logprobs)

    run(client, "massive_test", small_dataset(), ["0"], out)
    rows, _ = rep.read_rows(out)
    assert [(r["call"], r["status"]) for r in rows if r["model"] == P1] == [(c, "ok") for c in range(5)]


def admitted_study(tmp_path, monkeypatch):
    ledger = study(tmp_path, monkeypatch)
    rows, problems = rep.admission(list(ledger.values()), thresholds_sha=rep.sha256(tmp_path / rep.THRESHOLDS))
    assert not problems
    return rows, ra.load_data_sets()


def manifest_of(tmp_path):
    return json.loads((tmp_path / "data/replication_manifest.json").read_text())


def test_development_extras_are_reported(tmp_path, monkeypatch):
    rows, data_sets = admitted_study(tmp_path, monkeypatch)
    dev = next(r for r in rows if r["phase"] == "dev")
    frames, issues = ra.dev_frames([*rows, dict(dev, id="999")], manifest_of(tmp_path), data_sets)
    cell = f"{dev['dataset']} / {dev['model']}"
    assert cell in frames and issues == {cell: {"extra": [[*rep.key(dev)[:4], "999", 0]]}}
    assert json.loads((tmp_path / rep.THRESHOLDS).read_text())["issues"] == {}


def test_format_extras_are_reported_against_the_whole_test_manifest(tmp_path, monkeypatch):
    rows, data_sets = admitted_study(tmp_path, monkeypatch)
    manifest = manifest_of(tmp_path)
    manifest["sets"]["format_arm"] = manifest["sets"]["format_arm"][:30]
    bare = next(r for r in rows if r["phase"] == "format" and r["arm"] == "bare" and r["model"] == P1)
    test = next(r for r in rows if r["phase"] == "test" and r["dataset"] == "massive" and r["model"] == P1)
    added = [dict(bare, id="999"), dict(test, id="999")]
    result = ra.analyze([*rows, *added], manifest, data_sets, {"thresholds": {}})
    issues = result["format_arm"][P1]["issues"]
    assert result["format_arm"][P1]["status"] == "complete"
    assert [*rep.key(bare)[:4], "999", 0] in issues["bare"]["extra"]
    assert issues["labelled"] == {"extra": [[*rep.key(test)[:4], "999", test["call"]]]}
    clean = ra.analyze(rows, manifest_of(tmp_path), data_sets, {"thresholds": {}})
    assert clean["format_arm"][P1]["issues"] == {} and clean["development"] == {}
