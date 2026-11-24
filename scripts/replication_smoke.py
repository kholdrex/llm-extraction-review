"""End-to-end check of the replication pipeline on training utterances that are in no manifest and are left out of
retrieval: both primary models on both data sets with greedy plus four samples, an interruption inside an item and a
resume, an exhausted call, torn last lines in both files, both prompt arms for the four format models, then the
analysis code. Smoke phases are not study phases, so the admission check of the runner is off here.
Writes to runs/smoke-train-utterances/, which holds no study data."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import requests

from xreview import replication as rep
from xreview import replication_analysis as ra

OUT = Path(__file__).resolve().parents[1] / "runs" / "smoke-train-utterances"
PER_DATASET = 3


class Interrupt(KeyboardInterrupt):
    pass


def interrupting(after: int):
    calls = {"n": 0}

    def client(*args, **kwargs):
        if calls["n"] == after:
            raise Interrupt
        calls["n"] += 1
        return rep.ollama_chat(*args, **kwargs)

    return client


def failing(*args, **kwargs):
    raise requests.ConnectionError("refused")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = rep.load_manifest()
    used = {i for ids in manifest["sets"].values() for i in ids} | set(manifest["gate"])
    used |= {i for ids in manifest["fixed_examples"].values() for i in ids}
    rows = {"massive": rep.massive_rows(), "snips": rep.snips_rows()}
    picks, data_sets = {}, {}
    for name, all_rows in rows.items():
        pool = sorted(u.id for u in all_rows if u.partition == "train" and u.id not in used)
        picks[name] = random.Random(1).sample(pool, PER_DATASET)
        data_sets[name] = rep.build(name, all_rows, exclude=set(picks[name]))
        assert not {u.id for i in picks[name] for u in data_sets[name].retriever.nearest(data_sets[name].items[i].text)
                    } & set(picks[name])
    info = rep.server_info(rep.FORMAT_MODELS)
    problem = rep.environment_problem(info, rep.FORMAT_MODELS)
    if problem:
        raise SystemExit(problem)
    calls = OUT / "calls.jsonl"
    count = iter(range(100))

    def run(client, models, name, ids, arm, phase):
        header = {"invocation": f"smoke-{next(count)}", "phase": phase, "arm": arm, "models": list(models), **info}
        return rep.run(client, header, data_sets[name], ids, calls, check=False)

    try:
        run(interrupting(7), rep.PRIMARY, "massive", picks["massive"], "labelled", "smoke-test")
    except Interrupt:
        print("interrupted after 7 calls (second item, call 2)", file=sys.stderr)
    for name in ("massive", "snips"):
        run(rep.ollama_chat, rep.PRIMARY, name, picks[name], "labelled", "smoke-test")
    exhausted = run(failing, ["ibm/granite4.1:3b"], "massive", picks["massive"][:1], "labelled", "smoke-exhaust")
    before = len(rep.read_rows(calls)[0])
    again = run(rep.ollama_chat, ["ibm/granite4.1:3b"], "massive", picks["massive"][:1], "labelled", "smoke-exhaust")
    no_fourth = len(rep.read_rows(calls)[0]) == before and not again["header_written"]
    with calls.open("a") as f:
        f.write('{"phase": "smoke-format", "dataset"')
    with rep.meta_path(calls).open("a") as f:
        f.write('{"invocation": "half')
    torn_read = [rep.read_rows(calls)[1], rep.read_rows(rep.meta_path(calls))[1]]
    run(rep.ollama_chat, rep.FORMAT_MODELS, "massive", picks["massive"], "bare", "smoke-format")
    run(rep.ollama_chat, rep.FORMAT_ONLY, "massive", picks["massive"], "labelled", "smoke-format")
    stored, tail = rep.read_rows(calls)
    ok = [rep.key(r) for r in stored if r["status"] == "ok"]
    summary = {
        "picks": picks, "ok_rows": len(ok), "unique_ok_keys": len(set(ok)),
        "expected_ok": 2 * PER_DATASET * len(rep.PRIMARY) * 5 + PER_DATASET * (len(rep.FORMAT_MODELS) + 2),
        "failed_rows": sum(r["status"] != "ok" for r in stored), "exhausted": exhausted["exhausted"],
        "exhausted_again": again["exhausted"], "no_fourth_attempt": no_fourth, "torn_fragment_seen": torn_read,
        "torn_files": [(OUT / f).read_text() for f in ("calls.jsonl.torn", "calls.jsonl.meta.jsonl.torn")],
        "tail_after_resume": [tail, rep.read_rows(rep.meta_path(calls))[1]],
    }
    seconds = {}
    for r in stored:
        if r["status"] == "ok" and r["phase"] != "smoke-exhaust":
            kind = "greedy" if r["call"] == 0 else "sample"
            seconds.setdefault(f"{r['model']} {r['dataset']} {kind}", []).append(r["seconds"])
    summary["median_seconds"] = {k: float(np.median(v)) for k, v in sorted(seconds.items())}
    eligible = stored
    analysis = {}
    for name in ("massive", "snips"):
        ds = data_sets[name]
        for model in rep.PRIMARY:
            expected = ra.expected_keys("smoke-test", name, "labelled", model, picks[name], 5)
            chosen, issues = ra.select(eligible, expected)
            frame = ra.full_frame([(ds.items[i], chosen["smoke-test", name, "labelled", model, i, 0],
                                    [chosen["smoke-test", name, "labelled", model, i, k] for k in range(1, 5)])
                                   for i in picks[name]], ds.onto)
            analysis[f"{name} / {model}"] = {"issues": issues, "primary": ra.primary(frame),
                                             "descriptive": ra.descriptive(frame),
                                             "threshold": ra.threshold(0.9, frame)}
    for model in rep.FORMAT_MODELS:
        phase = "smoke-test" if model in rep.PRIMARY else "smoke-format"
        lab, _ = ra.select(eligible, ra.expected_keys(phase, "massive", "labelled", model, picks["massive"], 1))
        bare, _ = ra.select(eligible, ra.expected_keys("smoke-format", "massive", "bare", model, picks["massive"], 1))
        ds = data_sets["massive"]
        frames = [ra.greedy_frame([(ds.items[i], got[ph, "massive", arm, model, i, 0]) for i in picks["massive"]],
                                  ds.onto)
                  for got, ph, arm in ((lab, phase, "labelled"), (bare, "smoke-format", "bare"))]
        analysis[f"format / {model}"] = ra.format_contrast(*frames)
    summary["analysis"] = analysis
    (OUT / "summary.json").write_text(ra.dumps(summary))
    keys = ("ok_rows", "unique_ok_keys", "expected_ok", "failed_rows", "no_fourth_attempt", "tail_after_resume")
    print(json.dumps({k: summary[k] for k in keys}))


if __name__ == "__main__":
    main()
