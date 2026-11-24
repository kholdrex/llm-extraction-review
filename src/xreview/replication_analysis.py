"""Analyses of REPLICATION_PROTOCOL.md, Section 7, from the stored attempt rows."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import replication as rep
from . import stats
from .analysis import RESAMPLES, SEED, accepted_error, ties
from .controls import outside_text, strict_parse
from .data import Ontology, Utterance
from .detectors import score
from .metrics import score as score_record
from .reanalysis import budget_difference
from .replication import (
    ATTEMPTS,
    FORMAT_MODELS,
    FORMAT_ONLY,
    GATE_FILES,
    GATE_IDS,
    GATE_MODEL,
    GATE_WELL_FORMED,
    LEDGERS,
    PRIMARY,
    ROOT,
    SAMPLES,
    THRESHOLDS,
    Dataset,
    admission,
    build,
    committed_problem,
    environment_problem,
    history,
    massive_rows,
    meta_path,
    read_rows,
    reply_problem,
    sha256,
    snips_rows,
)
from .tokens import token_scores
from .validate import check

CELL_MIN = 3
FORMAT_MIN = 20
DEV_ITEMS = 100
THRESHOLD_QUANTILE = 0.30
DATASETS = ("snips", "massive")
FATAL = ("missing", "exhausted", "conflicting", "inconsistent")


def native(x):
    """JSON-ready copy: NumPy scalars to Python, NaN and infinities to None, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dumps(x) -> str:
    return json.dumps(native(x), indent=1, allow_nan=False) + "\n"


def select(rows: list[dict], expected: list[tuple]) -> tuple[dict, dict]:
    """The successful row of each expected call, and the issues. Missing, exhausted (`ATTEMPTS` failed attempts and
    no success), conflicting (more than one success) and inconsistent (a success after `ATTEMPTS` failed attempts)
    calls make the cell incomplete; extra successes for calls the manifest does not expect are reported and
    ignored."""
    ok, failures = history(rows)
    wanted = set(expected)
    scope = {k[:4] for k in expected}
    chosen, issues = {}, {name: [] for name in (*FATAL, "extra")}
    for k in expected:
        found = ok.get(k, [])
        if len(found) > 1:
            issues["conflicting"].append(list(k))
        elif found and failures[k] >= ATTEMPTS:
            issues["inconsistent"].append(list(k))
        elif found:
            chosen[k] = found[0]
        elif failures[k] >= ATTEMPTS:
            issues["exhausted"].append(list(k))
        else:
            issues["missing"].append(list(k))
    issues["extra"] = [list(k) for k in ok if k[:4] in scope and k not in wanted]
    return chosen, {k: v for k, v in issues.items() if v}


def fatal(issues: dict) -> dict:
    return {k: v for k, v in issues.items() if k in FATAL}


def expected_keys(phase: str, dataset: str, arm: str, model: str, ids: list[str], calls: int) -> list[tuple]:
    return [(phase, dataset, arm, model, i, c) for i in ids for c in range(calls)]


def greedy_frame(entries: list[tuple[Utterance, dict]], onto: Ontology) -> dict:
    """Outcome, one-call scores and format of greedy replies, in the given order."""
    wrong, tok, wrapped, malformed, strict = [], [], [], [], []
    for u, greedy in entries:
        record, _ = check(greedy["text"], u.text, onto)
        wrong.append(not score_record(record, u).record)
        tok.append(token_scores(greedy["text"], [tuple(t) for t in greedy["tokens"]]))
        wrapped.append(bool(outside_text(greedy["text"])))
        malformed.append(record is None)
        strict.append(strict_parse(greedy["text"]))
    names = tok[0] if tok else {}
    scores = {k: np.array([t[k] for t in tok], dtype=float) for k in names}
    scores["L-field"] = scores.get("field worst", np.array([]))
    return {"ids": [u.id for u, _ in entries], "wrong": np.array(wrong, dtype=bool), "scores": scores,
            "wrapped": np.array(wrapped, dtype=bool), "malformed": np.array(malformed, dtype=bool),
            "strict": np.array(strict, dtype=bool)}


def full_frame(entries: list[tuple[Utterance, dict, list[dict]]], onto: Ontology) -> dict:
    """`greedy_frame` plus R and D from the samples."""
    frame = greedy_frame([(u, g) for u, g, _ in entries], onto)
    rd = [score({"text": g["text"], "tokens": [tuple(t) for t in g["tokens"]]}, samples, u.text, onto)[1]
          for u, g, samples in entries]
    frame["scores"]["R"] = np.array([s.rules for s in rd])
    frame["scores"]["D"] = np.array([s.disagreement for s in rd])
    return frame


def subset(frame: dict, keep: np.ndarray) -> dict:
    out = {k: v[keep] for k, v in frame.items() if isinstance(v, np.ndarray)}
    out["ids"] = [i for i, k in zip(frame["ids"], keep, strict=True) if k]
    out["scores"] = {k: v[keep] for k, v in frame["scores"].items()}
    return out


def paired_difference(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """AUROC(a) − AUROC(b) on the same items; percentile interval (linear 2.5 and 97.5 percentiles) over 2,000
    resamples of items with replacement in manifest order, a fresh generator seeded 2026 for each interval; draws
    with an undefined AUROC are left out and counted."""
    boot = stats.bootstrap(lambda i: stats.auroc(a[i], y[i]) - stats.auroc(b[i], y[i]), len(y), RESAMPLES, SEED)
    finite = np.isfinite(boot)
    value = stats.auroc(a, y) - stats.auroc(b, y) if len(y) else float("nan")
    ci = stats.interval(boot) if finite.any() else (float("nan"), float("nan"))
    return {"value": value, "ci": ci, "undefined_draws": int((~finite).sum())}


def estimable(d: dict) -> bool:
    return all(v is not None and math.isfinite(v) for v in (d["value"], *d["ci"]))


def primary(frame: dict) -> dict:
    """Paired AUROC difference L-field − D. Replicated when the unrounded lower bound of the 95% interval is above
    0; not estimable when the cell has only one outcome class or the difference or its interval is undefined."""
    s, y = frame["scores"], frame["wrong"]
    d = paired_difference(s["L-field"], s["D"], y)
    ok = bool(y.any() and (~y).any()) and estimable(d)
    return {"n": len(y), "errors": int(y.sum()), "estimable": ok,
            "auroc_L_field": stats.auroc(s["L-field"], y), "auroc_D": stats.auroc(s["D"], y), "difference": d,
            "replicated": bool(ok and d["ci"][0] > 0)}


def cell_names() -> list[str]:
    return [f"{d} / {m}" for d in DATASETS for m in PRIMARY]


def overall(cells: dict[str, dict]) -> dict:
    """Over the four named cells: replicated when at least `CELL_MIN` replicate, not replicated otherwise; incomplete,
    without a verdict, when any named cell is missing, incomplete or not estimable."""
    reasons = {}
    for name in cell_names():
        cell = cells.get(name)
        if cell is None:
            reasons[name] = "no result"
        elif cell["status"] != "complete":
            reasons[name] = cell["status"]
        elif not cell["primary"]["estimable"]:
            reasons[name] = "not estimable"
    if reasons:
        return {"verdict": "incomplete", "reasons": reasons}
    hits = sum(cells[n]["primary"]["replicated"] for n in cell_names())
    return {"verdict": "replicated" if hits >= CELL_MIN else "not replicated", "replicated_cells": hits}


def descriptive(frame: dict) -> dict:
    s, y = frame["scores"], frame["wrong"]
    unanimous = s["D"] == 0
    flagged = s["R"] == 1
    return {
        "review_yield": budget_difference(s["L-field"], s["D"], y, ties(len(y))),
        "field_sum_minus_field": paired_difference(s["field sum"], s["L-field"], y),
        "json_open_worst_minus_field": paired_difference(s["json-open worst"], s["L-field"], y),
        "unanimous": {"requests": int(unanimous.sum()), "wrong": int(y[unanimous].sum())},
        "rules": {"flagged": int(flagged.sum()), "wrong_among_flagged": int(y[flagged].sum())},
        "malformed": {"count": int(frame["malformed"].sum()), "of": len(y)},
        "strict_parse": {"count": int(frame["strict"].sum()), "of": len(y)},
    }


def gap(frame: dict) -> float:
    s, y = frame["scores"], frame["wrong"]
    return stats.auroc(s["json-open worst"], y) - stats.auroc(s["all worst"], y)


def format_contrast(labelled: dict, bare: dict) -> dict:
    """Per model: W (share of replies containing a non-whitespace character outside the JSON span) and
    G = AUROC(json-open worst) − AUROC(all worst), each arm scored against its own outcomes; G_bare − G_labelled with
    a paired bootstrap over items.
    Informative when the labelled arm has a wrapped reply, each arm has at least `FORMAT_MIN` wrong and `FORMAT_MIN`
    correct records and the G difference is estimable; supported when informative, W_bare − W_labelled < 0 and the
    upper bound of G_bare − G_labelled < 0."""
    if labelled["ids"] != bare["ids"]:
        raise ValueError("the two arms must cover the same items in the same order")
    wl, wb = labelled["wrapped"], bare["wrapped"]

    def resampled(f: dict, i: np.ndarray) -> dict:
        return {"wrong": f["wrong"][i], "scores": {k: v[i] for k, v in f["scores"].items()}}

    boot = stats.bootstrap(lambda i: gap(resampled(bare, i)) - gap(resampled(labelled, i)), len(wl), RESAMPLES, SEED)
    finite = np.isfinite(boot)
    g = {"value": gap(bare) - gap(labelled),
         "ci": stats.interval(boot) if finite.any() else (float("nan"), float("nan")),
         "undefined_draws": int((~finite).sum())}
    reasons = []
    if not wl.any():
        reasons.append("no wrapped reply in the labelled arm")
    for arm, f in (("labelled", labelled), ("bare", bare)):
        wrong = int(f["wrong"].sum())
        if wrong < FORMAT_MIN:
            reasons.append(f"{arm} arm: {wrong} wrong records")
        if len(f["wrong"]) - wrong < FORMAT_MIN:
            reasons.append(f"{arm} arm: {len(f['wrong']) - wrong} correct records")
    if not estimable(g):
        reasons.append("G difference not estimable")
    w_diff = float(wb.mean() - wl.mean())
    informative = not reasons
    supported = bool(informative and w_diff < 0 and g["ci"][1] < 0)
    arms = {arm: {"n": len(f["wrong"]), "wrong": int(f["wrong"].sum()), "wrapped": int(f["wrapped"].sum()),
                  "not_well_formed": int(f["malformed"].sum()), "strict_parse": int(f["strict"].sum()), "G": gap(f)}
            for arm, f in (("labelled", labelled), ("bare", bare))}
    return {
        "arms": arms, "W_labelled": float(wl.mean()), "W_bare": float(wb.mean()), "W_bare_minus_labelled": w_diff,
        "wrapped_only_labelled": int((wl & ~wb).sum()), "wrapped_only_bare": int((wb & ~wl).sum()),
        "G_bare_minus_labelled": g, "informative": informative, "not_informative_because": reasons,
        "result": "supported" if supported else ("not supported" if informative else "not informative"),
    }


def shared_finding(results: dict[str, dict]) -> dict:
    """The dependence on the label is called shared by the two models that wrapped in October only when both are
    informative and both supported."""
    per = {m: results.get(m, {}).get("result", "incomplete") for m in FORMAT_ONLY}
    return {"models": per, "shared": all(r == "supported" for r in per.values())}


def thresholds_from(frames: dict[str, dict]) -> dict[str, float]:
    return {cell: float(np.quantile(np.exp(-f["scores"]["L-field"]), THRESHOLD_QUANTILE)) for cell, f in frames.items()}


def dev_frames(rows: list[dict], manifest: dict, data_sets: dict[str, Dataset]) -> tuple[dict, dict]:
    """Greedy frames of the complete development cells, and the issues of every cell, extras included."""
    frames, issues = {}, {}
    for name in DATASETS:
        ds, ids = data_sets[name], manifest["sets"][f"{name}_dev"]
        for model in PRIMARY:
            cell = f"{name} / {model}"
            chosen, problems = select(rows, expected_keys("dev", name, "labelled", model, ids, 1))
            if len(ids) != DEV_ITEMS:
                problems["items"] = len(ids)
            if problems:
                issues[cell] = problems
            if fatal(problems) or len(ids) != DEV_ITEMS:
                continue
            frames[cell] = greedy_frame([(ds.items[i], chosen["dev", name, "labelled", model, i, 0]) for i in ids],
                                        ds.onto)
    return frames, issues


def dev_inputs(root: Path) -> list[Path]:
    dev = root / LEDGERS["dev"]
    return [dev, meta_path(dev)]


def commit_thresholds(root: Path, manifest: dict, data_sets: dict[str, Dataset]) -> dict:
    """The four thresholds from the development ledger, with the SHA-256 of the ledger and of its metadata file;
    refused unless the ledger is fully admitted and all four cells have every development call and a finite
    threshold."""
    rows, problems = admission(dev_inputs(root)[:1])
    frames, issues = dev_frames(rows, manifest, data_sets)
    t = thresholds_from(frames)
    if problems or set(t) != set(cell_names()) or not all(math.isfinite(v) for v in t.values()):
        raise RuntimeError(f"thresholds not complete: {dict(problems) or issues or t}")
    return {"manifest_sha256": rep.MANIFEST_SHA256, "items": DEV_ITEMS, "thresholds": t,
            "dev_inputs": {str(p.relative_to(root)): sha256(p) for p in dev_inputs(root)}, "issues": issues}


def threshold_problem(root: Path = ROOT, manifest: dict | None = None,
                      data_sets: dict[str, Dataset] | None = None) -> str | None:
    """Why the thresholds cannot open the test phase or enter the analysis: the threshold file, the development
    ledger or its metadata file differ from the last commit; the file names another manifest, does not list exactly
    the two development inputs or their current hashes, or does not hold four finite values; or the development
    replies do not reproduce the values."""
    files = [THRESHOLDS, *(str(p.relative_to(root)) for p in dev_inputs(root))]
    problem = committed_problem(files, root)
    if problem:
        return problem
    committed = json.loads((root / THRESHOLDS).read_text())
    if committed.get("manifest_sha256") != rep.MANIFEST_SHA256:
        return "other manifest"
    inputs = committed.get("dev_inputs", {})
    if set(inputs) != set(files[1:]):
        return "the development ledger and its metadata file must both be listed"
    for name, digest in inputs.items():
        if sha256(root / name) != digest:
            return f"development input {name} changed"
    values = committed.get("thresholds", {})
    if set(values) != set(cell_names()) or not all(isinstance(v, float) and math.isfinite(v)
                                                   for v in values.values()):
        return "four finite thresholds required"
    manifest = manifest or rep.load_manifest(root / "data" / "replication_manifest.json")
    rows, _ = admission(dev_inputs(root)[:1])
    frames, _ = dev_frames(rows, manifest, data_sets or load_data_sets())
    if thresholds_from(frames) != values:
        return "the development replies do not reproduce the committed values"
    return None


def threshold(t: float, test: dict) -> dict:
    """A test record is reviewed when p_min < t; p_min = t is accepted."""
    p = np.exp(-test["scores"]["L-field"])
    review = p < t
    y = test["wrong"]
    return {"t": t, "review_share": float(review.mean()),
            "accepted_error": float(y[~review].mean()) if (~review).any() else None,
            "accepted_error_at_30_percent": accepted_error(test["scores"]["L-field"], y, 0.3, ties(len(y)))}


def format_frames(rows: list[dict], ds: Dataset, ids: list[str], model: str,
                  test_ids: list[str]) -> tuple[list[dict] | None, dict]:
    """Labelled and bare greedy replies of the format items, and the issues of each arm, extras included. The
    labelled replies of a primary model are its test greedy replies, judged against the whole test manifest; those of
    a format-only model come from the format phase."""
    if model in PRIMARY:
        phase, wanted = "test", expected_keys("test", "massive", "labelled", model, test_ids, 1 + SAMPLES)
        lab, p1 = select(rows, wanted)
        p1 = {k: [x for x in v if k == "extra" or (x[4] in ids and x[5] == 0)] for k, v in p1.items()}
        p1 = {k: v for k, v in p1.items() if v}
    else:
        phase = "format"
        lab, p1 = select(rows, expected_keys(phase, "massive", "labelled", model, ids, 1))
    bare, p2 = select(rows, expected_keys("format", "massive", "bare", model, ids, 1))
    issues = {k: v for k, v in (("labelled", p1), ("bare", p2)) if v}
    if fatal(p1) or fatal(p2):
        return None, issues
    frames = [greedy_frame([(ds.items[i], got[ph, "massive", arm, model, i, 0]) for i in ids], ds.onto)
              for got, ph, arm in ((lab, phase, "labelled"), (bare, "format", "bare"))]
    return frames, issues


def analyze(rows: list[dict], manifest: dict, data_sets: dict[str, Dataset], committed: dict) -> dict:
    sets = manifest["sets"]
    matches = set(manifest["snips_test_train_matches"])
    out = {"cells": {}, "format_arm": {}}
    for name in DATASETS:
        ds, ids = data_sets[name], sets[f"{name}_test"]
        for model in PRIMARY:
            cell = f"{name} / {model}"
            chosen, problems = select(rows, expected_keys("test", name, "labelled", model, ids, 1 + SAMPLES))
            if fatal(problems):
                out["cells"][cell] = {"status": "incomplete", "issues": problems}
                continue
            entries = [(ds.items[i], chosen["test", name, "labelled", model, i, 0],
                        [chosen["test", name, "labelled", model, i, k] for k in range(1, SAMPLES + 1)]) for i in ids]
            frame = full_frame(entries, ds.onto)
            result = {"status": "complete", "issues": problems, "primary": primary(frame),
                      "secondary": descriptive(frame),
                      "sensitivity": {"without_malformed": primary(subset(frame, ~frame["malformed"]))}}
            if name == "snips":
                result["sensitivity"]["without_train_matches"] = primary(
                    subset(frame, ~np.isin(frame["ids"], list(matches))))
            t = committed["thresholds"].get(cell)
            result["threshold_transport"] = threshold(t, frame) if t is not None else None
            out["cells"][cell] = result
    out["overall"] = overall(out["cells"])
    for model in FORMAT_MODELS:
        frames, problems = format_frames(rows, data_sets["massive"], sets["format_arm"], model, sets["massive_test"])
        out["format_arm"][model] = ({"status": "incomplete", "issues": problems} if frames is None
                                    else {"status": "complete", "issues": problems, **format_contrast(*frames)})
    out["development"] = dev_frames(rows, manifest, data_sets)[1]
    out["format_shared"] = shared_finding(out["format_arm"])
    return out


def gate_result(root: Path, dataset: Dataset) -> dict:
    """The gate verdict from the stored gate file: the 20 gate requests in order, all from the gate model with call 0,
    one metadata row with the protocol's Ollama version and digest, aligned log-probabilities and well-formed
    records."""
    rows, _ = read_rows(root / GATE_FILES[0])
    metas, _ = read_rows(root / GATE_FILES[1])
    expected = [r.get("id") for r in rows] == list(GATE_IDS) and all(
        (r.get("model"), r.get("dataset"), r.get("call")) == (GATE_MODEL, "massive", 0) for r in rows)
    environment = len(metas) == 1 and environment_problem(metas[0], [GATE_MODEL]) is None
    aligned = sum(reply_problem(r, 0) is None for r in rows)
    well_formed = sum(check(r["text"], dataset.items[r["id"]].text, dataset.onto)[0] is not None for r in rows)
    return {"requests": expected, "environment": environment, "aligned": aligned, "well_formed": well_formed,
            "passed": expected and environment and aligned == len(GATE_IDS) and well_formed >= GATE_WELL_FORMED}


def load_data_sets() -> dict[str, Dataset]:
    return {"massive": build("massive", massive_rows()), "snips": build("snips", snips_rows())}


def write_analysis(root: Path, out: Path, manifest: dict, data_sets: dict[str, Dataset]) -> dict:
    """The analysis of the three ledgers with the committed thresholds; refused when `threshold_problem` finds a
    problem."""
    problem = threshold_problem(root, manifest, data_sets)
    if problem:
        raise RuntimeError(f"thresholds: {problem}")
    committed = json.loads((root / THRESHOLDS).read_text())
    rows, problems = admission([root / LEDGERS[p] for p in ("dev", "test", "format")],
                               thresholds_sha=sha256(root / THRESHOLDS))
    result = analyze(rows, manifest, data_sets, committed)
    result["thresholds"] = committed["thresholds"]
    result["not_admitted"] = dict(problems)
    out.write_text(dumps(result))
    return result


def main(argv=None, root: Path = ROOT):
    parser = argparse.ArgumentParser(prog="python -m xreview.replication_analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("thresholds", help=f"write {THRESHOLDS} from the development ledger")
    a = sub.add_parser("analyze", help="analysis of the three ledgers with the committed thresholds")
    a.add_argument("--out", type=Path, required=True)
    sub.add_parser("gate", help="the gate verdict from the stored gate file")
    args = parser.parse_args(argv)
    manifest = rep.load_manifest(root / "data" / "replication_manifest.json")
    data_sets = load_data_sets()
    if args.command == "thresholds":
        (root / THRESHOLDS).write_text(dumps(commit_thresholds(root, manifest, data_sets)))
    elif args.command == "analyze":
        try:
            write_analysis(root, args.out, manifest, data_sets)
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
    else:
        print(dumps(gate_result(root, data_sets["massive"])), end="")


if __name__ == "__main__":
    main()
