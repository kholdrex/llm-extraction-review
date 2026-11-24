"""Exploratory analyses of the stored test outputs (ADDENDUM.md), written to reanalysis.json."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import stats
from .analysis import BUDGETS, LABELS, RESAMPLES, SEED, accepted_error, rules_first, ties
from .controls import entropy, jaccard_distance, outside_text, permissive_parse, retrieved_disagreement, strict_parse
from .data import Ontology, Utterance
from .detectors import score
from .experiment import load_rows
from .metrics import score as score_record
from .retrieval import Retriever
from .tokens import AGGREGATES, MASKS, token_scores, worst_in_span_is_opening, worst_position
from .validate import check, normalize

FOLDS = 5
FOLD_SEEDS = 20
ONE_CALL = ("field worst", "field mean", "field sum", "json-open worst", "json-open sum", "all sum", "output tokens",
            "R", "predicted slots", "request words", "nearest similarity", "retrieved intents", "retrieved slot types")
SAMPLING = ("D", "D jaccard", "D entropy")
NO_PROBABILITIES = ("R", "predicted slots", "request words", "output tokens", "nearest similarity", "retrieved intents",
                    "retrieved slot types")
COMBINED = {"C ranks": ("R", "D", "L-field"), "no probabilities": NO_PROBABILITIES, "one call": ONE_CALL,
            "one call + samples": ONE_CALL + SAMPLING}
CONTRASTS = (
    ("field sum", "field worst"),
    ("json-open worst", "field worst"),
    ("field worst", "all worst"),
    ("one call", "field worst"),
    ("one call + samples", "one call"),
    ("field worst", "retrieved slot types"),
    ("field worst", "no probabilities"),
    ("one call", "no probabilities"),
)
BUDGET_CONTRASTS = (("L-field", "D"), ("field sum", "D"), ("one call", "D"))
BUDGET_SCORES = ("R", "D", "D jaccard", "L-seq", "L-field", "R+L", "C", "field sum", "json-open worst",
                 "retrieved slot types", "C ranks", "no probabilities", "one call", "one call + samples")
STRATA = ("L-field", "field sum", "json-open worst", "all worst", "D", "D jaccard", "output tokens")


def record_features(row: dict, u: Utterance, onto: Ontology, retriever: Retriever) -> tuple[dict, dict]:
    greedy = row["greedy"]
    record, s = score(greedy, row["samples"], u.text, onto)
    samples = [check(x["text"], u.text, onto)[0] for x in row["samples"]]
    similarity = (retriever.matrix @ retriever.vectorizer.transform([u.text]).T).toarray().ravel().max()
    intents, types = retrieved_disagreement(record, retriever.nearest(u.text))
    f = {"R": s.rules, "D": s.disagreement, "L-seq": s.seq_nll, "L-field": s.field_nll}
    f.update(token_scores(greedy["text"], greedy["tokens"]))
    f.update({
        "output tokens": len(greedy["tokens"]),
        "predicted slots": len(record["slots"]) if record else 0,
        "request words": len(u.text.split()),
        "nearest similarity": 1.0 - similarity,
        "retrieved intents": intents,
        "retrieved slot types": types,
        "D jaccard": float(sum(jaccard_distance(record, x) for x in samples) / len(samples)),
        "D entropy": entropy(samples),
    })
    outside = outside_text(greedy["text"])
    info = {
        "wrong": not score_record(record, u).record,
        "outside": bool(outside),
        "fence": "```" in outside,
        "strict": strict_parse(greedy["text"]),
        "parsed": permissive_parse(greedy["text"]),
        "well_formed": record is not None,
        "worst_position": worst_position(greedy["text"], greedy["tokens"]),
        "opening_worst_in_span": worst_in_span_is_opening(greedy["text"], greedy["tokens"]),
        "api_count_plus_one": greedy["output_tokens"] == len(greedy["tokens"]) + 1,
    }
    return f, info


def model_frame(rows: list[dict], items: dict[str, Utterance], onto: Ontology, retriever: Retriever) -> dict:
    feats, info = zip(*(record_features(r, items[r["id"]], onto, retriever) for r in rows), strict=True)
    scores = {k: np.array([f[k] for f in feats], dtype=float) for k in feats[0]}
    y = np.array([i["wrong"] for i in info])
    scores["R+L"] = rules_first(scores["R"], scores["L-field"])
    scores["C"] = stats.cross_fitted(np.column_stack([scores[c] for c in ("R", "D", "L-field")]), y, FOLDS, SEED)
    for name, cols in COMBINED.items():
        scores[name] = stats.out_of_fold(np.column_stack([scores[c] for c in cols]), y, FOLDS, SEED)
    flags = {k: np.array([i[k] for i in info]) for k in info[0] if k != "wrong"}
    return {"ids": [r["id"] for r in rows], "wrong": y, "scores": scores, "flags": flags}


def auroc_ci(s: np.ndarray, y: np.ndarray) -> dict:
    boot = stats.bootstrap(lambda i: stats.auroc(s[i], y[i]), len(y), RESAMPLES, SEED)
    return {"value": stats.auroc(s, y), "ci": stats.interval(boot)}


def difference(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    boot = stats.bootstrap(lambda i: stats.auroc(a[i], y[i]) - stats.auroc(b[i], y[i]), len(y), RESAMPLES, SEED)
    return {"value": stats.auroc(a, y) - stats.auroc(b, y), "ci": stats.interval(boot)}


def caught(score: np.ndarray, wrong: np.ndarray, budget: float, tie: np.ndarray) -> int:
    """Erroneous records among the share `budget` with the highest scores (ties in the order of `tie`)."""
    order = np.lexsort((tie, -score))
    return int(wrong[order[: int(round(budget * len(score)))]].sum())


def budget_row(score: np.ndarray, y: np.ndarray, tie: np.ndarray) -> dict:
    out = {}
    for b in BUDGETS:
        boot = stats.bootstrap(lambda i, b=b: accepted_error(score[i], y[i], b, tie[i]), len(y), RESAMPLES, SEED)
        out[str(b)] = {"caught_per_1000": 1000 * caught(score, y, b, tie) / len(y),
                       "accepted_error": accepted_error(score, y, b, tie), "accepted_error_ci": stats.interval(boot)}
    return out


def budget_difference(a: np.ndarray, b: np.ndarray, y: np.ndarray, tie: np.ndarray) -> dict:
    def stat(i, budget):
        return [1000 * (caught(a[i], y[i], budget, tie[i]) - caught(b[i], y[i], budget, tie[i])) / len(i),
                accepted_error(a[i], y[i], budget, tie[i]) - accepted_error(b[i], y[i], budget, tie[i])]

    out = {}
    for budget in BUDGETS:
        point = stat(np.arange(len(y)), budget)
        boot = stats.bootstrap(lambda i, budget=budget: stat(i, budget), len(y), RESAMPLES, SEED)
        out[str(budget)] = {key: {"value": point[j], "ci": stats.interval(boot[:, j])}
                            for j, key in enumerate(("caught_per_1000", "accepted_error"))}
    return out


def subset(frame: dict, mask: np.ndarray, names: tuple[str, ...]) -> dict:
    y = frame["wrong"][mask]
    return {"n": int(mask.sum()), "errors": int(y.sum()),
            "auroc": {k: stats.auroc(frame["scores"][k][mask], y) for k in names}}


def summarize(frame: dict, duplicates: set[str]) -> dict:
    y, s, f = frame["wrong"], frame["scores"], frame["flags"]
    n = len(y)
    tie = ties(n)
    unanimous = s["D"] == 0
    slots = s["predicted slots"]
    out = {
        "n": n,
        "errors": int(y.sum()),
        "format": {
            "text_outside_json": int(f["outside"].sum()),
            "code_fence": int(f["fence"].sum()),
            "strict_parse": int(f["strict"].sum()),
            "permissive_parse": int(f["parsed"].sum()),
            "well_formed": int(f["well_formed"].sum()),
            "worst_token_position": {k: int((f["worst_position"] == k).sum()) for k in ("before", "opening",
                                                                                         "inside", "after")},
            "worst_json_token_is_opening": int(f["opening_worst_in_span"].sum()),
            "api_output_tokens_exceed_returned_by_one": int(f["api_count_plus_one"].sum()),
            "record_accuracy_permissive": float(1 - y.mean()),
            "record_accuracy_strict": float((~y & f["strict"]).mean()),
        },
        "unanimous": {"requests": int(unanimous.sum()), "wrong": int(y[unanimous].sum())},
        "auroc": {k: auroc_ci(v, y) for k, v in s.items()},
        "contrasts": {f"{a} - {b}": difference(s[a], s[b], y) for a, b in CONTRASTS},
        "budgets": {k: budget_row(s[k], y, tie) for k in BUDGET_SCORES},
        "budget_contrasts": {f"{a} - {b}": budget_difference(s[a], s[b], y, tie) for a, b in BUDGET_CONTRASTS},
        "rules_first": {"flagged": int(s["R"].sum()),
                        "flagged_reviewed": {str(b): int(min(s["R"].sum(), round(b * n))) for b in BUDGETS}},
        "by_predicted_slots": {label: subset(frame, m, STRATA) for label, m in
                               (("0", slots == 0), ("1", slots == 1), ("2+", slots >= 2))},
        "by_format": {label: subset(frame, m, STRATA) for label, m in
                      (("text outside JSON", f["outside"]), ("JSON only", ~f["outside"])) if 0 < y[m].sum() < m.sum()},
        "without_train_duplicates": subset(frame, np.array([i not in duplicates for i in frame["ids"]]),
                                           tuple(k for k in s if k not in COMBINED)),
    }
    rows = (np.column_stack([s[c] for c in cols]) for cols in COMBINED.values())
    out["combined_over_fold_seeds"] = {}
    for name, x in zip(COMBINED, rows, strict=True):
        values = [stats.auroc(stats.out_of_fold(x, y, FOLDS, seed), y) for seed in range(SEED, SEED + FOLD_SEEDS)]
        out["combined_over_fold_seeds"][name] = {"min": min(values), "max": max(values)}
    return out


def token_figure(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(summary), figsize=(10.5, 3.9), sharey=True)
    styles = {"worst": ("black", "-", "o"), "mean": ("0.45", "--", "s"), "sum": ("0.2", ":", "^")}
    x = np.arange(len(MASKS))
    names = ("all tokens", "JSON span", "JSON span\nwithout\nopening token", "first to last\nfield string",
             "field strings")
    for ax, (model, s) in zip(np.atleast_1d(axes), summary.items(), strict=True):
        for agg in AGGREGATES:
            color, ls, marker = styles[agg]
            ax.plot(x, [s["auroc"][f"{m} {agg}"]["value"] for m in MASKS], color=color, ls=ls, marker=marker,
                    label=agg)
        ax.axhline(s["auroc"]["D"]["value"], color="0.6", lw=1, label="D, disagreement")
        ax.set_xticks(x, names, fontsize=9)
        ax.set_title(LABELS.get(model, model), fontsize=11)
        ax.grid(color="0.9")
    np.atleast_1d(axes)[0].set_ylabel("AUROC")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def caught_figure(frames: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(10.5, 3.4), sharey=True)
    styles = {"D": ("0.55", ":", "D, disagreement"), "L-field": ("black", "-", "field worst"),
              "field sum": ("0.35", "--", "field sum"), "one call": ("black", "-.", "one-call combination")}
    budgets = np.linspace(0, 0.6, 61)
    for ax, (model, frame) in zip(np.atleast_1d(axes), frames.items(), strict=True):
        y = frame["wrong"]
        tie = ties(len(y))
        for name, (color, ls, label) in styles.items():
            ax.plot(budgets * 100, [1000 * caught(frame["scores"][name], y, b, tie) / len(y) for b in budgets],
                    color=color, ls=ls, lw=1.4, label=label)
        ax.plot(budgets * 100, budgets * 1000 * y.mean(), color="0.75", lw=0.8, label="random order")
        ax.set_title(LABELS.get(model, model), fontsize=11)
        ax.set_xlabel("Records sent to review, %")
        ax.grid(color="0.9")
    np.atleast_1d(axes)[0].set_ylabel("Errors caught per 1,000 requests")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(onto: Ontology, test: list[Utterance], retriever: Retriever, raw: Path, figures: Path) -> None:
    """reanalysis.json next to `raw`; figures to `figures`."""
    items = {u.id: u for u in test}
    rows = [r for r in load_rows(raw) if r["id"] in items]
    if any(n > 1 for n in Counter((r["model"], r["id"]) for r in rows).values()):
        raise ValueError("model/request pairs occur more than once")
    train = {normalize(u.text) for u in retriever.train}
    duplicates = {u.id for u in test if normalize(u.text) in train}
    models = [m for m in LABELS if any(r["model"] == m for r in rows)]
    frames = {m: model_frame([r for r in rows if r["model"] == m], items, onto, retriever) for m in models}
    summary = {"train_duplicates": sorted(duplicates, key=int)}
    summary.update({m: summarize(frame, duplicates) for m, frame in frames.items()})
    (raw.parent / "reanalysis.json").write_text(json.dumps(summary, indent=1) + "\n")
    figures.mkdir(parents=True, exist_ok=True)
    token_figure({m: summary[m] for m in models}, figures / "token_sets.png")
    caught_figure(frames, figures / "errors_caught.png")
