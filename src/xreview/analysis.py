"""Tables, figures and summary.json for the test run."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import stats
from .data import Ontology, Utterance
from .detectors import record_key, score
from .errors import KINDS, kinds
from .experiment import SAMPLES, load_rows
from .metrics import score as score_record
from .validate import check

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})

SEED = 2026
RESAMPLES = 2000
BUDGETS = (0.1, 0.2, 0.3)
CONTRASTS = (("L-field", "D"), ("L-field", "R"), ("D", "R"), ("C", "L-field"), ("R+L", "L-field"), ("L-seq", "L-field"))
LABELS = {
    "qwen2.5:3b-instruct": "Qwen2.5-3B",
    "gemma3:4b": "Gemma 3 4B",
    "qwen2.5:7b-instruct": "Qwen2.5-7B",
}


def majority(texts: list[str], utterance: str, onto: Ontology) -> tuple | None:
    """Most frequent record among the parsable outputs (ties: the earliest output); None if none can be parsed."""
    keys = [record_key(check(t, utterance, onto)[0]) for t in texts]
    counts = Counter(k for k in keys if k is not None)
    return counts.most_common(1)[0][0] if counts else None


def gold_key(u: Utterance) -> tuple:
    return record_key({"intent": u.intent, "slots": [{"type": t, "value": v} for t, v in u.slots]})


def model_frame(rows: list[dict], items: dict[str, Utterance], onto: Ontology) -> dict:
    wrong, scores, fields, seconds, tokens, vote, error_kinds, partial = [], [], [], [], [], [], [], []
    for r in rows:
        u = items[r["id"]]
        record, s = score(r["greedy"], r["samples"], u.text, onto)
        result = score_record(record, u)
        wrong.append(not result.record)
        scores.append([s.rules, s.disagreement, s.seq_nll, s.field_nll])
        fields.append((result.intent, result.tp, result.fp, result.fn))
        seconds.append([r["greedy"]["seconds"]] + [x["seconds"] for x in r["samples"]])
        tokens.append(r["greedy"]["prompt_tokens"] + r["greedy"]["output_tokens"])
        texts = [r["greedy"]["text"]] + [x["text"] for x in r["samples"]]
        vote.append(majority(texts, u.text, onto) == gold_key(u))
        error_kinds.append(kinds(record, u))
        samples = r["samples"]
        partial.append([score(r["greedy"], samples[:k], u.text, onto)[1].disagreement for k in range(1, SAMPLES + 1)])
    x = np.array(scores)
    y = np.array(wrong)
    combined = stats.cross_fitted(x[:, [0, 1, 3]], y, folds=5, seed=SEED)
    return {
        "wrong": y,
        "scores": {"R": x[:, 0], "D": x[:, 1], "L-seq": x[:, 2], "L-field": x[:, 3], "C": combined,
                   "R+L": rules_first(x[:, 0], x[:, 3])},
        "fields": np.array(fields),
        "seconds": np.array(seconds),
        "tokens": np.array(tokens),
        "vote": np.array(vote),
        "kinds": error_kinds,
        "partial": np.array(partial),
    }


def rules_first(rules: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Ranking score that puts every record flagged by the rules above all others, then orders by the field score."""
    return stats.ranks(field) + rules * len(field)


def accepted_error(score: np.ndarray, wrong: np.ndarray, budget: float, tie: np.ndarray) -> float:
    """Error rate among records left after the share `budget` with the highest scores is sent to review."""
    order = np.lexsort((tie, -score))
    kept = order[int(round(budget * len(score))) :]
    return float(wrong[kept].mean())


def ties(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).random(n)


def kind_shares(frame: dict) -> dict:
    """Share of erroneous records with each kind of error, and the share of them sent to review at a 30% budget."""
    y = frame["wrong"]
    out = {}
    for kind in KINDS:
        has = np.array([kind in k for k in frame["kinds"]])
        if not has.any():
            continue
        entry = {"share_of_errors": float(has[y].mean())}
        for name in ("R", "L-field"):
            s = frame["scores"][name]
            order = np.lexsort((ties(len(s)), -s))
            reviewed = np.zeros(len(s), dtype=bool)
            reviewed[order[: int(round(0.3 * len(s)))]] = True
            entry[f"reviewed_{name}"] = float(reviewed[has].mean())
        out[kind] = entry
    return out


def probability_bins(frame: dict) -> list[dict]:
    """Error rate of records grouped by the probability of their least certain field token."""
    p = np.exp(-frame["scores"]["L-field"])
    y = frame["wrong"]
    edges = [0.0, 0.5, 0.8, 0.95, 0.99, 1.0 + 1e-9]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        out.append({"from": lo, "to": min(hi, 1.0), "share": float(m.mean()),
                    "error": float(y[m].mean()) if m.any() else None})
    return out


def summarize(frame: dict) -> dict:
    y = frame["wrong"]
    n = len(y)
    intent, tp, fp, fn = frame["fields"].T
    out = {
        "n": n,
        "record_accuracy": float(1 - y.mean()),
        "intent_accuracy": float(intent.mean()),
        "slot_precision": float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
        "slot_recall": float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
        "flagged_by_rules": float(frame["scores"]["R"].mean()),
        "rules_precision": float(y[frame["scores"]["R"] == 1].mean()),
        "rules_recall": float((frame["scores"]["R"][y] == 1).mean()),
        "vote_accuracy": float(frame["vote"].mean()),
        "greedy_seconds": float(frame["seconds"][:, 0].mean()),
        "sample_seconds": float(frame["seconds"][:, 1:].sum(axis=1).mean()),
        "greedy_tokens": float(frame["tokens"].mean()),
        "auroc": {},
        "difference": {},
        "accepted_error": {},
    }
    out["error_kinds"] = kind_shares(frame)
    out["disagreement_by_k"] = [stats.auroc(frame["partial"][:, k], y) for k in range(SAMPLES)]
    out["probability_bins"] = probability_bins(frame)
    out["record_accuracy_ci"] = stats.interval(stats.bootstrap(lambda i: 1 - y[i].mean(), n, RESAMPLES, SEED))
    for name, s in frame["scores"].items():
        boot = stats.bootstrap(lambda i, s=s: stats.auroc(s[i], y[i]), n, RESAMPLES, SEED)
        out["auroc"][name] = {"value": stats.auroc(s, y), "ci": stats.interval(boot)}
        out["accepted_error"][name] = {str(b): accepted_error(s, y, b, ties(n)) for b in BUDGETS}
    for a, b in CONTRASTS:
        sa, sb = frame["scores"][a], frame["scores"][b]
        diff = stats.bootstrap(lambda i, sa=sa, sb=sb: stats.auroc(sa[i], y[i]) - stats.auroc(sb[i], y[i]), n,
                               RESAMPLES, SEED)
        out["difference"][f"{a} - {b}"] = {"value": stats.auroc(sa, y) - stats.auroc(sb, y), "ci": stats.interval(diff)}
    return out


def risk_coverage(frames: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(10.5, 3.6), sharey=True)
    styles = {"R": ("0.55", "--", "R, rules"), "D": ("0.35", ":", "D, disagreement"),
              "L-field": ("black", "-", "L field"), "R+L": ("black", "-.", "R, then L field")}
    for ax, (model, frame) in zip(np.atleast_1d(axes), frames.items(), strict=True):
        y = frame["wrong"]
        budgets = np.linspace(0, 0.6, 61)
        for name, (color, ls, label) in styles.items():
            tie = ties(len(y))
            err = [accepted_error(frame["scores"][name], y, b, tie) for b in budgets]
            ax.plot(budgets * 100, np.array(err) * 100, color=color, ls=ls, lw=1.4, label=label)
        ax.axhline(y.mean() * 100, color="0.7", lw=0.8)
        ax.set_title(LABELS.get(model, model), fontsize=11)
        ax.set_xlabel("Records sent to review, %")
        ax.grid(color="0.9")
    np.atleast_1d(axes)[0].set_ylabel("Errors among accepted, %")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def probability_histograms(frames: dict, path: Path) -> None:
    """Distribution of the least certain field probability for correct and erroneous records."""
    fig, axes = plt.subplots(1, len(frames), figsize=(10.5, 3.2), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (model, frame) in zip(np.atleast_1d(axes), frames.items(), strict=True):
        p = np.exp(-frame["scores"]["L-field"])
        y = frame["wrong"]
        for mask, color, label in ((~y, "0.75", "correct"), (y, "black", "erroneous")):
            ax.hist(p[mask], bins=bins, weights=np.full(mask.sum(), 100 / len(p)), color=color,
                    histtype="stepfilled" if color == "0.75" else "step", lw=1.4, label=label)
        ax.set_title(LABELS.get(model, model), fontsize=11)
        ax.set_xlabel("Probability of the least certain field")
        ax.grid(color="0.9")
    np.atleast_1d(axes)[0].set_ylabel("Records, %")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def samples_curve(summary: dict, path: Path) -> None:
    """AUROC of disagreement computed from the first k samples, with the field score as a reference."""
    fig, axes = plt.subplots(1, len(summary), figsize=(10.5, 2.9), sharey=True)
    k = np.arange(1, SAMPLES + 1)
    for ax, (model, s) in zip(np.atleast_1d(axes), summary.items(), strict=True):
        ax.plot(k, s["disagreement_by_k"], color="black", marker="o", label="D, disagreement")
        ax.axhline(s["auroc"]["L-field"]["value"], color="black", ls="--", label="L field")
        ax.axhline(s["auroc"]["R"]["value"], color="0.55", ls=":", label="R, rules")
        ax.set_title(LABELS.get(model, model), fontsize=11)
        ax.set_xticks(k)
        ax.set_xlabel("Number of samples")
        ax.grid(color="0.9")
    np.atleast_1d(axes)[0].set_ylabel("AUROC")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(onto: Ontology, test: list[Utterance], raw: Path, figures: Path) -> None:
    """Summary of the outputs in `raw`, written to summary.json next to it; figures to `figures`."""
    items = {u.id: u for u in test}
    rows = [r for r in load_rows(raw) if r["id"] in items]
    keys = Counter((r["model"], r["id"]) for r in rows)
    repeated = [k for k, n in keys.items() if n > 1]
    if repeated:
        raise ValueError(f"{len(repeated)} model/request pairs occur more than once, e.g. {repeated[0]}")
    order = list(LABELS)
    models = sorted({r["model"] for r in rows}, key=lambda m: (order.index(m) if m in order else len(order), m))
    if not models:
        raise ValueError(f"no outputs for the test requests in {raw}")
    frames = {m: model_frame([r for r in rows if r["model"] == m], items, onto) for m in models}
    summary = {model: summarize(frame) for model, frame in frames.items()}
    (raw.parent / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    figures.mkdir(parents=True, exist_ok=True)
    risk_coverage(frames, figures / "risk_coverage.png")
    probability_histograms(frames, figures / "field_probability.png")
    samples_curve(summary, figures / "samples.png")
