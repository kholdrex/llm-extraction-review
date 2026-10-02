import argparse
import json
from pathlib import Path

from . import data, experiment, prompts
from .retrieval import Retriever

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 2026
N_DEV = 100
N_TEST = 1000
NEIGHBOURS = 8
MODELS = ["qwen2.5:3b-instruct", "gemma3:4b", "qwen2.5:7b-instruct"]


def setup():
    rows = data.load(data.download(DATA / "massive"))
    train = [u for u in rows if u.partition == "train"]
    onto = data.ontology(train)
    shots = data.examples(train, SEED)
    split = {
        "dev": data.sample([u for u in rows if u.partition == "dev"], N_DEV, SEED),
        "test": data.sample([u for u in rows if u.partition == "test"], N_TEST, SEED),
    }
    return onto, prompts.system_prompt(onto, shots), Retriever(train, NEIGHBOURS), split


def cmd_prepare(_):
    onto, system, _, split = setup()
    manifest = {
        "seed": SEED,
        "dev": [u.id for u in split["dev"]],
        "test": [u.id for u in split["test"]],
        "intents": len(onto.intents),
        "slot_types": len(onto.slot_types),
    }
    (DATA / "sample.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (DATA / "system_prompt.txt").write_text(system + "\n")


def cmd_run(args):
    _, system, retriever, split = setup()
    items = split[args.split][: args.limit] if args.limit else split[args.split]
    out = args.out or RESULTS / f"raw_{args.split}.jsonl"
    experiment.run(args.models, items, system, retriever, out)


def cmd_analyze(args):
    from . import analysis

    onto, _, _, split = setup()
    analysis.main(onto, split["test"], args.raw, args.figures)


def main():
    parser = argparse.ArgumentParser(prog="xreview")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    run = sub.add_parser("run")
    run.add_argument("split", choices=["dev", "test"])
    run.add_argument("--models", nargs="+", default=MODELS)
    run.add_argument("--limit", type=int)
    run.add_argument("--out", type=Path, help="output file; an existing file is resumed")
    run.set_defaults(func=cmd_run)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--raw", type=Path, default=RESULTS / "raw_test.jsonl",
                         help="model outputs to analyse; summary.json is written next to this file")
    analyze.add_argument("--figures", type=Path, default=ROOT / "figures")
    analyze.set_defaults(func=cmd_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
