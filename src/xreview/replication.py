"""External replication (REPLICATION_PROTOCOL.md): data sets, manifest, prompt arms and a runner that stores one row
per attempt in one ledger per phase."""

import argparse
import fcntl
import hashlib
import json
import math
import random
import subprocess
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import requests

from . import data, prompts, snips
from .data import Ontology, Utterance
from .llm import OLLAMA_URL
from .retrieval import Retriever
from .validate import normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFEST = DATA / "replication_manifest.json"
MANIFEST_SHA256 = "5195cec4c044ec915af49707bb26868472f2163b2ac8d15c03bec7aee513322a"
LEDGERS = {"dev": "results/replication/dev.jsonl", "test": "results/replication/test.jsonl",
           "format": "results/replication/format.jsonl"}
THRESHOLDS = "results/replication/thresholds.json"
GATE_FILES = ("results/replication/gate.jsonl", "results/replication/gate.meta.jsonl")
FROZEN_FILES = ("REPLICATION_PROTOCOL.md", "data/replication_manifest.json", *GATE_FILES)
FROZEN_CODE = tuple(f"src/xreview/{m}.py" for m in (
    "__init__", "analysis", "cli", "controls", "data", "detectors", "errors", "experiment", "llm", "metrics",
    "prompts", "reanalysis", "replication", "replication_analysis", "retrieval", "snips", "stats", "tokens",
    "validate"))
SEED = 2027
FORMAT_SEED = 2028
EXAMPLE_SEED = 2026
NEIGHBOURS = 8
SAMPLES = 4
SAMPLE_TEMPERATURE = 0.7
ATTEMPTS = 3
OLLAMA_VERSION = "0.34.3"
DIGESTS = {
    "qwen2.5:7b-instruct": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
    "ibm/granite4.1:3b": "6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb",
    "qwen2.5:3b-instruct": "357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b",
    "gemma3:4b": "a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
}
PRIMARY = ("qwen2.5:7b-instruct", "ibm/granite4.1:3b")
FORMAT_ONLY = ("qwen2.5:3b-instruct", "gemma3:4b")
FORMAT_MODELS = PRIMARY + FORMAT_ONLY
GATE_MODEL = "ibm/granite4.1:3b"
GATE_IDS = ("251", "657", "1197", "1225", "1241", "1327", "1430", "1525", "1545", "1771", "1934", "2118", "2246",
            "2421", "2846", "2963", "3194", "3203", "3248", "3296")
GATE_WELL_FORMED = 18
SETS = {"massive_dev": ("massive", "dev"), "snips_dev": ("snips", "dev"), "massive_test": ("massive", "test"),
        "snips_test": ("snips", "test"), "format_arm": ("massive", "format")}


@dataclass
class Dataset:
    name: str
    items: dict[str, Utterance]
    onto: Ontology
    shots: list[Utterance]
    retriever: Retriever


def massive_rows() -> list[Utterance]:
    return data.load(data.download(DATA / "massive"))


def snips_rows() -> list[Utterance]:
    return snips.load(snips.download(DATA / "snips"))


def build(name: str, rows: list[Utterance], exclude: set[str] = frozenset()) -> Dataset:
    """Ontology and fixed examples from the training split; retrieval from the training split without `exclude`."""
    train = [u for u in rows if u.partition == "train"]
    return Dataset(name, {u.id: u for u in rows}, data.ontology(train), data.examples(train, EXAMPLE_SEED),
                   Retriever([u for u in train if u.id not in exclude], NEIGHBOURS))


def datasets() -> dict[str, Dataset]:
    return {"massive": build("massive", massive_rows()), "snips": build("snips", snips_rows())}


def texts(rows, ids=None) -> set[str]:
    return {normalize(u.text) for u in rows if ids is None or u.id in ids}


def by_number(u: Utterance) -> int:
    return int(u.id)


def by_line(u: Utterance) -> int:
    return snips.line(u.id)


def draw(pool: list[Utterance], n: int, seed: int, key) -> list[str]:
    return [u.id for u in sorted(random.Random(seed).sample(pool, n), key=key)]


def manifest(massive: list[Utterance], snips_all: list[Utterance], october: dict, gate: tuple[str, ...]) -> dict:
    """Item ids of every set, in run order. Drawn in this order, each pool excluding by normalised text the training
    split and every set it could collide with: fresh MASSIVE test, MASSIVE development, SNIPS development."""
    october_ids = set(october["test"]) | set(october["dev"])
    m_train = texts(u for u in massive if u.partition == "train")
    s_train = texts(u for u in snips_all if u.partition == "train")
    taken = m_train | texts(massive, october_ids) | texts(massive, set(gate))
    m_test_pool = sorted((u for u in massive if u.partition == "test" and u.id not in october_ids
                          and normalize(u.text) not in taken), key=by_number)
    massive_test = draw(m_test_pool, 500, SEED, by_number)
    taken |= texts(massive, set(massive_test))
    m_dev_pool = sorted((u for u in massive if u.partition == "dev" and u.id not in october_ids
                         and u.id not in gate and normalize(u.text) not in taken), key=by_number)
    s_test = [u for u in snips_all if u.partition == "test"]
    s_taken = s_train | texts(s_test)
    s_dev_pool = [u for u in snips_all if u.partition == "dev" and normalize(u.text) not in s_taken]
    shuffled = list(massive_test)
    random.Random(FORMAT_SEED).shuffle(shuffled)
    groups: dict[str, list[str]] = {}
    for u in s_test:
        groups.setdefault(normalize(u.text), []).append(u.id)
    return {
        "seeds": {"draw": SEED, "format_arm": FORMAT_SEED, "fixed_examples": EXAMPLE_SEED},
        "pools": {"massive_test": len(m_test_pool), "massive_dev": len(m_dev_pool), "snips_test": len(s_test),
                  "snips_dev": len(s_dev_pool)},
        "sets": {
            "snips_test": [u.id for u in s_test],
            "massive_test": massive_test,
            "snips_dev": draw(s_dev_pool, 100, SEED, by_line),
            "massive_dev": draw(m_dev_pool, 100, SEED, by_number),
            "format_arm": sorted(shuffled[:200], key=int),
        },
        "gate": list(gate),
        "snips_test_train_matches": [u.id for u in s_test if normalize(u.text) in s_train],
        "snips_test_text_duplicates": [ids for ids in groups.values() if len(ids) > 1],
        "fixed_examples": {
            "massive": [u.id for u in data.examples([u for u in massive if u.partition == "train"], EXAMPLE_SEED)],
            "snips": [u.id for u in data.examples([u for u in snips_all if u.partition == "train"], EXAMPLE_SEED)],
        },
        "snips_source": {"repository": "https://github.com/MiuLab/SlotGated-SLU", "commit": snips.COMMIT,
                         "sha256": snips.SHA256},
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST, expected: str | None = None) -> dict:
    if sha256(path) != (expected or MANIFEST_SHA256):
        raise RuntimeError(f"{path} does not match the manifest of the protocol")
    return json.loads(path.read_text())


def roster(set_name: str, arm: str) -> tuple[str, ...]:
    """Models of a run: the primary models for development and test, all four for the bare format arm, the two
    format-only models for the labelled format arm."""
    if arm not in ("labelled", "bare"):
        raise ValueError(f"unknown prompt arm {arm}")
    phase = SETS[set_name][1]
    if phase != "format":
        if arm != "labelled":
            raise ValueError("development and test runs use the labelled prompt")
        return PRIMARY
    return FORMAT_MODELS if arm == "bare" else FORMAT_ONLY


def calls_per_item(phase: str) -> int:
    return 1 + SAMPLES if phase in ("test", "smoke-test") else 1


def bare_example_block(shots: list[Utterance]) -> str:
    return "\n".join(f"Request: {u.text}\n{json.dumps(prompts.record(u))}" for u in shots)


def messages(ds: Dataset, u: Utterance, arm: str) -> list[dict]:
    """The October prompt (`labelled`), or the same prompt with the label "JSON: " removed from every worked example
    (`bare`)."""
    similar = ds.retriever.nearest(u.text)
    if arm == "labelled":
        return prompts.messages(prompts.system_prompt(ds.onto, ds.shots), u.text, similar)
    if arm != "bare":
        raise ValueError(f"unknown prompt arm {arm}")
    system = prompts.SYSTEM.format(intents=", ".join(ds.onto.intents), slot_types=", ".join(ds.onto.slot_types))
    system += "\n\nExamples:\n" + bare_example_block(ds.shots)
    user = f"Similar annotated requests:\n{bare_example_block(similar)}\n\nRequest: {u.text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def request_body(model: str, msgs: list[dict], temperature: float, seed: int, logprobs: bool) -> dict:
    """The request of `xreview.llm.chat`."""
    return {"model": model, "messages": msgs, "stream": False, "keep_alive": "30m", "logprobs": logprobs,
            "options": {"temperature": temperature, "seed": seed, "num_predict": 256, "num_ctx": 4096}}


def ollama_chat(model: str, msgs: list[dict], temperature: float = 0.0, seed: int = 0, logprobs: bool = False) -> dict:
    """The reply as received, with the stop reason; token metadata is read only when log-probabilities were
    requested. Its shape is checked by `reply_problem` before it can count."""
    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=request_body(model, msgs, temperature, seed, logprobs), timeout=600)
    response.raise_for_status()
    seconds = time.perf_counter() - start
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
        raise ValueError("reply is not an object with a message")
    tokens = (payload.get("logprobs") or []) if logprobs else []
    if not isinstance(tokens, list) or not all(isinstance(t, dict) for t in tokens):
        raise ValueError("log-probabilities are not a list of objects")
    return {"text": payload["message"].get("content"), "tokens": [[t.get("token"), t.get("logprob")] for t in tokens],
            "prompt_tokens": payload.get("prompt_eval_count", 0), "output_tokens": payload.get("eval_count", 0),
            "seconds": seconds, "done_reason": payload.get("done_reason")}


def reply_problem(reply, index: int) -> str | None:
    """Why a reply cannot count: not an object, content not a string, or, for call 0, a token list that is empty,
    has entries that are not (string, finite number), or does not concatenate to the content."""
    if not isinstance(reply, dict) or not isinstance(reply.get("text"), str):
        return "reply content is not a string"
    if index:
        return None
    tokens = reply.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        return "no token log-probabilities"
    for t in tokens:
        if not (isinstance(t, (list, tuple)) and len(t) == 2 and isinstance(t[0], str)
                and isinstance(t[1], (int, float)) and not isinstance(t[1], bool) and math.isfinite(t[1])):
            return "token log-probability is not a (string, finite number) pair"
    if "".join(t for t, _ in tokens) != reply["text"]:
        return "token log-probabilities not aligned with the reply"
    return None


def settings(index: int) -> dict:
    """Call 0 is greedy with log-probabilities; calls 1-4 are samples at temperature 0.7 with seed `index`."""
    if index == 0:
        return {"temperature": 0.0, "seed": 0, "logprobs": True}
    return {"temperature": SAMPLE_TEMPERATURE, "seed": index, "logprobs": False}


def key(row: dict) -> tuple:
    return row["phase"], row["dataset"], row["arm"], row["model"], row["id"], row["call"]


def meta_path(ledger: Path) -> Path:
    return ledger.with_name(ledger.name + ".meta.jsonl")


def read_rows(path: Path) -> tuple[list[dict], str]:
    """Complete rows and the incomplete fragment after the last newline ("" if none). A last line without newline
    that is a complete JSON object counts as a row."""
    if not path.exists():
        return [], ""
    text = path.read_text()
    body, last = (text, "") if text.endswith("\n") else (text.rpartition("\n")[0], text.rpartition("\n")[2])
    rows = [json.loads(line) for line in body.splitlines() if line]
    if last:
        try:
            rows.append(json.loads(last))
            last = ""
        except json.JSONDecodeError:
            pass
    return rows, last


def repair(path: Path) -> None:
    """Before appending: end a complete last row with a newline, or move an incomplete last line to `<path>.torn`
    with its byte offset and cut the file at the last newline."""
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    cut = raw.rfind(b"\n") + 1
    fragment = raw[cut:].decode("utf-8", errors="replace")
    try:
        json.loads(fragment)
        with path.open("ab") as f:
            f.write(b"\n")
    except json.JSONDecodeError:
        with path.with_name(path.name + ".torn").open("a") as f:
            f.write(json.dumps({"offset": cut, "fragment": fragment}) + "\n")
        with path.open("r+b") as f:
            f.truncate(cut)


@contextmanager
def single_writer(ledger: Path):
    """One lock over the ledger and its metadata file."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_name(ledger.name + ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"another process is writing {ledger}") from e
        yield


def environment_problem(info: dict, models) -> str | None:
    if info.get("ollama") != OLLAMA_VERSION:
        return f"Ollama {info.get('ollama')}, protocol {OLLAMA_VERSION}"
    for m in models:
        if m not in DIGESTS or info.get("digests", {}).get(m) != DIGESTS[m]:
            return f"{m}: digest {info.get('digests', {}).get(m)}, protocol {DIGESTS.get(m)}"
    return None


def header_problem(header: dict, manifest_sha: str, thresholds_sha: str | None) -> str | None:
    """Why an invocation header is not one the protocol admits."""
    set_name, arm = header.get("set"), header.get("arm")
    if set_name not in SETS or header.get("phase") != SETS[set_name][1]:
        return f"set {set_name} and phase {header.get('phase')} do not match"
    try:
        models = roster(set_name, arm)
    except ValueError as e:
        return str(e)
    if header.get("models") != list(models):
        return "models are not the roster of the set and arm"
    problem = environment_problem(header, models)
    if problem:
        return problem
    if header.get("manifest_sha256") != manifest_sha:
        return "other manifest"
    if not header.get("commit"):
        return "no commit recorded"
    bound = thresholds_sha and header.get("thresholds_sha256") == thresholds_sha
    if header["phase"] in ("test", "format") and not bound:
        return "not bound to the committed thresholds"
    return None


def history(rows: list[dict]) -> tuple[dict, Counter]:
    """Per call, in ledger order: the successes and the failures that preceded the first success."""
    ok: dict[tuple, list[dict]] = {}
    failures = Counter()
    for r in rows:
        k = key(r)
        if r["status"] == "ok":
            ok.setdefault(k, []).append(r)
        elif k not in ok:
            failures[k] += 1
    return ok, failures


def admission(ledgers: list[Path], manifest_sha: str | None = None,
              thresholds_sha: str | None = None) -> tuple[list[dict], Counter]:
    """The rows the analysis admits and what it does not admit. Headers are collected from all ledgers first; an
    invocation with any header the protocol does not admit, or with two different headers, is invalid wherever it
    appears. A row counts only if its invocation is valid and its phase, data set, arm and model agree with that
    header. Incomplete last lines are reported."""
    manifest_sha = manifest_sha or MANIFEST_SHA256
    headers, bad, problems = {}, set(), Counter()
    for ledger in ledgers:
        metas, tail = read_rows(meta_path(ledger))
        if tail:
            problems[f"{meta_path(ledger).name}: incomplete last line"] += 1
        for h in metas:
            inv = h.get("invocation")
            problem = header_problem(h, manifest_sha, thresholds_sha)
            if problem is None and inv in headers and headers[inv] != h:
                problem = "two different headers"
            if problem:
                bad.add(inv)
                problems[f"invocation {inv}: {problem}"] += 1
            headers.setdefault(inv, h)
    rows = []
    for ledger in ledgers:
        got, tail = read_rows(ledger)
        if tail:
            problems[f"{ledger.name}: incomplete last line"] += 1
        for r in got:
            h = headers.get(r.get("invocation"))
            if h is None or r.get("invocation") in bad:
                problems[f"{ledger.name}: row of an unknown or invalid invocation"] += 1
            elif ((r.get("phase"), r.get("dataset"), r.get("arm")) != (h["phase"], SETS[h["set"]][0], h["arm"])
                  or r.get("model") not in h["models"]):
                problems[f"{ledger.name}: row does not match its invocation"] += 1
            else:
                rows.append(r)
    ok, failures = history(rows)
    for k, found in ok.items():
        if failures[k] >= ATTEMPTS:
            problems["success after three failed attempts"] += 1
        if len(found) > 1:
            problems["more than one success for a call"] += 1
    return rows, problems


def run(client, header: dict, dataset: Dataset, ids: list[str], out: Path, manifest_sha: str | None = None,
        thresholds_sha: str | None = None, check: bool = True) -> dict:
    """One row per attempt. Under one lock over the ledger and its metadata file: both tails are repaired; if
    `check`, the run stops before any call when the header or anything already in the ledger is not admitted
    (`admission`); the header is appended before the first call, so an invocation without calls writes nothing. A
    call with an `ok` row is not sent again. A failed attempt (`error`: the request failed or raised; `invalid`: the
    reply fails `reply_problem`, kept) is retried; a call with `ATTEMPTS` failed attempts is exhausted: it is not
    sent again and the rest of that model's cell is skipped. Other models continue."""
    phase, arm = header["phase"], header["arm"]
    calls = calls_per_item(phase)
    exhausted, written = [], False
    with single_writer(out):
        repair(out)
        repair(meta_path(out))
        if check:
            problem = header_problem(header, manifest_sha or MANIFEST_SHA256, thresholds_sha)
            _, problems = admission([out], manifest_sha, thresholds_sha)
            if problem or problems:
                raise RuntimeError(f"{out} not admitted: {problem or dict(problems)}")
        rows, _ = read_rows(out)
        done = {key(r) for r in rows if r["status"] == "ok"}
        attempts = Counter(key(r) for r in rows)
        failures = Counter(key(r) for r in rows if r["status"] != "ok")
        with out.open("a") as f:
            for model in header["models"]:
                stopped = False
                for item in ids:
                    msgs = messages(dataset, dataset.items[item], arm)
                    for index in range(calls):
                        k = (phase, dataset.name, arm, model, item, index)
                        while k not in done and not stopped:
                            if failures[k] >= ATTEMPTS:
                                exhausted.append(list(k))
                                stopped = True
                                break
                            if not written:
                                with meta_path(out).open("a") as m:
                                    m.write(json.dumps(header) + "\n")
                                written = True
                            attempts[k] += 1
                            row = dict(zip(("phase", "dataset", "arm", "model", "id", "call"), k, strict=True))
                            row.update(invocation=header["invocation"], attempt=attempts[k])
                            try:
                                reply = client(model, msgs, **settings(index))
                                problem = reply_problem(reply, index)
                            except Exception as e:
                                row.update(status="error", error=f"{type(e).__name__}: {e}")
                            else:
                                if isinstance(reply, dict):
                                    row.update(reply)
                                row.update(status="invalid" if problem else "ok")
                                if problem:
                                    row["error"] = problem
                            f.write(json.dumps(row) + "\n")
                            f.flush()
                            if row["status"] == "ok":
                                done.add(k)
                            else:
                                failures[k] += 1
                        if stopped:
                            break
                    if stopped:
                        break
    return {"exhausted": exhausted, "header_written": written}


def server_info(models) -> dict:
    """Ollama version and the full digest of each model, as reported by the server."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    version = requests.get(f"{base}/api/version", timeout=30).json()["version"]
    tags = {m["name"]: m["digest"] for m in requests.get(f"{base}/api/tags", timeout=30).json()["models"]}
    return {"ollama": version, "digests": {m: tags.get(m) or tags.get(f"{m}:latest") for m in models}}


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} exited with {result.returncode}")
    return result.stdout


def committed_problem(paths, root: Path = ROOT) -> str | None:
    """None when every path has the bytes of its blob at HEAD; any git error is a refusal."""
    try:
        for p in paths:
            blob = git(root, "cat-file", "blob", f"HEAD:{p}")
            if not (root / p).is_file() or (root / p).read_bytes() != blob:
                return f"{p} differs from the last commit"
    except RuntimeError as e:
        return str(e)
    return None


def preflight(set_name: str, arm: str, client_info=server_info, root: Path = ROOT) -> dict:
    """Checks before an invocation; returns its header."""
    phase = SETS[set_name][1]
    models = roster(set_name, arm)
    manifest_sha = sha256(root / "data" / "replication_manifest.json")
    if manifest_sha != MANIFEST_SHA256:
        raise RuntimeError("the manifest does not match the protocol")
    problem = committed_problem((*FROZEN_FILES, *FROZEN_CODE), root)
    if problem:
        raise RuntimeError(problem)
    header = {"invocation": uuid.uuid4().hex, "phase": phase, "set": set_name, "arm": arm, "models": list(models),
              "manifest_sha256": manifest_sha, "commit": git(root, "rev-parse", "HEAD").decode().strip(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if phase in ("test", "format"):
        from .replication_analysis import threshold_problem

        problem = threshold_problem(root)
        if problem:
            raise RuntimeError(f"thresholds: {problem}")
        header["thresholds_sha256"] = sha256(root / THRESHOLDS)
    info = client_info(models)
    problem = environment_problem(info, models)
    if problem:
        raise RuntimeError(problem)
    return {**header, **info}


def main():
    parser = argparse.ArgumentParser(prog="python -m xreview.replication")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help=f"write {MANIFEST.relative_to(ROOT)}")
    r = sub.add_parser("run", help="calls of one set into the ledger of its phase")
    r.add_argument("set", choices=sorted(SETS))
    r.add_argument("--arm", choices=["labelled", "bare"], default="labelled")
    args = parser.parse_args()
    if args.command == "manifest":
        october = json.loads((DATA / "sample.json").read_text())
        MANIFEST.write_text(json.dumps(manifest(massive_rows(), snips_rows(), october, GATE_IDS), indent=1) + "\n")
        return
    header = preflight(args.set, args.arm)
    name, phase = SETS[args.set]
    ids = load_manifest()["sets"][args.set]
    rows = massive_rows() if name == "massive" else snips_rows()
    result = run(ollama_chat, header, build(name, rows), ids, ROOT / LEDGERS[phase],
                 thresholds_sha=header.get("thresholds_sha256"))
    print(json.dumps(result))
    if result["exhausted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
