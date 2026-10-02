"""MASSIVE en-US utterances, gold intent/slot records and the ontology derived from the training split."""

import hashlib
import json
import random
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MEMBER = "1.1/data/en-US.jsonl"

SLOT_RE = re.compile(r"\[(\w+) : ([^\]]+)\]")


@dataclass(frozen=True)
class Utterance:
    id: str
    partition: str
    scenario: str
    text: str
    intent: str
    slots: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Ontology:
    intents: tuple[str, ...]
    slot_types: tuple[str, ...]
    allowed: dict[str, frozenset[str]]


def parse_annotation(annot: str) -> tuple[tuple[str, str], ...]:
    return tuple((t, v.strip()) for t, v in SLOT_RE.findall(annot))


def download(dest: Path) -> Path:
    archive = dest / "amazon-massive-dataset-1.1.tar.gz"
    if not archive.exists():
        dest.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MASSIVE_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != MASSIVE_SHA256:
        raise ValueError(f"unexpected checksum of {archive}: {digest}")
    return archive


def load(archive: Path) -> list[Utterance]:
    with tarfile.open(archive) as tar:
        lines = tar.extractfile(MEMBER).read().decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    return [
        Utterance(r["id"], r["partition"], r["scenario"], r["utt"], r["intent"], parse_annotation(r["annot_utt"]))
        for r in rows
    ]


def ontology(train: list[Utterance]) -> Ontology:
    allowed: dict[str, set[str]] = {}
    for u in train:
        allowed.setdefault(u.intent, set()).update(t for t, _ in u.slots)
    types = sorted({t for u in train for t, _ in u.slots})
    return Ontology(tuple(sorted(allowed)), tuple(types), {k: frozenset(v) for k, v in allowed.items()})


def examples(train: list[Utterance], seed: int) -> list[Utterance]:
    """One training utterance with at least one slot per scenario, drawn reproducibly."""
    rng = random.Random(seed)
    by_scenario: dict[str, list[Utterance]] = {}
    for u in train:
        if u.slots:
            by_scenario.setdefault(u.scenario, []).append(u)
    return [rng.choice(by_scenario[s]) for s in sorted(by_scenario)]


def sample(items: list[Utterance], n: int, seed: int) -> list[Utterance]:
    rng = random.Random(seed)
    return sorted(rng.sample(items, n), key=lambda u: int(u.id))
