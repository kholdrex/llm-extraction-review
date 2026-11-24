"""SNIPS (Coucke et al., 2018) in the train/valid/test split of the SlotGated-SLU repository, as gold records."""

import hashlib
import urllib.request
from pathlib import Path

from .data import Utterance

COMMIT = "941bc27b6312d110345bedc036c6b533f4b217a7"
URL = "https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/{commit}/data/snips/{split}/{name}"
SPLITS = {"train": "train", "valid": "dev", "test": "test"}
FILES = ("seq.in", "seq.out", "label")
SHA256 = {
    "train/seq.in": "04ad1e3f2121d83c6478f9a4b17d2dbdc21d06d6127f73cb7af835ed733f9855",
    "train/seq.out": "fd5f2121d8055827e5d6b3d3bac30f3d7e94e82a5d437751960cbdd9e1b4f6d3",
    "train/label": "bfd7df6cf9dd76c5da5d8b61bc85f90386197bdd3d7510c1d324c051537b962f",
    "valid/seq.in": "64e0f6d8ad8aab3a3828164d7d53731a0993d3ca1aed437ceaca6d3d568d410e",
    "valid/seq.out": "712e6d34feb8d52d208b820a5f8bae6e6ff990f08374507dfde22de389088319",
    "valid/label": "9c1dd041216f647bcfca838aaf1a48eb5781e4404256d31ff07cd38a5436c52f",
    "test/seq.in": "c1e341173fcea2951a8ca371956a6f81c76de87b9583c62eb87edd6158fa3a7c",
    "test/seq.out": "58c07c9097d5e1f78889765c2a535a160f6455b90b3142055c8051847afcca47",
    "test/label": "37a26da3ad079bb41c42c85a3a8f827eb47efec4c5917ce5d08f4b2736e4f830",
}


def download(dest: Path) -> Path:
    """The nine files under `dest`, fetched once at the pinned commit and checked by SHA-256."""
    for key, digest in SHA256.items():
        path = dest / key
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            split, name = key.split("/")
            urllib.request.urlretrieve(URL.format(commit=COMMIT, split=split, name=name), path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"unexpected checksum of {path}: {actual}")
    return dest


def slots_from_bio(words: list[str], tags: list[str]) -> tuple[tuple[str, str], ...]:
    """Typed slots from BIO tags: a B- tag starts a slot, I- tags of the same type extend it; an I- tag that does not
    continue a slot of its type starts a new one."""
    if len(words) != len(tags):
        raise ValueError(f"{len(words)} words but {len(tags)} tags")
    slots: list[tuple[str, list[str]]] = []
    previous = "O"
    for word, tag in zip(words, tags, strict=True):
        if tag == "O":
            previous = "O"
            continue
        prefix, kind = tag.split("-", 1)
        if prefix == "I" and previous == kind:
            slots[-1][1].append(word)
        else:
            slots.append((kind, [word]))
        previous = kind
    return tuple((kind, " ".join(value)) for kind, value in slots)


def load(dest: Path) -> list[Utterance]:
    """All utterances; the id is the split and the 0-based line number, the intent doubles as the scenario."""
    rows = []
    for split, partition in SPLITS.items():
        texts, tags, labels = ((dest / split / name).read_text(encoding="utf-8").splitlines() for name in FILES)
        if not len(texts) == len(tags) == len(labels):
            raise ValueError(f"line counts differ in {dest / split}")
        for i, (text, tag, label) in enumerate(zip(texts, tags, labels, strict=True)):
            words = text.split()
            rows.append(Utterance(f"{split}-{i}", partition, label, " ".join(words), label,
                                  slots_from_bio(words, tag.split())))
    return rows


def line(item_id: str) -> int:
    return int(item_id.rsplit("-", 1)[1])
