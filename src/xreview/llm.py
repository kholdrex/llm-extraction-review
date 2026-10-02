"""Minimal client for the Ollama chat API."""

import time
from dataclasses import dataclass, field

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    tokens: list[tuple[str, float]] = field(default_factory=list)


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    seed: int = 0,
    logprobs: bool = False,
    max_tokens: int = 256,
    num_ctx: int = 4096,
) -> Completion:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "logprobs": logprobs,
        "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens, "num_ctx": num_ctx},
    }
    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=body, timeout=600)
    response.raise_for_status()
    seconds = time.perf_counter() - start
    data = response.json()
    tokens = [(t["token"], t["logprob"]) for t in data.get("logprobs") or []]
    if logprobs and "".join(t for t, _ in tokens) != data["message"]["content"]:
        raise ValueError(f"{model}: token log-probabilities missing or not aligned with the output")
    return Completion(
        data["message"]["content"], data.get("prompt_eval_count", 0), data.get("eval_count", 0), seconds, tokens
    )
