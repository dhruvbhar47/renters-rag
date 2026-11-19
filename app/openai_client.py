"""Thin wrappers around OpenAI-compatible APIs so the rest of the code stays testable.

Chat and embeddings use separate clients, so you can mix providers
(for example Groq for chat and a local Ollama model for embeddings).
"""
from functools import lru_cache

from openai import OpenAI

from .config import settings


@lru_cache(maxsize=1)
def _chat_client() -> OpenAI:
    # api_key / base_url of None make the SDK fall back to OPENAI_API_KEY / OPENAI_BASE_URL.
    return OpenAI(api_key=settings.chat_api_key, base_url=settings.chat_base_url)


@lru_cache(maxsize=1)
def _embed_client() -> OpenAI:
    return OpenAI(api_key=settings.embed_api_key, base_url=settings.embed_base_url)


def embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 64):
        batch = [t.replace("\n", " ") for t in texts[i : i + 64]]
        resp = _embed_client().embeddings.create(model=settings.embed_model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def chat_json(system: str, user: str) -> str:
    """Chat completion that must return a JSON object (returned as a string)."""
    resp = _chat_client().chat.completions.create(
        model=settings.chat_model,
        temperature=0,  # remove this line if you switch to a model that rejects it
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or "{}"