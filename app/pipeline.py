"""The full RAG flow: retrieve -> gate -> generate -> cite."""
import time
from functools import lru_cache

from .config import settings
from .generate import REFUSAL, generate_answer
from .openai_client import chat_json, embed
from .retrieval import make_cross_encoder_reranker, retrieve
from .store import PgStore, connect


@lru_cache(maxsize=1)
def get_reranker():
    if not settings.use_reranker:
        return None
    return make_cross_encoder_reranker(settings.reranker_model)


def run_rag(question: str, store, *, mode: str = "vector") -> dict:
    result = retrieve(
        question,
        store,
        embed,
        mode=mode,
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        rerank_fn=get_reranker(),
    )
    base = {"chunks": result.chunks, "best_score": result.best_vector_score}

    # Gate 1: nothing close enough in the corpus -> refuse without spending an LLM call.
    if not result.chunks or result.best_vector_score < settings.min_vector_score:
        return {"answerable": False, "answer": REFUSAL, "cited": [], **base}

    # Gate 2 (inside generate_answer): the model must cite real sources or we refuse.
    return {**generate_answer(question, result.chunks, chat_json), **base}


def answer_question(question: str) -> dict:
    start = time.perf_counter()
    with connect() as conn:
        out = run_rag(question, PgStore(conn))

    sources = [
        {
            "n": n,
            "title": out["chunks"][n - 1]["title"],
            "url": out["chunks"][n - 1]["url"],
            "updated_at": out["chunks"][n - 1].get("updated_at"),
            "snippet": out["chunks"][n - 1]["content"][:300],
        }
        for n in out["cited"]
    ]
    return {
        "answerable": out["answerable"],
        "answer": out["answer"],
        "sources": sources,
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }
