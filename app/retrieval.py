"""Hybrid retrieval: vector + keyword search fused with Reciprocal Rank Fusion (RRF)."""
from dataclasses import dataclass
from typing import Callable, Protocol


class SearchStore(Protocol):
    def vector_search(self, query_vector: list[float], k: int) -> list[dict]: ...
    def keyword_search(self, query: str, k: int) -> list[dict]: ...


@dataclass
class Retrieval:
    chunks: list[dict]
    best_vector_score: float  # cosine similarity of the closest chunk; used to refuse off-topic questions


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Combine several ranked id lists. Each list adds 1 / (k + rank) to an id's score."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def retrieve(
    query: str,
    store: SearchStore,
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    mode: str = "hybrid",  # "vector" | "keyword" | "hybrid"
    top_k: int = 5,
    candidate_k: int = 20,
    rerank_fn: Callable[[str, list[dict]], list[dict]] | None = None,
) -> Retrieval:
    if mode not in {"vector", "keyword", "hybrid"}:
        raise ValueError(f"unknown mode: {mode}")

    vec_rows = store.vector_search(embed_fn([query])[0], candidate_k) if mode != "keyword" else []
    kw_rows = store.keyword_search(query, candidate_k) if mode != "vector" else []

    by_id: dict[int, dict] = {}
    for row in kw_rows:
        by_id[row["id"]] = row
    for row in vec_rows:
        by_id[row["id"]] = {**by_id.get(row["id"], {}), **row}

    if mode == "hybrid":
        fused = reciprocal_rank_fusion([[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]])
        ordered_ids = sorted(fused, key=fused.get, reverse=True)
    elif mode == "vector":
        ordered_ids = [r["id"] for r in vec_rows]
    else:
        ordered_ids = [r["id"] for r in kw_rows]

    candidates = [by_id[i] for i in ordered_ids]
    if rerank_fn and candidates:
        candidates = rerank_fn(query, candidates[:candidate_k])

    best = max((r["vector_score"] for r in vec_rows), default=0.0)
    return Retrieval(chunks=candidates[:top_k], best_vector_score=float(best))


def make_cross_encoder_reranker(model_name: str) -> Callable[[str, list[dict]], list[dict]]:
    """Optional second-stage reranker (needs requirements-rerank.txt and a model download)."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def rerank(query: str, candidates: list[dict]) -> list[dict]:
        scores = model.predict([(query, f"{c['title']}\n{c['content']}") for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [c for _, c in ranked]

    return rerank
