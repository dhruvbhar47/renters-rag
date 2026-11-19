from app.retrieval import reciprocal_rank_fusion, retrieve
from app.store import keyword_query


class FakeStore:
    def __init__(self, vec, kw):
        self.vec, self.kw = vec, kw

    def vector_search(self, query_vector, k):
        return self.vec[:k]

    def keyword_search(self, query, k):
        return self.kw[:k]


def row(i, **extra):
    return {"id": i, "url": f"https://x/{i}", "title": f"t{i}", "content": f"c{i}", **extra}


def fake_embed(texts):
    return [[0.0] for _ in texts]


def test_rrf_rewards_agreement_between_rankers():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 9]])
    assert scores[2] > scores[1]  # 2 is ranked by both, 1 only by one
    assert scores[3] > scores[9]


def test_hybrid_surfaces_chunk_found_by_both():
    vec = [row(1, vector_score=0.9), row(2, vector_score=0.8), row(3, vector_score=0.7)]
    kw = [row(3, keyword_score=2.0), row(2, keyword_score=1.0)]
    result = retrieve("q", FakeStore(vec, kw), fake_embed, mode="hybrid", top_k=2)
    # Chunks 2 and 3 appear in both rankings, so they beat chunk 1 (vector-only).
    assert [c["id"] for c in result.chunks] == [3, 2]
    assert result.best_vector_score == 0.9


def test_vector_mode_never_calls_keyword_search():
    class Boom(FakeStore):
        def keyword_search(self, query, k):
            raise AssertionError("keyword search should not run")

    vec = [row(1, vector_score=0.5)]
    result = retrieve("q", Boom(vec, []), fake_embed, mode="vector")
    assert result.chunks[0]["id"] == 1


def test_rerank_function_is_applied():
    vec = [row(1, vector_score=0.9), row(2, vector_score=0.8)]
    result = retrieve("q", FakeStore(vec, []), fake_embed, mode="vector", rerank_fn=lambda q, c: list(reversed(c)))
    assert result.chunks[0]["id"] == 2


def test_keyword_query_is_or_style_and_safe():
    q = keyword_query("How do I get my deposit back? '; DROP TABLE chunks;--")
    assert "deposit" in q and " | " in q
    assert ";" not in q and "'" not in q
