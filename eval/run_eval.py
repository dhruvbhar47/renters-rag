"""Measure retrieval quality and end-to-end answer quality.

Usage:
    python -m eval.run_eval                 # retrieval only: hit@k and MRR for vector / keyword / hybrid
    python -m eval.run_eval --generate      # also test refusals and answer faithfulness (LLM-as-judge)

Question file format (one JSON object per line):
    {"question": "...", "answerable": true,  "expected_url_contains": ["part-of-url"], "must_mention": ["optional", "keywords"]}
    {"question": "...", "answerable": false, "expected_url_contains": []}
"""
import argparse
import json
import statistics
from pathlib import Path

from app.config import settings
from app.openai_client import chat_json, embed
from app.pipeline import get_reranker, run_rag
from app.retrieval import retrieve
from app.store import PgStore, connect

JUDGE_SYSTEM = (
    "You are a strict fact-checker. Given SOURCES and an ANSWER, decide whether every factual claim "
    "in the ANSWER is supported by the SOURCES. Respond with JSON only: "
    '{"supported": true or false, "unsupported_claims": ["..."]}'
)


def load_questions(path: str) -> list[dict]:
    lines = Path(path).read_text().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip() and not ln.startswith("#")]


def retrieval_eval(store, questions, mode, k, rerank_fn=None) -> dict:
    hits, reciprocal_ranks, n = 0, [], 0
    for q in questions:
        if not q["answerable"]:
            continue
        n += 1
        res = retrieve(q["question"], store, embed, mode=mode, top_k=k, candidate_k=settings.candidate_k, rerank_fn=rerank_fn)
        rank = next(
            (i for i, c in enumerate(res.chunks, 1) if any(e in c["url"] for e in q["expected_url_contains"])),
            None,
        )
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return {
        "setup": mode + ("+rerank" if rerank_fn else ""),
        f"hit@{k}": round(hits / n, 3) if n else 0,
        "mrr": round(statistics.mean(reciprocal_ranks), 3) if n else 0,
    }


def generation_eval(store, questions) -> dict:
    correct_behaviour, faithful, judged, mention_ok, mention_n = 0, 0, 0, 0, 0
    failures = []
    for q in questions:
        out = run_rag(q["question"], store)
        if out["answerable"] == q["answerable"]:
            correct_behaviour += 1
        else:
            failures.append({"question": q["question"], "expected_answerable": q["answerable"], "got": out["answer"][:120]})
        if out["answerable"]:
            sources = "\n\n".join(f"[{i}] {c['content']}" for i, c in enumerate(out["chunks"], 1))
            verdict = json.loads(chat_json(JUDGE_SYSTEM, f"SOURCES:\n{sources}\n\nANSWER:\n{out['answer']}"))
            judged += 1
            faithful += bool(verdict.get("supported"))
            if q.get("must_mention"):
                mention_n += 1
                mention_ok += all(m.lower() in out["answer"].lower() for m in q["must_mention"])
    return {
        "answer_or_refuse_correct": round(correct_behaviour / len(questions), 3),
        "faithfulness": round(faithful / judged, 3) if judged else None,
        "must_mention_coverage": round(mention_ok / mention_n, 3) if mention_n else None,
        "failures": failures,
    }


def to_markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    with connect() as conn:
        store = PgStore(conn)
        rows = [retrieval_eval(store, questions, m, args.k) for m in ("vector", "keyword", "hybrid")]
        if settings.use_reranker:
            rows.append(retrieval_eval(store, questions, "hybrid", args.k, rerank_fn=get_reranker()))
        print("\nRetrieval\n" + to_markdown(rows))

        results = {"chunk_words": settings.chunk_words, "retrieval": rows}
        if args.generate:
            results["generation"] = generation_eval(store, questions)
            print("\nGeneration\n" + json.dumps(results["generation"], indent=2))

    Path("eval/results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved eval/results.json")


if __name__ == "__main__":
    main()
