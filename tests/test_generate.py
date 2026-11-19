import json

from app.generate import REFUSAL, generate_answer

CHUNKS = [
    {"title": "A", "content": "Deposits must be protected.", "updated_at": "2025-01-01"},
    {"title": "B", "content": "Landlords must give notice.", "updated_at": None},
]


def chat(payload):
    return lambda system, user: json.dumps(payload)


def test_valid_cited_answer_passes():
    out = generate_answer("q", CHUNKS, chat({"answerable": True, "answer": "Protect it [1]."}))
    assert out == {"answerable": True, "answer": "Protect it [1].", "cited": [1]}


def test_answer_without_citation_is_refused():
    out = generate_answer("q", CHUNKS, chat({"answerable": True, "answer": "Protect it."}))
    assert out["answerable"] is False and out["answer"] == REFUSAL


def test_invalid_citation_numbers_are_removed():
    out = generate_answer("q", CHUNKS, chat({"answerable": True, "answer": "Yes [1][7]."}))
    assert out["cited"] == [1] and "[7]" not in out["answer"]


def test_model_can_decline():
    out = generate_answer("q", CHUNKS, chat({"answerable": False, "answer": "no idea"}))
    assert out["answerable"] is False and out["answer"] == REFUSAL


def test_bad_json_is_refused():
    out = generate_answer("q", CHUNKS, lambda s, u: "not json")
    assert out["answerable"] is False
