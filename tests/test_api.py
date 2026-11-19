from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def fake_answer(question):
    return {"answerable": True, "answer": "Yes [1].", "sources": [], "latency_ms": 1}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_pipeline_result(monkeypatch):
    monkeypatch.setattr(main, "answer_question", fake_answer)
    main._recent.clear()
    resp = client.post("/ask", json={"question": "How are deposits protected?"})
    assert resp.status_code == 200 and resp.json()["answer"] == "Yes [1]."


def test_short_question_is_rejected():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(main, "answer_question", fake_answer)
    main._recent.clear()
    limit = main.settings.rate_limit_per_minute
    codes = [client.post("/ask", json={"question": "How are deposits protected?"}).status_code for _ in range(limit + 1)]
    assert codes[-1] == 429


def test_errors_do_not_leak_details(monkeypatch):
    def boom(question):
        raise RuntimeError("secret db password")

    monkeypatch.setattr(main, "answer_question", boom)
    main._recent.clear()
    resp = client.post("/ask", json={"question": "How are deposits protected?"})
    assert resp.status_code == 502 and "secret" not in resp.text
