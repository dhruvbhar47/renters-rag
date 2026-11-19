"""FastAPI app: POST /ask, GET /health, and (in production) the built React UI."""
import logging
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import answer_question

log = logging.getLogger("api")
app = FastAPI(title="Renters RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


# --- Cost protection for a public demo (in-memory: fine for one instance) ---------------------
_recent: dict[str, deque] = defaultdict(deque)
_daily = {"day": date.today(), "count": 0}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "?")


def _enforce_limits(ip: str) -> None:
    now = time.time()
    window = _recent[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "Too many questions. Wait a minute and try again.")

    today = date.today()
    if _daily["day"] != today:
        _daily.update(day=today, count=0)
    if _daily["count"] >= settings.daily_request_cap:
        raise HTTPException(429, "This demo has reached its daily question limit. Try again tomorrow.")

    window.append(now)
    _daily["count"] += 1


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskRequest, request: Request) -> dict:
    _enforce_limits(_client_ip(request))
    try:
        result = answer_question(body.question.strip())
    except Exception:  # noqa: BLE001 - never leak internals to the client
        log.exception("ask failed")
        raise HTTPException(502, "Something went wrong while answering. Please try again.")
    log.info("answered answerable=%s latency_ms=%s", result["answerable"], result["latency_ms"])
    return result


# Serve the built frontend when it exists (the Docker image copies it to ./static).
_STATIC = Path(__file__).resolve().parent.parent / "static"
if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
